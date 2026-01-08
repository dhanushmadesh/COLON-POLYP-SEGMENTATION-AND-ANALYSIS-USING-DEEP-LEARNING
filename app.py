import os
import io
import zipfile
import traceback
import time  

import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from skimage.measure import label, regionprops
import pandas as pd



# ==============================
# CPU PERFORMANCE FIX (IMPORTANT)
# ==============================
CPU = os.cpu_count() or 4
torch.set_num_threads(max(1, CPU // 2))
torch.set_num_interop_threads(1)

torch.set_grad_enabled(False)
torch.backends.mkldnn.enabled = True

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "model/b7_unetpp.pth"
IMG_SIZE = 512
THRESH = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# LOAD MODEL
# ==============================
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b7",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    # Load trained weights safely
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        # Backward compatibility for older checkpoints
        state = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ==============================
# LOAD MODEL
# ==============================
model = load_model()

# ==============================
# MODEL WARM-UP (IMPORTANT)
# ==============================
with torch.inference_mode():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    _ = model(dummy)

print("Model loaded and warmed up")



# ==============================
# TRANSFORM
# ==============================
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])


# ==============================
# HELPERS
# ==============================

def smooth_mask(mask_small):
    mask_small = cv2.GaussianBlur(mask_small, (9, 9), 0)
    mask_small = (mask_small > 127).astype("uint8") * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_small


def overlay_mask_on_image(original, mask_img, alpha=0.4):
    original_np = np.array(original.convert("RGB"))
    mask_np = np.array(mask_img)

    red = [255, 0, 0]
    faint_green = [100, 180, 100]

    color_mask = np.zeros_like(original_np)
    color_mask[mask_np <= 127] = faint_green
    color_mask[mask_np > 127] = red

    blended = cv2.addWeighted(original_np, 0.6, color_mask, alpha, 0)
    return Image.fromarray(blended)


def analyze_mask(mask_array, threshold=100):
    binary = (mask_array > 127).astype(np.uint8)
    lbl = label(binary)
    props = regionprops(lbl)

    total_area, count = 0, 0
    features = []

    for p in props:
        if p.area < threshold:
            continue
        count += 1
        total_area += p.area
        features.append({
            "Area": p.area,
            "Perimeter": round(p.perimeter, 2),
            "Solidity": round(p.solidity, 3),
            "Eccentricity": round(p.eccentricity, 3),
            "Aspect Ratio": round((p.bbox[2]-p.bbox[0]) / max(1, p.bbox[3]-p.bbox[1]), 3),
        })

    return total_area, count, features


def predict_mask_pytorch(image_pil):
    start = time.time()

    img_np = np.array(image_pil)
    aug = transform(image=img_np)
    t = aug["image"].unsqueeze(0).contiguous().to(DEVICE, dtype=torch.float32)

    with torch.inference_mode():
        pred = model(t)
        prob = torch.sigmoid(pred).squeeze().cpu().numpy()

    mask_small = (prob > THRESH).astype("uint8") * 255
    mask_small = smooth_mask(mask_small)

    mask_pil_small = Image.fromarray(mask_small)
    mask_full_res = mask_pil_small.resize(image_pil.size, Image.NEAREST)

    total_time = time.time() - start
    print(f"Total time per image: {total_time:.2f} seconds")

    return mask_full_res, prob



# ==============================
# MAIN PROCESSING
# ==============================
def process_images(files):
    numeric_rows = []
    image_rows = []

    buffer_zip = io.BytesIO()
    with zipfile.ZipFile(buffer_zip, "w") as zf:
        for file in files:
            image = Image.open(file).convert("RGB")
            w, h = image.size
            area_img = w * h
            image_size_str = f"{w}×{h}"  # Format as "width×height"

            mask, prob = predict_mask_pytorch(image)
            mask_np = np.array(mask)

            total_area, count, feat = analyze_mask(mask_np)
            coverage = (total_area / area_img) * 100 if area_img > 0 else 0.0

            polyp_pixels = prob[prob > THRESH]
            conf = float(np.mean(polyp_pixels) * 100) if polyp_pixels.size else 0.0

            name = os.path.basename(file)
            base, _ = os.path.splitext(name)
            outname = f"{base}_mask.png"

            buf = io.BytesIO()
            mask.save(buf, format="PNG")
            zf.writestr(outname, buf.getvalue())

            if total_area > 0 and feat:
                ws = round(sum(f["Solidity"] * f["Area"] for f in feat) / total_area, 3)
                we = round(sum(f["Eccentricity"] * f["Area"] for f in feat) / total_area, 3)
            else:
                ws, we = 0.0, 0.0

            dominant = max(feat, key=lambda f: f["Area"]) if feat else None
            dominant_ar = dominant["Aspect Ratio"] if dominant else 0.0

            numeric_rows.append({
                "Image Name": name,
                "Image Size": image_size_str,  # NEW COLUMN
                "Polyp Count": count,
                "Polyp Area (px)": int(total_area),
                "Solidity": ws,
                "Eccentricity": we,
                "% Coverage": f"{coverage:.2f}",
                "Aspect Ratio": dominant_ar,
                "Confidence (%)": f"{conf:.2f}",
                # Hidden fields for sorting
                "_width": w,
                "_height": h,
                "_total_pixels": area_img
            })

            overlay = overlay_mask_on_image(image, mask)
            image_rows.append([
                [image, f"{name} - Original ({image_size_str})"],  # Show size in label too
                [overlay, f"{name} - Overlay"],
                [mask, f"{name} - Mask"]
            ])

    buffer_zip.seek(0)

    BASE_DIR = r"C:\Users\dhanu\Desktop\MAJOR PROJECT\Generated_files"
    os.makedirs(BASE_DIR, exist_ok=True)

    auto_csv_path = os.path.join(BASE_DIR, "polyp_summary.csv")
    auto_zip_path = os.path.join(BASE_DIR, "masks.zip")

    # SMART SORTING: Group similar sizes together for better comparison
    df = pd.DataFrame(numeric_rows)
    
    # Sort by total pixels (image area) - SMALL TO LARGE (recommended)
    df = df.sort_values("_total_pixels", ascending=True)
    
    # Remove hidden sorting columns from final display
    df_display = df.drop(["_width", "_height", "_total_pixels"], axis=1)
    
    df_display.to_csv(auto_csv_path, index=False)
    with open(auto_zip_path, "wb") as f:
        f.write(buffer_zip.getvalue())

    gallery_data = []
    for row in image_rows:
        for col in row:
            gallery_data.append(col)

    return gallery_data, df_display, auto_csv_path, auto_zip_path


def run_segmentation(files):
    try:
        if not files:
            return [], pd.DataFrame([{"Error": "No files uploaded"}]), None, None
        return process_images(files)
    except Exception as e:
        traceback.print_exc()
        return [], pd.DataFrame([{"Error": str(e)}]), None, None

# ==============================
# GRADIO UI
# ==============================

css = """
* { transition: 0.2s ease-in-out; }

/* ===============================
   BACKGROUND + GLOBAL TEXT
================================*/
.gradio-container {
    background: radial-gradient(circle at top, #1f2933 0, #05070b 55%, #000000 100%) !important;
    color: #e5e5e5 !important;
    font-family: 'Inter', sans-serif;
}

/* ===============================
   HERO HEADER
================================*/
.hero-header {
    width: 100%;
    padding: 40px 20px;
    margin: 10px auto 30px auto;
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(148,163,184,0.28);
    border-radius: 18px;
    box-shadow: 0 25px 60px rgba(30,64,175,0.55);
    text-align: center;
}

.hero-header h1 {
    font-size: 48px;
    font-weight: 900;
    color: #f0f4ff;
    letter-spacing: 0.02em;
    text-shadow: 0 0 12px rgba(59,130,246,1),
                 0 0 22px rgba(59,130,246,0.8),
                 0 0 32px rgba(59,130,246,0.6);
    margin: 0;
}

/* ===============================
   DESCRIPTION BOX
================================*/
.desc-box {
    padding: 25px;
    border-radius: 16px;
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 12px 35px rgba(30,64,175,0.3);
    font-size: 18px;
    height: 100%;
    color: #e5e7eb;
}

/* ===============================
   UPLOAD BOX
================================*/
.upload-box {
    border: 2px dashed rgba(148,163,184,0.8) !important;
    background: rgba(15,23,42,0.9) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

.upload-box .gr-button {
    background: radial-gradient(circle at top left, #1d4ed8, #0f172a) !important;
    color: white !important;
    border: 1px solid rgba(59,130,246,0.9) !important;
}

/* ===============================
   PRIMARY BUTTON
================================*/
.primary-btn {
    background: radial-gradient(circle at top left, #1d4ed8, #0f172a) !important;
    color: white !important;
    border: 1px solid rgba(59,130,246,0.9) !important;
    border-radius: 999px !important;
    padding: 16px !important;
    font-weight: 800 !important;
    font-size: 18px !important;
    box-shadow: 0 14px 40px rgba(37,99,235,0.6) !important;
    transition: all 0.2s ease-in-out !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 55px rgba(59,130,246,0.9) !important;
}

/* ===============================
   ANALYSIS TABLE
================================*/
.gr-dataframe {
    background: rgba(15,23,42,0.45) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    box-shadow: 0 12px 35px rgba(30,64,175,0.3) !important;
}

.gr-dataframe table {
    background: transparent !important;
    color: #e5e7eb !important;
}

.gr-dataframe thead tr th {
    background: rgba(59,130,246,0.15) !important;
    color: #cfe1ff !important;
    font-weight: 700 !important;
    border: none !important;
}

.gr-dataframe tbody tr td {
    background: transparent !important;
    color: #e5e7eb !important;
    border-color: rgba(148,163,184,0.2) !important;
}

/* ===============================
   DOWNLOAD BUTTONS
================================*/
.download-btn {
    background: rgba(15,23,42,0.95) !important;
    border: 1px solid rgba(148,163,184,0.7) !important;
    color: #dbeafe !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

.download-btn:hover {
    background: rgba(25,35,55,0.95) !important;
    transform: translateY(-3px) !important;
}

/* ===============================
   GALLERY STYLING
================================*/
.gr-gallery {
    background: rgba(15,23,42,0.45) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
}

/* ===============================
   LABEL STYLING
================================*/
.gr-label {
    color: #e5e7eb !important;
    font-weight: 600 !important;
}

/* ===============================
   HIDDEN ELEMENTS INITIALLY
================================*/
.hidden-initially {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

.hidden-initially.visible {
    display: block !important;
    opacity: 1 !important;
    visibility: visible !important;
    transition: opacity 0.3s ease !important;
}
"""

with gr.Blocks(css=css, title="Colon Polyp Segmentation & Analysis") as demo:

    # Header with proper hero styling
    gr.HTML("""
    <div class="hero-header">
        <h1>Colon Polyp Segmentation and Analysis</h1>
    </div>
    """)

    # 2-column layout
    with gr.Row():

        # LEFT - Description box
        with gr.Column(scale=7):
            gr.HTML("""
            <div class="desc-box">
            <b>Project Overview:</b><br><br>
            Colon polyps are abnormal tissue growths in the colon that require early
            identification and detailed morphological analysis during colonoscopic
            examination. Accurate identification and
            morphological analysis of polyps—such as size, shape, and structure—are
            critical for early diagnosis, treatment planning, and risk assessment.<br><br>

            <b>How to use this tool:</b>
            <ol>
            <li>Upload one or more colonoscopy images.</li>
            <li>Click on <b>Run Segmentation</b> to start the analysis.</li>
            <li>View the generated segmentation mask and overlay.</li>
            <li>Analyze quantitative metrics such as polyp area, count, shape features,
                and confidence score, and download the results.</li>
            </ol>
            </div>
        """)


        # RIGHT - Upload + button
        with gr.Column(scale=4):
            file_input = gr.File(
                file_types=[".jpg", ".jpeg", ".png"],
                type="filepath",
                label="Upload Image",
                file_count="multiple",
                elem_classes=["upload-box"]
            )

            run_btn = gr.Button(
                "Run Segmentation",
                elem_classes=["primary-btn"]
            )

    # ANALYSIS TABLE FULL WIDTH - HIDDEN INITIALLY
    with gr.Row(visible=False) as results_row:
        with gr.Column():
            results_table = gr.Dataframe(
                wrap=True,
                interactive=False,
                label="Analysis Results",
                elem_classes=["analysis-table"]
            )

    # GALLERY - HIDDEN INITIALLY
    with gr.Row(visible=False) as gallery_row:
        with gr.Column():
            gallery = gr.Gallery(
                columns=3,
                rows=5,
                label="Segmentation Results",
                show_label=True
            )

    # DOWNLOAD BUTTONS (bottom) - HIDDEN INITIALLY
    with gr.Row(visible=False) as download_row:
        with gr.Column(scale=1):
            zip_download = gr.File(
                label="Download Masks (ZIP)",
                elem_classes=["download-btn"]
            )
        with gr.Column(scale=1):
            csv_download = gr.File(
                label="Download Analysis (CSV)",
                elem_classes=["download-btn"]
            )

    # Function to show results after processing
    def show_results(gallery_data, df_display, csv_path, zip_path):
        # Return all outputs and also make the rows visible
        return (
            gallery_data, 
            df_display, 
            csv_path, 
            zip_path,
            gr.Row(visible=True),  # results_row
            gr.Row(visible=True),  # gallery_row
            gr.Row(visible=True)   # download_row
        )

    # BUTTON ACTION - UPDATED
    run_btn.click(
        fn=run_segmentation,
        inputs=file_input,
        outputs=[gallery, results_table, csv_download, zip_download]
    ).then(
        fn=lambda g, d, c, z: (
            g, d, c, z, 
            gr.Row(visible=True),  # Show results table
            gr.Row(visible=True),  # Show gallery
            gr.Row(visible=True)   # Show download buttons
        ),
        inputs=[gallery, results_table, csv_download, zip_download],
        outputs=[gallery, results_table, csv_download, zip_download, results_row, gallery_row, download_row]
    )


if __name__ == "__main__":
    print("Starting Colon Polyp App...")
    demo.launch(inbrowser=True, debug=False, share=False)



