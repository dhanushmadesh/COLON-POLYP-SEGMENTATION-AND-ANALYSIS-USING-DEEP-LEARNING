import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# =========================================================
# CONFIG
# =========================================================
IMG_HEIGHT, IMG_WIDTH = 512, 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IMG_DIR  = r"C:\Users\user\Desktop\MAJOR PROJECT MODEL TRAINING\MODEL_DATASET\test\images"
TEST_MASK_DIR = r"C:\Users\user\Desktop\MAJOR PROJECT MODEL TRAINING\MODEL_DATASET\test\masks"

MODEL_PATH = r"model/b7_unetpp.pth"

SAVE_DIR = "EVALUATION_RESULTS"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/samples", exist_ok=True)

# =========================================================
# MODEL LOADING
# =========================================================
def load_model():
    print("Loading the model for TEST evaluation...")
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b7",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# =========================================================
# PREPROCESSING
# =========================================================
transform = Compose([
    Resize(IMG_HEIGHT, IMG_WIDTH),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def load_test_data():
    img_paths  = sorted(glob(os.path.join(TEST_IMG_DIR, "*")))
    mask_paths = sorted(glob(os.path.join(TEST_MASK_DIR, "*")))
    return img_paths, mask_paths

# =========================================================
# METRICS
# =========================================================
def compute_metrics(pred, mask):
    tp = ((pred == 1) & (mask == 1)).sum()
    fp = ((pred == 1) & (mask == 0)).sum()
    fn = ((pred == 0) & (mask == 1)).sum()

    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    f1  = (2 * tp) / (2 * tp + fp + fn + eps)

    return iou, f1

# =========================================================
# EVALUATION LOOP
# =========================================================
def evaluate():
    model = load_model()

    img_paths, mask_paths = load_test_data()
    print(f"Total TEST images: {len(img_paths)}")

    all_preds = []
    all_targets = []

    total_iou, total_f1 = 0, 0

    for idx, (img_path, mask_path) in enumerate(tqdm(zip(img_paths, mask_paths), total=len(img_paths))):

        # Read image + mask
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask_bin = (mask > 127).astype("uint8")

        # Transform
        aug = transform(image=img_rgb, mask=mask_bin)
        image = aug["image"].unsqueeze(0).float().to(DEVICE)
        mask_t = aug["mask"].reshape(-1)

        with torch.no_grad():
            pred = model(image)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            pred_np = pred.cpu().numpy().reshape(-1)

        # Metrics
        iou, f1 = compute_metrics(pred_np, mask_t.numpy())
        total_iou += iou
        total_f1 += f1

        all_preds.extend(pred_np)
        all_targets.extend(mask_t.numpy())

        # Save first 10 prediction overlays
        if idx < 10:
            pred_mask = pred.cpu().numpy()[0, 0]
            pred_mask = (pred_mask * 255).astype("uint8")

            # Resize predicted mask to original image size
            orig_h, orig_w = img.shape[:2]
            pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h))

            overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            colored_mask = np.zeros_like(overlay)
            colored_mask[:, :, 1] = pred_mask_resized

            result = cv2.addWeighted(overlay, 0.8, colored_mask, 0.4, 0)
            plt.imsave(f"{SAVE_DIR}/samples/sample_{idx}.png", result)

    # Average metrics
    avg_iou = total_iou / len(img_paths)
    avg_f1  = total_f1  / len(img_paths)

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png")
    plt.close()

    # =========================================================
    # CLASSIFICATION REPORT
    # =========================================================
    report = classification_report(all_targets, all_preds, digits=4)
    with open(f"{SAVE_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    # =========================================================
    # METRIC SUMMARY
    # =========================================================
    with open(f"{SAVE_DIR}/metrics_summary.txt", "w") as f:
        f.write(f"Test IoU: {avg_iou:.4f}\n")
        f.write(f"Test F1 : {avg_f1:.4f}\n")

    print("\n============================")
    print("  FINAL TEST RESULTS")
    print("============================")
    print(f"Test IoU: {avg_iou:.4f}")
    print(f"Test F1 : {avg_f1:.4f}")
    print("Results saved in:", SAVE_DIR)

# Run
if __name__ == "__main__":
    evaluate()
