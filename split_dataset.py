import os
import shutil
import random

# ============================
# ROOT PATH
# ============================
ROOT = r"C:\Users\user\Desktop\MAJOR PROJECT MODEL TRAINING\MODEL_DATASET"

IMG_DIR = os.path.join(ROOT, "images")
MASK_DIR = os.path.join(ROOT, "masks")

TRAIN_IMG = os.path.join(ROOT, "train", "images")
TRAIN_MASK = os.path.join(ROOT, "train", "masks")

VAL_IMG = os.path.join(ROOT, "val", "images")
VAL_MASK = os.path.join(ROOT, "val", "masks")

TEST_IMG = os.path.join(ROOT, "test", "images")
TEST_MASK = os.path.join(ROOT, "test", "masks")

for path in [TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK, TEST_IMG, TEST_MASK]:
    os.makedirs(path, exist_ok=True)

# ============================
# LIST ALL FILES
# ============================
files = sorted(os.listdir(IMG_DIR))
random.shuffle(files)

total = len(files)
train_n = int(total * 0.8)
val_n = int(total * 0.1)
test_n = total - train_n - val_n

train_files = files[:train_n]
val_files = files[train_n : train_n + val_n]
test_files = files[train_n + val_n :]

# ============================
# COPY FILES
# ============================
def copy_files(file_list, dst_img, dst_mask):
    for f in file_list:
        shutil.copy(os.path.join(IMG_DIR, f),  os.path.join(dst_img, f))
        shutil.copy(os.path.join(MASK_DIR, f), os.path.join(dst_mask, f))

print("\n🔁 Splitting dataset...")
copy_files(train_files, TRAIN_IMG, TRAIN_MASK)
copy_files(val_files,   VAL_IMG,   VAL_MASK)
copy_files(test_files,  TEST_IMG,  TEST_MASK)

# ============================
# PRINT SUMMARY
# ============================
print("\n====== 🎉 SPLIT COMPLETE (80/10/10) ======")
print(f"Total images: {total}")
print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")

print("\n🔍 Example train files:")
print(train_files[:10])

print("\n🔥 Final dataset saved inside MODEL_DATASET/")
