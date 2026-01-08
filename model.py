import os
from glob import glob  
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


IMG_HEIGHT, IMG_WIDTH = 512, 512
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = r"C:\Users\user\Desktop\MAJOR PROJECT MODEL TRAINING\MODEL_DATASET\train\images"
TRAIN_MASK_DIR = r"C:\Users\user\Desktop\MAJOR PROJECT MODEL TRAINING\MODEL_DATASET\train\masks"
VAL_IMG_DIR   = r"C:\Users\user\Desktop\MAJOR PROJECT MODEL TRAINING\MODEL_DATASET\val\images"
VAL_MASK_DIR  = r"C:\Users\user\Desktop\MAJOR PROJECT MODEL TRAINING\MODEL_DATASET\val\masks"


class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*")))
        self.mask_paths  = sorted(glob(os.path.join(mask_dir, "*")))
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.Resize(IMG_HEIGHT, IMG_WIDTH),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.2),
                A.CLAHE(p=0.1),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                ),
                A.GaussianBlur(p=0.1),
                A.GridDistortion(p=0.05),

                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(IMG_HEIGHT, IMG_WIDTH),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")

        aug = self.transform(image=img, mask=mask)
        image = aug["image"].float()
        mask  = aug["mask"].float()

        mask = (mask > 0.5).float()
        mask = mask.unsqueeze(0)

        return image, mask


def build_model():
    return smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
focal_loss = smp.losses.FocalLoss(mode="binary")

def final_loss(pred, target):
    return (
        0.6 * dice_loss(pred, target) +
        0.2 * bce_loss(pred, target) +
        0.2 * focal_loss(pred, target)
    )

def calc_metrics(pred, target):
    prob = torch.sigmoid(pred)
    pred_bin = (prob > 0.5).float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    eps = 1e-7
    f1  = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return iou.item(), f1.item()


def train():
    print(f"Using device: {DEVICE}")

    train_dataset = PolypDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, augment=True)
    val_dataset   = PolypDataset(VAL_IMG_DIR, VAL_MASK_DIR, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_f1 = 0
    warmup_epochs = 5

    for epoch in range(1, EPOCHS + 1):

        if epoch <= warmup_epochs:
            for g in optimizer.param_groups:
                g['lr'] = LR * (epoch / warmup_epochs)

        if epoch == 1:
            for param in model.encoder.parameters():
                param.requires_grad = False

        if epoch == warmup_epochs + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True

        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Training]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds = model(imgs)
                loss = final_loss(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch)

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()

        val_loss = val_iou = val_f1 = 0
        n = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="Validation"):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

                preds = model(imgs)
                loss = final_loss(preds, masks)

                iou, f1 = calc_metrics(preds, masks)

                val_loss += loss.item()
                val_iou += iou
                val_f1  += f1
                n += 1

        avg_val_f1  = val_f1 / n
        avg_val_iou = val_iou / n

        print(f"Epoch {epoch}/{EPOCHS} | Val F1: {avg_val_f1:.4f} | IoU: {avg_val_iou:.4f}")

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            os.makedirs("model", exist_ok=True)
            torch.save(model.state_dict(), "model/b7_unetpp.pth")
            print("Saved new BEST model!")


if __name__ == "__main__":
    train()
