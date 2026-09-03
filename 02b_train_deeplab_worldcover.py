"""
DeepLabV3 Fine-tuning on ESA WorldCover 10m Labels (11 classes)
Identical to 02_train_deeplab.py but uses _wc_ labels instead of _lc_ labels.
"""

import os
import json
import logging
import rasterio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
TRAIN_SPLIT_JSON = "./splits/train_files.json"
VAL_SPLIT_JSON = "./splits/val_files.json"
OUTPUT_MODEL = "deeplabv3_finetuned_worldcover.pth"
LOG_FILE = "deeplab_worldcover_training.log"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_CLASSES = 11  # WorldCover has 11 classes (0-10)

CLASS_NAMES = [
    "Tree Cover", "Shrubland", "Grassland", "Cropland",
    "Urban / Built-up", "Barren", "Snow and Ice", "Water Bodies",
    "Wetlands", "Mangroves", "Moss and Lichen"
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class SEN12MS_WorldCover_Dataset(Dataset):
    def __init__(self, sar_json_path):
        with open(sar_json_path, "r") as f:
            all_files = json.load(f)

        # Only keep patches that have a _wc_ label file
        self.sar_files = []
        for s1_path in all_files:
            wc_path = s1_path.replace('_s1_', '_wc_').replace('/s1_', '/wc_').replace('\\s1_', '\\wc_')
            if os.path.exists(wc_path):
                self.sar_files.append(s1_path)

        logging.info(f"Loaded {len(self.sar_files)} patches with WorldCover labels (out of {len(all_files)} total)")

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        s1_path = self.sar_files[idx]
        s2_path = s1_path.replace('_s1_', '_s2_').replace('/s1_', '/s2_').replace('\\s1_', '\\s2_')
        wc_path = s1_path.replace('_s1_', '_wc_').replace('/s1_', '/wc_').replace('\\s1_', '\\wc_')

        # Load Optical RGB
        with rasterio.open(s2_path) as src:
            opt_img = src.read()
            if opt_img.shape[0] >= 3:
                if opt_img.shape[0] == 13:
                    opt_img = opt_img[[3, 2, 1], :, :]
                else:
                    opt_img = opt_img[:3, :, :]
            opt_img = opt_img.astype(np.float32) / 10000.0
            opt_tensor = torch.from_numpy(opt_img)

        # Load WorldCover label (already 0-indexed, 0-10)
        with rasterio.open(wc_path) as src:
            wc_mask = src.read(1).astype(np.int64)
            wc_tensor = torch.from_numpy(wc_mask)

        return opt_tensor, wc_tensor


def get_model(num_classes):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting DeepLabV3 Fine-tuning on WorldCover 10m labels ({NUM_CLASSES} classes) on {device}")

    logging.info("Loading dataset splits...")
    train_dataset = SEN12MS_WorldCover_Dataset(TRAIN_SPLIT_JSON)
    val_dataset = SEN12MS_WorldCover_Dataset(VAL_SPLIT_JSON)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = get_model(NUM_CLASSES).to(device)

    # Auto-resume logic
    if os.path.exists(OUTPUT_MODEL):
        logging.info(f"Found existing checkpoint {OUTPUT_MODEL}. Resuming training!")
        model.load_state_dict(torch.load(OUTPUT_MODEL, map_location=device, weights_only=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignores unmapped pixels
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for opt_imgs, wc_masks in pbar:
            opt_imgs = opt_imgs.to(device, non_blocking=True)
            wc_masks = wc_masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(opt_imgs)['out']
                loss = criterion(outputs, wc_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Avg Train Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for opt_imgs, wc_masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                opt_imgs, wc_masks = opt_imgs.to(device), wc_masks.to(device)
                outputs = model(opt_imgs)['out']
                loss = criterion(outputs, wc_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_MODEL)
            logging.info(f"Saved new best model with Val Loss: {best_val_loss:.4f}")

    logging.info("Training complete. Best model saved to " + OUTPUT_MODEL)


if __name__ == "__main__":
    main()
