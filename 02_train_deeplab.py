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
OUTPUT_MODEL = "deeplabv3_finetuned.pth"
LOG_FILE = "deeplab_training.log"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_CLASSES = 17 # IGBP Land Cover has 17 classes (0-16)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class SEN12MS_SegmentationDataset(Dataset):
    def __init__(self, sar_json_path):
        with open(sar_json_path, "r") as f:
            self.sar_files = json.load(f)
            
    def __len__(self):
        return len(self.sar_files)
        
    def __getitem__(self, idx):
        # Infer Optical (s2) and Land Cover (lc) paths from SAR (s1) path
        s1_path = self.sar_files[idx]
        s2_path = s1_path.replace('_s1_', '_s2_').replace('/s1_', '/s2_').replace('\\s1_', '\\s2_')
        lc_path = s1_path.replace('_s1_', '_lc_').replace('/s1_', '/lc_').replace('\\s1_', '\\lc_')
        
        # Load Optical RGB (Sentinel-2 typically has RGB at bands 4, 3, 2, which is idx 3, 2, 1)
        # Note: If your pre-processed data only has 3 bands, we just take all 3.
        with rasterio.open(s2_path) as src:
            opt_img = src.read()
            if opt_img.shape[0] >= 3:
                # Attempt standard RGB extraction if 13-band, else assume it's already RGB
                if opt_img.shape[0] == 13:
                    opt_img = opt_img[[3, 2, 1], :, :] 
                else:
                    opt_img = opt_img[:3, :, :]
            opt_img = opt_img.astype(np.float32) / 10000.0 # Standard Sentinel-2 normalization
            opt_tensor = torch.from_numpy(opt_img)

        # Load Land Cover mask
        with rasterio.open(lc_path) as src:
            lc_mask = src.read(1) # shape (H, W)
            # Ensure labels are 0-16 for PyTorch CrossEntropyLoss
            lc_mask = np.clip(lc_mask - 1, 0, NUM_CLASSES - 1).astype(np.int64)
            lc_tensor = torch.from_numpy(lc_mask)
            
        return opt_tensor, lc_tensor

def get_model(num_classes):
    # Load pre-trained DeepLabV3 and modify the classifier head
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    # Modify aux classifier as well just in case
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting DeepLabV3 Fine-tuning on {device}")
    
    # Datasets and Loaders
    logging.info("Loading dataset splits...")
    train_dataset = SEN12MS_SegmentationDataset(TRAIN_SPLIT_JSON)
    val_dataset = SEN12MS_SegmentationDataset(VAL_SPLIT_JSON)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_model(NUM_CLASSES).to(device)
    
    # Auto-resume logic if laptop dies!
    if os.path.exists(OUTPUT_MODEL):
        logging.info(f"Found existing checkpoint {OUTPUT_MODEL}. Resuming training to save time!")
        model.load_state_dict(torch.load(OUTPUT_MODEL, map_location=device, weights_only=True))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255) # Ignore masking if needed
    scaler = torch.amp.GradScaler('cuda') # ADDED: Mixed Precision Scaler for 3x speedup!
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar in terminal
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for opt_imgs, lc_masks in pbar:
            opt_imgs = opt_imgs.to(device, non_blocking=True)
            lc_masks = lc_masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) # Faster than standard zero_grad
            
            # ADDED: Automatic Mixed Precision (AMP) uses the T4 GPU's Tensor Cores
            with torch.amp.autocast('cuda'):
                outputs = model(opt_imgs)['out']
                loss = criterion(outputs, lc_masks)
            
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
            for opt_imgs, lc_masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                opt_imgs, lc_masks = opt_imgs.to(device), lc_masks.to(device)
                outputs = model(opt_imgs)['out']
                loss = criterion(outputs, lc_masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Avg Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_MODEL)
            logging.info(f"Saved new best model with Val Loss: {best_val_loss:.4f}")

    logging.info("Training complete. Best model saved to " + OUTPUT_MODEL)

if __name__ == "__main__":
    main()
