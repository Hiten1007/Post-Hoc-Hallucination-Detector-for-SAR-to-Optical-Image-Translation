import os
import json
import torch
import rasterio
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import logging
from pathlib import Path

from models import GeneratorUNet, Discriminator

# --- CONFIGURATION ---
TRAIN_SPLIT_JSON = "./splits/train_files.json"
VAL_SPLIT_JSON = "./splits/val_files.json"
CHECKPOINT_FILE = "pix2pix_checkpoint.pth"
FINAL_MODEL = "pix2pix_gen_global.pth"
OUTPUT_IMAGES = Path("./training_progress_images")

EPOCHS = 20
BATCH_SIZE = 16
LR = 0.0002
B1 = 0.5
B2 = 0.999
LAMBDA_PIXEL = 100

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SAR2OptDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.sar_files = json.load(f)
            
    def __len__(self): return len(self.sar_files)
    
    def __getitem__(self, idx):
        s1_path = self.sar_files[idx]
        s2_path = s1_path.replace('_s1_', '_s2_').replace('/s1_', '/s2_').replace('\\s1_', '\\s2_')
        
        with rasterio.open(s1_path) as src:
            sar = src.read().astype(np.float32)
            
        with rasterio.open(s2_path) as src:
            opt = src.read()
            if opt.shape[0] >= 3:
                opt = opt[[3, 2, 1], :, :] if opt.shape[0] == 13 else opt[:3, :, :]
            opt = opt.astype(np.float32) / 10000.0 # Scale [0, 1]
            
        return torch.from_numpy(sar), torch.from_numpy(opt)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting Pix2Pix Training on {device}")
    
    train_dataset = SAR2OptDataset(TRAIN_SPLIT_JSON)
    val_dataset = SAR2OptDataset(VAL_SPLIT_JSON)
    
    # Using 4 workers, memory pinning, and prefetching for massive T4 speedups
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    generator = GeneratorUNet(in_channels=2, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=5).to(device)
    
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(B1, B2))
    
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')
    
    start_epoch = 0
    
    # SAFE RESUME LOGIC (All 4 states explicitly loaded to prevent collapse!)
    if os.path.exists(CHECKPOINT_FILE):
        logging.info(f"Found checkpoint {CHECKPOINT_FILE}. Restoring ALL states to safely resume...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        discriminator.load_state_dict(checkpoint['disc_state_dict'])
        optimizer_G.load_state_dict(checkpoint['gen_optimizer'])
        optimizer_D.load_state_dict(checkpoint['disc_optimizer'])
        scaler_G.load_state_dict(checkpoint['scaler_G'])
        scaler_D.load_state_dict(checkpoint['scaler_D'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming from Epoch {start_epoch+1}")
        
    for epoch in range(start_epoch, EPOCHS):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for sar, real_opt in pbar:
            sar, real_opt = sar.to(device, non_blocking=True), real_opt.to(device, non_blocking=True)
            valid = torch.ones((sar.size(0), 1, 16, 16), device=device, requires_grad=False)
            fake = torch.zeros((sar.size(0), 1, 16, 16), device=device, requires_grad=False)
            
            # --- Train Generator ---
            optimizer_G.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                fake_opt = generator(sar)
                pred_fake = discriminator(sar, fake_opt)
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_pixel = criterion_pixel(fake_opt, real_opt)
                loss_G = loss_GAN + LAMBDA_PIXEL * loss_pixel
                
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pred_real = discriminator(sar, real_opt)
                loss_real = criterion_GAN(pred_real, valid)
                pred_fake = discriminator(sar, fake_opt.detach())
                loss_fake = criterion_GAN(pred_fake, fake)
                loss_D = 0.5 * (loss_real + loss_fake)
                
            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            
            pbar.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item(), L1=loss_pixel.item())
            
        # --- Validation & Visual Progress ---
        generator.eval()
        val_l1_loss = 0.0
        with torch.no_grad():
            for i, (sar, real_opt) in enumerate(val_loader):
                sar, real_opt = sar.to(device), real_opt.to(device)
                with torch.amp.autocast('cuda'):
                    fake_opt = generator(sar)
                    val_l1_loss += criterion_pixel(fake_opt, real_opt).item()
                
                # Save a visual grid from the first batch
                if i == 0:
                    img_sample = torch.cat((real_opt.data[:4], fake_opt.data[:4]), -2)
                    save_image(img_sample, OUTPUT_IMAGES / f"epoch_{epoch+1}.png", nrow=4, normalize=False)
                    
        val_l1_loss /= len(val_loader)
        logging.info(f"Epoch {epoch+1} Complete. Validation L1 Pixel Loss: {val_l1_loss:.4f}")
        
        # --- SAVE CHECKPOINT (All 4 components!) ---
        checkpoint = {
            'epoch': epoch,
            'gen_state_dict': generator.state_dict(),
            'disc_state_dict': discriminator.state_dict(),
            'gen_optimizer': optimizer_G.state_dict(),
            'disc_optimizer': optimizer_D.state_dict(),
            'scaler_G': scaler_G.state_dict(),
            'scaler_D': scaler_D.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_FILE)
        logging.info("Checkpoint saved safely (Generator, Discriminator, and both Optimizers).")

    # Export final lightweight weights for the detector script
    torch.save(generator.state_dict(), FINAL_MODEL)
    logging.info(f"Training finished! Best generator saved to {FINAL_MODEL}")

if __name__ == "__main__":
    main()
