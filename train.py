import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_valid_triplets, SEN12MSDataset
from models import GeneratorUNet, Discriminator

# --- CONFIGURATION ---
DATA_DIR = "./data/sen12ms"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 16 # Suitable for 16GB T4 GPU
EPOCHS = 150
LR = 0.0002
LAMBDA_PIXEL = 100

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- DATA LOADING ---
triplets = get_valid_triplets(DATA_DIR)
dataset = SEN12MSDataset(triplets)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- MODELS & OPTIMIZERS ---
G = GeneratorUNet().to(device)
D = Discriminator().to(device)

optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
criterion_GAN = nn.BCELoss()
criterion_pixelwise = nn.L1Loss()

# --- RESUME FROM CHECKPOINT (FOR SPOT INSTANCES) ---
start_epoch = 0
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
if os.path.exists(checkpoint_path):
    print("Found checkpoint! Resuming training...")
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from Epoch {start_epoch}")

# --- TRAINING LOOP ---
print("--- STARTING PIX2PIX TRAINING ---")
for epoch in range(start_epoch, EPOCHS):
    loop = tqdm(dataloader, leave=True)
    loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
    
    for i, (sar, optical_real) in enumerate(loop):
        sar, optical_real = sar.to(device), optical_real.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        optical_fake = G(sar)
        pred_fake = D(sar, optical_fake)
        
        valid = torch.ones_like(pred_fake, device=device)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_pixel = criterion_pixelwise(optical_fake, optical_real)
        loss_G = loss_GAN + (LAMBDA_PIXEL * loss_pixel)
        
        loss_G.backward()
        optimizer_G.step()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = D(sar, optical_real)
        loss_real = criterion_GAN(pred_real, valid)
        
        fake_labels = torch.zeros_like(pred_fake, device=device)
        loss_fake = criterion_GAN(D(sar, optical_fake.detach()), fake_labels)
        
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # Update progress bar
        loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

    # --- CHECKPOINT SAVING ---
    torch.save({
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)
    print(f"Saved Checkpoint for Epoch {epoch}")

print("Training Complete!")
