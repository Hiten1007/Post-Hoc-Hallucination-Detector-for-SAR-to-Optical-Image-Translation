"""
Universal Fake Image Generator for All Models
Generates fake optical images from SAR inputs using any trained generator model.
Supports: pix2pix, cyclegan, palette

Usage:
    python3 generate_new_fakes.py --model pix2pix
    python3 generate_new_fakes.py --model cyclegan
    python3 generate_new_fakes.py --model palette
"""

import os
import sys
import json
import argparse
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Import Pix2Pix generator from the existing models.py
from models import GeneratorUNet

# ============================
# CycleGAN Generator (must match training architecture exactly)
# ============================

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )
    def forward(self, x):
        return x + self.block(x)

class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_residual_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7), nn.Sigmoid()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

# ============================
# Palette UNet (must match training architecture exactly)
# ============================

import math
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bij,bcj->bci', attn, v).reshape(B, C, H, W)
        return x + self.proj(out)

class PaletteUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.enc1 = ConvBlock(in_channels, 64, time_emb_dim)
        self.enc2 = ConvBlock(64, 128, time_emb_dim)
        self.enc3 = ConvBlock(128, 256, time_emb_dim)
        self.enc4 = ConvBlock(256, 512, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.bot = ConvBlock(512, 1024, time_emb_dim)
        self.attn = AttentionBlock(1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64, time_emb_dim)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)
        b = self.bot(self.pool(e4), t_emb)
        b = self.attn(b)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.final(d1)


# ============================
# DIFFUSION SAMPLING UTILITIES
# ============================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def get_diffusion_params(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }

@torch.no_grad()
def palette_sample(model, sar, diffusion_params, device, T=1000, num_steps=100):
    """Generate optical image from SAR using iterative denoising."""
    b = sar.shape[0]
    img = torch.randn(b, 3, 256, 256, device=device)
    step_size = T // num_steps
    timesteps = list(range(T - 1, -1, -step_size))

    for t_val in timesteps:
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        model_input = torch.cat([img, sar], dim=1)
        predicted_noise = model(model_input, t)
        betas_t = diffusion_params['betas'][t_val].to(device)
        sqrt_one_minus_t = diffusion_params['sqrt_one_minus_alphas_cumprod'][t_val].to(device)
        sqrt_recip_t = diffusion_params['sqrt_recip_alphas'][t_val].to(device)
        model_mean = sqrt_recip_t * (img - betas_t / sqrt_one_minus_t * predicted_noise)
        if t_val > 0:
            posterior_var = diffusion_params['posterior_variance'][t_val].to(device)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_var) * noise
        else:
            img = model_mean

    return img.clamp(0, 1)


# ============================
# MAIN GENERATION LOGIC
# ============================

def main():
    parser = argparse.ArgumentParser(description="Generate fake optical images from SAR using a trained model.")
    parser.add_argument("--model", type=str, required=True, choices=["pix2pix", "cyclegan", "palette"],
                        help="Which generator model to use.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. If not specified, uses default for the model.")
    parser.add_argument("--inference-steps", type=int, default=100,
                        help="Number of denoising steps for Palette (ignored for GAN models).")
    args = parser.parse_args()

    # Default weight paths
    weight_defaults = {
        "pix2pix": "pix2pix_gen_global.pth",
        "cyclegan": "cyclegan_gen_global.pth",
        "palette": "palette_gen_global.pth",
    }
    weights_path = args.weights or weight_defaults[args.model]

    # Output directory (model-specific)
    output_dir = Path(f"./fake_optical_test_set_{args.model}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating fake optical images using [{args.model.upper()}] on {device}")
    print(f"Loading weights from: {weights_path}")

    # Load test split
    with open("./splits/test_files.json", "r") as f:
        test_files = json.load(f)

    # Load the appropriate model
    if args.model == "pix2pix":
        generator = GeneratorUNet(in_channels=2, out_channels=3).to(device)
        generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        generator.eval()

    elif args.model == "cyclegan":
        generator = CycleGANGenerator(in_channels=2, out_channels=3).to(device)
        generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        generator.eval()

    elif args.model == "palette":
        generator = PaletteUNet(in_channels=5, out_channels=3).to(device)
        generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        generator.eval()
        betas = linear_beta_schedule(1000)
        diffusion_params = get_diffusion_params(betas)

    # Generate fake optical images for every test SAR file
    with torch.no_grad():
        for s1_path in tqdm(test_files, desc=f"Generating [{args.model}] fakes"):
            filename = os.path.basename(s1_path)
            out_filename = filename.replace('_s1_', '_fake_opt_')
            out_path = output_dir / out_filename

            if out_path.exists():
                continue  # Skip already generated files (resume-safe)

            # Load SAR image
            with rasterio.open(s1_path) as src:
                sar = src.read().astype(np.float32)
                profile = src.profile.copy()

            sar_tensor = torch.from_numpy(sar).unsqueeze(0).to(device)

            # Generate fake optical
            if args.model in ("pix2pix", "cyclegan"):
                fake_opt = generator(sar_tensor).squeeze(0).cpu().numpy()
            elif args.model == "palette":
                fake_opt = palette_sample(generator, sar_tensor, diffusion_params, device,
                                          num_steps=args.inference_steps).squeeze(0).cpu().numpy()

            # Scale back to uint16 range [0, 10000] to match real S2 data format
            fake_opt = (fake_opt * 10000.0).clip(0, 65535).astype(np.uint16)

            # Write output .tif
            profile.update(count=3, dtype='uint16')
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(fake_opt)

    print(f"\nDone! {len(test_files)} fake optical images saved to: {output_dir}")


if __name__ == "__main__":
    main()
