import os
import json
import torch
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import the actual generator from the repo's models.py!
from models import GeneratorUNet 

# --- CONFIGURATION ---
WEIGHTS_PATH = "pix2pix_gen_180.pth" 
TEST_SPLIT_JSON = "./splits/test_files.json"
OUTPUT_DIR = Path("./fake_optical_test_set") 

def load_sar_image(file_path):
    with rasterio.open(file_path) as src:
        img = src.read()
        # SEN12MS SAR has 2 bands. We keep it as 2 bands since GeneratorUNet expects in_channels=2.
        img = torch.from_numpy(img).float()
    return img

def save_rgb_image(tensor, file_path):
    img_np = tensor.squeeze(0).cpu().numpy() 
    # GeneratorUNet outputs [0, 1]. We scale to 255 for standard TIFF viewing if needed, 
    # but the simplest is just saving it as float or uint8. Let's save as uint8 RGB.
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    with rasterio.open(
        file_path, 'w', driver='GTiff',
        height=img_np.shape[1], width=img_np.shape[2],
        count=3, dtype=img_np.dtype
    ) as dst:
        dst.write(img_np)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(TEST_SPLIT_JSON, "r") as f:
        test_sar_files = json.load(f)
    print(f"Loaded {len(test_sar_files)} SAR files for testing.")

    # Load Model
    print("Initializing GeneratorUNet...")
    model = GeneratorUNet(in_channels=2, out_channels=3).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True), strict=False)
    model.eval()

    with torch.no_grad():
        for sar_path in tqdm(test_sar_files, desc="Generating Fakes"):
            sar_tensor = load_sar_image(sar_path).unsqueeze(0).to(device)
            fake_optical = model(sar_tensor)
            
            filename = os.path.basename(sar_path).replace('_s1_', '_fake_opt_')
            save_rgb_image(fake_optical, OUTPUT_DIR / filename)

if __name__ == "__main__":
    main()
