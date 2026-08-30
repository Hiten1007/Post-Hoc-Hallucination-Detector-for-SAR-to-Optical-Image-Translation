import os
import json
import torch
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import the Pix2Pix class from your model architecture
from model import Pix2Pix 

# --- CONFIGURATION ---
WEIGHTS_PATH = "pix2pix_gen_180.pth" 
TEST_SPLIT_JSON = "./splits/test_files.json"
OUTPUT_DIR = Path("./fake_optical_test_set") 

def load_sar_image(file_path):
    with rasterio.open(file_path) as src:
        img = src.read()
        if img.shape[0] == 2:
            img = np.concatenate([img, img[0:1, :, :]], axis=0) # Make it 3 channels
        img = torch.from_numpy(img).float()
    return img

def save_rgb_image(tensor, file_path):
    img_np = tensor.squeeze(0).cpu().numpy() 
    with rasterio.open(
        file_path, 'w', driver='GTiff',
        height=img_np.shape[1], width=img_np.shape[2],
        count=3, dtype=img_np.dtype
    ) as dst:
        dst.write(img_np)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load strict test split
    with open(TEST_SPLIT_JSON, "r") as f:
        test_sar_files = json.load(f)
    print(f"Loaded {len(test_sar_files)} SAR files for testing.")

    # Load Model
    print("Initializing Pix2Pix generator...")
    model = Pix2Pix(c_in=3, c_out=3, is_train=False).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True), strict=False)
    model.eval()

    with torch.no_grad():
        for sar_path in tqdm(test_sar_files, desc="Generating Fakes"):
            sar_tensor = load_sar_image(sar_path).unsqueeze(0).to(device)
            fake_optical = model.generate(sar_tensor, is_scaled=False, to_uint8=False)
            
            # Preserve original filename but mark it as fake optical
            filename = os.path.basename(sar_path).replace('_s1_', '_fake_opt_')
            save_rgb_image(fake_optical, OUTPUT_DIR / filename)

if __name__ == "__main__":
    main()
