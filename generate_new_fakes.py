import os
import json
import torch
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from models import GeneratorUNet
def main():
    device = torch.device("cpu")
    print("Loading Epoch 6 Generator weights...")
    generator = GeneratorUNet(in_channels=2, out_channels=3).to(device)
    generator.load_state_dict(torch.load('pix2pix_gen_global.pth', map_location=device))
    generator.eval()
    with open('./splits/test_files.json', 'r') as f:
        test_files = json.load(f)
        
    out_dir = Path('./fake_optical_test_set')
    out_dir.mkdir(exist_ok=True)
    print(f"Generating new fake images for {len(test_files)} test files...")
    with torch.no_grad():
        for s1_path in tqdm(test_files):
            with rasterio.open(s1_path) as src:
                sar = src.read().astype(np.float32)
                profile = src.profile
                
            sar_tensor = torch.from_numpy(sar).unsqueeze(0).to(device)
            fake_opt = generator(sar_tensor).squeeze(0).cpu().numpy()
            
            fake_opt = np.clip((fake_opt * 10000.0), 0, 10000).astype(np.uint16)
            
            profile.update(count=3, dtype=rasterio.uint16)
            fake_name = os.path.basename(s1_path).replace('_s1_', '_fake_opt_')
            with rasterio.open(out_dir / fake_name, 'w', **profile) as dst:
                dst.write(fake_opt)
if __name__ == "__main__":
    main()
