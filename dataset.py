import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

def get_valid_triplets(data_dir):
    print(f"Scanning dataset directory: {data_dir}")
    s1_files, s2_files, lc_files = [], [], []
    
    # Bulletproof scanner that ignores folder structures
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.tif'):
                full_path = os.path.join(root, f)
                if '_s1_' in f: s1_files.append(full_path)
                elif '_s2_' in f: s2_files.append(full_path)
                elif '_lc_' in f: lc_files.append(full_path)

    s2_dict = {f.split(os.sep)[-1].replace('_s2_', ''): f for f in s2_files}
    lc_dict = {f.split(os.sep)[-1].replace('_lc_', ''): f for f in lc_files}

    valid_triplets = []
    for s1_path in s1_files:
        base_id = s1_path.split(os.sep)[-1].replace('_s1_', '')
        if base_id in s2_dict and base_id in lc_dict:
            valid_triplets.append((s1_path, s2_dict[base_id], lc_dict[base_id]))
            
    print(f"Total Complete Triplets found: {len(valid_triplets)}")
    return valid_triplets

class SEN12MSDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        s1_path, s2_path, lc_path = self.triplets[idx]
        
        with rasterio.open(s1_path) as src_s1:
            s1_data = src_s1.read() 
            
        with rasterio.open(s2_path) as src_s2:
            s2_data = src_s2.read()
            
        # FILTER: No-Data Filter (>5% zeros in SAR)
        if np.sum(s1_data == 0) / s1_data.size > 0.05:
            # Skip corrupted patch and load a random valid one
            return self.__getitem__(np.random.randint(len(self.triplets)))

        # NORMALISATION: SAR
        vv = np.clip(s1_data[0], -25, 0)
        vv = (vv - (-25)) / (0 - (-25))
        vh = np.clip(s1_data[1], -32, -5)
        vh = (vh - (-32)) / (-5 - (-32))
        s1_tensor = torch.tensor(np.stack([vv, vh]), dtype=torch.float32)

        # NORMALISATION: Optical (B04, B03, B02)
        rgb = np.stack([s2_data[3], s2_data[2], s2_data[1]])
        rgb = np.clip(rgb, 0, 10000) / 10000.0
        s2_tensor = torch.tensor(rgb, dtype=torch.float32)

        return s1_tensor, s2_tensor
