import os
import glob
import json
import random
from pathlib import Path

# --- CONFIGURATION ---
# Change this to where you extracted the 510GB SEN12MS dataset on your EC2 instance
DATASET_ROOT = Path("./data/sen12ms") 
OUTPUT_DIR = Path("./splits")
random.seed(42) # Fixed seed for thesis reproducibility!

def get_all_rois(base_path):
    """Finds all unique ROI folders in the SEN12MS dataset."""
    rois = []
    print("Scanning dataset for Sentinel-1 files (this may take a minute)...")
    s1_files = glob.glob(os.path.join(base_path, "**", "s1_*", "*.tif"), recursive=True)
    
    roi_dict = {}
    for f in s1_files:
        basename = os.path.basename(f)
        parts = basename.split('_s1_')
        if len(parts) == 2:
            season_roi = parts[0] + "_" + parts[1].split('_p')[0]
            if season_roi not in roi_dict:
                roi_dict[season_roi] = []
            roi_dict[season_roi].append(f)
            
    print(f"Found {len(roi_dict)} unique geographic ROIs containing {len(s1_files)} patches.")
    return list(roi_dict.keys()), roi_dict

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_rois, roi_files_map = get_all_rois(DATASET_ROOT)
    
    # Shuffle ROIs geographically
    random.shuffle(unique_rois)
    
    # 80 / 10 / 10 Split
    num_rois = len(unique_rois)
    train_split_idx = int(num_rois * 0.8)
    val_split_idx = int(num_rois * 0.9)
    
    train_rois = unique_rois[:train_split_idx]
    val_rois = unique_rois[train_split_idx:val_split_idx]
    test_rois = unique_rois[val_split_idx:]
    
    splits = {"train": [], "val": [], "test": []}
    
    for roi in train_rois: splits["train"].extend(roi_files_map[roi])
    for roi in val_rois: splits["val"].extend(roi_files_map[roi])
    for roi in test_rois: splits["test"].extend(roi_files_map[roi])
        
    for split_name in ["train", "val", "test"]:
        with open(OUTPUT_DIR / f"{split_name}_files.json", "w") as f:
            json.dump(splits[split_name], f)
            
    print(f"Splits saved to {OUTPUT_DIR}/")
    print(f"Train: {len(splits['train'])} patches")
    print(f"Val:   {len(splits['val'])} patches")
    print(f"Test:  {len(splits['test'])} patches (We will generate fakes for these!)")

if __name__ == "__main__":
    main()
