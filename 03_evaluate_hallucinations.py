import os
import json
import logging
import rasterio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# --- CONFIGURATION ---
TEST_SPLIT_JSON = "./splits/test_files.json"
FAKE_OPTICAL_DIR = Path("./fake_optical_test_set")
DEEPLAB_WEIGHTS = "deeplabv3_finetuned.pth"
LOG_FILE = "hallucination_evaluation.log"
CSV_OUTPUT = "hallucination_benchmark.csv"

NUM_CLASSES = 17
EROSION_RADIUS = 2 # Excludes a 2-pixel margin around class boundaries to prevent false positives

# IGBP Land Cover Class Names (Simplified for readability)
CLASS_NAMES = [
    "Evergreen Needleleaf Forest", "Evergreen Broadleaf Forest", 
    "Deciduous Needleleaf Forest", "Deciduous Broadleaf Forest",
    "Mixed Forest", "Closed Shrublands", "Open Shrublands",
    "Woody Savannas", "Savannas", "Grasslands", "Permanent Wetlands",
    "Croplands", "Urban and Built-up", "Cropland/Natural Vegetation Mosaics",
    "Snow and Ice", "Barren or Sparsely Vegetated", "Water Bodies"
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def get_boundary_mask(label_mask, radius=2):
    """
    Implements the 2-pixel morphological erosion on class boundaries as defined in the proposal.
    Returns a boolean mask where True = VALID pixel, False = BOUNDARY pixel (to be ignored).
    """
    # Create a mask of class boundaries by shifting the image
    up = np.roll(label_mask, 1, axis=0)
    down = np.roll(label_mask, -1, axis=0)
    left = np.roll(label_mask, 1, axis=1)
    right = np.roll(label_mask, -1, axis=1)
    
    # Boundary is anywhere a pixel's class doesn't match its neighbor
    boundaries = (label_mask != up) | (label_mask != down) | (label_mask != left) | (label_mask != right)
    
    # Dilate the boundaries by the specified radius to create an exclusion zone
    exclusion_zone = binary_dilation(boundaries, iterations=radius)
    
    # Return the inverse (Valid pixels)
    return ~exclusion_zone

def load_rgb_tensor(path):
    with rasterio.open(path) as src:
        img = src.read()
        if img.shape[0] == 13:
            img = img[[3, 2, 1], :, :]
        elif img.shape[0] > 3:
            img = img[:3, :, :]
        
        # Scale to match DeepLab training norms
        if img.dtype != np.uint8:
            img = img.astype(np.float32) / 10000.0
        else:
            img = img.astype(np.float32) / 255.0
            
    return torch.from_numpy(img).unsqueeze(0) # Add batch dim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting Hallucination Audit Pipeline on {device}")
    logging.info("Applying Double-Condition Filter with 2-pixel morphological boundary exclusion.")
    
    with open(TEST_SPLIT_JSON, "r") as f:
        test_sar_files = json.load(f)

    # Load Fine-tuned DeepLabV3
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(DEEPLAB_WEIGHTS, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Tracking dictionaries
    class_total_valid_pixels = {i: 0 for i in range(NUM_CLASSES)}
    class_hallucination_pixels = {i: 0 for i in range(NUM_CLASSES)}
    
    with torch.no_grad():
        for s1_path in tqdm(test_sar_files, desc="Auditing Hallucinations"):
            filename = os.path.basename(s1_path)
            fake_rgb_filename = filename.replace('_s1_', '_fake_opt_')
            
            s2_path = s1_path.replace('_s1_', '_s2_').replace(os.sep + 's1' + os.sep, os.sep + 's2' + os.sep)
            lc_path = s1_path.replace('_s1_', '_lc_').replace(os.sep + 's1' + os.sep, os.sep + 'lc' + os.sep)
            fake_path = FAKE_OPTICAL_DIR / fake_rgb_filename
            
            if not fake_path.exists():
                continue # Skip if generator failed or didn't run on this file
                
            # 1. Load Data
            real_rgb = load_rgb_tensor(s2_path).to(device)
            fake_rgb = load_rgb_tensor(fake_path).to(device)
            
            with rasterio.open(lc_path) as src:
                gt_mask = src.read(1)
                gt_mask = np.clip(gt_mask - 1, 0, NUM_CLASSES - 1).astype(np.int64)

            # 2. Get DeepLab Predictions
            pred_real = model(real_rgb)['out'].argmax(1).squeeze().cpu().numpy()
            pred_fake = model(fake_rgb)['out'].argmax(1).squeeze().cpu().numpy()
            
            # 3. Create morphological exclusion mask (exclude edges)
            valid_mask = get_boundary_mask(gt_mask, radius=EROSION_RADIUS)
            
            # 4. Apply the Double-Condition Filter
            # Condition 1: Model accurately predicts the Real Optical
            cond_1 = (pred_real == gt_mask)
            # Condition 2: Model inaccurately predicts the Fake Optical (The hallucination)
            cond_2 = (pred_fake != gt_mask)
            
            # True hallucinations are where C1 and C2 are true, AND we are not on a boundary edge
            hallucinations = cond_1 & cond_2 & valid_mask
            valid_baseline = cond_1 & valid_mask # Only count pixels where the baseline filter works
            
            # 5. Aggregate Statistics
            for c in range(NUM_CLASSES):
                class_mask = (gt_mask == c)
                class_total_valid_pixels[c] += np.sum(valid_baseline & class_mask)
                class_hallucination_pixels[c] += np.sum(hallucinations & class_mask)

    # 6. Generate the Final Benchmark Table
    logging.info("\n=======================================================")
    logging.info("  FINAL PER-LAND-COVER-CLASS HALLUCINATION BENCHMARK")
    logging.info("=======================================================")
    
    results = []
    for c in range(NUM_CLASSES):
        valid = class_total_valid_pixels[c]
        hallucinated = class_hallucination_pixels[c]
        
        # Calculate rate, guarding against division by zero
        rate = (hallucinated / valid * 100) if valid > 0 else 0.0
        
        results.append({
            "Class ID": c,
            "Land Cover Type": CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Unknown ({c})",
            "Valid Audited Pixels": valid,
            "Hallucinated Pixels": hallucinated,
            "Hallucination Rate (%)": round(rate, 2)
        })
        
    df = pd.DataFrame(results)
    df.to_csv(CSV_OUTPUT, index=False)
    
    # Log the table beautifully to the console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    logging.info("\n" + df.to_string(index=False))
    logging.info(f"\nAudit complete! Full benchmark saved to {CSV_OUTPUT}")
    
if __name__ == "__main__":
    main()
