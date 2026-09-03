"""
Hallucination Benchmark Evaluation using ESA WorldCover 10m Ground Truth.
Identical logic to 03_evaluate_hallucinations.py but uses:
  - _wc_ labels (10m WorldCover) instead of _lc_ labels (500m MODIS IGBP)
  - deeplabv3_finetuned_worldcover.pth (11-class judge) instead of the 17-class judge

Usage:
    python3 03b_evaluate_hallucinations_worldcover.py --model pix2pix
    python3 03b_evaluate_hallucinations_worldcover.py --model cyclegan
    python3 03b_evaluate_hallucinations_worldcover.py --model palette
"""

import os
import json
import logging
import argparse
import rasterio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import binary_dilation
import torch.nn as nn

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="WorldCover Hallucination Benchmark")
parser.add_argument("--model", type=str, default="pix2pix", choices=["pix2pix", "cyclegan", "palette"],
                    help="Which generator's fake images to evaluate.")
args = parser.parse_args()

# --- CONFIGURATION ---
MODEL_NAME = args.model
TEST_SPLIT_JSON = "./splits/test_files.json"
FAKE_OPTICAL_DIR = Path(f"./fake_optical_test_set_{MODEL_NAME}")
DEEPLAB_WEIGHTS = "deeplabv3_finetuned_worldcover.pth"
LOG_FILE = f"hallucination_worldcover_{MODEL_NAME}.log"
CSV_OUTPUT = f"hallucination_benchmark_worldcover_{MODEL_NAME}.csv"

NUM_CLASSES = 11
EROSION_RADIUS = 2

CLASS_NAMES = [
    "Tree Cover", "Shrubland", "Grassland", "Cropland",
    "Urban / Built-up", "Barren", "Snow and Ice", "Water Bodies",
    "Wetlands", "Mangroves", "Moss and Lichen"
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)


def get_boundary_mask(label_mask, radius=2):
    """2-pixel morphological erosion on class boundaries."""
    up = np.roll(label_mask, 1, axis=0)
    down = np.roll(label_mask, -1, axis=0)
    left = np.roll(label_mask, 1, axis=1)
    right = np.roll(label_mask, -1, axis=1)
    boundaries = (label_mask != up) | (label_mask != down) | (label_mask != left) | (label_mask != right)
    exclusion_zone = binary_dilation(boundaries, iterations=radius)
    return ~exclusion_zone


def load_rgb_tensor(path):
    with rasterio.open(path) as src:
        img = src.read()
        if img.shape[0] == 13:
            img = img[[3, 2, 1], :, :]
        elif img.shape[0] > 3:
            img = img[:3, :, :]
        if img.dtype != np.uint8:
            img = img.astype(np.float32) / 10000.0
        else:
            img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def get_model(num_classes):
    model = deeplabv3_resnet50(weights=None, num_classes=21)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting WorldCover Hallucination Audit on {device} for model: [{MODEL_NAME.upper()}]")
    logging.info(f"Using 10m WorldCover ground truth ({NUM_CLASSES} classes)")
    logging.info("Applying Double-Condition Filter with 2-pixel morphological boundary exclusion.")

    with open(TEST_SPLIT_JSON, "r") as f:
        test_sar_files = json.load(f)

    # Load WorldCover-trained DeepLabV3
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(DEEPLAB_WEIGHTS, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Tracking dictionaries
    class_total_gt_pixels = {i: 0 for i in range(NUM_CLASSES)}
    class_total_valid_pixels = {i: 0 for i in range(NUM_CLASSES)}
    class_hallucination_pixels = {i: 0 for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for s1_path in tqdm(test_sar_files, desc="Auditing Hallucinations (WorldCover)"):
            filename = os.path.basename(s1_path)
            fake_rgb_filename = filename.replace('_s1_', '_fake_opt_')

            s2_path = s1_path.replace('_s1_', '_s2_').replace('/s1_', '/s2_').replace('\\s1_', '\\s2_')
            wc_path = s1_path.replace('_s1_', '_wc_').replace('/s1_', '/wc_').replace('\\s1_', '\\wc_')
            fake_path = FAKE_OPTICAL_DIR / fake_rgb_filename

            if not fake_path.exists() or not os.path.exists(wc_path):
                continue

            # 1. Load Data
            real_rgb = load_rgb_tensor(s2_path).to(device)
            fake_rgb = load_rgb_tensor(fake_path).to(device)

            with rasterio.open(wc_path) as src:
                gt_mask = src.read(1).astype(np.int64)
                # WorldCover labels are already 0-indexed (0-10), no need to subtract 1

            # Skip patches where all labels are 255 (unmapped)
            if np.all(gt_mask == 255):
                continue

            # 2. Get DeepLab Predictions
            pred_real = model(real_rgb)['out'].argmax(1).squeeze().cpu().numpy()
            pred_fake = model(fake_rgb)['out'].argmax(1).squeeze().cpu().numpy()

            # 3. Create morphological exclusion mask
            valid_mask = get_boundary_mask(gt_mask, radius=EROSION_RADIUS)

            # Also exclude any pixels with label 255 (unmapped)
            valid_mask = valid_mask & (gt_mask != 255)

            # 4. Apply the Double-Condition Filter
            cond_1 = (pred_real == gt_mask)
            cond_2 = (pred_fake != gt_mask)
            hallucinations = cond_1 & cond_2 & valid_mask
            valid_baseline = cond_1 & valid_mask

            # 5. Aggregate Statistics
            for c in range(NUM_CLASSES):
                class_mask = (gt_mask == c)
                class_total_gt_pixels[c] += np.sum(class_mask & valid_mask)
                class_total_valid_pixels[c] += np.sum(valid_baseline & class_mask)
                class_hallucination_pixels[c] += np.sum(hallucinations & class_mask)

    # 6. Generate the Final Benchmark Table
    logging.info("\n=======================================================")
    logging.info("  WORLDCOVER 10m HALLUCINATION BENCHMARK")
    logging.info("=======================================================")

    results = []
    for c in range(NUM_CLASSES):
        gt = class_total_gt_pixels[c]
        valid = class_total_valid_pixels[c]
        hallucinated = class_hallucination_pixels[c]

        detector_acc = (valid / gt * 100) if gt > 0 else 0.0
        rate = (hallucinated / valid * 100) if valid > 0 else 0.0

        results.append({
            "Class ID": c,
            "Land Cover Type": CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Unknown ({c})",
            "Total GT Pixels": gt,
            "Detector Baseline Acc (%)": round(detector_acc, 2),
            "Valid Audited Pixels": valid,
            "Hallucinated Pixels": hallucinated,
            "Hallucination Rate (%)": round(rate, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv(CSV_OUTPUT, index=False)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    logging.info("\n" + df.to_string(index=False))
    logging.info(f"\nAudit complete! Benchmark saved to {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
