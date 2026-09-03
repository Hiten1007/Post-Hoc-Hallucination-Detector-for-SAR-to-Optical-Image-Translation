"""
Downloads ESA WorldCover 10m tiles from the public AWS S3 bucket.
Scans all SEN12MS patches to determine which tiles are needed, then downloads only the unique ones.
"""

import os
import json
import rasterio
import numpy as np
import subprocess
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm

SPLIT_FILES = ["./splits/train_files.json", "./splits/val_files.json", "./splits/test_files.json"]
TILE_DIR = Path("./worldcover_tiles")
S3_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

os.makedirs(TILE_DIR, exist_ok=True)


def get_tile_name(lat, lon):
    """Convert lat/lon to the ESA WorldCover 3x3 degree tile name."""
    # Tile grid is based on lower-left corner, snapped to 3-degree intervals
    lat_floor = int(np.floor(lat / 3.0) * 3)
    lon_floor = int(np.floor(lon / 3.0) * 3)

    lat_prefix = "N" if lat_floor >= 0 else "S"
    lon_prefix = "E" if lon_floor >= 0 else "W"

    return f"ESA_WorldCover_10m_2021_v200_{lat_prefix}{abs(lat_floor):02d}{lon_prefix}{abs(lon_floor):03d}_Map.tif"


def main():
    print("=" * 60)
    print("  ESA WorldCover Tile Downloader")
    print("=" * 60)

    # Collect all unique SAR file paths from all splits
    all_files = []
    for split_file in SPLIT_FILES:
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                all_files.extend(json.load(f))
    print(f"Total patches across all splits: {len(all_files)}")

    # Determine which WorldCover tiles we need
    print("Scanning patch coordinates to find required tiles...")
    needed_tiles = set()
    errors = 0

    for s1_path in tqdm(all_files, desc="Scanning patches"):
        try:
            with rasterio.open(s1_path) as src:
                bounds = src.bounds
                crs = src.crs

                # Convert patch center to WGS84 lat/lon
                if crs and str(crs) != "EPSG:4326":
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    center_x = (bounds.left + bounds.right) / 2
                    center_y = (bounds.bottom + bounds.top) / 2
                    lon, lat = transformer.transform(center_x, center_y)
                else:
                    lat = (bounds.bottom + bounds.top) / 2
                    lon = (bounds.left + bounds.right) / 2

                tile_name = get_tile_name(lat, lon)
                needed_tiles.add(tile_name)
        except Exception as e:
            errors += 1
            continue

    print(f"\nUnique WorldCover tiles needed: {len(needed_tiles)}")
    if errors > 0:
        print(f"Skipped {errors} files due to read errors")

    # Download tiles
    downloaded = 0
    skipped = 0
    failed = 0

    for tile_name in sorted(needed_tiles):
        tile_path = TILE_DIR / tile_name

        if tile_path.exists():
            skipped += 1
            print(f"  [SKIP] {tile_name} (already downloaded)")
            continue

        url = f"{S3_BASE}/{tile_name}"
        print(f"  [DOWNLOAD] {tile_name} ... ", end="", flush=True)

        try:
            result = subprocess.run(
                ["wget", "-q", "-O", str(tile_path), url],
                timeout=300, capture_output=True
            )
            if result.returncode == 0 and tile_path.exists() and tile_path.stat().st_size > 1000:
                downloaded += 1
                size_mb = tile_path.stat().st_size / (1024 * 1024)
                print(f"OK ({size_mb:.1f} MB)")
            else:
                failed += 1
                if tile_path.exists():
                    tile_path.unlink()
                print(f"FAILED (HTTP error or empty file)")
        except subprocess.TimeoutExpired:
            failed += 1
            if tile_path.exists():
                tile_path.unlink()
            print("TIMEOUT")
        except FileNotFoundError:
            # wget not available, try curl
            try:
                result = subprocess.run(
                    ["curl", "-s", "-o", str(tile_path), url],
                    timeout=300, capture_output=True
                )
                if result.returncode == 0 and tile_path.exists() and tile_path.stat().st_size > 1000:
                    downloaded += 1
                    size_mb = tile_path.stat().st_size / (1024 * 1024)
                    print(f"OK ({size_mb:.1f} MB)")
                else:
                    failed += 1
                    if tile_path.exists():
                        tile_path.unlink()
                    print("FAILED")
            except Exception:
                failed += 1
                print("FAILED (neither wget nor curl available)")

    print(f"\n{'=' * 60}")
    print(f"  Download Summary")
    print(f"  Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
    print(f"  Tiles saved to: {TILE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
