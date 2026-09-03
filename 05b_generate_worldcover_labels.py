"""
Generates pixel-accurate ESA WorldCover labels for every SEN12MS patch.
For each _s1_ file, reads its geographic bounds, crops the matching WorldCover tile,
remaps the class values to 0-indexed, and saves as a _wc_ label file.
"""

import os
import json
import rasterio
import rasterio.warp
import numpy as np
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from pathlib import Path
from tqdm import tqdm

SPLIT_FILES = ["./splits/train_files.json", "./splits/val_files.json", "./splits/test_files.json"]
TILE_DIR = Path("./worldcover_tiles")

# WorldCover class value -> 0-indexed class ID
REMAP = np.full(101, fill_value=255, dtype=np.int64)
REMAP[10]  = 0   # Tree Cover
REMAP[20]  = 1   # Shrubland
REMAP[30]  = 2   # Grassland
REMAP[40]  = 3   # Cropland
REMAP[50]  = 4   # Urban / Built-up
REMAP[60]  = 5   # Barren
REMAP[70]  = 6   # Snow and Ice
REMAP[80]  = 7   # Water Bodies
REMAP[90]  = 8   # Wetlands
REMAP[95]  = 9   # Mangroves
REMAP[100] = 10  # Moss and Lichen

NUM_CLASSES = 11


def get_tile_name(lat, lon):
    """Convert lat/lon to the ESA WorldCover 3x3 degree tile name."""
    lat_floor = int(np.floor(lat / 3.0) * 3)
    lon_floor = int(np.floor(lon / 3.0) * 3)
    lat_prefix = "N" if lat_floor >= 0 else "S"
    lon_prefix = "E" if lon_floor >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{lat_prefix}{abs(lat_floor):02d}{lon_prefix}{abs(lon_floor):03d}_Map.tif"


def generate_label(s1_path, tile_cache):
    """Generate a WorldCover label file for a single SEN12MS patch."""
    # Determine output path: replace _s1_ with _wc_
    wc_path = s1_path.replace('_s1_', '_wc_').replace('/s1_', '/wc_').replace('\\s1_', '\\wc_')

    # Create output directory if needed
    wc_dir = os.path.dirname(wc_path)
    if wc_dir and not os.path.exists(wc_dir):
        os.makedirs(wc_dir, exist_ok=True)

    # Skip if already generated (resume-safe)
    if os.path.exists(wc_path):
        return True

    # Read the SAR patch metadata to get its exact geographic footprint
    with rasterio.open(s1_path) as src:
        dst_crs = src.crs
        dst_transform = src.transform
        dst_height = src.height
        dst_width = src.width
        bounds = src.bounds

    # Convert center to WGS84 to find the right WorldCover tile
    if dst_crs and str(dst_crs) != "EPSG:4326":
        transformer = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        lon, lat = transformer.transform(center_x, center_y)
    else:
        lat = (bounds.bottom + bounds.top) / 2
        lon = (bounds.left + bounds.right) / 2

    tile_name = get_tile_name(lat, lon)
    tile_path = TILE_DIR / tile_name

    if not tile_path.exists():
        return False  # Tile not downloaded

    # Open WorldCover tile (use cache to avoid re-opening the same tile repeatedly)
    if tile_name not in tile_cache:
        tile_cache[tile_name] = rasterio.open(tile_path)

    wc_src = tile_cache[tile_name]

    # Reproject and crop the WorldCover data to match the exact SEN12MS patch grid
    wc_data = np.zeros((1, dst_height, dst_width), dtype=np.uint8)

    try:
        reproject(
            source=rasterio.band(wc_src, 1),
            destination=wc_data[0],
            src_transform=wc_src.transform,
            src_crs=wc_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,  # Nearest neighbor for categorical data
        )
    except Exception as e:
        return False

    # Remap WorldCover values to 0-indexed class IDs
    wc_data_clipped = np.clip(wc_data[0], 0, 100)
    remapped = REMAP[wc_data_clipped].astype(np.uint8)

    # Save the label file with the same geospatial metadata as the SAR patch
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': dst_width,
        'height': dst_height,
        'count': 1,
        'crs': dst_crs,
        'transform': dst_transform,
    }

    with rasterio.open(wc_path, 'w', **profile) as dst:
        dst.write(remapped, 1)

    return True


def main():
    print("=" * 60)
    print("  WorldCover Label Generator for SEN12MS")
    print("=" * 60)

    # Collect all SAR files
    all_files = []
    for split_file in SPLIT_FILES:
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                all_files.extend(json.load(f))

    print(f"Total patches to process: {len(all_files)}")

    tile_cache = {}  # Cache open tile file handles
    success = 0
    skipped_no_tile = 0
    errors = 0

    for s1_path in tqdm(all_files, desc="Generating WorldCover labels"):
        try:
            result = generate_label(s1_path, tile_cache)
            if result:
                success += 1
            else:
                skipped_no_tile += 1
        except Exception as e:
            errors += 1

    # Close cached tile handles
    for name, handle in tile_cache.items():
        handle.close()

    print(f"\n{'=' * 60}")
    print(f"  Label Generation Summary")
    print(f"  Success: {success} | Missing tile: {skipped_no_tile} | Errors: {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
