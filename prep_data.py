import os
import argparse
import numpy as np
import laspy
import torch
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from functools import partial

def voxelize(coord, feat, label, grid_size):
    """
    Grid sampling (voxelization) of point cloud.
    Returns the first point in each voxel.
    """
    idx_grid = np.floor(coord / grid_size).astype(np.int64)
    # Create a unique hash for each voxel
    unique_voxels, indices = np.unique(idx_grid, axis=0, return_index=True)
    return coord[indices], feat[indices], label[indices]

def process_single_tile(tile_info, coords, feat, labels, grid_size, output_dir, min_points=100):
    """
    Process a single tile - designed to be called by worker processes.
    Returns the tile filename if successful, None if skipped.
    """
    tile_idx, x, y, tile_size = tile_info

    # Mask for points in this 2D tile (vertical column)
    mask = (coords[:, 0] >= x) & (coords[:, 0] < x + tile_size) & \
           (coords[:, 1] >= y) & (coords[:, 1] < y + tile_size)

    if np.sum(mask) < min_points:
        return None

    tile_coords = coords[mask]
    tile_feat = feat[mask]
    tile_labels = labels[mask]

    # Voxelize
    v_coords, v_feat, v_labels = voxelize(tile_coords, tile_feat, tile_labels, grid_size)

    # Prepare data_dict compatible with PTv3 Point class
    data_dict = {
        "coord": v_coords.astype(np.float32),
        "feat": v_feat.astype(np.float32),
        "segment": v_labels.astype(np.int64),
        "grid_size": grid_size
    }

    tile_name = f"tile_{tile_idx:04d}_{int(x)}_{int(y)}.pth"
    torch.save(data_dict, os.path.join(output_dir, tile_name))
    return tile_name

def process_las_to_tiles(las_path, output_dir, tile_size=10.0, grid_size=0.02, workers=None):
    """
    Slices a large .las file into vertical columns and voxelizes each.
    Uses threading for parallel tile processing (shares memory efficiently).
    """
    if workers is None:
        workers = max(1, cpu_count() // 3)  # Use 1/3 of available cores

    print(f"Reading {las_path}...")
    las = laspy.read(las_path)

    # Extract coordinates
    coords = np.vstack((las.x, las.y, las.z)).T

    # Extract features (Intensity and potentially colors if available)
    # For now, let's use Intensity as a primary feature.
    # We normalize intensity to [0, 1] if possible.
    if hasattr(las, 'intensity'):
        feat = np.array(las.intensity).astype(np.float32).reshape(-1, 1)
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
    else:
        # Fallback to a dummy feature if intensity is missing
        feat = np.ones((coords.shape[0], 1), dtype=np.float32)

    # Extract classification (ground truth labels)
    # RF pipeline should have populated 'classification'
    if hasattr(las, 'classification'):
        labels = np.array(las.classification).astype(np.int64)
    else:
        labels = np.zeros(coords.shape[0], dtype=np.int64)

    # Determine tiling bounds
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()

    x_steps = np.arange(min_x, max_x, tile_size)
    y_steps = np.arange(min_y, max_y, tile_size)

    os.makedirs(output_dir, exist_ok=True)

    # Create list of all tile bounds to process
    tile_list = []
    tile_idx = 0
    for x in x_steps:
        for y in y_steps:
            tile_list.append((tile_idx, x, y, tile_size))
            tile_idx += 1

    print(f"Processing {len(tile_list)} potential tiles with {workers} workers (using 1/3 of {cpu_count()} cores)...")
    print(f"Tile size: {tile_size}m x {tile_size}m, Voxel size: {grid_size}m")
    print(f"Point cloud loaded into shared memory: {coords.shape[0]:,} points")

    # Create a partial function with fixed arguments
    process_func = partial(
        process_single_tile,
        coords=coords,
        feat=feat,
        labels=labels,
        grid_size=grid_size,
        output_dir=output_dir
    )

    # Process tiles in parallel using threads (shares memory, no copying)
    with ThreadPool(workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, tile_list),
            total=len(tile_list),
            desc="Creating tiles"
        ))

    # Count successful tiles (filter out None values)
    tile_count = sum(1 for r in results if r is not None)

    print(f"Finished. Saved {tile_count} tiles to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LAS files into PTv3 tiles")
    parser.add_argument("--input", type=str, required=True, help="Path to input .las file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .pth tiles")
    parser.add_argument("--tile_size", type=float, default=10.0, help="Size of vertical tiles in meters (default: 10.0)")
    parser.add_argument("--grid_size", type=float, default=0.02, help="Voxel size for downsampling (default: 0.02)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: auto-detect)")

    args = parser.parse_args()
    process_las_to_tiles(args.input, args.output, args.tile_size, args.grid_size, args.workers)
