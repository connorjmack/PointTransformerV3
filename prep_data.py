import os
import argparse
import numpy as np
import laspy
import torch
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from functools import partial

def voxelize_fast(coord, feat, label, grid_size):
    """
    Fast grid sampling (voxelization) using hash-based method.
    Returns the first point in each voxel.
    Much faster than np.unique for large point clouds.
    """
    idx_grid = np.floor(coord / grid_size).astype(np.int64)

    # Hash the 3D grid indices to 1D
    # This is much faster than np.unique on 3D arrays
    offset = idx_grid.min(axis=0)
    idx_grid -= offset

    # Create unique hash for each voxel (handles up to 10000 voxels per dimension)
    hash_vals = idx_grid[:, 0] + idx_grid[:, 1] * 10000 + idx_grid[:, 2] * 100000000

    # Find first occurrence of each hash (much faster than np.unique)
    _, indices = np.unique(hash_vals, return_index=True)

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

    # Voxelize using fast hash-based method
    v_coords, v_feat, v_labels = voxelize_fast(tile_coords, tile_feat, tile_labels, grid_size)

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

def process_las_to_tiles(las_path, output_dir, tile_size=10.0, grid_size=0.02, workers=None, subsample=None):
    """
    Slices a large .las file into vertical columns and voxelizes each.
    Uses threading for parallel tile processing (shares memory efficiently).

    Args:
        subsample: Optional random subsampling ratio (e.g., 0.5 = keep 50% of points).
                   Useful for very dense point clouds. Applied before tiling.
    """
    if workers is None:
        workers = max(1, cpu_count() // 2)  # Use 1/2 of available cores

    print(f"Reading {las_path}...")
    las = laspy.read(las_path)

    n_points = len(las.x)
    print(f"Loaded {n_points:,} points")

    # Extract coordinates (optimized: direct array creation)
    coords = np.column_stack((las.x, las.y, las.z)).astype(np.float32)

    # Extract features (Intensity and potentially colors if available)
    if hasattr(las, 'intensity'):
        feat = np.array(las.intensity, dtype=np.float32).reshape(-1, 1)
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
    else:
        feat = np.ones((n_points, 1), dtype=np.float32)

    # Extract classification (ground truth labels)
    if hasattr(las, 'classification'):
        labels = np.array(las.classification, dtype=np.int64)
    else:
        labels = np.zeros(n_points, dtype=np.int64)

    # Optional random subsampling for very dense point clouds
    if subsample is not None and 0 < subsample < 1:
        n_subsample = int(n_points * subsample)
        print(f"Subsampling {subsample*100:.1f}%: {n_points:,} → {n_subsample:,} points")
        idx = np.random.choice(n_points, n_subsample, replace=False)
        coords = coords[idx]
        feat = feat[idx]
        labels = labels[idx]

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

    print(f"Processing {len(tile_list)} potential tiles with {workers} workers (using 1/2 of {cpu_count()} cores)...")
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
    parser.add_argument("--subsample", type=float, default=None, help="Random subsample ratio before tiling (e.g., 0.3 = keep 30%%). Good for very dense clouds.")

    args = parser.parse_args()
    process_las_to_tiles(args.input, args.output, args.tile_size, args.grid_size, args.workers, args.subsample)
