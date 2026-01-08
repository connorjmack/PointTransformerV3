import os
import argparse
import numpy as np
import laspy
import torch
from tqdm import tqdm

def voxelize(coord, feat, label, grid_size):
    """
    Grid sampling (voxelization) of point cloud.
    Returns the first point in each voxel.
    """
    idx_grid = np.floor(coord / grid_size).astype(np.int64)
    # Create a unique hash for each voxel
    unique_voxels, indices = np.unique(idx_grid, axis=0, return_index=True)
    return coord[indices], feat[indices], label[indices]

def process_las_to_tiles(las_path, output_dir, tile_size=5.0, grid_size=0.02):
    """
    Slices a large .las file into vertical columns and voxelizes each.
    """
    print(f"Reading {las_path}...")
    las = laspy.read(las_path)
    
    # Extract coordinates
    coords = np.vstack((las.x, las.y, las.z)).T
    
    # Extract features (Intensity and potentially colors if available)
    # For now, let's use Intensity as a primary feature.
    # We normalize intensity to [0, 1] if possible.
    if hasattr(las, 'intensity'):
        feat = las.intensity.astype(np.float32).reshape(-1, 1)
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
    else:
        # Fallback to a dummy feature if intensity is missing
        feat = np.ones((coords.shape[0], 1), dtype=np.float32)
        
    # Extract classification (ground truth labels)
    # RF pipeline should have populated 'classification'
    if hasattr(las, 'classification'):
        labels = las.classification.astype(np.int64)
    else:
        labels = np.zeros(coords.shape[0], dtype=np.int64)

    # Determine tiling bounds
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
    
    x_steps = np.arange(min_x, max_x, tile_size)
    y_steps = np.arange(min_y, max_y, tile_size)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tile_count = 0
    print(f"Tiling into {tile_size}m blocks...")
    
    for x in tqdm(x_steps):
        for y in y_steps:
            # Mask for points in this 2D tile (vertical column)
            mask = (coords[:, 0] >= x) & (coords[:, 0] < x + tile_size) & \
                   (coords[:, 1] >= y) & (coords[:, 1] < y + tile_size)
            
            if np.sum(mask) < 100: # Skip empty or very small tiles
                continue
                
            tile_coords = coords[mask]
            tile_feat = feat[mask]
            tile_labels = labels[mask]
            
            # Center the coordinates locally for better precision in voxelization
            # (Optional, depends on how PTv3 handles global coords)
            # tile_coords -= np.array([x, y, tile_coords[:, 2].min()])
            
            # Voxelize
            v_coords, v_feat, v_labels = voxelize(tile_coords, tile_feat, tile_labels, grid_size)
            
            # Prepare data_dict compatible with PTv3 Point class
            # PTv3 Point class expects: coord, feat, grid_size, (optionally labels/segment)
            data_dict = {
                "coord": v_coords.astype(np.float32),
                "feat": v_feat.astype(np.float32),
                "segment": v_labels.astype(np.int64),
                "grid_size": grid_size
            }
            
            tile_name = f"tile_{tile_count:04d}_{int(x)}_{int(y)}.pth"
            torch.save(data_dict, os.path.join(output_dir, tile_name))
            tile_count += 1
            
    print(f"Finished. Saved {tile_count} tiles to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LAS files into PTv3 tiles")
    parser.add_argument("--input", type=str, required=True, help="Path to input .las file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .pth tiles")
    parser.add_argument("--tile_size", type=float, default=5.0, help="Size of vertical tiles in meters")
    parser.add_argument("--grid_size", type=float, default=0.02, help="Voxel size for downsampling")
    
    args = parser.parse_args()
    process_las_to_tiles(args.input, args.output, args.tile_size, args.grid_size)
