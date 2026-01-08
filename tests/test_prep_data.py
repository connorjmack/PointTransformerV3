import os
import numpy as np
import laspy
import torch
import pytest
from prep_data import process_las_to_tiles, voxelize

def create_dummy_las(path, num_points=1000):
    """Creates a dummy .las file for testing."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    
    # Create a simple 10x10 area with points
    x = np.random.uniform(0, 10, num_points)
    y = np.random.uniform(0, 10, num_points)
    z = np.random.uniform(0, 5, num_points)
    
    las.x = x
    las.y = y
    las.z = z
    las.intensity = np.random.randint(0, 255, num_points, dtype=np.uint16)
    las.classification = np.random.randint(0, 3, num_points, dtype=np.uint8)
    
    las.write(path)

def test_voxelize():
    coords = np.array([[0, 0, 0], [0.01, 0.01, 0.01], [1, 1, 1]], dtype=np.float32)
    feat = np.array([[1], [2], [3]], dtype=np.float32)
    labels = np.array([0, 0, 1], dtype=np.int64)
    grid_size = 0.1
    
    v_coords, v_feat, v_labels = voxelize(coords, feat, labels, grid_size)
    
    # Points [0,0,0] and [0.01, 0.01, 0.01] should fall into the same 0.1m voxel
    assert len(v_coords) == 2
    assert len(v_feat) == 2
    assert len(v_labels) == 2

def test_full_pipeline(tmp_path):
    las_path = os.path.join(tmp_path, "test.las")
    output_dir = os.path.join(tmp_path, "tiles")
    
    create_dummy_las(las_path, num_points=2000)
    
    # Run the processor
    # We use a 5m tile size on a 10m x 10m area, so we expect ~4 tiles
    process_las_to_tiles(las_path, output_dir, tile_size=5.0, grid_size=0.1)
    
    tiles = os.listdir(output_dir)
    assert len(tiles) > 0
    
    # Load one tile and check structure
    tile_path = os.path.join(output_dir, tiles[0])
    data = torch.load(tile_path)
    
    assert "coord" in data
    assert "feat" in data
    assert "segment" in data
    assert "grid_size" in data
    assert isinstance(data["coord"], np.ndarray)
    assert data["coord"].shape[1] == 3
    assert data["feat"].shape[1] == 1

if __name__ == "__main__":
    # If run directly without pytest
    import sys
    from tempfile import TemporaryDirectory
    
    with TemporaryDirectory() as tmpdir:
        print("Running manual tests...")
        test_voxelize()
        print("Voxelization test passed.")
        test_full_pipeline(tmpdir)
        print("Pipeline test passed.")
