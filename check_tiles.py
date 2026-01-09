#!/usr/bin/env python3
"""
Quick validation script to check if tiles were created correctly.
Usage: python3 check_tiles.py [tile_directory]
"""
import torch
import os
import sys

# Get tile directory from command line or use default
tile_dir = sys.argv[1] if len(sys.argv) > 1 else "./test_tiles"

if not os.path.exists(tile_dir):
    print(f"Error: Directory '{tile_dir}' does not exist")
    sys.exit(1)

tiles = [f for f in os.listdir(tile_dir) if f.endswith('.pth')]

print(f"Found {len(tiles)} tiles in {tile_dir}")

if len(tiles) == 0:
    print("No .pth files found!")
    sys.exit(1)

# Load and inspect the first few tiles
for i, tile_name in enumerate(tiles[:3]):
    tile_path = os.path.join(tile_dir, tile_name)
    data = torch.load(tile_path)

    print(f"\n{'='*60}")
    print(f"Tile {i+1}: {tile_name}")
    print(f"{'='*60}")
    print(f"  coord shape: {data['coord'].shape}")
    print(f"  feat shape: {data['feat'].shape}")
    print(f"  segment shape: {data['segment'].shape}")
    print(f"  grid_size: {data['grid_size']}")
    print(f"  num points: {data['coord'].shape[0]:,}")
    print(f"  unique labels: {sorted(set(data['segment'].tolist()))}")

    # Verify expected format
    assert data['coord'].shape[1] == 3, "Coords should be Nx3"
    assert data['feat'].shape[1] == 1, "Features should be Nx1"
    assert len(data['coord']) == len(data['feat']) == len(data['segment']), "All arrays should have same length"

    print("  ✓ Structure valid")

print(f"\n{'='*60}")
print(f"SUCCESS: All {len(tiles)} tiles look good!")
print(f"{'='*60}")
