#!/usr/bin/env python3
"""
Visualize point cloud tiles to verify preprocessing quality.

Usage:
    python3 visualize_tiles.py ./tests/test_parallel/20170301_00590_00612/tile_0000_*.pth
    python3 visualize_tiles.py ./tests/test_parallel/20170301_00590_00612 --random 3
    python3 visualize_tiles.py ./tests/test_parallel/20170301_00590_00612/tile_0001_*.pth --method open3d
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
import random

def visualize_matplotlib(tile_path, show_labels=True):
    """Visualize using matplotlib (simple, works everywhere)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print(f"\nLoading {tile_path}...")
    data = torch.load(tile_path, weights_only=False)

    coords = data['coord']
    feat = data['feat']
    labels = data['segment']

    print(f"Points: {len(coords):,}")
    print(f"Coord range: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
          f"Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
          f"Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    print(f"Feature range: [{feat.min():.3f}, {feat.max():.3f}]")
    print(f"Unique labels: {sorted(set(labels.tolist()))}")
    print(f"Grid size: {data['grid_size']}m")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))

    # 3D scatter plot colored by height (Z)
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=coords[:, 2], cmap='terrain', s=1, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Colored by Elevation')
    plt.colorbar(scatter1, ax=ax1, label='Z (m)', shrink=0.5)

    # 3D scatter plot colored by intensity
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=feat.flatten(), cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Colored by Intensity')
    plt.colorbar(scatter2, ax=ax2, label='Intensity (normalized)', shrink=0.5)

    # 3D scatter plot colored by classification label
    if show_labels:
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               c=labels, cmap='tab10', s=1, alpha=0.6)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('Colored by Classification')
        plt.colorbar(scatter3, ax=ax3, label='Class ID', shrink=0.5)

    plt.suptitle(f'Tile: {Path(tile_path).name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_open3d(tile_path):
    """Visualize using open3d (interactive, better quality)."""
    try:
        import open3d as o3d
    except ImportError:
        print("ERROR: open3d not installed. Install with: pip install open3d")
        print("Falling back to matplotlib...")
        visualize_matplotlib(tile_path)
        return

    print(f"\nLoading {tile_path}...")
    data = torch.load(tile_path, weights_only=False)

    coords = data['coord']
    feat = data['feat']
    labels = data['segment']

    print(f"Points: {len(coords):,}")
    print(f"Coord range: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
          f"Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
          f"Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    print(f"Unique labels: {sorted(set(labels.tolist()))}")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # Color by classification labels using a colormap
    unique_labels = np.unique(labels)
    colormap = plt.cm.get_cmap('tab10')
    colors = np.zeros((len(labels), 3))
    for label in unique_labels:
        mask = labels == label
        color = colormap(int(label) % 10)[:3]  # RGB only
        colors[mask] = color

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals for better visualization
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    print("\nOpen3D Controls:")
    print("  - Mouse drag: Rotate")
    print("  - Scroll: Zoom")
    print("  - Ctrl+Mouse drag: Pan")
    print("  - Q or ESC: Close window")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Tile: {Path(tile_path).name}",
        width=1200,
        height=800,
        point_show_normal=False
    )

def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed point cloud tiles"
    )
    parser.add_argument("path", type=str,
                       help="Path to .pth file or directory containing tiles")
    parser.add_argument("--method", type=str, default="matplotlib",
                       choices=["matplotlib", "open3d"],
                       help="Visualization method (default: matplotlib)")
    parser.add_argument("--random", type=int, default=None,
                       help="Visualize N random tiles from directory")
    parser.add_argument("--no-labels", action="store_true",
                       help="Don't show classification labels")

    args = parser.parse_args()

    # Collect tile paths
    path = Path(args.path)
    tiles = []

    if path.is_file() and path.suffix == '.pth':
        tiles = [str(path)]
    elif path.is_dir():
        tiles = sorted([str(f) for f in path.glob("*.pth")])
        if not tiles:
            print(f"ERROR: No .pth files found in {path}")
            sys.exit(1)
    else:
        print(f"ERROR: {path} is not a valid .pth file or directory")
        sys.exit(1)

    # Select random subset if requested
    if args.random is not None:
        if len(tiles) > args.random:
            tiles = random.sample(tiles, args.random)
            print(f"Selected {args.random} random tiles from {len(tiles)} total")

    print(f"Found {len(tiles)} tile(s) to visualize")

    # Visualize each tile
    visualize_func = visualize_open3d if args.method == "open3d" else visualize_matplotlib

    for tile_path in tiles:
        if args.method == "open3d":
            visualize_func(tile_path)
        else:
            visualize_func(tile_path, show_labels=not args.no_labels)

if __name__ == "__main__":
    main()
