"""
Enhanced Preprocessing for Coastal LiDAR Data

Extends basic tiling with:
- Surface normal estimation
- Curvature computation
- Multi-resolution output
- Feature extraction (intensity, return number, etc.)

Usage:
    python -m coastal_contrast.scripts.preprocess_surveys \
        --input /path/to/registered_surveys/ \
        --output /path/to/processed/ \
        --resolutions 0.02 0.05 0.1
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TileMetadata:
    """Metadata for a processed tile."""
    tile_id: str
    survey_id: str
    resolution: float
    bounds: Tuple[float, float, float, float, float, float]
    point_count: int
    file_path: str
    timestamp: str


def voxelize(
    coords: np.ndarray,
    features: np.ndarray,
    labels: Optional[np.ndarray],
    grid_size: float,
    aggregation: str = 'first'
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Voxelize point cloud with feature aggregation.

    Args:
        coords: (N, 3) coordinates
        features: (N, F) features
        labels: (N,) labels or None
        grid_size: Voxel size
        aggregation: 'first', 'mean', or 'random'

    Returns:
        Tuple of (coords, features, labels, voxel_indices)
    """
    # Compute voxel indices
    voxel_coords = np.floor(coords / grid_size).astype(np.int64)

    # Find unique voxels
    _, unique_indices, inverse_indices = np.unique(
        voxel_coords, axis=0, return_index=True, return_inverse=True
    )

    if aggregation == 'first':
        out_coords = coords[unique_indices]
        out_features = features[unique_indices]
        out_labels = labels[unique_indices] if labels is not None else None
    elif aggregation == 'mean':
        # Mean aggregation
        n_voxels = len(unique_indices)
        out_coords = np.zeros((n_voxels, 3), dtype=np.float32)
        out_features = np.zeros((n_voxels, features.shape[1]), dtype=np.float32)

        np.add.at(out_coords, inverse_indices, coords)
        np.add.at(out_features, inverse_indices, features)

        counts = np.bincount(inverse_indices)
        out_coords /= counts[:, np.newaxis]
        out_features /= counts[:, np.newaxis]

        # For labels, take mode (most common)
        if labels is not None:
            out_labels = np.zeros(n_voxels, dtype=np.int64)
            for i in range(n_voxels):
                mask = inverse_indices == i
                values, counts = np.unique(labels[mask], return_counts=True)
                out_labels[i] = values[counts.argmax()]
        else:
            out_labels = None
    elif aggregation == 'random':
        # Random point from each voxel
        out_coords = coords[unique_indices]
        out_features = features[unique_indices]
        out_labels = labels[unique_indices] if labels is not None else None
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return out_coords, out_features, out_labels, inverse_indices


def estimate_normals_pca(
    coords: np.ndarray,
    k_neighbors: int = 30,
    radius: Optional[float] = None
) -> np.ndarray:
    """
    Estimate surface normals using PCA on local neighborhoods.

    Args:
        coords: (N, 3) point coordinates
        k_neighbors: Number of neighbors for PCA
        radius: Optional search radius (uses k-NN if None)

    Returns:
        (N, 3) surface normals
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for normal estimation")

    n_points = len(coords)
    normals = np.zeros((n_points, 3), dtype=np.float32)

    # Build KD-tree
    tree = cKDTree(coords)

    for i in range(n_points):
        # Find neighbors
        if radius:
            idx = tree.query_ball_point(coords[i], radius)
            if len(idx) < 3:
                idx = tree.query(coords[i], k=k_neighbors)[1]
        else:
            _, idx = tree.query(coords[i], k=k_neighbors)

        # Get neighbor points
        neighbors = coords[idx]

        # PCA
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Normal is eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]

        # Orient normals consistently (pointing "up" on average)
        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals


def estimate_normals_vectorized(
    coords: np.ndarray,
    k_neighbors: int = 30
) -> np.ndarray:
    """
    Faster vectorized normal estimation (less accurate but much faster).

    Args:
        coords: (N, 3) coordinates
        k_neighbors: Number of neighbors

    Returns:
        (N, 3) normals
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required")

    n_points = len(coords)
    tree = cKDTree(coords)

    # Query all neighbors at once
    _, all_idx = tree.query(coords, k=k_neighbors)

    normals = np.zeros((n_points, 3), dtype=np.float32)

    # Process in batches for memory efficiency
    batch_size = 10000
    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch_idx = all_idx[start:end]

        for i, idx in enumerate(batch_idx):
            neighbors = coords[idx]
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]

            if normal[2] < 0:
                normal = -normal

            normals[start + i] = normal

    return normals


def estimate_curvature(
    coords: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int = 30
) -> np.ndarray:
    """
    Estimate local curvature from normal variation.

    Curvature is computed as: λ_min / (λ_0 + λ_1 + λ_2)
    where λ are eigenvalues of local covariance.

    Args:
        coords: (N, 3) coordinates
        normals: (N, 3) normals
        k_neighbors: Number of neighbors

    Returns:
        (N,) curvature values
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required")

    n_points = len(coords)
    curvature = np.zeros(n_points, dtype=np.float32)

    tree = cKDTree(coords)
    _, all_idx = tree.query(coords, k=k_neighbors)

    for i, idx in enumerate(all_idx):
        neighbors = coords[idx]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)

        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)

        # Curvature = smallest eigenvalue / sum
        total = eigenvalues.sum()
        if total > 1e-10:
            curvature[i] = eigenvalues[0] / total
        else:
            curvature[i] = 0.0

    return curvature


def compute_geometric_features(
    coords: np.ndarray,
    k_neighbors: int = 30
) -> Dict[str, np.ndarray]:
    """
    Compute full set of geometric features.

    Returns dictionary with:
    - normals: (N, 3) surface normals
    - curvature: (N,) local curvature
    - planarity: (N,) planarity score
    - linearity: (N,) linearity score
    - sphericity: (N,) sphericity score
    - verticality: (N,) verticality (normal alignment with Z)

    Args:
        coords: (N, 3) coordinates
        k_neighbors: Number of neighbors for local analysis

    Returns:
        Dictionary of geometric features
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required")

    n_points = len(coords)
    tree = cKDTree(coords)
    _, all_idx = tree.query(coords, k=k_neighbors)

    # Initialize outputs
    normals = np.zeros((n_points, 3), dtype=np.float32)
    curvature = np.zeros(n_points, dtype=np.float32)
    planarity = np.zeros(n_points, dtype=np.float32)
    linearity = np.zeros(n_points, dtype=np.float32)
    sphericity = np.zeros(n_points, dtype=np.float32)

    for i, idx in enumerate(all_idx):
        neighbors = coords[idx]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues descending
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        λ1, λ2, λ3 = eigenvalues

        # Normal (smallest eigenvalue direction)
        normal = eigenvectors[:, 2]
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal

        # Curvature
        total = λ1 + λ2 + λ3
        if total > 1e-10:
            curvature[i] = λ3 / total

            # Linearity: (λ1 - λ2) / λ1
            if λ1 > 1e-10:
                linearity[i] = (λ1 - λ2) / λ1

            # Planarity: (λ2 - λ3) / λ1
            if λ1 > 1e-10:
                planarity[i] = (λ2 - λ3) / λ1

            # Sphericity: λ3 / λ1
            if λ1 > 1e-10:
                sphericity[i] = λ3 / λ1

    # Verticality: alignment of normal with Z axis
    verticality = np.abs(normals[:, 2])

    return {
        'normals': normals,
        'curvature': curvature,
        'planarity': planarity,
        'linearity': linearity,
        'sphericity': sphericity,
        'verticality': verticality
    }


def extract_las_features(las: 'laspy.LasData') -> Dict[str, np.ndarray]:
    """
    Extract available features from LAS file.

    Args:
        las: laspy LasData object

    Returns:
        Dictionary of features
    """
    features = {}

    # Intensity (normalize to 0-1)
    if hasattr(las, 'intensity'):
        intensity = las.intensity.astype(np.float32)
        if intensity.max() > intensity.min():
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        features['intensity'] = intensity

    # Return number
    if hasattr(las, 'return_number'):
        features['return_number'] = las.return_number.astype(np.float32)

    # Number of returns
    if hasattr(las, 'number_of_returns'):
        features['number_of_returns'] = las.number_of_returns.astype(np.float32)

    # Classification (if present)
    if hasattr(las, 'classification'):
        features['classification'] = las.classification.astype(np.int64)

    # RGB (if present, normalize to 0-1)
    if all(hasattr(las, c) for c in ['red', 'green', 'blue']):
        r = las.red.astype(np.float32)
        g = las.green.astype(np.float32)
        b = las.blue.astype(np.float32)
        max_val = max(r.max(), g.max(), b.max())
        if max_val > 0:
            features['rgb'] = np.stack([r, g, b], axis=1) / max_val

    return features


def process_las_tile(
    coords: np.ndarray,
    features: Dict[str, np.ndarray],
    labels: Optional[np.ndarray],
    grid_size: float,
    compute_normals: bool = True,
    compute_curvature: bool = True,
    k_neighbors: int = 30
) -> Dict[str, np.ndarray]:
    """
    Process a single tile: voxelize and compute geometric features.

    Args:
        coords: (N, 3) coordinates
        features: Dictionary of point features
        labels: (N,) classification labels or None
        grid_size: Voxel size for downsampling
        compute_normals: Whether to compute normals
        compute_curvature: Whether to compute curvature
        k_neighbors: Neighbors for geometric features

    Returns:
        Dictionary with processed tile data
    """
    # Combine features into array
    feat_list = []
    feat_names = []

    if 'intensity' in features:
        feat_list.append(features['intensity'].reshape(-1, 1))
        feat_names.append('intensity')

    if 'return_number' in features:
        feat_list.append(features['return_number'].reshape(-1, 1))
        feat_names.append('return_number')

    if 'rgb' in features:
        feat_list.append(features['rgb'])
        feat_names.extend(['r', 'g', 'b'])

    if feat_list:
        feat_array = np.hstack(feat_list).astype(np.float32)
    else:
        # Default feature: Z coordinate normalized
        z_norm = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min() + 1e-6)
        feat_array = z_norm.reshape(-1, 1).astype(np.float32)
        feat_names.append('z_norm')

    # Voxelize
    v_coords, v_feat, v_labels, _ = voxelize(
        coords, feat_array, labels, grid_size, aggregation='mean'
    )

    # Compute geometric features on voxelized points
    result = {
        'coord': v_coords.astype(np.float32),
        'feat': v_feat.astype(np.float32),
        'grid_size': grid_size,
        'feat_names': feat_names
    }

    if v_labels is not None:
        result['segment'] = v_labels.astype(np.int64)

    if compute_normals or compute_curvature:
        geo_features = compute_geometric_features(v_coords, k_neighbors)

        if compute_normals:
            result['normal'] = geo_features['normals'].astype(np.float32)
            result['verticality'] = geo_features['verticality'].astype(np.float32)

        if compute_curvature:
            result['curvature'] = geo_features['curvature'].astype(np.float32)
            result['planarity'] = geo_features['planarity'].astype(np.float32)

    return result


def process_las_to_tiles_multiresolution(
    las_path: str,
    output_dir: str,
    tile_size: float = 10.0,
    resolutions: List[float] = [0.02, 0.05, 0.1],
    compute_normals: bool = True,
    compute_curvature: bool = True,
    k_neighbors: int = 30,
    min_points: int = 100
) -> List[TileMetadata]:
    """
    Process LAS file into multi-resolution tiles.

    Args:
        las_path: Input LAS file path
        output_dir: Output directory for tiles
        tile_size: Size of tiles in meters
        resolutions: List of voxel sizes (grid_size)
        compute_normals: Whether to compute normals
        compute_curvature: Whether to compute curvature
        k_neighbors: Neighbors for geometric features
        min_points: Minimum points per tile

    Returns:
        List of TileMetadata for created tiles
    """
    if not HAS_LASPY:
        raise ImportError("laspy required")

    from datetime import datetime

    logger.info(f"Processing {las_path}")
    las = laspy.read(las_path)

    # Extract coordinates and features
    coords = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    features = extract_las_features(las)
    labels = features.pop('classification', None)

    # Get survey ID from filename
    survey_id = Path(las_path).stem

    # Create output directories
    output_path = Path(output_dir)
    for res in resolutions:
        (output_path / f"res_{res:.3f}").mkdir(parents=True, exist_ok=True)

    # Determine tiling bounds
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()

    x_steps = np.arange(min_x, max_x, tile_size)
    y_steps = np.arange(min_y, max_y, tile_size)

    tile_metadata = []
    tile_count = 0

    logger.info(f"Creating tiles of size {tile_size}m at resolutions {resolutions}")

    for x in x_steps:
        for y in y_steps:
            # Get points in tile
            mask = (
                (coords[:, 0] >= x) & (coords[:, 0] < x + tile_size) &
                (coords[:, 1] >= y) & (coords[:, 1] < y + tile_size)
            )

            if mask.sum() < min_points:
                continue

            tile_coords = coords[mask]
            tile_features = {k: v[mask] for k, v in features.items()}
            tile_labels = labels[mask] if labels is not None else None

            # Process at each resolution
            for res in resolutions:
                try:
                    tile_data = process_las_tile(
                        tile_coords, tile_features, tile_labels,
                        grid_size=res,
                        compute_normals=compute_normals,
                        compute_curvature=compute_curvature,
                        k_neighbors=k_neighbors
                    )

                    if len(tile_data['coord']) < min_points:
                        continue

                    # Add metadata
                    tile_id = f"{survey_id}_tile_{tile_count:05d}"
                    tile_data['survey_id'] = survey_id
                    tile_data['tile_id'] = tile_id
                    tile_data['bounds'] = (
                        float(tile_coords[:, 0].min()),
                        float(tile_coords[:, 1].min()),
                        float(tile_coords[:, 2].min()),
                        float(tile_coords[:, 0].max()),
                        float(tile_coords[:, 1].max()),
                        float(tile_coords[:, 2].max())
                    )

                    # Save tile
                    tile_path = output_path / f"res_{res:.3f}" / f"{tile_id}.pth"

                    if HAS_TORCH:
                        torch.save(tile_data, tile_path)
                    else:
                        import pickle
                        with open(str(tile_path).replace('.pth', '.pkl'), 'wb') as f:
                            pickle.dump(tile_data, f)
                        tile_path = str(tile_path).replace('.pth', '.pkl')

                    # Record metadata
                    meta = TileMetadata(
                        tile_id=tile_id,
                        survey_id=survey_id,
                        resolution=res,
                        bounds=tile_data['bounds'],
                        point_count=len(tile_data['coord']),
                        file_path=str(tile_path),
                        timestamp=datetime.now().isoformat()
                    )
                    tile_metadata.append(meta)

                except Exception as e:
                    logger.warning(f"Error processing tile at ({x}, {y}) res={res}: {e}")
                    continue

            tile_count += 1

            if tile_count % 100 == 0:
                logger.info(f"Processed {tile_count} tile locations...")

    logger.info(f"Created {len(tile_metadata)} tiles total")
    return tile_metadata


def build_tile_index(
    tile_metadata: List[TileMetadata],
    output_path: str
) -> None:
    """
    Build index of tiles for efficient lookup.

    Args:
        tile_metadata: List of TileMetadata
        output_path: Output path for index file (JSON)
    """
    import json

    index = {
        'tiles': [
            {
                'tile_id': m.tile_id,
                'survey_id': m.survey_id,
                'resolution': m.resolution,
                'bounds': m.bounds,
                'point_count': m.point_count,
                'file_path': m.file_path,
                'timestamp': m.timestamp
            }
            for m in tile_metadata
        ],
        'by_survey': {},
        'by_resolution': {}
    }

    # Index by survey
    for m in tile_metadata:
        if m.survey_id not in index['by_survey']:
            index['by_survey'][m.survey_id] = []
        index['by_survey'][m.survey_id].append(m.tile_id)

    # Index by resolution
    for m in tile_metadata:
        res_key = f"{m.resolution:.3f}"
        if res_key not in index['by_resolution']:
            index['by_resolution'][res_key] = []
        index['by_resolution'][res_key].append(m.tile_id)

    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)

    logger.info(f"Saved tile index to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LAS files into tiles")
    parser.add_argument("--input", "-i", required=True, help="Input LAS file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--tile-size", type=float, default=10.0, help="Tile size in meters")
    parser.add_argument("--resolutions", "-r", type=float, nargs='+',
                       default=[0.02, 0.05, 0.1], help="Voxel resolutions")
    parser.add_argument("--no-normals", action="store_true", help="Skip normal computation")
    parser.add_argument("--no-curvature", action="store_true", help="Skip curvature computation")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Parallel workers")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        metadata = process_las_to_tiles_multiresolution(
            str(input_path),
            args.output,
            tile_size=args.tile_size,
            resolutions=args.resolutions,
            compute_normals=not args.no_normals,
            compute_curvature=not args.no_curvature
        )
        build_tile_index(metadata, f"{args.output}/tile_index.json")

    elif input_path.is_dir():
        # Directory of files
        las_files = list(input_path.glob("*.las")) + list(input_path.glob("*.laz"))
        all_metadata = []

        for las_file in las_files:
            try:
                metadata = process_las_to_tiles_multiresolution(
                    str(las_file),
                    args.output,
                    tile_size=args.tile_size,
                    resolutions=args.resolutions,
                    compute_normals=not args.no_normals,
                    compute_curvature=not args.no_curvature
                )
                all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Error processing {las_file}: {e}")

        build_tile_index(all_metadata, f"{args.output}/tile_index.json")

    else:
        logger.error(f"Input not found: {input_path}")
