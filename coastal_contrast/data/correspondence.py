"""
Temporal Correspondence Mining

Finds point correspondences between co-registered temporal surveys.
These correspondences form the positive pairs for contrastive learning.

Key insight: Points at the same location across time should have
similar features (unless actual change occurred).

Usage:
    from coastal_contrast.data.correspondence import CorrespondenceMiner

    miner = CorrespondenceMiner(max_distance=0.5, normal_threshold=0.8)
    correspondences = miner.mine_tile_pair(tile_t1, tile_t2)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

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
class CorrespondenceSet:
    """Set of correspondences between two point clouds."""
    source_indices: np.ndarray      # (M,) indices in source
    target_indices: np.ndarray      # (M,) indices in target
    distances: np.ndarray           # (M,) spatial distances
    confidences: np.ndarray         # (M,) confidence scores (0-1)
    change_mask: np.ndarray         # (M,) True if likely change
    source_survey_id: str
    target_survey_id: str
    source_tile_id: str
    target_tile_id: str

    def __len__(self) -> int:
        return len(self.source_indices)

    @property
    def stable_correspondences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return only stable (non-change) correspondences."""
        mask = ~self.change_mask
        return self.source_indices[mask], self.target_indices[mask]

    @property
    def change_correspondences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return only change correspondences."""
        mask = self.change_mask
        return self.source_indices[mask], self.target_indices[mask]

    def filter_by_confidence(self, min_confidence: float) -> 'CorrespondenceSet':
        """Return new set with only high-confidence correspondences."""
        mask = self.confidences >= min_confidence
        return CorrespondenceSet(
            source_indices=self.source_indices[mask],
            target_indices=self.target_indices[mask],
            distances=self.distances[mask],
            confidences=self.confidences[mask],
            change_mask=self.change_mask[mask],
            source_survey_id=self.source_survey_id,
            target_survey_id=self.target_survey_id,
            source_tile_id=self.source_tile_id,
            target_tile_id=self.target_tile_id
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_indices': self.source_indices.tolist(),
            'target_indices': self.target_indices.tolist(),
            'distances': self.distances.tolist(),
            'confidences': self.confidences.tolist(),
            'change_mask': self.change_mask.tolist(),
            'source_survey_id': self.source_survey_id,
            'target_survey_id': self.target_survey_id,
            'source_tile_id': self.source_tile_id,
            'target_tile_id': self.target_tile_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrespondenceSet':
        """Create from dictionary."""
        return cls(
            source_indices=np.array(data['source_indices'], dtype=np.int64),
            target_indices=np.array(data['target_indices'], dtype=np.int64),
            distances=np.array(data['distances'], dtype=np.float32),
            confidences=np.array(data['confidences'], dtype=np.float32),
            change_mask=np.array(data['change_mask'], dtype=bool),
            source_survey_id=data['source_survey_id'],
            target_survey_id=data['target_survey_id'],
            source_tile_id=data['source_tile_id'],
            target_tile_id=data['target_tile_id']
        )


class CorrespondenceMiner:
    """
    Mine temporal correspondences between point clouds.

    For each point in source, finds potential correspondences in target
    based on spatial proximity and geometric similarity.
    """

    def __init__(
        self,
        max_distance: float = 0.5,
        k_neighbors: int = 5,
        normal_threshold: float = 0.8,
        change_confidence_threshold: float = 0.3,
        use_normals: bool = True,
        use_curvature: bool = True
    ):
        """
        Initialize correspondence miner.

        Args:
            max_distance: Maximum distance for correspondence (meters)
            k_neighbors: Number of neighbors to consider
            normal_threshold: Minimum normal agreement (dot product)
            change_confidence_threshold: Below this = potential change
            use_normals: Use normals for confidence scoring
            use_curvature: Use curvature for confidence scoring
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required. Install with: pip install scipy")

        self.max_distance = max_distance
        self.k_neighbors = k_neighbors
        self.normal_threshold = normal_threshold
        self.change_threshold = change_confidence_threshold
        self.use_normals = use_normals
        self.use_curvature = use_curvature

    def mine_correspondences(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        source_normals: Optional[np.ndarray] = None,
        target_normals: Optional[np.ndarray] = None,
        source_curvature: Optional[np.ndarray] = None,
        target_curvature: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Mine correspondences between two point clouds.

        Args:
            source_coords: (N, 3) source coordinates
            target_coords: (M, 3) target coordinates
            source_normals: (N, 3) source normals (optional)
            target_normals: (M, 3) target normals (optional)
            source_curvature: (N,) source curvature (optional)
            target_curvature: (M,) target curvature (optional)

        Returns:
            Tuple of (source_idx, target_idx, distances, confidences, change_mask)
        """
        # Build KD-tree on target
        tree = cKDTree(target_coords)

        # Query for each source point
        distances, indices = tree.query(
            source_coords,
            k=self.k_neighbors,
            distance_upper_bound=self.max_distance
        )

        # Process correspondences
        source_idx_list = []
        target_idx_list = []
        distance_list = []
        confidence_list = []

        for i in range(len(source_coords)):
            for j in range(self.k_neighbors):
                d = distances[i, j]
                t_idx = indices[i, j]

                # Skip if no valid correspondence
                if np.isinf(d) or t_idx >= len(target_coords):
                    continue

                # Compute confidence
                confidence = self._compute_confidence(
                    d, i, t_idx,
                    source_normals, target_normals,
                    source_curvature, target_curvature
                )

                source_idx_list.append(i)
                target_idx_list.append(t_idx)
                distance_list.append(d)
                confidence_list.append(confidence)

        if not source_idx_list:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=bool)
            )

        source_idx = np.array(source_idx_list, dtype=np.int64)
        target_idx = np.array(target_idx_list, dtype=np.int64)
        dist = np.array(distance_list, dtype=np.float32)
        conf = np.array(confidence_list, dtype=np.float32)

        # Flag low-confidence as potential change
        change_mask = conf < self.change_threshold

        return source_idx, target_idx, dist, conf, change_mask

    def _compute_confidence(
        self,
        distance: float,
        source_idx: int,
        target_idx: int,
        source_normals: Optional[np.ndarray],
        target_normals: Optional[np.ndarray],
        source_curvature: Optional[np.ndarray],
        target_curvature: Optional[np.ndarray]
    ) -> float:
        """Compute confidence score for a correspondence."""
        # Distance component (closer = higher confidence)
        dist_conf = 1.0 - (distance / self.max_distance)

        # Normal agreement component
        if self.use_normals and source_normals is not None and target_normals is not None:
            normal_dot = np.dot(source_normals[source_idx], target_normals[target_idx])
            normal_conf = max(0.0, normal_dot)  # Clip negative
        else:
            normal_conf = 1.0

        # Curvature similarity component
        if self.use_curvature and source_curvature is not None and target_curvature is not None:
            curv_diff = abs(source_curvature[source_idx] - target_curvature[target_idx])
            curv_conf = max(0.0, 1.0 - curv_diff * 10)  # Scale factor
        else:
            curv_conf = 1.0

        # Combined confidence
        confidence = dist_conf * normal_conf * curv_conf

        return float(confidence)

    def mine_tile_pair(
        self,
        source_tile: Dict[str, Any],
        target_tile: Dict[str, Any]
    ) -> CorrespondenceSet:
        """
        Mine correspondences between two tile dictionaries.

        Args:
            source_tile: Source tile dict with 'coord', optionally 'normal', 'curvature'
            target_tile: Target tile dict

        Returns:
            CorrespondenceSet object
        """
        source_idx, target_idx, distances, confidences, change_mask = self.mine_correspondences(
            source_coords=source_tile['coord'],
            target_coords=target_tile['coord'],
            source_normals=source_tile.get('normal'),
            target_normals=target_tile.get('normal'),
            source_curvature=source_tile.get('curvature'),
            target_curvature=target_tile.get('curvature')
        )

        return CorrespondenceSet(
            source_indices=source_idx,
            target_indices=target_idx,
            distances=distances,
            confidences=confidences,
            change_mask=change_mask,
            source_survey_id=source_tile.get('survey_id', 'unknown'),
            target_survey_id=target_tile.get('survey_id', 'unknown'),
            source_tile_id=source_tile.get('tile_id', 'unknown'),
            target_tile_id=target_tile.get('tile_id', 'unknown')
        )


def mine_temporal_correspondences_batch(
    tile_pairs: List[Tuple[str, str]],
    output_dir: str,
    miner: Optional[CorrespondenceMiner] = None,
    n_workers: int = 4
) -> int:
    """
    Mine correspondences for multiple tile pairs.

    Args:
        tile_pairs: List of (source_path, target_path) tuples
        output_dir: Output directory for correspondence files
        miner: CorrespondenceMiner instance (creates default if None)
        n_workers: Number of parallel workers

    Returns:
        Number of correspondence files created
    """
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if miner is None:
        miner = CorrespondenceMiner()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created = 0

    for source_path, target_path in tile_pairs:
        try:
            # Load tiles
            if HAS_TORCH:
                source_tile = torch.load(source_path)
                target_tile = torch.load(target_path)
            else:
                import pickle
                with open(source_path, 'rb') as f:
                    source_tile = pickle.load(f)
                with open(target_path, 'rb') as f:
                    target_tile = pickle.load(f)

            # Mine correspondences
            corr = miner.mine_tile_pair(source_tile, target_tile)

            if len(corr) == 0:
                continue

            # Save
            source_id = source_tile.get('tile_id', Path(source_path).stem)
            target_id = target_tile.get('tile_id', Path(target_path).stem)
            output_file = output_path / f"corr_{source_id}_{target_id}.json"

            with open(output_file, 'w') as f:
                json.dump(corr.to_dict(), f)

            created += 1

            if created % 100 == 0:
                logger.info(f"Created {created} correspondence files...")

        except Exception as e:
            logger.warning(f"Error processing {source_path} -> {target_path}: {e}")
            continue

    logger.info(f"Created {created} correspondence files")
    return created


def load_correspondence_set(path: str) -> CorrespondenceSet:
    """Load correspondence set from JSON file."""
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return CorrespondenceSet.from_dict(data)


def find_overlapping_tiles(
    tile_index: Dict[str, Any],
    survey_a_id: str,
    survey_b_id: str,
    min_overlap: float = 0.5
) -> List[Tuple[str, str]]:
    """
    Find pairs of tiles from two surveys that spatially overlap.

    Args:
        tile_index: Tile index dictionary
        survey_a_id: First survey ID
        survey_b_id: Second survey ID
        min_overlap: Minimum overlap ratio (IoU)

    Returns:
        List of (tile_a_path, tile_b_path) tuples
    """
    # Get tiles for each survey
    tiles_a = [t for t in tile_index['tiles'] if t['survey_id'] == survey_a_id]
    tiles_b = [t for t in tile_index['tiles'] if t['survey_id'] == survey_b_id]

    pairs = []

    for ta in tiles_a:
        for tb in tiles_b:
            # Check same resolution
            if abs(ta['resolution'] - tb['resolution']) > 0.001:
                continue

            # Check overlap
            overlap = _compute_tile_overlap(ta['bounds'], tb['bounds'])
            if overlap >= min_overlap:
                pairs.append((ta['file_path'], tb['file_path']))

    return pairs


def _compute_tile_overlap(bounds_a: Tuple, bounds_b: Tuple) -> float:
    """Compute IoU overlap between two tile bounds."""
    # bounds = (min_x, min_y, min_z, max_x, max_y, max_z)
    a_min_x, a_min_y, _, a_max_x, a_max_y, _ = bounds_a
    b_min_x, b_min_y, _, b_max_x, b_max_y, _ = bounds_b

    # Intersection
    i_min_x = max(a_min_x, b_min_x)
    i_min_y = max(a_min_y, b_min_y)
    i_max_x = min(a_max_x, b_max_x)
    i_max_y = min(a_max_y, b_max_y)

    if i_max_x <= i_min_x or i_max_y <= i_min_y:
        return 0.0

    intersection = (i_max_x - i_min_x) * (i_max_y - i_min_y)

    # Union
    area_a = (a_max_x - a_min_x) * (a_max_y - a_min_y)
    area_b = (b_max_x - b_min_x) * (b_max_y - b_min_y)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Mine temporal correspondences")
    parser.add_argument("--source", "-s", required=True, help="Source tile path")
    parser.add_argument("--target", "-t", required=True, help="Target tile path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--max-distance", type=float, default=0.5, help="Max correspondence distance")
    parser.add_argument("--k-neighbors", type=int, default=5, help="K neighbors")
    parser.add_argument("--normal-threshold", type=float, default=0.8, help="Normal agreement threshold")

    args = parser.parse_args()

    miner = CorrespondenceMiner(
        max_distance=args.max_distance,
        k_neighbors=args.k_neighbors,
        normal_threshold=args.normal_threshold
    )

    # Load tiles
    if HAS_TORCH:
        source_tile = torch.load(args.source)
        target_tile = torch.load(args.target)
    else:
        import pickle
        with open(args.source, 'rb') as f:
            source_tile = pickle.load(f)
        with open(args.target, 'rb') as f:
            target_tile = pickle.load(f)

    # Mine
    corr = miner.mine_tile_pair(source_tile, target_tile)

    print(f"\nCorrespondence Mining Results:")
    print(f"  Total correspondences: {len(corr)}")
    print(f"  Stable correspondences: {(~corr.change_mask).sum()}")
    print(f"  Change candidates: {corr.change_mask.sum()}")
    print(f"  Mean confidence: {corr.confidences.mean():.3f}")
    print(f"  Mean distance: {corr.distances.mean():.4f}m")

    # Save
    with open(args.output, 'w') as f:
        json.dump(corr.to_dict(), f, indent=2)

    print(f"\nSaved to {args.output}")
