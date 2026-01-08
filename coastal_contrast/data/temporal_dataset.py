"""
Temporal Pair Dataset for Contrastive Learning

Samples pairs of point clouds from different time steps for
temporal contrastive pre-training.

Usage:
    from coastal_contrast.data.temporal_dataset import TemporalPairDataset

    dataset = TemporalPairDataset(
        tile_index_path="./data/tile_index.json",
        correspondence_dir="./data/correspondences/",
        temporal_pairs_path="./data/temporal_pairs.json"
    )

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

from .transforms import get_pretrain_transforms, Compose
from .correspondence import CorrespondenceSet, load_correspondence_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalPairDataset(Dataset):
    """
    Dataset that yields pairs of temporally related point clouds
    with their correspondences for contrastive learning.
    """

    def __init__(
        self,
        tile_index_path: str,
        correspondence_dir: str,
        temporal_pairs_path: Optional[str] = None,
        resolution: float = 0.02,
        max_points: int = 50000,
        min_correspondences: int = 100,
        transform: Optional[Callable] = None,
        augment_both: bool = True
    ):
        """
        Initialize dataset.

        Args:
            tile_index_path: Path to tile index JSON
            correspondence_dir: Directory containing correspondence files
            temporal_pairs_path: Path to temporal pairs JSON (optional)
            resolution: Target resolution to use
            max_points: Maximum points per tile
            min_correspondences: Minimum correspondences required
            transform: Data transform (applied to both tiles)
            augment_both: Apply independent augmentations to both tiles
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for dataset")

        self.correspondence_dir = Path(correspondence_dir)
        self.resolution = resolution
        self.max_points = max_points
        self.min_correspondences = min_correspondences
        self.transform = transform or get_pretrain_transforms()
        self.augment_both = augment_both

        # Load tile index
        with open(tile_index_path, 'r') as f:
            self.tile_index = json.load(f)

        # Build tile lookup by ID
        self.tiles_by_id = {t['tile_id']: t for t in self.tile_index['tiles']}

        # Filter by resolution
        self.tiles_by_id = {
            k: v for k, v in self.tiles_by_id.items()
            if abs(v['resolution'] - resolution) < 0.001
        }

        # Load temporal pairs or discover from correspondences
        if temporal_pairs_path and Path(temporal_pairs_path).exists():
            with open(temporal_pairs_path, 'r') as f:
                temporal_pairs_data = json.load(f)
            self.temporal_pairs = [
                (p['file_a'], p['file_b'], p.get('delta_days', 0))
                for p in temporal_pairs_data
            ]
        else:
            self.temporal_pairs = self._discover_pairs_from_correspondences()

        logger.info(f"Loaded {len(self.temporal_pairs)} temporal pairs")

    def _discover_pairs_from_correspondences(self) -> List[Tuple[str, str, int]]:
        """Discover pairs from correspondence files."""
        pairs = []

        for corr_file in self.correspondence_dir.glob("corr_*.json"):
            try:
                with open(corr_file, 'r') as f:
                    data = json.load(f)

                source_id = data.get('source_tile_id')
                target_id = data.get('target_tile_id')

                if source_id in self.tiles_by_id and target_id in self.tiles_by_id:
                    source_path = self.tiles_by_id[source_id]['file_path']
                    target_path = self.tiles_by_id[target_id]['file_path']
                    pairs.append((source_path, target_path, 0))

            except Exception as e:
                logger.warning(f"Error loading {corr_file}: {e}")
                continue

        return pairs

    def __len__(self) -> int:
        return len(self.temporal_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a temporal pair sample.

        Returns:
            Dictionary with:
                - source_coord, source_feat, source_normal (if available)
                - target_coord, target_feat, target_normal (if available)
                - correspondences: (M, 2) tensor of [source_idx, target_idx]
                - confidence: (M,) tensor of confidence scores
                - delta_days: time between surveys
        """
        source_path, target_path, delta_days = self.temporal_pairs[idx]

        # Load tiles
        source_tile = torch.load(source_path)
        target_tile = torch.load(target_path)

        # Convert to numpy if needed
        source_tile = self._to_numpy(source_tile)
        target_tile = self._to_numpy(target_tile)

        # Apply transforms
        if self.transform:
            source_tile = self.transform(source_tile)
            if self.augment_both:
                target_tile = self.transform(target_tile)

        # Load correspondences
        corr = self._load_correspondences(source_tile, target_tile)

        # Subsample if needed
        source_tile, target_tile, corr = self._subsample(
            source_tile, target_tile, corr
        )

        # Build output
        output = {
            'source_coord': torch.from_numpy(source_tile['coord']).float(),
            'source_feat': torch.from_numpy(source_tile['feat']).float(),
            'target_coord': torch.from_numpy(target_tile['coord']).float(),
            'target_feat': torch.from_numpy(target_tile['feat']).float(),
            'delta_days': delta_days
        }

        # Add normals if available
        if 'normal' in source_tile and source_tile['normal'] is not None:
            output['source_normal'] = torch.from_numpy(source_tile['normal']).float()
        if 'normal' in target_tile and target_tile['normal'] is not None:
            output['target_normal'] = torch.from_numpy(target_tile['normal']).float()

        # Add correspondences
        if corr is not None and len(corr.source_indices) > 0:
            output['correspondences'] = torch.stack([
                torch.from_numpy(corr.source_indices).long(),
                torch.from_numpy(corr.target_indices).long()
            ], dim=1)
            output['confidence'] = torch.from_numpy(corr.confidences).float()
        else:
            # Empty correspondences
            output['correspondences'] = torch.zeros((0, 2), dtype=torch.long)
            output['confidence'] = torch.zeros(0, dtype=torch.float)

        return output

    def _to_numpy(self, tile: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tile data to numpy arrays."""
        result = {}
        for key, value in tile.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.numpy()
            else:
                result[key] = value
        return result

    def _load_correspondences(
        self,
        source_tile: Dict[str, Any],
        target_tile: Dict[str, Any]
    ) -> Optional[CorrespondenceSet]:
        """Load or compute correspondences for tile pair."""
        source_id = source_tile.get('tile_id', 'unknown')
        target_id = target_tile.get('tile_id', 'unknown')

        # Try to load pre-computed correspondences
        corr_path = self.correspondence_dir / f"corr_{source_id}_{target_id}.json"

        if corr_path.exists():
            try:
                return load_correspondence_set(str(corr_path))
            except Exception:
                pass

        # Fall back to simple nearest-neighbor correspondences
        return self._compute_simple_correspondences(source_tile, target_tile)

    def _compute_simple_correspondences(
        self,
        source_tile: Dict[str, Any],
        target_tile: Dict[str, Any],
        max_distance: float = 0.5
    ) -> CorrespondenceSet:
        """Compute simple nearest-neighbor correspondences."""
        from scipy.spatial import cKDTree

        source_coords = source_tile['coord']
        target_coords = target_tile['coord']

        tree = cKDTree(target_coords)
        distances, indices = tree.query(source_coords, k=1)

        # Filter by distance
        valid = distances < max_distance
        source_idx = np.where(valid)[0]
        target_idx = indices[valid]
        dists = distances[valid]

        # Simple confidence based on distance
        confidences = 1.0 - (dists / max_distance)

        return CorrespondenceSet(
            source_indices=source_idx.astype(np.int64),
            target_indices=target_idx.astype(np.int64),
            distances=dists.astype(np.float32),
            confidences=confidences.astype(np.float32),
            change_mask=np.zeros(len(source_idx), dtype=bool),
            source_survey_id=source_tile.get('survey_id', 'unknown'),
            target_survey_id=target_tile.get('survey_id', 'unknown'),
            source_tile_id=source_tile.get('tile_id', 'unknown'),
            target_tile_id=target_tile.get('tile_id', 'unknown')
        )

    def _subsample(
        self,
        source_tile: Dict[str, Any],
        target_tile: Dict[str, Any],
        corr: Optional[CorrespondenceSet]
    ) -> Tuple[Dict, Dict, Optional[CorrespondenceSet]]:
        """Subsample tiles to max_points."""
        # Subsample source
        n_source = len(source_tile['coord'])
        if n_source > self.max_points:
            idx = np.random.choice(n_source, self.max_points, replace=False)
            idx = np.sort(idx)

            # Create mapping from old to new indices
            old_to_new = {old: new for new, old in enumerate(idx)}

            for key in ['coord', 'feat', 'normal', 'curvature']:
                if key in source_tile and source_tile[key] is not None:
                    source_tile[key] = source_tile[key][idx]

            # Update correspondences
            if corr is not None:
                valid_corr = np.isin(corr.source_indices, idx)
                new_source_idx = np.array([
                    old_to_new[i] for i in corr.source_indices[valid_corr]
                ])
                corr = CorrespondenceSet(
                    source_indices=new_source_idx,
                    target_indices=corr.target_indices[valid_corr],
                    distances=corr.distances[valid_corr],
                    confidences=corr.confidences[valid_corr],
                    change_mask=corr.change_mask[valid_corr],
                    source_survey_id=corr.source_survey_id,
                    target_survey_id=corr.target_survey_id,
                    source_tile_id=corr.source_tile_id,
                    target_tile_id=corr.target_tile_id
                )

        # Subsample target
        n_target = len(target_tile['coord'])
        if n_target > self.max_points:
            idx = np.random.choice(n_target, self.max_points, replace=False)
            idx = np.sort(idx)

            old_to_new = {old: new for new, old in enumerate(idx)}

            for key in ['coord', 'feat', 'normal', 'curvature']:
                if key in target_tile and target_tile[key] is not None:
                    target_tile[key] = target_tile[key][idx]

            if corr is not None:
                valid_corr = np.isin(corr.target_indices, idx)
                new_target_idx = np.array([
                    old_to_new[i] for i in corr.target_indices[valid_corr]
                ])
                corr = CorrespondenceSet(
                    source_indices=corr.source_indices[valid_corr],
                    target_indices=new_target_idx,
                    distances=corr.distances[valid_corr],
                    confidences=corr.confidences[valid_corr],
                    change_mask=corr.change_mask[valid_corr],
                    source_survey_id=corr.source_survey_id,
                    target_survey_id=corr.target_survey_id,
                    source_tile_id=corr.source_tile_id,
                    target_tile_id=corr.target_tile_id
                )

        return source_tile, target_tile, corr

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for variable-size point clouds.

        Uses offset-based batching (Pointcept style).
        """
        collated = {
            'source_coord': [],
            'source_feat': [],
            'target_coord': [],
            'target_feat': [],
            'source_offset': [0],
            'target_offset': [0],
            'correspondences': [],
            'confidence': [],
            'corr_batch': [],
            'delta_days': []
        }

        source_cumsum = 0
        target_cumsum = 0

        for i, sample in enumerate(batch):
            n_source = len(sample['source_coord'])
            n_target = len(sample['target_coord'])

            collated['source_coord'].append(sample['source_coord'])
            collated['source_feat'].append(sample['source_feat'])
            collated['target_coord'].append(sample['target_coord'])
            collated['target_feat'].append(sample['target_feat'])

            source_cumsum += n_source
            target_cumsum += n_target
            collated['source_offset'].append(source_cumsum)
            collated['target_offset'].append(target_cumsum)

            # Offset correspondences
            if len(sample['correspondences']) > 0:
                corr = sample['correspondences'].clone()
                corr[:, 0] += source_cumsum - n_source
                corr[:, 1] += target_cumsum - n_target
                collated['correspondences'].append(corr)
                collated['confidence'].append(sample['confidence'])
                collated['corr_batch'].append(
                    torch.full((len(corr),), i, dtype=torch.long)
                )

            collated['delta_days'].append(sample['delta_days'])

            # Optional fields
            for key in ['source_normal', 'target_normal']:
                if key in sample:
                    if key not in collated:
                        collated[key] = []
                    collated[key].append(sample[key])

        # Concatenate
        collated['source_coord'] = torch.cat(collated['source_coord'], dim=0)
        collated['source_feat'] = torch.cat(collated['source_feat'], dim=0)
        collated['target_coord'] = torch.cat(collated['target_coord'], dim=0)
        collated['target_feat'] = torch.cat(collated['target_feat'], dim=0)
        collated['source_offset'] = torch.tensor(collated['source_offset'][1:], dtype=torch.long)
        collated['target_offset'] = torch.tensor(collated['target_offset'][1:], dtype=torch.long)

        if collated['correspondences']:
            collated['correspondences'] = torch.cat(collated['correspondences'], dim=0)
            collated['confidence'] = torch.cat(collated['confidence'], dim=0)
            collated['corr_batch'] = torch.cat(collated['corr_batch'], dim=0)
        else:
            collated['correspondences'] = torch.zeros((0, 2), dtype=torch.long)
            collated['confidence'] = torch.zeros(0, dtype=torch.float)
            collated['corr_batch'] = torch.zeros(0, dtype=torch.long)

        collated['delta_days'] = torch.tensor(collated['delta_days'], dtype=torch.long)

        for key in ['source_normal', 'target_normal']:
            if key in collated:
                collated[key] = torch.cat(collated[key], dim=0)

        return collated


class SingleTileDataset(Dataset):
    """
    Simple dataset for fine-tuning on individual tiles with labels.
    """

    def __init__(
        self,
        tile_paths: List[str],
        transform: Optional[Callable] = None,
        max_points: int = 50000
    ):
        self.tile_paths = tile_paths
        self.transform = transform
        self.max_points = max_points

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tile = torch.load(self.tile_paths[idx])

        # Convert to numpy
        for key in tile:
            if isinstance(tile[key], torch.Tensor):
                tile[key] = tile[key].numpy()

        # Apply transform
        if self.transform:
            tile = self.transform(tile)

        # Subsample
        n_points = len(tile['coord'])
        if n_points > self.max_points:
            idx = np.random.choice(n_points, self.max_points, replace=False)
            for key in ['coord', 'feat', 'normal', 'curvature', 'segment']:
                if key in tile and tile[key] is not None:
                    tile[key] = tile[key][idx]

        # Convert to tensors
        output = {
            'coord': torch.from_numpy(tile['coord']).float(),
            'feat': torch.from_numpy(tile['feat']).float()
        }

        if 'segment' in tile and tile['segment'] is not None:
            output['segment'] = torch.from_numpy(tile['segment']).long()

        if 'normal' in tile and tile['normal'] is not None:
            output['normal'] = torch.from_numpy(tile['normal']).float()

        return output

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate with offset-based batching."""
        collated = {
            'coord': [],
            'feat': [],
            'offset': [0]
        }

        cumsum = 0
        for sample in batch:
            n = len(sample['coord'])
            collated['coord'].append(sample['coord'])
            collated['feat'].append(sample['feat'])
            cumsum += n
            collated['offset'].append(cumsum)

            for key in ['segment', 'normal']:
                if key in sample:
                    if key not in collated:
                        collated[key] = []
                    collated[key].append(sample[key])

        collated['coord'] = torch.cat(collated['coord'], dim=0)
        collated['feat'] = torch.cat(collated['feat'], dim=0)
        collated['offset'] = torch.tensor(collated['offset'][1:], dtype=torch.long)

        for key in ['segment', 'normal']:
            if key in collated:
                collated[key] = torch.cat(collated[key], dim=0)

        return collated
