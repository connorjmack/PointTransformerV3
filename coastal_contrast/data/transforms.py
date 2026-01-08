"""
Point Cloud Data Augmentation Transforms

Transformations for training data augmentation during
self-supervised pre-training and fine-tuning.

Usage:
    from coastal_contrast.data.transforms import Compose, RandomRotation, RandomJitter

    transform = Compose([
        RandomRotation(axis='z'),
        RandomJitter(sigma=0.01),
        RandomScale(scale_range=(0.9, 1.1))
    ])

    augmented_data = transform(data_dict)
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import copy


class BaseTransform:
    """Base class for point cloud transforms."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(BaseTransform):
    """Compose multiple transforms."""

    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        transform_str = ', '.join([repr(t) for t in self.transforms])
        return f"Compose([{transform_str}])"


class RandomApply(BaseTransform):
    """Apply transform with given probability."""

    def __init__(self, transform: BaseTransform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() < self.p:
            return self.transform(data)
        return data


class CopyData(BaseTransform):
    """Create a deep copy of the data dictionary."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(data)


class NormalizeCoordinates(BaseTransform):
    """
    Normalize coordinates to zero-mean and unit scale.
    """

    def __init__(self, center: bool = True, scale: bool = False):
        self.center = center
        self.scale = scale

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        coord = data['coord'].copy()

        if self.center:
            centroid = coord.mean(axis=0)
            coord = coord - centroid
            data['centroid'] = centroid

        if self.scale:
            max_dist = np.max(np.linalg.norm(coord, axis=1))
            if max_dist > 0:
                coord = coord / max_dist
                data['scale'] = max_dist

        data['coord'] = coord
        return data


class RandomRotation(BaseTransform):
    """
    Random rotation around specified axis or axes.
    """

    def __init__(
        self,
        axis: str = 'z',
        angle_range: Tuple[float, float] = (0, 2 * np.pi),
        p: float = 1.0
    ):
        """
        Args:
            axis: 'x', 'y', 'z', or 'xyz' for random 3D rotation
            angle_range: Range of rotation angles in radians
            p: Probability of applying transform
        """
        self.axis = axis
        self.angle_range = angle_range
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        coord = data['coord'].copy()
        angle = np.random.uniform(*self.angle_range)

        if self.axis == 'z':
            R = self._rotation_matrix_z(angle)
        elif self.axis == 'x':
            R = self._rotation_matrix_x(angle)
        elif self.axis == 'y':
            R = self._rotation_matrix_y(angle)
        elif self.axis == 'xyz':
            angles = np.random.uniform(*self.angle_range, size=3)
            Rx = self._rotation_matrix_x(angles[0])
            Ry = self._rotation_matrix_y(angles[1])
            Rz = self._rotation_matrix_z(angles[2])
            R = Rz @ Ry @ Rx
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

        data['coord'] = (R @ coord.T).T

        # Rotate normals if present
        if 'normal' in data and data['normal'] is not None:
            data['normal'] = (R @ data['normal'].T).T

        return data

    @staticmethod
    def _rotation_matrix_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def _rotation_matrix_x(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _rotation_matrix_y(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class RandomFlip(BaseTransform):
    """Random flip along specified axis."""

    def __init__(self, axis: str = 'x', p: float = 0.5):
        self.axis = axis
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        coord = data['coord'].copy()
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.axis]

        coord[:, axis_idx] = -coord[:, axis_idx]
        data['coord'] = coord

        # Flip normals if present
        if 'normal' in data and data['normal'] is not None:
            normal = data['normal'].copy()
            normal[:, axis_idx] = -normal[:, axis_idx]
            data['normal'] = normal

        return data


class RandomScale(BaseTransform):
    """Random uniform scaling."""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        anisotropic: bool = False,
        p: float = 1.0
    ):
        """
        Args:
            scale_range: (min_scale, max_scale)
            anisotropic: If True, scale each axis independently
            p: Probability of applying transform
        """
        self.scale_range = scale_range
        self.anisotropic = anisotropic
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        if self.anisotropic:
            scale = np.random.uniform(*self.scale_range, size=3)
        else:
            scale = np.random.uniform(*self.scale_range)

        data['coord'] = data['coord'] * scale
        return data


class RandomJitter(BaseTransform):
    """Add Gaussian noise to coordinates."""

    def __init__(
        self,
        sigma: float = 0.01,
        clip: Optional[float] = 0.05,
        p: float = 1.0
    ):
        """
        Args:
            sigma: Standard deviation of Gaussian noise
            clip: Clip noise to this range (None for no clipping)
            p: Probability of applying transform
        """
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        noise = np.random.normal(0, self.sigma, size=data['coord'].shape)

        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)

        data['coord'] = data['coord'] + noise.astype(data['coord'].dtype)
        return data


class RandomTranslation(BaseTransform):
    """Random translation."""

    def __init__(
        self,
        shift_range: float = 0.2,
        p: float = 1.0
    ):
        self.shift_range = shift_range
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        shift = np.random.uniform(-self.shift_range, self.shift_range, size=3)
        data['coord'] = data['coord'] + shift
        return data


class RandomDropout(BaseTransform):
    """Randomly drop points."""

    def __init__(
        self,
        dropout_ratio: float = 0.1,
        p: float = 0.5
    ):
        """
        Args:
            dropout_ratio: Fraction of points to drop
            p: Probability of applying transform
        """
        self.dropout_ratio = dropout_ratio
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        n_points = len(data['coord'])
        n_keep = int(n_points * (1 - self.dropout_ratio))

        if n_keep < 10:  # Keep minimum points
            return data

        indices = np.random.choice(n_points, n_keep, replace=False)
        indices = np.sort(indices)

        # Apply to all arrays
        for key in ['coord', 'feat', 'normal', 'curvature', 'segment',
                    'planarity', 'verticality']:
            if key in data and data[key] is not None:
                data[key] = data[key][indices]

        return data


class RandomSubsample(BaseTransform):
    """Subsample to fixed number of points."""

    def __init__(self, num_points: int = 50000):
        self.num_points = num_points

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        n_points = len(data['coord'])

        if n_points <= self.num_points:
            return data

        indices = np.random.choice(n_points, self.num_points, replace=False)
        indices = np.sort(indices)

        for key in ['coord', 'feat', 'normal', 'curvature', 'segment',
                    'planarity', 'verticality']:
            if key in data and data[key] is not None:
                data[key] = data[key][indices]

        return data


class ElasticDistortion(BaseTransform):
    """
    Elastic distortion for point clouds.
    Applies smooth random deformation to coordinates.
    """

    def __init__(
        self,
        granularity: float = 0.2,
        magnitude: float = 0.4,
        p: float = 0.5
    ):
        """
        Args:
            granularity: Size of distortion grid
            magnitude: Strength of distortion
            p: Probability of applying transform
        """
        self.granularity = granularity
        self.magnitude = magnitude
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        coord = data['coord'].copy()

        # Create noise grid
        min_coord = coord.min(axis=0)
        max_coord = coord.max(axis=0)

        grid_size = ((max_coord - min_coord) / self.granularity + 1).astype(int)
        noise = np.random.randn(*grid_size, 3) * self.magnitude

        # Interpolate noise at point locations
        normalized = (coord - min_coord) / self.granularity
        indices = normalized.astype(int)
        indices = np.clip(indices, 0, np.array(grid_size) - 1)

        displacement = noise[indices[:, 0], indices[:, 1], indices[:, 2]]
        data['coord'] = coord + displacement.astype(coord.dtype)

        return data


class ChromaticAutoContrast(BaseTransform):
    """
    Random contrast adjustment for intensity/color features.
    """

    def __init__(
        self,
        blend_factor: Tuple[float, float] = (0.5, 1.0),
        p: float = 0.5
    ):
        self.blend_factor = blend_factor
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return data

        if 'feat' not in data:
            return data

        feat = data['feat'].copy()

        # Auto-contrast: stretch to full range
        lo = feat.min(axis=0)
        hi = feat.max(axis=0)

        scale = np.where(hi - lo > 1e-6, 1.0 / (hi - lo), 1.0)
        contrast_feat = (feat - lo) * scale

        # Blend with original
        blend = np.random.uniform(*self.blend_factor)
        data['feat'] = (blend * contrast_feat + (1 - blend) * feat).astype(feat.dtype)

        return data


class MaskPoints(BaseTransform):
    """
    Mask (zero out) random points for masked autoencoding pre-training.
    """

    def __init__(
        self,
        mask_ratio: float = 0.6,
        mask_type: str = 'random'
    ):
        """
        Args:
            mask_ratio: Fraction of points to mask
            mask_type: 'random' or 'block' masking
        """
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        n_points = len(data['coord'])
        n_mask = int(n_points * self.mask_ratio)

        if self.mask_type == 'random':
            mask_indices = np.random.choice(n_points, n_mask, replace=False)
        elif self.mask_type == 'block':
            # Block masking: mask contiguous region
            center_idx = np.random.randint(n_points)
            center = data['coord'][center_idx]

            # Mask nearest points to center
            distances = np.linalg.norm(data['coord'] - center, axis=1)
            mask_indices = np.argsort(distances)[:n_mask]
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        # Create mask
        mask = np.zeros(n_points, dtype=bool)
        mask[mask_indices] = True

        data['mask'] = mask
        data['mask_indices'] = mask_indices

        return data


class ToTensor(BaseTransform):
    """Convert numpy arrays to PyTorch tensors."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:
            return data

        for key in ['coord', 'feat', 'normal', 'curvature', 'segment',
                    'planarity', 'verticality', 'mask']:
            if key in data and data[key] is not None:
                if isinstance(data[key], np.ndarray):
                    if key == 'segment':
                        data[key] = torch.from_numpy(data[key]).long()
                    elif key == 'mask':
                        data[key] = torch.from_numpy(data[key]).bool()
                    else:
                        data[key] = torch.from_numpy(data[key]).float()

        return data


# =============================================================================
# Pre-built transform pipelines
# =============================================================================

def get_pretrain_transforms(
    jitter_sigma: float = 0.01,
    rotation: bool = True,
    scale: bool = True,
    dropout: bool = True
) -> Compose:
    """
    Get standard transforms for self-supervised pre-training.
    """
    transforms = [CopyData()]

    if rotation:
        transforms.append(RandomRotation(axis='z', p=0.8))

    if scale:
        transforms.append(RandomScale(scale_range=(0.9, 1.1), p=0.5))

    transforms.append(RandomJitter(sigma=jitter_sigma, p=0.8))

    if dropout:
        transforms.append(RandomDropout(dropout_ratio=0.1, p=0.5))

    transforms.append(NormalizeCoordinates(center=True))

    return Compose(transforms)


def get_finetune_transforms(strong: bool = False) -> Compose:
    """
    Get transforms for fine-tuning.
    """
    transforms = [
        CopyData(),
        RandomRotation(axis='z', p=0.5),
        RandomFlip(axis='x', p=0.5),
        RandomJitter(sigma=0.005, p=0.5),
        NormalizeCoordinates(center=True)
    ]

    if strong:
        transforms.insert(3, RandomScale(scale_range=(0.8, 1.2), p=0.5))
        transforms.insert(4, ElasticDistortion(p=0.3))
        transforms.insert(5, RandomDropout(dropout_ratio=0.2, p=0.3))

    return Compose(transforms)


def get_eval_transforms() -> Compose:
    """
    Get transforms for evaluation (minimal augmentation).
    """
    return Compose([
        CopyData(),
        NormalizeCoordinates(center=True)
    ])
