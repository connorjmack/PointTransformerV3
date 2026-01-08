"""
Loss functions for CoastalContrast pre-training.

Includes:
- TemporalContrastiveLoss: InfoNCE for temporal correspondences
- SpatialContrastiveLoss: Contrastive loss for spatial contexts
- GeometricLoss: Masked reconstruction of geometric properties
"""

from .contrastive import TemporalContrastiveLoss, SpatialContrastiveLoss
from .geometric import GeometricLoss, MaskedGeometricLoss

__all__ = [
    'TemporalContrastiveLoss',
    'SpatialContrastiveLoss',
    'GeometricLoss',
    'MaskedGeometricLoss'
]
