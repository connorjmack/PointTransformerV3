"""
Model components for CoastalContrast.

Includes:
- CoastalContrast: Self-supervised pre-training model
- ProjectionHead: MLP projection for contrastive learning
- SegmentationHead: Semantic segmentation head
- GeometricHead: Geometric property prediction head
"""

from .coastal_contrast import CoastalContrast
from .heads import ProjectionHead, SegmentationHead, GeometricHead

__all__ = [
    'CoastalContrast',
    'ProjectionHead',
    'SegmentationHead',
    'GeometricHead'
]
