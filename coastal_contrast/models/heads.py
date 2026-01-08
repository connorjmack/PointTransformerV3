"""
Model Heads for Pre-Training and Fine-Tuning

Projection heads for contrastive learning and task-specific heads
for downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Projects backbone features to a lower-dimensional space
    for computing contrastive loss.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_layers: int = 2,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: Input feature dimension (from backbone)
            hidden_channels: Hidden layer dimension
            out_channels: Output projection dimension
            num_layers: Number of MLP layers
            use_bn: Use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_channels, hidden_channels))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_channels = hidden_channels

        layers.append(nn.Linear(current_channels, out_channels))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features.

        Args:
            x: (N, in_channels) features

        Returns:
            (N, out_channels) projected features
        """
        return self.mlp(x)


class SegmentationHead(nn.Module):
    """
    Semantic segmentation head.

    Simple linear classifier on top of backbone features.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels: Input feature dimension
            num_classes: Number of segmentation classes
            hidden_channels: Hidden layer dimension (None = linear)
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_channels is None:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_channels, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits.

        Args:
            x: (N, in_channels) features

        Returns:
            (N, num_classes) class logits
        """
        return self.head(x)


class GeometricHead(nn.Module):
    """
    Head for predicting geometric properties.

    Predicts surface normals, curvature, and optionally occupancy.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        predict_normals: bool = True,
        predict_curvature: bool = True,
        predict_occupancy: bool = False,
        occupancy_grid_size: int = 27  # 3x3x3 grid
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            predict_normals: Predict surface normals
            predict_curvature: Predict local curvature
            predict_occupancy: Predict local occupancy grid
            occupancy_grid_size: Size of occupancy grid (e.g., 27 for 3x3x3)
        """
        super().__init__()

        self.predict_normals = predict_normals
        self.predict_curvature = predict_curvature
        self.predict_occupancy = predict_occupancy

        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Task-specific heads
        if predict_normals:
            self.normal_head = nn.Linear(hidden_channels, 3)

        if predict_curvature:
            self.curvature_head = nn.Linear(hidden_channels, 1)

        if predict_occupancy:
            self.occupancy_head = nn.Linear(hidden_channels, occupancy_grid_size)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Predict geometric properties.

        Args:
            x: (N, in_channels) features

        Returns:
            Dictionary with predictions:
                - normals: (N, 3) if predict_normals
                - curvature: (N, 1) if predict_curvature
                - occupancy: (N, grid_size) if predict_occupancy
        """
        hidden = self.shared(x)

        outputs = {}

        if self.predict_normals:
            normals = self.normal_head(hidden)
            outputs['normals'] = F.normalize(normals, dim=-1)

        if self.predict_curvature:
            curvature = self.curvature_head(hidden)
            outputs['curvature'] = torch.sigmoid(curvature)  # Curvature in [0, 1]

        if self.predict_occupancy:
            occupancy = self.occupancy_head(hidden)
            outputs['occupancy'] = occupancy  # Raw logits

        return outputs


class ChangeDetectionHead(nn.Module):
    """
    Head for change detection from temporal features.

    Takes features from two time steps and predicts change category.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 4,  # no_change, erosion, accretion, anthropogenic
        hidden_channels: int = 128,
        feature_fusion: str = 'concat'  # 'concat', 'diff', 'both'
    ):
        """
        Args:
            in_channels: Feature dimension per time step
            num_classes: Number of change classes
            hidden_channels: Hidden layer dimension
            feature_fusion: How to combine temporal features
        """
        super().__init__()

        self.feature_fusion = feature_fusion

        if feature_fusion == 'concat':
            fusion_dim = in_channels * 2
        elif feature_fusion == 'diff':
            fusion_dim = in_channels
        elif feature_fusion == 'both':
            fusion_dim = in_channels * 3  # concat + diff
        else:
            raise ValueError(f"Unknown feature_fusion: {feature_fusion}")

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(
        self,
        feat_t1: torch.Tensor,
        feat_t2: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict change class.

        Args:
            feat_t1: (N, C) features from time 1
            feat_t2: (N, C) features from time 2

        Returns:
            (N, num_classes) change class logits
        """
        if self.feature_fusion == 'concat':
            fused = torch.cat([feat_t1, feat_t2], dim=-1)
        elif self.feature_fusion == 'diff':
            fused = feat_t2 - feat_t1
        elif self.feature_fusion == 'both':
            fused = torch.cat([feat_t1, feat_t2, feat_t2 - feat_t1], dim=-1)

        return self.head(fused)
