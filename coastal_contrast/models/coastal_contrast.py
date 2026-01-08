"""
CoastalContrast: Self-Supervised Pre-Training Model

Combines Point Transformer V3 backbone with contrastive and
geometric prediction objectives for temporal self-supervised learning.

Usage:
    from coastal_contrast.models import CoastalContrast

    model = CoastalContrast(
        backbone_config={...},
        projection_dim=128,
        temperature=0.1
    )

    loss, metrics = model(batch)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

# Add parent for model import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .heads import ProjectionHead, GeometricHead


class CoastalContrast(nn.Module):
    """
    Self-supervised pre-training model combining:
    - PTv3 backbone for point cloud encoding
    - Temporal contrastive learning
    - Geometric property prediction (optional)

    The model processes pairs of temporally-related point clouds
    and optimizes:
    1. Temporal InfoNCE loss on corresponding points
    2. Geometric prediction loss (normals, curvature) as auxiliary task
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int = 64,
        projection_dim: int = 128,
        projection_hidden: int = 256,
        temperature: float = 0.1,
        enable_geometric: bool = True,
        geometric_weight: float = 0.5,
        use_momentum_encoder: bool = False,
        momentum: float = 0.999
    ):
        """
        Initialize CoastalContrast.

        Args:
            backbone: PTv3 backbone (encoder-decoder)
            in_channels: Output channels from backbone
            projection_dim: Contrastive projection dimension
            projection_hidden: Projection MLP hidden dimension
            temperature: InfoNCE temperature
            enable_geometric: Enable geometric prediction head
            geometric_weight: Weight for geometric loss
            use_momentum_encoder: Use momentum encoder (MoCo-style)
            momentum: Momentum coefficient for momentum encoder
        """
        super().__init__()

        self.backbone = backbone
        self.temperature = temperature
        self.enable_geometric = enable_geometric
        self.geometric_weight = geometric_weight
        self.use_momentum_encoder = use_momentum_encoder
        self.momentum = momentum

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            in_channels=in_channels,
            hidden_channels=projection_hidden,
            out_channels=projection_dim,
            num_layers=2
        )

        # Geometric prediction head (optional)
        if enable_geometric:
            self.geometric_head = GeometricHead(
                in_channels=in_channels,
                predict_normals=True,
                predict_curvature=True,
                predict_occupancy=False
            )

        # Momentum encoder (optional, for MoCo-style training)
        if use_momentum_encoder:
            self.momentum_backbone = self._copy_backbone(backbone)
            self.momentum_projection = ProjectionHead(
                in_channels=in_channels,
                hidden_channels=projection_hidden,
                out_channels=projection_dim,
                num_layers=2
            )
            # Initialize momentum encoder
            self._init_momentum_encoder()

    def _copy_backbone(self, backbone: nn.Module) -> nn.Module:
        """Create a copy of backbone for momentum encoder."""
        import copy
        momentum_backbone = copy.deepcopy(backbone)
        for param in momentum_backbone.parameters():
            param.requires_grad = False
        return momentum_backbone

    def _init_momentum_encoder(self):
        """Initialize momentum encoder with same weights."""
        for param_q, param_k in zip(
            self.backbone.parameters(),
            self.momentum_backbone.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(
            self.projection_head.parameters(),
            self.momentum_projection.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Update momentum encoder with exponential moving average."""
        for param_q, param_k in zip(
            self.backbone.parameters(),
            self.momentum_backbone.parameters()
        ):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

        for param_q, param_k in zip(
            self.projection_head.parameters(),
            self.momentum_projection.parameters()
        ):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    def encode(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Encode a single point cloud.

        Args:
            data_dict: Dictionary with 'coord', 'feat', 'offset' or 'batch'

        Returns:
            Point features (N, C)
        """
        point = self.backbone(data_dict)
        return point.feat

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for pre-training.

        Args:
            batch: Dictionary containing:
                - source_coord, source_feat, source_offset
                - target_coord, target_feat, target_offset
                - correspondences: (K, 2) tensor
                - confidence: (K,) tensor
                - source_normal, target_normal (optional)
                - source_curvature, target_curvature (optional)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        device = batch['source_coord'].device

        # Encode source
        source_data = {
            'coord': batch['source_coord'],
            'feat': batch['source_feat'],
            'offset': batch['source_offset'],
            'grid_size': batch.get('grid_size', 0.02)
        }
        source_point = self.backbone(source_data)
        source_features = source_point.feat

        # Encode target
        if self.use_momentum_encoder:
            with torch.no_grad():
                self._update_momentum_encoder()
                target_data = {
                    'coord': batch['target_coord'],
                    'feat': batch['target_feat'],
                    'offset': batch['target_offset'],
                    'grid_size': batch.get('grid_size', 0.02)
                }
                target_point = self.momentum_backbone(target_data)
                target_features = target_point.feat
        else:
            target_data = {
                'coord': batch['target_coord'],
                'feat': batch['target_feat'],
                'offset': batch['target_offset'],
                'grid_size': batch.get('grid_size', 0.02)
            }
            target_point = self.backbone(target_data)
            target_features = target_point.feat

        # Project features
        source_proj = self.projection_head(source_features)
        if self.use_momentum_encoder:
            with torch.no_grad():
                target_proj = self.momentum_projection(target_features)
        else:
            target_proj = self.projection_head(target_features)

        # Normalize projections
        source_proj = F.normalize(source_proj, dim=-1)
        target_proj = F.normalize(target_proj, dim=-1)

        # Compute temporal contrastive loss
        correspondences = batch['correspondences']
        confidence = batch.get('confidence')

        contrastive_loss, contrastive_metrics = self._contrastive_loss(
            source_proj, target_proj, correspondences, confidence
        )

        total_loss = contrastive_loss
        metrics = {'contrastive_loss': contrastive_loss.item()}
        metrics.update({f'contrastive_{k}': v for k, v in contrastive_metrics.items()})

        # Geometric prediction loss (optional)
        if self.enable_geometric:
            geo_loss, geo_metrics = self._geometric_loss(
                source_features,
                batch.get('source_normal'),
                batch.get('source_curvature')
            )

            if geo_loss is not None:
                total_loss = total_loss + self.geometric_weight * geo_loss
                metrics['geometric_loss'] = geo_loss.item()
                metrics.update({f'geo_{k}': v for k, v in geo_metrics.items()})

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def _contrastive_loss(
        self,
        source_proj: torch.Tensor,
        target_proj: torch.Tensor,
        correspondences: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute InfoNCE contrastive loss."""
        if len(correspondences) == 0:
            return torch.tensor(0.0, device=source_proj.device), {'n_pairs': 0}

        source_idx = correspondences[:, 0]
        target_idx = correspondences[:, 1]

        anchor = source_proj[source_idx]
        positive = target_proj[target_idx]

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # All negatives (all target points)
        all_sim = torch.mm(anchor, target_proj.t()) / self.temperature

        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        log_sum_exp = torch.logsumexp(all_sim, dim=-1)
        loss_per_pair = -pos_sim + log_sum_exp

        # Weight by confidence
        if confidence is not None:
            loss_per_pair = loss_per_pair * confidence

        loss = loss_per_pair.mean()

        # Metrics
        with torch.no_grad():
            accuracy = (all_sim.argmax(dim=-1) == target_idx).float().mean()

        metrics = {
            'n_pairs': len(correspondences),
            'accuracy': accuracy.item()
        }

        return loss, metrics

    def _geometric_loss(
        self,
        features: torch.Tensor,
        gt_normals: Optional[torch.Tensor],
        gt_curvature: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Dict]:
        """Compute geometric prediction loss."""
        if gt_normals is None and gt_curvature is None:
            return None, {}

        predictions = self.geometric_head(features)
        loss = torch.tensor(0.0, device=features.device)
        metrics = {}

        # Normal loss
        if 'normals' in predictions and gt_normals is not None:
            pred_n = predictions['normals']
            gt_n = F.normalize(gt_normals, dim=-1)

            # Cosine loss with sign ambiguity
            cos_sim = torch.sum(pred_n * gt_n, dim=-1)
            normal_loss = (1 - cos_sim.abs()).mean()
            loss = loss + normal_loss
            metrics['normal_loss'] = normal_loss.item()

        # Curvature loss
        if 'curvature' in predictions and gt_curvature is not None:
            pred_c = predictions['curvature'].squeeze(-1)
            gt_c = gt_curvature
            if gt_c.dim() > 1:
                gt_c = gt_c.squeeze(-1)

            curv_loss = F.mse_loss(pred_c, gt_c)
            loss = loss + curv_loss
            metrics['curvature_loss'] = curv_loss.item()

        return loss, metrics

    def get_backbone_features(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Get backbone features for downstream tasks.

        Args:
            data_dict: Point cloud data dictionary

        Returns:
            Point features from backbone
        """
        with torch.no_grad():
            point = self.backbone(data_dict)
            return point.feat


def create_coastal_contrast(
    backbone_checkpoint: Optional[str] = None,
    in_channels: int = 6,
    projection_dim: int = 128,
    temperature: float = 0.1,
    enable_geometric: bool = True,
    **backbone_kwargs
) -> CoastalContrast:
    """
    Factory function to create CoastalContrast model.

    Args:
        backbone_checkpoint: Path to pre-trained backbone weights
        in_channels: Input feature channels
        projection_dim: Projection dimension
        temperature: Contrastive temperature
        enable_geometric: Enable geometric head
        **backbone_kwargs: Additional backbone configuration

    Returns:
        CoastalContrast model
    """
    # Import PTv3 backbone
    try:
        from model import PointTransformerV3
    except ImportError:
        raise ImportError("PointTransformerV3 not found. Ensure model.py is in the path.")

    # Default backbone config
    default_config = {
        'in_channels': in_channels,
        'enc_depths': (2, 2, 2, 6, 2),
        'enc_channels': (32, 64, 128, 256, 512),
        'enc_num_head': (2, 4, 8, 16, 32),
        'dec_depths': (2, 2, 2, 2),
        'dec_channels': (64, 64, 128, 256),
        'dec_num_head': (4, 4, 8, 16),
        'enable_flash': True,
        'enable_rpe': False,
    }
    default_config.update(backbone_kwargs)

    backbone = PointTransformerV3(**default_config)

    # Load checkpoint if provided
    if backbone_checkpoint and Path(backbone_checkpoint).exists():
        state_dict = torch.load(backbone_checkpoint, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        backbone.load_state_dict(state_dict, strict=False)

    # Get output channels from decoder
    out_channels = default_config['dec_channels'][0]

    model = CoastalContrast(
        backbone=backbone,
        in_channels=out_channels,
        projection_dim=projection_dim,
        temperature=temperature,
        enable_geometric=enable_geometric
    )

    return model
