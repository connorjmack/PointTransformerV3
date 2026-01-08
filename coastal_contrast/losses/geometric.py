"""
Geometric Loss Functions for Point Cloud Pre-Training

Losses for predicting geometric properties:
- Surface normals
- Local curvature
- Occupancy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GeometricLoss(nn.Module):
    """
    Combined loss for geometric property prediction.

    Predicts surface normals, curvature, and optionally occupancy
    as auxiliary tasks during pre-training.
    """

    def __init__(
        self,
        normal_weight: float = 1.0,
        curvature_weight: float = 0.5,
        occupancy_weight: float = 0.5,
        normal_loss_type: str = 'cosine'
    ):
        """
        Args:
            normal_weight: Weight for normal prediction loss
            curvature_weight: Weight for curvature prediction loss
            occupancy_weight: Weight for occupancy prediction loss
            normal_loss_type: 'cosine' or 'mse' for normal loss
        """
        super().__init__()
        self.normal_weight = normal_weight
        self.curvature_weight = curvature_weight
        self.occupancy_weight = occupancy_weight
        self.normal_loss_type = normal_loss_type

    def forward(
        self,
        pred_normals: Optional[torch.Tensor],
        gt_normals: Optional[torch.Tensor],
        pred_curvature: Optional[torch.Tensor],
        gt_curvature: Optional[torch.Tensor],
        pred_occupancy: Optional[torch.Tensor] = None,
        gt_occupancy: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined geometric loss.

        Args:
            pred_normals: (N, 3) predicted normals
            gt_normals: (N, 3) ground truth normals
            pred_curvature: (N, 1) or (N,) predicted curvature
            gt_curvature: (N, 1) or (N,) ground truth curvature
            pred_occupancy: (N, K) predicted occupancy logits
            gt_occupancy: (N, K) ground truth occupancy (binary)
            mask: (N,) boolean mask for which points to include

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        device = pred_normals.device if pred_normals is not None else pred_curvature.device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        # Apply mask if provided
        if mask is not None:
            if pred_normals is not None:
                pred_normals = pred_normals[mask]
                gt_normals = gt_normals[mask]
            if pred_curvature is not None:
                pred_curvature = pred_curvature[mask]
                gt_curvature = gt_curvature[mask]
            if pred_occupancy is not None:
                pred_occupancy = pred_occupancy[mask]
                gt_occupancy = gt_occupancy[mask]

        # Normal loss
        if pred_normals is not None and gt_normals is not None and self.normal_weight > 0:
            normal_loss = self._normal_loss(pred_normals, gt_normals)
            total_loss = total_loss + self.normal_weight * normal_loss
            metrics['normal_loss'] = normal_loss.item()

            # Normal accuracy (angle error)
            with torch.no_grad():
                pred_n = F.normalize(pred_normals, dim=-1)
                gt_n = F.normalize(gt_normals, dim=-1)
                cos_sim = torch.sum(pred_n * gt_n, dim=-1).clamp(-1, 1)
                angle_error = torch.acos(cos_sim.abs()) * 180 / 3.14159
                metrics['normal_angle_error'] = angle_error.mean().item()

        # Curvature loss
        if pred_curvature is not None and gt_curvature is not None and self.curvature_weight > 0:
            pred_curv = pred_curvature.view(-1)
            gt_curv = gt_curvature.view(-1)
            curvature_loss = F.mse_loss(pred_curv, gt_curv)
            total_loss = total_loss + self.curvature_weight * curvature_loss
            metrics['curvature_loss'] = curvature_loss.item()

            with torch.no_grad():
                metrics['curvature_mae'] = F.l1_loss(pred_curv, gt_curv).item()

        # Occupancy loss
        if pred_occupancy is not None and gt_occupancy is not None and self.occupancy_weight > 0:
            occupancy_loss = F.binary_cross_entropy_with_logits(
                pred_occupancy.float(),
                gt_occupancy.float()
            )
            total_loss = total_loss + self.occupancy_weight * occupancy_loss
            metrics['occupancy_loss'] = occupancy_loss.item()

            with torch.no_grad():
                pred_occ = (torch.sigmoid(pred_occupancy) > 0.5).float()
                accuracy = (pred_occ == gt_occupancy).float().mean()
                metrics['occupancy_accuracy'] = accuracy.item()

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def _normal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute normal prediction loss."""
        if self.normal_loss_type == 'cosine':
            # Cosine similarity loss (1 - |cos_sim|)
            # Use absolute value since normals can be flipped
            pred_n = F.normalize(pred, dim=-1)
            target_n = F.normalize(target, dim=-1)
            cos_sim = torch.sum(pred_n * target_n, dim=-1)
            loss = (1 - cos_sim.abs()).mean()
        elif self.normal_loss_type == 'mse':
            pred_n = F.normalize(pred, dim=-1)
            target_n = F.normalize(target, dim=-1)
            # MSE with sign ambiguity handling
            loss_pos = F.mse_loss(pred_n, target_n)
            loss_neg = F.mse_loss(pred_n, -target_n)
            loss = torch.min(loss_pos, loss_neg)
        else:
            raise ValueError(f"Unknown normal_loss_type: {self.normal_loss_type}")

        return loss


class MaskedGeometricLoss(nn.Module):
    """
    Masked autoencoding loss for geometric properties.

    Mask a portion of points and predict their geometric properties
    from context (similar to MAE/BERT).
    """

    def __init__(
        self,
        mask_ratio: float = 0.6,
        coord_weight: float = 1.0,
        normal_weight: float = 1.0,
        curvature_weight: float = 0.5
    ):
        """
        Args:
            mask_ratio: Fraction of points to mask
            coord_weight: Weight for coordinate reconstruction
            normal_weight: Weight for normal reconstruction
            curvature_weight: Weight for curvature reconstruction
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.coord_weight = coord_weight
        self.normal_weight = normal_weight
        self.curvature_weight = curvature_weight

    def forward(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        pred_normals: Optional[torch.Tensor],
        gt_normals: Optional[torch.Tensor],
        pred_curvature: Optional[torch.Tensor],
        gt_curvature: Optional[torch.Tensor],
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute masked reconstruction loss.

        Args:
            pred_coords: (N, 3) predicted coordinates for masked points
            gt_coords: (N, 3) ground truth coordinates
            pred_normals: (N, 3) predicted normals (optional)
            gt_normals: (N, 3) ground truth normals (optional)
            pred_curvature: (N,) predicted curvature (optional)
            gt_curvature: (N,) ground truth curvature (optional)
            mask: (N,) boolean mask (True = masked/to predict)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        device = pred_coords.device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        n_masked = mask.sum().item()
        if n_masked == 0:
            return total_loss, {'n_masked': 0}

        metrics['n_masked'] = n_masked
        metrics['mask_ratio'] = n_masked / len(mask)

        # Coordinate reconstruction loss
        if self.coord_weight > 0:
            coord_loss = F.mse_loss(
                pred_coords[mask],
                gt_coords[mask]
            )
            total_loss = total_loss + self.coord_weight * coord_loss
            metrics['coord_loss'] = coord_loss.item()

            with torch.no_grad():
                # Mean reconstruction error in original units
                coord_error = torch.norm(
                    pred_coords[mask] - gt_coords[mask],
                    dim=-1
                ).mean()
                metrics['coord_error'] = coord_error.item()

        # Normal reconstruction
        if pred_normals is not None and gt_normals is not None and self.normal_weight > 0:
            pred_n = F.normalize(pred_normals[mask], dim=-1)
            gt_n = F.normalize(gt_normals[mask], dim=-1)

            # Cosine loss with sign ambiguity
            cos_sim = torch.sum(pred_n * gt_n, dim=-1)
            normal_loss = (1 - cos_sim.abs()).mean()

            total_loss = total_loss + self.normal_weight * normal_loss
            metrics['normal_loss'] = normal_loss.item()

        # Curvature reconstruction
        if pred_curvature is not None and gt_curvature is not None and self.curvature_weight > 0:
            curv_loss = F.mse_loss(
                pred_curvature[mask],
                gt_curvature[mask]
            )
            total_loss = total_loss + self.curvature_weight * curv_loss
            metrics['curvature_loss'] = curv_loss.item()

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


class ChamferLoss(nn.Module):
    """
    Chamfer distance loss for point cloud reconstruction.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Chamfer distance.

        Args:
            pred: (N, 3) predicted points
            target: (M, 3) target points

        Returns:
            Chamfer distance
        """
        # Distance from pred to target
        diff_p2t = pred.unsqueeze(1) - target.unsqueeze(0)  # (N, M, 3)
        dist_p2t = torch.norm(diff_p2t, dim=-1)  # (N, M)
        min_dist_p2t = dist_p2t.min(dim=1)[0]  # (N,)

        # Distance from target to pred
        min_dist_t2p = dist_p2t.min(dim=0)[0]  # (M,)

        # Chamfer distance
        if self.reduction == 'mean':
            chamfer = min_dist_p2t.mean() + min_dist_t2p.mean()
        elif self.reduction == 'sum':
            chamfer = min_dist_p2t.sum() + min_dist_t2p.sum()
        else:
            chamfer = (min_dist_p2t, min_dist_t2p)

        return chamfer
