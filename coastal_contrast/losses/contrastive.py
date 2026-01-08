"""
Contrastive Loss Functions for Temporal Point Cloud Learning

Implements InfoNCE-style losses for:
- Temporal correspondence learning
- Spatial context learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalContrastiveLoss(nn.Module):
    """
    InfoNCE loss for temporal correspondences.

    For each correspondence (source_i, target_j), the corresponding
    target point is treated as positive, and other target points
    in the batch are negatives.

    Loss = -log(exp(sim(z_s, z_t+) / τ) / Σ exp(sim(z_s, z_t) / τ))
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_confidence_weighting: bool = True,
        hard_negative_weight: float = 1.0
    ):
        """
        Args:
            temperature: Temperature parameter τ for softmax
            use_confidence_weighting: Weight loss by correspondence confidence
            hard_negative_weight: Extra weight for hard negatives
        """
        super().__init__()
        self.temperature = temperature
        self.use_confidence_weighting = use_confidence_weighting
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        correspondences: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        source_offset: Optional[torch.Tensor] = None,
        target_offset: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute temporal contrastive loss.

        Args:
            source_features: (N, D) source point features
            target_features: (M, D) target point features
            correspondences: (K, 2) tensor of [source_idx, target_idx]
            confidence: (K,) confidence weights (optional)
            source_offset: Batch offsets for source (optional)
            target_offset: Batch offsets for target (optional)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if len(correspondences) == 0:
            return torch.tensor(0.0, device=source_features.device), {'n_pairs': 0}

        # Normalize features
        source_features = F.normalize(source_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)

        # Get corresponding features
        source_idx = correspondences[:, 0]
        target_idx = correspondences[:, 1]

        anchor_features = source_features[source_idx]  # (K, D)
        positive_features = target_features[target_idx]  # (K, D)

        # Compute positive similarities
        pos_sim = torch.sum(anchor_features * positive_features, dim=-1) / self.temperature  # (K,)

        # Compute similarities with all target points (negatives)
        # For memory efficiency, use batched computation
        all_sim = torch.mm(anchor_features, target_features.t()) / self.temperature  # (K, M)

        # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
        # = -pos + log(sum(exp(all)))
        log_sum_exp = torch.logsumexp(all_sim, dim=-1)  # (K,)
        loss_per_pair = -pos_sim + log_sum_exp

        # Weight by confidence if provided
        if self.use_confidence_weighting and confidence is not None:
            loss_per_pair = loss_per_pair * confidence

        loss = loss_per_pair.mean()

        # Compute metrics
        with torch.no_grad():
            # Accuracy: is the positive the most similar?
            predictions = all_sim.argmax(dim=-1)
            accuracy = (predictions == target_idx).float().mean()

            # Mean positive similarity
            mean_pos_sim = (pos_sim * self.temperature).mean()

        metrics = {
            'n_pairs': len(correspondences),
            'accuracy': accuracy.item(),
            'mean_pos_sim': mean_pos_sim.item(),
            'loss': loss.item()
        }

        return loss, metrics


class SpatialContrastiveLoss(nn.Module):
    """
    Contrastive loss for spatial context learning.

    Points from overlapping regions of the same scene
    (different tiles, different augmentations) should be similar.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        n_negatives: int = 1024
    ):
        """
        Args:
            temperature: Temperature parameter
            n_negatives: Number of negatives to sample
        """
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        correspondences: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute spatial contrastive loss.

        Args:
            features_a: (N, D) features from view A
            features_b: (M, D) features from view B
            correspondences: (K, 2) pairs of corresponding point indices

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if len(correspondences) == 0:
            return torch.tensor(0.0, device=features_a.device), {'n_pairs': 0}

        # Normalize
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)

        # Get corresponding features
        idx_a = correspondences[:, 0]
        idx_b = correspondences[:, 1]

        anchor = features_a[idx_a]  # (K, D)
        positive = features_b[idx_b]  # (K, D)

        # Sample negatives from features_b
        n_neg = min(self.n_negatives, len(features_b))
        neg_idx = torch.randperm(len(features_b), device=features_b.device)[:n_neg]
        negatives = features_b[neg_idx]  # (n_neg, D)

        # Positive similarities
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature  # (K, 1)

        # Negative similarities
        neg_sim = torch.mm(anchor, negatives.t()) / self.temperature  # (K, n_neg)

        # Concatenate
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (K, 1 + n_neg)

        # Labels: positive is always at index 0
        labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        # Metrics
        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == 0).float().mean()

        metrics = {
            'n_pairs': len(correspondences),
            'accuracy': accuracy.item(),
            'loss': loss.item()
        }

        return loss, metrics


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Used in SimCLR-style contrastive learning where each sample
    has one positive pair and 2(N-1) negatives in a batch.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss for paired embeddings.

        Args:
            z_i: (N, D) embeddings from first augmentation
            z_j: (N, D) embeddings from second augmentation

        Returns:
            Scalar loss
        """
        N = len(z_i)
        device = z_i.device

        # Normalize
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device)
        ])

        loss = F.cross_entropy(sim, labels)

        return loss


class InfoNCELoss(nn.Module):
    """
    General InfoNCE loss implementation.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: (N, D) anchor embeddings
            positive: (N, D) positive embeddings
            negatives: (N, K, D) or (K, D) negative embeddings

        Returns:
            Scalar loss
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature

        # Negative similarity
        if negatives.dim() == 2:
            # Shared negatives: (K, D)
            neg_sim = torch.mm(anchor, negatives.t()) / self.temperature
        else:
            # Per-sample negatives: (N, K, D)
            neg_sim = torch.bmm(
                anchor.unsqueeze(1),
                negatives.transpose(1, 2)
            ).squeeze(1) / self.temperature

        # InfoNCE
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)

        return loss
