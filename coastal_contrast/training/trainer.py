"""
Training Loop Implementations

PreTrainer: Self-supervised pre-training with temporal contrastive learning
FineTuner: Supervised fine-tuning for segmentation
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .utils import (
    AverageMeter,
    MetricTracker,
    get_scheduler,
    save_checkpoint,
    load_checkpoint,
    log_metrics
)

logger = logging.getLogger(__name__)


class PreTrainer:
    """
    Self-supervised pre-training trainer.

    Handles training loop, validation, checkpointing, and logging
    for CoastalContrast pre-training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        num_epochs: int = 100,
        warmup_epochs: int = 10,
        scheduler_type: str = 'cosine',
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10,
        device: str = 'cuda',
        mixed_precision: bool = True
    ):
        """
        Initialize pre-trainer.

        Args:
            model: CoastalContrast model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            lr: Learning rate
            weight_decay: Weight decay
            num_epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
            scheduler_type: LR scheduler type
            checkpoint_dir: Directory for checkpoints
            log_interval: Steps between logging
            device: Device to train on
            mixed_precision: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.device = device
        self.mixed_precision = mixed_precision

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type,
            num_epochs,
            warmup_epochs
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Metrics
        self.metric_names = ['total_loss', 'contrastive_loss', 'contrastive_accuracy']
        if model.enable_geometric:
            self.metric_names.extend(['geometric_loss'])

        # Best metric tracking
        self.best_loss = float('inf')

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricTracker(self.metric_names)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for step, batch in enumerate(pbar):
            # Move to device
            batch = self._to_device(batch)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                loss, metrics = self.model(batch)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Update metrics
            tracker.update(metrics)

            # Update progress bar
            if step % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{tracker.metrics["total_loss"].avg:.4f}',
                    'acc': f'{tracker.metrics.get("contrastive_accuracy", AverageMeter()).avg:.4f}'
                })

        return tracker.get_averages()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        tracker = MetricTracker(self.metric_names)

        for batch in tqdm(self.val_loader, desc='Validation'):
            batch = self._to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                loss, metrics = self.model(batch)

            tracker.update(metrics)

        return tracker.get_averages()

    def train(self, resume_from: Optional[str] = None):
        """
        Full training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0

        # Resume from checkpoint
        if resume_from and Path(resume_from).exists():
            start_epoch = load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler
            ) + 1
            logger.info(f"Resumed from epoch {start_epoch}")

        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            log_metrics(train_metrics, epoch, 'train')

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                log_metrics(val_metrics, epoch, 'val')

            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('total_loss', train_metrics['total_loss']))
            else:
                self.scheduler.step()

            # Checkpoint
            current_loss = val_metrics.get('total_loss', train_metrics['total_loss'])
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss

            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                {'train': train_metrics, 'val': val_metrics},
                self.checkpoint_dir / f'checkpoint_{epoch:04d}.pth',
                is_best=is_best
            )

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch} | LR: {current_lr:.6f}")

        logger.info(f"Training complete. Best loss: {self.best_loss:.4f}")

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch


class FineTuner:
    """
    Supervised fine-tuning trainer for segmentation.
    """

    def __init__(
        self,
        model: nn.Module,
        seg_head: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_classes: int = 8,
        lr: float = 1e-4,
        backbone_lr_mult: float = 0.1,
        weight_decay: float = 0.05,
        num_epochs: int = 50,
        warmup_epochs: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = './checkpoints_finetune',
        freeze_backbone: bool = False,
        progressive_unfreeze: bool = True,
        unfreeze_epoch: int = 10,
        device: str = 'cuda',
        mixed_precision: bool = True
    ):
        """
        Initialize fine-tuner.

        Args:
            model: Pre-trained backbone (or CoastalContrast.backbone)
            seg_head: Segmentation head
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of segmentation classes
            lr: Learning rate
            backbone_lr_mult: LR multiplier for backbone
            weight_decay: Weight decay
            num_epochs: Training epochs
            warmup_epochs: Warmup epochs
            class_weights: Class weights for imbalanced data
            checkpoint_dir: Checkpoint directory
            freeze_backbone: Freeze backbone entirely
            progressive_unfreeze: Progressively unfreeze backbone
            unfreeze_epoch: Epoch to unfreeze backbone
            device: Training device
            mixed_precision: Use AMP
        """
        self.backbone = model.to(device)
        self.seg_head = seg_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.mixed_precision = mixed_precision
        self.freeze_backbone = freeze_backbone
        self.progressive_unfreeze = progressive_unfreeze
        self.unfreeze_epoch = unfreeze_epoch

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")

        # Optimizer with different LR for backbone
        param_groups = []
        if not freeze_backbone:
            param_groups.append({
                'params': self.backbone.parameters(),
                'lr': lr * backbone_lr_mult
            })
        param_groups.append({
            'params': self.seg_head.parameters(),
            'lr': lr
        })

        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)

        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            'cosine',
            num_epochs,
            warmup_epochs
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Best metric
        self.best_miou = 0.0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.backbone.train()
        self.seg_head.train()

        loss_meter = AverageMeter('loss')
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            batch = self._to_device(batch)

            # Forward
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                # Get backbone features
                data_dict = {
                    'coord': batch['coord'],
                    'feat': batch['feat'],
                    'offset': batch['offset'],
                    'grid_size': batch.get('grid_size', 0.02)
                }
                point = self.backbone(data_dict)
                features = point.feat

                # Segmentation prediction
                logits = self.seg_head(features)

                # Loss
                labels = batch['segment']
                loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Metrics
            loss_meter.update(loss.item())
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += len(labels)

            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{correct/total:.4f}'
            })

        return {
            'loss': loss_meter.avg,
            'accuracy': correct / total
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute mIoU."""
        if self.val_loader is None:
            return {}

        self.backbone.eval()
        self.seg_head.eval()

        loss_meter = AverageMeter('loss')
        confusion = torch.zeros(self.num_classes, self.num_classes, device=self.device)

        for batch in tqdm(self.val_loader, desc='Validation'):
            batch = self._to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                data_dict = {
                    'coord': batch['coord'],
                    'feat': batch['feat'],
                    'offset': batch['offset'],
                    'grid_size': batch.get('grid_size', 0.02)
                }
                point = self.backbone(data_dict)
                features = point.feat
                logits = self.seg_head(features)

                labels = batch['segment']
                loss = self.criterion(logits, labels)

            loss_meter.update(loss.item())

            # Update confusion matrix
            pred = logits.argmax(dim=-1)
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion[t.long(), p.long()] += 1

        # Compute mIoU
        intersection = torch.diag(confusion)
        union = confusion.sum(dim=1) + confusion.sum(dim=0) - intersection
        iou = intersection / (union + 1e-10)
        miou = iou.mean().item()

        # Per-class IoU
        per_class_iou = {f'iou_class_{i}': iou[i].item() for i in range(self.num_classes)}

        return {
            'loss': loss_meter.avg,
            'miou': miou,
            'accuracy': (torch.diag(confusion).sum() / confusion.sum()).item(),
            **per_class_iou
        }

    def train(self, resume_from: Optional[str] = None):
        """Full fine-tuning loop."""
        start_epoch = 0

        if resume_from and Path(resume_from).exists():
            start_epoch = self._load_checkpoint(resume_from) + 1

        for epoch in range(start_epoch, self.num_epochs):
            # Progressive unfreeze
            if self.progressive_unfreeze and epoch == self.unfreeze_epoch:
                logger.info(f"Unfreezing backbone at epoch {epoch}")
                for param in self.backbone.parameters():
                    param.requires_grad = True

            # Train
            train_metrics = self.train_epoch(epoch)
            log_metrics(train_metrics, epoch, 'train')

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                log_metrics(val_metrics, epoch, 'val')

            # Scheduler
            self.scheduler.step()

            # Checkpoint
            is_best = val_metrics.get('miou', 0) > self.best_miou
            if is_best:
                self.best_miou = val_metrics.get('miou', 0)

            self._save_checkpoint(epoch, train_metrics, val_metrics, is_best)

        logger.info(f"Fine-tuning complete. Best mIoU: {self.best_miou:.4f}")

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, is_best):
        checkpoint = {
            'epoch': epoch,
            'backbone': self.backbone.state_dict(),
            'seg_head': self.seg_head.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_{epoch:04d}.pth')

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')

    def _load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location='cpu')
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.seg_head.load_state_dict(checkpoint['seg_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_miou = checkpoint.get('best_miou', 0)
        return checkpoint['epoch']
