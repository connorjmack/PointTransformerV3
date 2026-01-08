"""
Training Utilities

Helper functions and classes for training.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __repr__(self):
        return f'{self.name}: {self.avg:.4f}'


class MetricTracker:
    """Track multiple metrics."""

    def __init__(self, metric_names: list):
        self.metrics = {name: AverageMeter(name) for name in metric_names}

    def update(self, values: Dict[str, float], n: int = 1):
        for name, value in values.items():
            if name in self.metrics:
                self.metrics[name].update(value, n)

    def reset(self):
        for meter in self.metrics.values():
            meter.reset()

    def get_averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.metrics.items()}

    def __repr__(self):
        return ' | '.join([repr(m) for m in self.metrics.values()])


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    **kwargs
) -> _LRScheduler:
    """
    Get learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_type: 'cosine', 'step', 'plateau', 'linear'
        num_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        StepLR,
        ReduceLROnPlateau,
        LinearLR,
        SequentialLR
    )

    # Main scheduler
    if scheduler_type == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', num_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=min_lr
        )
    elif scheduler_type == 'linear':
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.defaults['lr'],
            total_iters=num_epochs - warmup_epochs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Add warmup
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
        return scheduler

    return main_scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics: Dict[str, Any],
    output_path: str,
    is_best: bool = False
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state (optional)
        epoch: Current epoch
        metrics: Metrics dictionary
        output_path: Output file path
        is_best: If True, also save as 'best.pth'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    torch.save(checkpoint, output_path)
    logger.info(f"Saved checkpoint to {output_path}")

    if is_best:
        best_path = output_path.parent / 'best.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    strict: bool = True
) -> int:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        strict: Whether to strictly enforce state dict matching

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'], strict=strict)
    logger.info(f"Loaded model from {checkpoint_path}")

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint.get('epoch', 0)


def get_parameter_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 0.1
) -> list:
    """
    Get parameter groups with different learning rates.

    Args:
        model: Model with backbone and heads
        lr: Base learning rate
        weight_decay: Weight decay
        backbone_lr_mult: Multiplier for backbone learning rate

    Returns:
        List of parameter groups for optimizer
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': lr * backbone_lr_mult, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr, 'weight_decay': weight_decay}
    ]

    return param_groups


def log_metrics(
    metrics: Dict[str, float],
    epoch: int,
    phase: str = 'train',
    logger_obj: Optional[logging.Logger] = None,
    tensorboard_writer: Any = None
):
    """
    Log metrics to console and tensorboard.

    Args:
        metrics: Dictionary of metric values
        epoch: Current epoch
        phase: 'train' or 'val'
        logger_obj: Logger instance
        tensorboard_writer: TensorBoard SummaryWriter
    """
    if logger_obj is None:
        logger_obj = logger

    # Console logging
    metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    logger_obj.info(f"Epoch {epoch} [{phase}] {metrics_str}")

    # TensorBoard logging
    if tensorboard_writer is not None:
        for key, value in metrics.items():
            tensorboard_writer.add_scalar(f'{phase}/{key}', value, epoch)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
