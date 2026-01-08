"""
Training infrastructure for CoastalContrast.

Includes:
- Trainer: Main training loop with logging and checkpointing
- utils: Training utilities (learning rate schedulers, metrics, etc.)
"""

from .trainer import PreTrainer, FineTuner
from .utils import AverageMeter, get_scheduler, save_checkpoint, load_checkpoint

__all__ = [
    'PreTrainer',
    'FineTuner',
    'AverageMeter',
    'get_scheduler',
    'save_checkpoint',
    'load_checkpoint'
]
