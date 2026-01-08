#!/usr/bin/env python3
"""
Train CoastalContrast: Self-Supervised Pre-Training

Pre-trains the model using temporal contrastive learning on
unlabeled LiDAR surveys.

Usage:
    python -m coastal_contrast.scripts.train_pretrain \
        --tile-index ./data/tiles/tile_index.json \
        --correspondence-dir ./data/correspondences/ \
        --output ./checkpoints/pretrain/ \
        --epochs 100 \
        --batch-size 4 \
        --lr 1e-3

    # Resume training
    python -m coastal_contrast.scripts.train_pretrain \
        --tile-index ./data/tiles/tile_index.json \
        --correspondence-dir ./data/correspondences/ \
        --output ./checkpoints/pretrain/ \
        --resume ./checkpoints/pretrain/checkpoint_0050.pth
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train CoastalContrast model"
    )

    # Data
    parser.add_argument("--tile-index", required=True,
                       help="Path to tile index JSON")
    parser.add_argument("--correspondence-dir", required=True,
                       help="Directory with correspondence files")
    parser.add_argument("--temporal-pairs", default=None,
                       help="Path to temporal pairs JSON (optional)")
    parser.add_argument("--resolution", type=float, default=0.02,
                       help="Tile resolution to use")
    parser.add_argument("--max-points", type=int, default=50000,
                       help="Max points per tile")

    # Model
    parser.add_argument("--in-channels", type=int, default=4,
                       help="Input feature channels")
    parser.add_argument("--projection-dim", type=int, default=128,
                       help="Projection dimension for contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="InfoNCE temperature")
    parser.add_argument("--enable-geometric", action="store_true",
                       help="Enable geometric prediction head")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                       help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                       help="Warmup epochs")
    parser.add_argument("--scheduler", choices=['cosine', 'step', 'linear'],
                       default='cosine', help="LR scheduler")

    # Output
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--resume", default=None,
                       help="Resume from checkpoint")

    # Hardware
    parser.add_argument("--device", default='cuda',
                       help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=4,
                       help="DataLoader workers")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use automatic mixed precision")

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.info("="*60)
    logger.info("CoastalContrast Pre-Training")
    logger.info("="*60)

    # Load dataset
    logger.info("Loading dataset...")
    from coastal_contrast.data.temporal_dataset import TemporalPairDataset
    from coastal_contrast.data.transforms import get_pretrain_transforms

    transform = get_pretrain_transforms()

    train_dataset = TemporalPairDataset(
        tile_index_path=args.tile_index,
        correspondence_dir=args.correspondence_dir,
        temporal_pairs_path=args.temporal_pairs,
        resolution=args.resolution,
        max_points=args.max_points,
        transform=transform
    )

    logger.info(f"Dataset size: {len(train_dataset)} pairs")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )

    # Create model
    logger.info("Creating model...")
    from coastal_contrast.models.coastal_contrast import create_coastal_contrast

    model = create_coastal_contrast(
        in_channels=args.in_channels,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        enable_geometric=args.enable_geometric
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} ({n_trainable:,} trainable)")

    # Create trainer
    from coastal_contrast.training.trainer import PreTrainer

    trainer = PreTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Could add validation split
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        scheduler_type=args.scheduler,
        checkpoint_dir=str(output_dir),
        device=args.device,
        mixed_precision=args.mixed_precision
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from=args.resume)

    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
