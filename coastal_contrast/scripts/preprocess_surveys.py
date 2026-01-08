#!/usr/bin/env python3
"""
Preprocess Surveys: Full Pipeline

Runs the complete preprocessing pipeline:
1. Build survey catalog
2. Preprocess LAS files to multi-resolution tiles
3. Mine temporal correspondences

Usage:
    python -m coastal_contrast.scripts.preprocess_surveys \
        --input /path/to/surveys/ \
        --output /path/to/processed/ \
        --catalog ./data/catalog.db \
        --resolutions 0.02 0.05 0.1 \
        --tile-size 10.0

    # Or run individual steps:
    python -m coastal_contrast.scripts.preprocess_surveys \
        --input /path/to/surveys/ \
        --output /path/to/processed/ \
        --step tiles  # Only create tiles
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coastal_contrast.data.catalog import SurveyCatalog
from coastal_contrast.data.preprocess import (
    process_las_to_tiles_multiresolution,
    build_tile_index
)
from coastal_contrast.data.correspondence import (
    CorrespondenceMiner,
    find_overlapping_tiles,
    mine_temporal_correspondences_batch
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LiDAR surveys for CoastalContrast"
    )

    parser.add_argument("--input", "-i", required=True,
                       help="Input directory with LAS files")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for processed data")
    parser.add_argument("--catalog", "-c", default=None,
                       help="Catalog database path (default: output/catalog.db)")
    parser.add_argument("--step", choices=['all', 'catalog', 'tiles', 'correspondences'],
                       default='all', help="Which step to run")

    # Catalog options
    parser.add_argument("--pattern", default="*.las",
                       help="File pattern for LAS files")
    parser.add_argument("--platform", default="mobile",
                       help="Platform type (mobile, terrestrial, airborne)")

    # Tile options
    parser.add_argument("--tile-size", type=float, default=10.0,
                       help="Tile size in meters")
    parser.add_argument("--resolutions", "-r", type=float, nargs='+',
                       default=[0.02, 0.05, 0.1],
                       help="Voxel resolutions")
    parser.add_argument("--no-normals", action="store_true",
                       help="Skip normal computation")
    parser.add_argument("--no-curvature", action="store_true",
                       help="Skip curvature computation")

    # Correspondence options
    parser.add_argument("--max-distance", type=float, default=0.5,
                       help="Max correspondence distance")
    parser.add_argument("--min-days", type=int, default=7,
                       help="Min days between temporal pairs")
    parser.add_argument("--max-days", type=int, default=30,
                       help="Max days between temporal pairs")

    # General options
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of parallel workers")

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = args.catalog or str(output_dir / "catalog.db")
    tiles_dir = output_dir / "tiles"
    corr_dir = output_dir / "correspondences"

    start_time = datetime.now()
    logger.info(f"Starting preprocessing pipeline")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Step 1: Build catalog
    if args.step in ['all', 'catalog']:
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Building survey catalog")
        logger.info("="*60)

        catalog = SurveyCatalog(catalog_path)
        n_added = catalog.add_survey_directory(
            args.input,
            pattern=args.pattern,
            platform=args.platform,
            n_workers=args.workers
        )

        stats = catalog.get_statistics()
        logger.info(f"Catalog complete: {stats['total_surveys']} surveys")

    # Step 2: Create tiles
    if args.step in ['all', 'tiles']:
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Creating multi-resolution tiles")
        logger.info("="*60)

        catalog = SurveyCatalog(catalog_path)
        surveys = catalog.get_all_surveys()

        all_metadata = []
        for i, survey in enumerate(surveys):
            logger.info(f"Processing survey {i+1}/{len(surveys)}: {survey.filename}")
            try:
                metadata = process_las_to_tiles_multiresolution(
                    survey.file_path,
                    str(tiles_dir),
                    tile_size=args.tile_size,
                    resolutions=args.resolutions,
                    compute_normals=not args.no_normals,
                    compute_curvature=not args.no_curvature
                )
                all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Error processing {survey.filename}: {e}")
                continue

        # Build tile index
        build_tile_index(all_metadata, str(tiles_dir / "tile_index.json"))
        logger.info(f"Created {len(all_metadata)} tiles")

    # Step 3: Mine correspondences
    if args.step in ['all', 'correspondences']:
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Mining temporal correspondences")
        logger.info("="*60)

        import json
        tile_index_path = tiles_dir / "tile_index.json"

        if not tile_index_path.exists():
            logger.error("Tile index not found. Run tiles step first.")
            sys.exit(1)

        with open(tile_index_path, 'r') as f:
            tile_index = json.load(f)

        # Get temporal pairs from catalog
        catalog = SurveyCatalog(catalog_path)
        temporal_pairs = catalog.get_temporal_pairs(
            min_days=args.min_days,
            max_days=args.max_days
        )

        logger.info(f"Found {len(temporal_pairs)} temporal survey pairs")

        # Find overlapping tiles for each survey pair
        corr_dir.mkdir(parents=True, exist_ok=True)
        miner = CorrespondenceMiner(max_distance=args.max_distance)

        tile_pairs = []
        for survey_a, survey_b in temporal_pairs[:100]:  # Limit for now
            pairs = find_overlapping_tiles(
                tile_index,
                survey_a.survey_id,
                survey_b.survey_id
            )
            tile_pairs.extend(pairs)

        logger.info(f"Found {len(tile_pairs)} tile pairs to process")

        n_created = mine_temporal_correspondences_batch(
            tile_pairs,
            str(corr_dir),
            miner=miner,
            n_workers=args.workers
        )

        logger.info(f"Created {n_created} correspondence files")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
