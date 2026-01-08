#!/usr/bin/env python3
"""
Build Survey Catalog from LAS Files

Scans directories containing LAS/LAZ files and builds a SQLite catalog
with metadata for each survey. Extracts dates from filenames, reads
spatial bounds from headers, and indexes for efficient temporal queries.

Usage:
    # Basic usage - scan a directory
    python -m coastal_contrast.scripts.build_catalog \
        --input /path/to/lidar/surveys/ \
        --output ./data/catalog.db

    # Recursive scan with LAZ files
    python -m coastal_contrast.scripts.build_catalog \
        --input /path/to/lidar/ \
        --output ./data/catalog.db \
        --pattern "*.laz" \
        --recursive

    # Specify platform type
    python -m coastal_contrast.scripts.build_catalog \
        --input /path/to/mobile_surveys/ \
        --output ./data/catalog.db \
        --platform mobile

    # Multiple directories
    python -m coastal_contrast.scripts.build_catalog \
        --input /path/to/dir1/ /path/to/dir2/ \
        --output ./data/catalog.db

Example Output:
    Found 423 files to process
    Added 50/423 surveys...
    Added 100/423 surveys...
    ...
    Successfully added 420 surveys to catalog

    === Catalog Statistics ===
    total_surveys: 420
    total_points: 15,234,567,890
    date_range: {'min': '2018-01-05', 'max': '2025-12-20'}
    by_platform: {'mobile': 420}
    registered: 0
    unregistered: 420
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coastal_contrast.data.catalog import SurveyCatalog

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build survey catalog from LAS/LAZ files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan directory for LAS files
  python -m coastal_contrast.scripts.build_catalog --input ./surveys/ --output ./catalog.db

  # Scan for LAZ files, non-recursive
  python -m coastal_contrast.scripts.build_catalog --input ./surveys/ --output ./catalog.db --pattern "*.laz" --no-recursive

  # Multiple input directories
  python -m coastal_contrast.scripts.build_catalog --input ./mobile/ ./airborne/ --output ./catalog.db
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs='+',
        required=True,
        help="Input directory/directories containing LAS/LAZ files"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for SQLite catalog database"
    )

    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="*.las",
        help="Glob pattern for files (default: *.las). Use '*.laz' for compressed files, or '*.la[sz]' for both."
    )

    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Recursively search subdirectories (default: True)"
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories"
    )

    parser.add_argument(
        "--platform",
        type=str,
        choices=["mobile", "terrestrial", "airborne", "unknown"],
        default="unknown",
        help="Platform type for all surveys (default: unknown)"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing entries if survey_id already exists"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show catalog statistics after building"
    )

    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export catalog to CSV file after building"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be processed, don't add to catalog"
    )

    args = parser.parse_args()

    # Handle recursive flag
    recursive = args.recursive and not args.no_recursive

    # Validate input directories
    for input_dir in args.input:
        if not Path(input_dir).exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)

    # Dry run - just list files
    if args.dry_run:
        print("\n=== Dry Run - Files to Process ===\n")
        total_files = 0
        for input_dir in args.input:
            input_path = Path(input_dir)
            if recursive:
                files = list(input_path.rglob(args.pattern))
            else:
                files = list(input_path.glob(args.pattern))

            print(f"{input_dir}:")
            for f in files[:10]:
                print(f"  - {f.name}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
            total_files += len(files)
            print()

        print(f"Total files: {total_files}")
        return

    # Initialize catalog
    logger.info(f"Initializing catalog at {args.output}")
    catalog = SurveyCatalog(args.output)

    # Add surveys from each input directory
    total_added = 0
    start_time = datetime.now()

    for input_dir in args.input:
        logger.info(f"\nProcessing directory: {input_dir}")
        added = catalog.add_survey_directory(
            directory=input_dir,
            pattern=args.pattern,
            recursive=recursive,
            platform=args.platform,
            n_workers=args.workers,
            replace=args.replace
        )
        total_added += added

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nCompleted in {elapsed:.1f} seconds")
    logger.info(f"Total surveys added: {total_added}")

    # Show statistics
    if args.stats or True:  # Always show stats
        stats = catalog.get_statistics()
        print("\n" + "=" * 50)
        print("CATALOG STATISTICS")
        print("=" * 50)
        print(f"Total surveys:      {stats['total_surveys']:,}")
        print(f"Total points:       {stats['total_points']:,}")
        print(f"Surveys with dates: {stats['surveys_with_dates']:,}")
        print(f"Date range:         {stats['date_range']['min']} to {stats['date_range']['max']}")
        print(f"Registered:         {stats['registered']:,}")
        print(f"Unregistered:       {stats['unregistered']:,}")
        print("\nBy platform:")
        for platform, count in stats['by_platform'].items():
            print(f"  {platform}: {count:,}")
        print("\nTop locations:")
        for location, count in list(stats['top_locations'].items())[:5]:
            print(f"  {location or '(unknown)'}: {count:,}")
        print("=" * 50)

    # Export to CSV if requested
    if args.export_csv:
        catalog.export_to_csv(args.export_csv)
        logger.info(f"Exported catalog to {args.export_csv}")


if __name__ == "__main__":
    main()
