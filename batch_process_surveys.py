#!/usr/bin/env python3
"""
Batch process all LiDAR surveys from CSV file.

Usage:
    python3 batch_process_surveys.py --output ./processed_tiles --subsample 0.3

    # Or with custom settings:
    python3 batch_process_surveys.py \
        --csv coastal_contrast/data/survey_paths.csv \
        --output ./processed_tiles \
        --tile-size 10.0 \
        --grid-size 0.02 \
        --subsample 0.3 \
        --workers 8
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from prep_data import process_las_to_tiles

def batch_process(csv_path, output_base, tile_size=10.0, grid_size=0.02,
                  workers=None, subsample=None, resume=True, path_replace=None):
    """
    Process all surveys listed in CSV file.

    Args:
        csv_path: Path to CSV with 'full_path' column
        output_base: Base directory for all processed tiles
        tile_size: Tile size in meters
        grid_size: Voxel size in meters
        workers: Number of worker threads (None = auto)
        subsample: Random subsample ratio (None = no subsampling)
        resume: Skip already processed surveys
        path_replace: Tuple of (old_prefix, new_prefix) to replace in paths
                      e.g., ('/Volumes', '/project') to fix mount point differences
    """

    # Read survey list
    print(f"Reading survey list from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'full_path' not in df.columns:
        print(f"Error: CSV must have 'full_path' column")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    surveys = df['full_path'].tolist()

    # Apply path replacement if specified
    if path_replace is not None:
        old_prefix, new_prefix = path_replace
        surveys = [s.replace(old_prefix, new_prefix) for s in surveys]
        print(f"Replacing '{old_prefix}' with '{new_prefix}' in all paths")

    print(f"Found {len(surveys)} surveys to process\n")

    # Create output directory
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = output_base / "processing_log.txt"

    # Track progress
    success_count = 0
    skip_count = 0
    error_count = 0
    errors = []

    start_time = datetime.now()

    print("="*80)
    print(f"BATCH PROCESSING")
    print("="*80)
    print(f"Total surveys: {len(surveys)}")
    print(f"Output directory: {output_base}")
    print(f"Tile size: {tile_size}m x {tile_size}m")
    print(f"Voxel size: {grid_size}m")
    if subsample:
        print(f"Subsampling: {subsample*100:.1f}%")
    print(f"Workers: {workers if workers else 'auto'}")
    print(f"Resume mode: {resume}")
    print("="*80)
    print()

    with open(log_file, 'a') as log:
        log.write(f"\n{'='*80}\n")
        log.write(f"Batch started: {start_time}\n")
        log.write(f"{'='*80}\n")

        for i, survey_path in enumerate(surveys, 1):
            # Create unique output directory for this survey
            # Use the survey date and MOP range from the filename
            survey_name = Path(survey_path).stem  # Removes .las extension
            output_dir = output_base / survey_name

            # Check if already processed (resume mode)
            if resume and output_dir.exists():
                tiles = list(output_dir.glob("*.pth"))
                if len(tiles) > 0:
                    print(f"[{i}/{len(surveys)}] SKIP: {survey_name} (already has {len(tiles)} tiles)")
                    log.write(f"[{i}/{len(surveys)}] SKIP: {survey_name} (already processed)\n")
                    skip_count += 1
                    continue

            print(f"\n{'='*80}")
            print(f"[{i}/{len(surveys)}] Processing: {survey_name}")
            print(f"{'='*80}")
            print(f"Input: {survey_path}")
            print(f"Output: {output_dir}")

            # Check if file exists
            if not os.path.exists(survey_path):
                print(f"ERROR: File not found: {survey_path}")
                log.write(f"[{i}/{len(surveys)}] ERROR: File not found - {survey_path}\n")
                errors.append((survey_name, "File not found"))
                error_count += 1
                continue

            # Process the survey
            try:
                survey_start = datetime.now()

                process_las_to_tiles(
                    las_path=survey_path,
                    output_dir=str(output_dir),
                    tile_size=tile_size,
                    grid_size=grid_size,
                    workers=workers,
                    subsample=subsample
                )

                survey_elapsed = (datetime.now() - survey_start).total_seconds()

                # Count created tiles
                tiles = list(output_dir.glob("*.pth"))

                print(f"✓ SUCCESS: Created {len(tiles)} tiles in {survey_elapsed:.1f}s")
                log.write(f"[{i}/{len(surveys)}] SUCCESS: {survey_name} - {len(tiles)} tiles in {survey_elapsed:.1f}s\n")
                success_count += 1

            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                log.write(f"[{i}/{len(surveys)}] ERROR: {survey_name} - {str(e)}\n")
                errors.append((survey_name, str(e)))
                error_count += 1

                # Clean up partial output on error
                if output_dir.exists():
                    import shutil
                    shutil.rmtree(output_dir)

    # Final summary
    total_elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)")
    print(f"Surveys processed: {success_count}/{len(surveys)}")
    print(f"Skipped (already done): {skip_count}")
    print(f"Errors: {error_count}")

    if errors:
        print(f"\nErrors encountered:")
        for survey_name, error_msg in errors:
            print(f"  - {survey_name}: {error_msg}")

    print(f"\nOutput directory: {output_base}")
    print(f"Log file: {log_file}")
    print("="*80)

    # Write summary to log
    with open(log_file, 'a') as log:
        log.write(f"\nBatch completed: {datetime.now()}\n")
        log.write(f"Total time: {total_elapsed:.1f}s\n")
        log.write(f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}\n")
        log.write(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process LiDAR surveys from CSV file"
    )

    parser.add_argument("--csv", type=str,
                       default="coastal_contrast/data/survey_paths.csv",
                       help="CSV file with 'full_path' column")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output base directory for all tiles")
    parser.add_argument("--tile-size", type=float, default=10.0,
                       help="Tile size in meters (default: 10.0)")
    parser.add_argument("--grid-size", type=float, default=0.02,
                       help="Voxel size for downsampling (default: 0.02)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto = 1/2 cores)")
    parser.add_argument("--subsample", type=float, default=None,
                       help="Random subsample ratio (e.g., 0.3 = keep 30%%). Good for dense clouds.")
    parser.add_argument("--no-resume", action="store_true",
                       help="Reprocess all surveys (don't skip already processed)")
    parser.add_argument("--path-prefix", type=str, nargs=2, default=None,
                       metavar=('OLD', 'NEW'),
                       help="Replace path prefix (e.g., --path-prefix /Volumes /project)")

    args = parser.parse_args()

    # Parse path replacement
    path_replace = None
    if args.path_prefix:
        path_replace = tuple(args.path_prefix)

    batch_process(
        csv_path=args.csv,
        output_base=args.output,
        tile_size=args.tile_size,
        grid_size=args.grid_size,
        workers=args.workers,
        subsample=args.subsample,
        resume=not args.no_resume,
        path_replace=path_replace
    )
