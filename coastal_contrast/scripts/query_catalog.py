#!/usr/bin/env python3
"""
Query Survey Catalog

Provides various queries against the survey catalog including:
- List all surveys
- Find temporal pairs for contrastive learning
- Search by date range, location, or spatial bounds
- Export filtered results

Usage:
    # Show catalog statistics
    python -m coastal_contrast.scripts.query_catalog --db ./catalog.db --stats

    # List all surveys
    python -m coastal_contrast.scripts.query_catalog --db ./catalog.db --list

    # Find temporal pairs (7-30 days apart)
    python -m coastal_contrast.scripts.query_catalog --db ./catalog.db --pairs --min-days 7 --max-days 30

    # Filter by date range
    python -m coastal_contrast.scripts.query_catalog --db ./catalog.db --date-range 2020-01-01 2020-12-31

    # Filter by location
    python -m coastal_contrast.scripts.query_catalog --db ./catalog.db --location del_mar

    # Export temporal pairs to JSON
    python -m coastal_contrast.scripts.query_catalog --db ./catalog.db --pairs --export-pairs pairs.json
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coastal_contrast.data.catalog import SurveyCatalog


def format_size(num_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Query the survey catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--db", "-d",
        type=str,
        required=True,
        help="Path to catalog database"
    )

    # Query modes
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show catalog statistics"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all surveys"
    )

    parser.add_argument(
        "--pairs", "-p",
        action="store_true",
        help="Find temporal pairs for contrastive learning"
    )

    # Filters
    parser.add_argument(
        "--date-range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        help="Filter by date range (YYYY-MM-DD YYYY-MM-DD)"
    )

    parser.add_argument(
        "--location",
        type=str,
        help="Filter by location name"
    )

    parser.add_argument(
        "--platform",
        type=str,
        choices=["mobile", "terrestrial", "airborne"],
        help="Filter by platform"
    )

    parser.add_argument(
        "--unregistered",
        action="store_true",
        help="Show only unregistered surveys"
    )

    # Temporal pair options
    parser.add_argument(
        "--min-days",
        type=int,
        default=7,
        help="Minimum days between pairs (default: 7)"
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=30,
        help="Maximum days between pairs (default: 30)"
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.5,
        help="Minimum spatial overlap ratio for pairs (default: 0.5)"
    )

    # Output options
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV"
    )

    parser.add_argument(
        "--export-json",
        type=str,
        help="Export results to JSON"
    )

    parser.add_argument(
        "--export-pairs",
        type=str,
        help="Export temporal pairs to JSON"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of results shown (default: 50)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )

    args = parser.parse_args()

    # Validate database
    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    catalog = SurveyCatalog(args.db)

    # Show statistics
    if args.stats:
        stats = catalog.get_statistics()
        print("\n" + "=" * 60)
        print("CATALOG STATISTICS")
        print("=" * 60)
        print(f"{'Total surveys:':<25} {stats['total_surveys']:,}")
        print(f"{'Total points:':<25} {stats['total_points']:,}")
        print(f"{'Surveys with dates:':<25} {stats['surveys_with_dates']:,}")

        if stats['date_range']['min']:
            print(f"{'Date range:':<25} {stats['date_range']['min']} to {stats['date_range']['max']}")

            # Calculate span
            start = datetime.fromisoformat(stats['date_range']['min'])
            end = datetime.fromisoformat(stats['date_range']['max'])
            span_years = (end - start).days / 365.25
            print(f"{'Time span:':<25} {span_years:.1f} years")

        print(f"\n{'Registration Status:'}")
        print(f"  {'Registered:':<23} {stats['registered']:,}")
        print(f"  {'Unregistered:':<23} {stats['unregistered']:,}")

        print(f"\n{'By Platform:'}")
        for platform, count in stats['by_platform'].items():
            print(f"  {platform or '(unknown)':<23} {count:,}")

        print(f"\n{'Top Locations:'}")
        for location, count in list(stats['top_locations'].items())[:10]:
            print(f"  {location or '(unknown)':<23} {count:,}")

        print("=" * 60)
        return

    # Get surveys based on filters
    surveys = []

    if args.date_range:
        surveys = catalog.get_surveys_by_date_range(args.date_range[0], args.date_range[1])
    elif args.location:
        surveys = catalog.get_surveys_by_location(args.location)
    elif args.unregistered:
        surveys = catalog.get_unregistered_surveys()
    else:
        surveys = catalog.get_all_surveys()

    # Apply platform filter
    if args.platform:
        surveys = [s for s in surveys if s.platform == args.platform]

    # Find temporal pairs
    if args.pairs:
        pairs = catalog.get_temporal_pairs(
            min_days=args.min_days,
            max_days=args.max_days,
            min_overlap_ratio=args.min_overlap
        )

        print(f"\n=== Temporal Pairs ({len(pairs)} found) ===")
        print(f"Criteria: {args.min_days}-{args.max_days} days apart, {args.min_overlap:.0%} min overlap\n")

        # Group by time delta
        delta_groups = {}
        for a, b in pairs:
            delta = (datetime.fromisoformat(b.date) - datetime.fromisoformat(a.date)).days
            week = delta // 7
            key = f"{week}-{week+1} weeks"
            if key not in delta_groups:
                delta_groups[key] = 0
            delta_groups[key] += 1

        print("Distribution by time delta:")
        for key, count in sorted(delta_groups.items()):
            print(f"  {key}: {count:,} pairs")

        print(f"\nSample pairs (showing {min(args.limit, len(pairs))}):")
        print("-" * 80)
        print(f"{'Survey A':<30} {'Survey B':<30} {'Days':<8} {'Location'}")
        print("-" * 80)

        for a, b in pairs[:args.limit]:
            delta = (datetime.fromisoformat(b.date) - datetime.fromisoformat(a.date)).days
            loc = a.location_name or "(unknown)"
            print(f"{a.survey_id[:28]:<30} {b.survey_id[:28]:<30} {delta:<8} {loc}")

        if len(pairs) > args.limit:
            print(f"\n... and {len(pairs) - args.limit} more pairs")

        # Export pairs if requested
        if args.export_pairs:
            pairs_data = []
            for a, b in pairs:
                delta = (datetime.fromisoformat(b.date) - datetime.fromisoformat(a.date)).days
                pairs_data.append({
                    'survey_a': a.survey_id,
                    'survey_b': b.survey_id,
                    'file_a': a.file_path,
                    'file_b': b.file_path,
                    'date_a': a.date,
                    'date_b': b.date,
                    'delta_days': delta,
                    'location': a.location_name
                })

            with open(args.export_pairs, 'w') as f:
                json.dump(pairs_data, f, indent=2)
            print(f"\nExported {len(pairs)} pairs to {args.export_pairs}")

        return

    # List surveys
    if args.list or surveys:
        print(f"\n=== Surveys ({len(surveys)} found) ===\n")

        if args.verbose:
            for s in surveys[:args.limit]:
                print(f"ID:       {s.survey_id}")
                print(f"File:     {s.file_path}")
                print(f"Date:     {s.date or '(unknown)'}")
                print(f"Points:   {s.point_count:,}" if s.point_count else "Points:   (unknown)")
                print(f"Location: {s.location_name or '(unknown)'}")
                print(f"Platform: {s.platform}")
                print(f"Bounds:   ({s.min_x:.1f}, {s.min_y:.1f}) - ({s.max_x:.1f}, {s.max_y:.1f})" if s.bounds_2d else "Bounds:   (unknown)")
                print(f"Registered: {'Yes' if s.is_registered else 'No'}")
                print("-" * 60)
        else:
            print(f"{'ID':<35} {'Date':<12} {'Points':>12} {'Location':<15}")
            print("-" * 80)
            for s in surveys[:args.limit]:
                points = f"{s.point_count:,}" if s.point_count else "-"
                print(f"{s.survey_id[:33]:<35} {s.date or '-':<12} {points:>12} {(s.location_name or '-')[:15]:<15}")

        if len(surveys) > args.limit:
            print(f"\n... and {len(surveys) - args.limit} more surveys")

    # Export if requested
    if args.export_csv:
        # Write filtered surveys to CSV
        import csv
        with open(args.export_csv, 'w', newline='') as f:
            if surveys:
                fieldnames = list(surveys[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in surveys:
                    writer.writerow(s.to_dict())
        print(f"\nExported {len(surveys)} surveys to {args.export_csv}")

    if args.export_json:
        with open(args.export_json, 'w') as f:
            json.dump([s.to_dict() for s in surveys], f, indent=2)
        print(f"\nExported {len(surveys)} surveys to {args.export_json}")


if __name__ == "__main__":
    main()
