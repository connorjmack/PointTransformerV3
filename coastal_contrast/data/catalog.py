"""
Survey Catalog System for Coastal LiDAR Data

Manages metadata for hundreds of LiDAR surveys, enabling efficient
temporal and spatial queries for contrastive learning.

Usage:
    from coastal_contrast.data.catalog import SurveyCatalog

    catalog = SurveyCatalog("./data/catalog.db")
    catalog.add_survey_directory("/path/to/surveys/")

    # Find temporal pairs
    pairs = catalog.get_temporal_pairs(min_days=7, max_days=30)
"""

import os
import re
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Iterator, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    logging.warning("laspy not installed. Install with: pip install laspy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SurveyMetadata:
    """Metadata for a single LiDAR survey."""

    survey_id: str                          # Unique identifier
    file_path: str                          # Absolute path to LAS/LAZ file
    filename: str                           # Just the filename
    date: Optional[str] = None              # ISO format date (YYYY-MM-DD)

    # Spatial bounds (from LAS header)
    min_x: Optional[float] = None
    min_y: Optional[float] = None
    min_z: Optional[float] = None
    max_x: Optional[float] = None
    max_y: Optional[float] = None
    max_z: Optional[float] = None

    # Point cloud statistics
    point_count: Optional[int] = None
    point_format: Optional[int] = None      # LAS point format (0-10)

    # Coordinate reference system
    crs_epsg: Optional[int] = None          # EPSG code if available
    crs_wkt: Optional[str] = None           # WKT string if available

    # Data characteristics
    has_intensity: bool = False
    has_rgb: bool = False
    has_classification: bool = False
    has_return_number: bool = False

    # Platform and collection info
    platform: str = "unknown"               # mobile, terrestrial, airborne
    location_name: Optional[str] = None     # e.g., "del_mar", "solana_beach"

    # Processing status
    is_registered: bool = False             # Has been co-registered
    reference_survey_id: Optional[str] = None  # Reference epoch ID
    registration_rmse: Optional[float] = None  # Registration quality

    # Quality metrics
    quality_score: float = 1.0              # Overall quality (0-1)
    notes: Optional[str] = None             # Any notes

    # Timestamps
    added_at: Optional[str] = None          # When added to catalog
    processed_at: Optional[str] = None      # When last processed

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return (min_x, min_y, min_z, max_x, max_y, max_z)."""
        if all(v is not None for v in [self.min_x, self.min_y, self.min_z,
                                        self.max_x, self.max_y, self.max_z]):
            return (self.min_x, self.min_y, self.min_z,
                    self.max_x, self.max_y, self.max_z)
        return None

    @property
    def bounds_2d(self) -> Optional[Tuple[float, float, float, float]]:
        """Return (min_x, min_y, max_x, max_y)."""
        if all(v is not None for v in [self.min_x, self.min_y,
                                        self.max_x, self.max_y]):
            return (self.min_x, self.min_y, self.max_x, self.max_y)
        return None

    @property
    def date_parsed(self) -> Optional[datetime]:
        """Return date as datetime object."""
        if self.date:
            try:
                return datetime.fromisoformat(self.date)
            except ValueError:
                return None
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurveyMetadata":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# Date Parsing from Filenames
# =============================================================================

class DateParser:
    """Parse dates from various filename formats."""

    # Common date patterns in filenames
    PATTERNS = [
        # ISO format: 2020-01-05, 2020_01_05
        (r'(\d{4})[-_](\d{2})[-_](\d{2})', '%Y-%m-%d'),
        # US format: 01-05-2020, 01_05_2020
        (r'(\d{2})[-_](\d{2})[-_](\d{4})', '%m-%d-%Y'),
        # Compact: 20200105
        (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
        # With time: 2020-01-05T10-30
        (r'(\d{4})[-_](\d{2})[-_](\d{2})[T_](\d{2})[-_](\d{2})', '%Y-%m-%dT%H-%M'),
        # Year and day of year: 2020_005 (day 5 of 2020)
        (r'(\d{4})[-_](\d{3})(?!\d)', 'doy'),
    ]

    @classmethod
    def parse_date(cls, filename: str) -> Optional[str]:
        """
        Extract date from filename, return ISO format string.

        Args:
            filename: The filename to parse

        Returns:
            Date in ISO format (YYYY-MM-DD) or None if not found
        """
        # Remove extension and path
        name = Path(filename).stem

        for pattern, fmt in cls.PATTERNS:
            match = re.search(pattern, name)
            if match:
                try:
                    if fmt == 'doy':
                        # Day of year format
                        year = int(match.group(1))
                        doy = int(match.group(2))
                        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                    elif fmt == '%Y-%m-%dT%H-%M':
                        date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}T{match.group(4)}-{match.group(5)}"
                        date = datetime.strptime(date_str, fmt)
                    elif fmt == '%Y%m%d':
                        date_str = f"{match.group(1)}{match.group(2)}{match.group(3)}"
                        date = datetime.strptime(date_str, fmt)
                    elif fmt == '%m-%d-%Y':
                        date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                        date = datetime.strptime(date_str, fmt)
                    else:
                        date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                        date = datetime.strptime(date_str, '%Y-%m-%d')

                    return date.strftime('%Y-%m-%d')
                except (ValueError, IndexError):
                    continue

        return None

    @classmethod
    def parse_location(cls, filename: str) -> Optional[str]:
        """
        Extract location name from filename.

        Looks for common location patterns like:
        - 2020-01-05_del_mar.las -> del_mar
        - delmar_20200105.las -> delmar
        """
        name = Path(filename).stem.lower()

        # Remove date patterns first
        for pattern, _ in cls.PATTERNS:
            name = re.sub(pattern, '', name)

        # Clean up
        name = re.sub(r'^[-_]+|[-_]+$', '', name)  # Remove leading/trailing separators
        name = re.sub(r'[-_]+', '_', name)          # Normalize separators

        if name and len(name) > 1:
            return name
        return None


# =============================================================================
# LAS File Header Reader
# =============================================================================

def read_las_header(file_path: str) -> Optional[SurveyMetadata]:
    """
    Read LAS file header without loading point data.

    Args:
        file_path: Path to LAS/LAZ file

    Returns:
        SurveyMetadata with header information, or None on error
    """
    if not HAS_LASPY:
        raise ImportError("laspy is required. Install with: pip install laspy")

    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    try:
        # Open in read mode without loading points
        with laspy.open(file_path) as las_file:
            header = las_file.header

            # Generate survey ID from filename
            survey_id = file_path.stem.replace(' ', '_').lower()

            # Parse date from filename
            date = DateParser.parse_date(file_path.name)

            # Parse location from filename
            location = DateParser.parse_location(file_path.name)

            # Get point format info
            point_format = header.point_format

            # Check available fields
            has_intensity = 'intensity' in [dim.name for dim in point_format.dimensions]
            has_rgb = all(c in [dim.name for dim in point_format.dimensions]
                         for c in ['red', 'green', 'blue'])
            has_classification = 'classification' in [dim.name for dim in point_format.dimensions]
            has_return_number = 'return_number' in [dim.name for dim in point_format.dimensions]

            # Try to get CRS
            crs_epsg = None
            crs_wkt = None
            for vlr in header.vlrs:
                if hasattr(vlr, 'string') and 'EPSG' in str(vlr.string):
                    # Try to extract EPSG code
                    match = re.search(r'EPSG[:\s]*(\d+)', str(vlr.string))
                    if match:
                        crs_epsg = int(match.group(1))
                if hasattr(vlr, 'string') and 'PROJCS' in str(vlr.string):
                    crs_wkt = str(vlr.string)

            metadata = SurveyMetadata(
                survey_id=survey_id,
                file_path=str(file_path.absolute()),
                filename=file_path.name,
                date=date,
                min_x=header.x_min,
                min_y=header.y_min,
                min_z=header.z_min,
                max_x=header.x_max,
                max_y=header.y_max,
                max_z=header.z_max,
                point_count=header.point_count,
                point_format=point_format.id,
                crs_epsg=crs_epsg,
                crs_wkt=crs_wkt,
                has_intensity=has_intensity,
                has_rgb=has_rgb,
                has_classification=has_classification,
                has_return_number=has_return_number,
                location_name=location,
                added_at=datetime.now().isoformat(),
            )

            return metadata

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def read_las_header_worker(file_path: str) -> Tuple[str, Optional[Dict]]:
    """Worker function for parallel header reading."""
    metadata = read_las_header(file_path)
    if metadata:
        return file_path, metadata.to_dict()
    return file_path, None


# =============================================================================
# Survey Catalog Database
# =============================================================================

class SurveyCatalog:
    """
    SQLite-backed catalog for managing LiDAR survey metadata.

    Features:
    - Fast temporal and spatial queries
    - Automatic date parsing from filenames
    - Parallel survey ingestion
    - Export to CSV/JSON

    Example:
        catalog = SurveyCatalog("./catalog.db")
        catalog.add_survey_directory("/data/surveys/", pattern="*.las")

        # Find surveys from January 2020
        surveys = catalog.get_surveys_by_date_range("2020-01-01", "2020-01-31")

        # Find temporal pairs (same location, 1-4 weeks apart)
        pairs = catalog.get_temporal_pairs(min_days=7, max_days=28)
    """

    def __init__(self, db_path: str):
        """
        Initialize catalog.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS surveys (
                    survey_id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    date TEXT,
                    min_x REAL,
                    min_y REAL,
                    min_z REAL,
                    max_x REAL,
                    max_y REAL,
                    max_z REAL,
                    point_count INTEGER,
                    point_format INTEGER,
                    crs_epsg INTEGER,
                    crs_wkt TEXT,
                    has_intensity INTEGER,
                    has_rgb INTEGER,
                    has_classification INTEGER,
                    has_return_number INTEGER,
                    platform TEXT,
                    location_name TEXT,
                    is_registered INTEGER DEFAULT 0,
                    reference_survey_id TEXT,
                    registration_rmse REAL,
                    quality_score REAL DEFAULT 1.0,
                    notes TEXT,
                    added_at TEXT,
                    processed_at TEXT
                )
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON surveys(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_location ON surveys(location_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bounds ON surveys(min_x, min_y, max_x, max_y)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_registered ON surveys(is_registered)")

            conn.commit()

    def _metadata_to_row(self, m: SurveyMetadata) -> Tuple:
        """Convert SurveyMetadata to database row."""
        return (
            m.survey_id, m.file_path, m.filename, m.date,
            m.min_x, m.min_y, m.min_z, m.max_x, m.max_y, m.max_z,
            m.point_count, m.point_format, m.crs_epsg, m.crs_wkt,
            int(m.has_intensity), int(m.has_rgb), int(m.has_classification),
            int(m.has_return_number), m.platform, m.location_name,
            int(m.is_registered), m.reference_survey_id, m.registration_rmse,
            m.quality_score, m.notes, m.added_at, m.processed_at
        )

    def _row_to_metadata(self, row: sqlite3.Row) -> SurveyMetadata:
        """Convert database row to SurveyMetadata."""
        return SurveyMetadata(
            survey_id=row['survey_id'],
            file_path=row['file_path'],
            filename=row['filename'],
            date=row['date'],
            min_x=row['min_x'],
            min_y=row['min_y'],
            min_z=row['min_z'],
            max_x=row['max_x'],
            max_y=row['max_y'],
            max_z=row['max_z'],
            point_count=row['point_count'],
            point_format=row['point_format'],
            crs_epsg=row['crs_epsg'],
            crs_wkt=row['crs_wkt'],
            has_intensity=bool(row['has_intensity']),
            has_rgb=bool(row['has_rgb']),
            has_classification=bool(row['has_classification']),
            has_return_number=bool(row['has_return_number']),
            platform=row['platform'],
            location_name=row['location_name'],
            is_registered=bool(row['is_registered']),
            reference_survey_id=row['reference_survey_id'],
            registration_rmse=row['registration_rmse'],
            quality_score=row['quality_score'],
            notes=row['notes'],
            added_at=row['added_at'],
            processed_at=row['processed_at'],
        )

    # =========================================================================
    # Add Surveys
    # =========================================================================

    def add_survey(self, metadata: SurveyMetadata, replace: bool = False) -> bool:
        """
        Add a single survey to the catalog.

        Args:
            metadata: SurveyMetadata object
            replace: If True, replace existing entry with same survey_id

        Returns:
            True if added successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            try:
                if replace:
                    conn.execute("DELETE FROM surveys WHERE survey_id = ?", (metadata.survey_id,))

                conn.execute("""
                    INSERT INTO surveys VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, self._metadata_to_row(metadata))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                logger.warning(f"Survey {metadata.survey_id} already exists. Use replace=True to update.")
                return False

    def add_survey_from_file(self, file_path: str, replace: bool = False,
                             platform: str = "unknown") -> Optional[str]:
        """
        Add survey from LAS file path.

        Args:
            file_path: Path to LAS/LAZ file
            replace: If True, replace existing entry
            platform: Platform type (mobile, terrestrial, airborne)

        Returns:
            survey_id if successful, None otherwise
        """
        metadata = read_las_header(file_path)
        if metadata:
            metadata.platform = platform
            if self.add_survey(metadata, replace=replace):
                return metadata.survey_id
        return None

    def add_survey_directory(self, directory: str, pattern: str = "*.las",
                             recursive: bool = True, platform: str = "unknown",
                             n_workers: int = 4, replace: bool = False) -> int:
        """
        Add all LAS files from a directory.

        Args:
            directory: Directory path to scan
            pattern: Glob pattern for files (*.las, *.laz, etc.)
            recursive: Search subdirectories
            platform: Platform type for all surveys
            n_workers: Number of parallel workers
            replace: Replace existing entries

        Returns:
            Number of surveys added
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return 0

        # Find all matching files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return 0

        logger.info(f"Found {len(files)} files to process")

        # Process files in parallel
        added = 0
        failed = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(read_las_header_worker, str(f)): f for f in files}

            for future in as_completed(futures):
                file_path, metadata_dict = future.result()
                if metadata_dict:
                    metadata = SurveyMetadata.from_dict(metadata_dict)
                    metadata.platform = platform
                    if self.add_survey(metadata, replace=replace):
                        added += 1
                        if added % 50 == 0:
                            logger.info(f"Added {added}/{len(files)} surveys...")
                else:
                    failed.append(file_path)

        if failed:
            logger.warning(f"Failed to process {len(failed)} files")
            for f in failed[:5]:
                logger.warning(f"  - {f}")
            if len(failed) > 5:
                logger.warning(f"  ... and {len(failed) - 5} more")

        logger.info(f"Successfully added {added} surveys to catalog")
        return added

    # =========================================================================
    # Query Surveys
    # =========================================================================

    def get_survey(self, survey_id: str) -> Optional[SurveyMetadata]:
        """Get survey by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM surveys WHERE survey_id = ?", (survey_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_metadata(row)
        return None

    def get_all_surveys(self) -> List[SurveyMetadata]:
        """Get all surveys in catalog."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM surveys ORDER BY date")
            return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def get_surveys_by_date_range(self, start_date: str, end_date: str) -> List[SurveyMetadata]:
        """
        Get surveys within a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of matching surveys
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM surveys WHERE date >= ? AND date <= ? ORDER BY date",
                (start_date, end_date)
            )
            return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def get_surveys_by_location(self, location_name: str) -> List[SurveyMetadata]:
        """Get all surveys for a location."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM surveys WHERE location_name = ? ORDER BY date",
                (location_name,)
            )
            return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def get_surveys_in_bounds(self, min_x: float, min_y: float,
                               max_x: float, max_y: float) -> List[SurveyMetadata]:
        """
        Get surveys that overlap with given bounding box.

        Args:
            min_x, min_y, max_x, max_y: Bounding box coordinates

        Returns:
            List of overlapping surveys
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Check for overlap: NOT (survey_max < query_min OR survey_min > query_max)
            cursor = conn.execute("""
                SELECT * FROM surveys
                WHERE NOT (max_x < ? OR min_x > ? OR max_y < ? OR min_y > ?)
                ORDER BY date
            """, (min_x, max_x, min_y, max_y))
            return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def get_unregistered_surveys(self) -> List[SurveyMetadata]:
        """Get surveys that haven't been co-registered."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM surveys WHERE is_registered = 0 ORDER BY date"
            )
            return [self._row_to_metadata(row) for row in cursor.fetchall()]

    # =========================================================================
    # Temporal Pair Queries (Key for Contrastive Learning)
    # =========================================================================

    def get_temporal_pairs(self, min_days: int = 1, max_days: int = 365,
                           same_location: bool = True,
                           require_overlap: bool = True,
                           min_overlap_ratio: float = 0.5) -> List[Tuple[SurveyMetadata, SurveyMetadata]]:
        """
        Find pairs of surveys suitable for temporal contrastive learning.

        Args:
            min_days: Minimum days between surveys
            max_days: Maximum days between surveys
            same_location: Require same location_name
            require_overlap: Require spatial overlap
            min_overlap_ratio: Minimum overlap ratio (0-1)

        Returns:
            List of (survey_a, survey_b) tuples
        """
        surveys = self.get_all_surveys()
        surveys = [s for s in surveys if s.date is not None]  # Filter surveys without dates
        surveys.sort(key=lambda s: s.date)

        pairs = []

        for i, survey_a in enumerate(surveys):
            date_a = datetime.fromisoformat(survey_a.date)

            for survey_b in surveys[i+1:]:
                date_b = datetime.fromisoformat(survey_b.date)
                delta_days = (date_b - date_a).days

                # Check time constraint
                if delta_days < min_days:
                    continue
                if delta_days > max_days:
                    break  # Sorted by date, so no more valid pairs for survey_a

                # Check location constraint
                if same_location and survey_a.location_name != survey_b.location_name:
                    continue

                # Check spatial overlap
                if require_overlap:
                    overlap = self._compute_overlap_ratio(survey_a, survey_b)
                    if overlap < min_overlap_ratio:
                        continue

                pairs.append((survey_a, survey_b))

        logger.info(f"Found {len(pairs)} temporal pairs with {min_days}-{max_days} day separation")
        return pairs

    def _compute_overlap_ratio(self, a: SurveyMetadata, b: SurveyMetadata) -> float:
        """Compute 2D overlap ratio between two surveys."""
        if a.bounds_2d is None or b.bounds_2d is None:
            return 0.0

        a_min_x, a_min_y, a_max_x, a_max_y = a.bounds_2d
        b_min_x, b_min_y, b_max_x, b_max_y = b.bounds_2d

        # Intersection
        i_min_x = max(a_min_x, b_min_x)
        i_min_y = max(a_min_y, b_min_y)
        i_max_x = min(a_max_x, b_max_x)
        i_max_y = min(a_max_y, b_max_y)

        if i_max_x <= i_min_x or i_max_y <= i_min_y:
            return 0.0

        intersection = (i_max_x - i_min_x) * (i_max_y - i_min_y)

        # Union
        area_a = (a_max_x - a_min_x) * (a_max_y - a_min_y)
        area_b = (b_max_x - b_min_x) * (b_max_y - b_min_y)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def get_temporal_stack(self, location_name: str) -> List[SurveyMetadata]:
        """
        Get all surveys for a location ordered by date (temporal stack).

        Useful for mining correspondences across all time steps.
        """
        surveys = self.get_surveys_by_location(location_name)
        return sorted([s for s in surveys if s.date], key=lambda s: s.date)

    # =========================================================================
    # Update and Delete
    # =========================================================================

    def update_registration_status(self, survey_id: str, is_registered: bool,
                                    reference_id: Optional[str] = None,
                                    rmse: Optional[float] = None) -> bool:
        """Update registration status of a survey."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE surveys
                SET is_registered = ?, reference_survey_id = ?, registration_rmse = ?,
                    processed_at = ?
                WHERE survey_id = ?
            """, (int(is_registered), reference_id, rmse,
                  datetime.now().isoformat(), survey_id))
            conn.commit()
            return conn.total_changes > 0

    def update_platform(self, survey_id: str, platform: str) -> bool:
        """Update platform type of a survey."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE surveys SET platform = ? WHERE survey_id = ?",
                (platform, survey_id)
            )
            conn.commit()
            return conn.total_changes > 0

    def delete_survey(self, survey_id: str) -> bool:
        """Remove a survey from the catalog."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM surveys WHERE survey_id = ?", (survey_id,))
            conn.commit()
            return conn.total_changes > 0

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Total surveys
            cursor = conn.execute("SELECT COUNT(*) FROM surveys")
            stats['total_surveys'] = cursor.fetchone()[0]

            # Total points
            cursor = conn.execute("SELECT SUM(point_count) FROM surveys")
            result = cursor.fetchone()[0]
            stats['total_points'] = result if result else 0

            # Date range
            cursor = conn.execute("SELECT MIN(date), MAX(date) FROM surveys WHERE date IS NOT NULL")
            row = cursor.fetchone()
            stats['date_range'] = {'min': row[0], 'max': row[1]}

            # By platform
            cursor = conn.execute(
                "SELECT platform, COUNT(*) FROM surveys GROUP BY platform"
            )
            stats['by_platform'] = dict(cursor.fetchall())

            # By location
            cursor = conn.execute(
                "SELECT location_name, COUNT(*) FROM surveys GROUP BY location_name ORDER BY COUNT(*) DESC LIMIT 10"
            )
            stats['top_locations'] = dict(cursor.fetchall())

            # Registration status
            cursor = conn.execute(
                "SELECT is_registered, COUNT(*) FROM surveys GROUP BY is_registered"
            )
            reg_counts = dict(cursor.fetchall())
            stats['registered'] = reg_counts.get(1, 0)
            stats['unregistered'] = reg_counts.get(0, 0)

            # Surveys with dates
            cursor = conn.execute(
                "SELECT COUNT(*) FROM surveys WHERE date IS NOT NULL"
            )
            stats['surveys_with_dates'] = cursor.fetchone()[0]

            return stats

    def export_to_csv(self, output_path: str) -> None:
        """Export catalog to CSV file."""
        import csv

        surveys = self.get_all_surveys()
        if not surveys:
            logger.warning("No surveys to export")
            return

        fieldnames = list(surveys[0].to_dict().keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for survey in surveys:
                writer.writerow(survey.to_dict())

        logger.info(f"Exported {len(surveys)} surveys to {output_path}")

    def export_to_json(self, output_path: str) -> None:
        """Export catalog to JSON file."""
        surveys = self.get_all_surveys()

        with open(output_path, 'w') as f:
            json.dump([s.to_dict() for s in surveys], f, indent=2)

        logger.info(f"Exported {len(surveys)} surveys to {output_path}")

    def __len__(self) -> int:
        """Return number of surveys in catalog."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM surveys")
            return cursor.fetchone()[0]

    def __repr__(self) -> str:
        return f"SurveyCatalog({self.db_path}, n_surveys={len(self)})"


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Survey Catalog Management")
    parser.add_argument("--db", type=str, default="./catalog.db", help="Database path")
    parser.add_argument("--add", type=str, help="Add surveys from directory")
    parser.add_argument("--pattern", type=str, default="*.las", help="File pattern")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--export-csv", type=str, help="Export to CSV")
    parser.add_argument("--list-pairs", action="store_true", help="List temporal pairs")
    parser.add_argument("--min-days", type=int, default=7, help="Min days for pairs")
    parser.add_argument("--max-days", type=int, default=30, help="Max days for pairs")

    args = parser.parse_args()

    catalog = SurveyCatalog(args.db)

    if args.add:
        catalog.add_survey_directory(args.add, pattern=args.pattern)

    if args.stats:
        stats = catalog.get_statistics()
        print("\n=== Catalog Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

    if args.export_csv:
        catalog.export_to_csv(args.export_csv)

    if args.list_pairs:
        pairs = catalog.get_temporal_pairs(min_days=args.min_days, max_days=args.max_days)
        print(f"\n=== Temporal Pairs ({len(pairs)}) ===")
        for a, b in pairs[:10]:
            delta = (datetime.fromisoformat(b.date) - datetime.fromisoformat(a.date)).days
            print(f"  {a.survey_id} -> {b.survey_id} ({delta} days)")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")
