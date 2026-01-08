"""
Tests for Survey Catalog System

Run with: pytest tests/test_catalog.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coastal_contrast.data.catalog import (
    SurveyCatalog,
    SurveyMetadata,
    DateParser,
    read_las_header
)


# =============================================================================
# DateParser Tests
# =============================================================================

class TestDateParser:
    """Tests for date parsing from filenames."""

    def test_iso_format_dash(self):
        """Test ISO format with dashes: 2020-01-05"""
        assert DateParser.parse_date("survey_2020-01-05.las") == "2020-01-05"
        assert DateParser.parse_date("2020-01-05_delmar.las") == "2020-01-05"

    def test_iso_format_underscore(self):
        """Test ISO format with underscores: 2020_01_05"""
        assert DateParser.parse_date("survey_2020_01_05.las") == "2020-01-05"

    def test_compact_format(self):
        """Test compact format: 20200105"""
        assert DateParser.parse_date("survey_20200105.las") == "2020-01-05"
        assert DateParser.parse_date("20200105_delmar.las") == "2020-01-05"

    def test_us_format(self):
        """Test US format: 01-05-2020"""
        assert DateParser.parse_date("survey_01-05-2020.las") == "2020-01-05"

    def test_day_of_year(self):
        """Test day of year format: 2020_005 (5th day of 2020)"""
        assert DateParser.parse_date("survey_2020_005.las") == "2020-01-05"
        assert DateParser.parse_date("2020_365.las") == "2020-12-30"

    def test_no_date(self):
        """Test filename without date."""
        assert DateParser.parse_date("survey.las") is None
        assert DateParser.parse_date("delmar_scan.las") is None

    def test_location_parsing(self):
        """Test location extraction from filenames."""
        assert DateParser.parse_location("2020-01-05_del_mar.las") == "del_mar"
        assert DateParser.parse_location("delmar_20200105.las") == "delmar"

    def test_location_empty(self):
        """Test location parsing when only date present."""
        # When filename is only date, location should be None or empty
        result = DateParser.parse_location("20200105.las")
        assert result is None or result == ""


# =============================================================================
# SurveyMetadata Tests
# =============================================================================

class TestSurveyMetadata:
    """Tests for SurveyMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating basic metadata."""
        meta = SurveyMetadata(
            survey_id="test_001",
            file_path="/path/to/file.las",
            filename="file.las"
        )
        assert meta.survey_id == "test_001"
        assert meta.file_path == "/path/to/file.las"

    def test_bounds_property(self):
        """Test bounds property."""
        meta = SurveyMetadata(
            survey_id="test",
            file_path="/path/to/file.las",
            filename="file.las",
            min_x=0.0, min_y=0.0, min_z=0.0,
            max_x=100.0, max_y=100.0, max_z=50.0
        )
        assert meta.bounds == (0.0, 0.0, 0.0, 100.0, 100.0, 50.0)
        assert meta.bounds_2d == (0.0, 0.0, 100.0, 100.0)

    def test_bounds_none(self):
        """Test bounds when not all values present."""
        meta = SurveyMetadata(
            survey_id="test",
            file_path="/path/to/file.las",
            filename="file.las",
            min_x=0.0, min_y=0.0  # Missing z and max values
        )
        assert meta.bounds is None
        assert meta.bounds_2d is None

    def test_date_parsed(self):
        """Test date parsing to datetime."""
        meta = SurveyMetadata(
            survey_id="test",
            file_path="/path/to/file.las",
            filename="file.las",
            date="2020-01-05"
        )
        assert meta.date_parsed == datetime(2020, 1, 5)

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = SurveyMetadata(
            survey_id="test_001",
            file_path="/path/to/file.las",
            filename="file.las",
            date="2020-01-05",
            point_count=1000000,
            platform="mobile"
        )

        data = original.to_dict()
        restored = SurveyMetadata.from_dict(data)

        assert restored.survey_id == original.survey_id
        assert restored.date == original.date
        assert restored.point_count == original.point_count


# =============================================================================
# SurveyCatalog Tests
# =============================================================================

class TestSurveyCatalog:
    """Tests for SurveyCatalog database operations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database."""
        db_path = tmp_path / "test_catalog.db"
        return SurveyCatalog(str(db_path))

    @pytest.fixture
    def sample_metadata(self):
        """Create sample survey metadata."""
        return SurveyMetadata(
            survey_id="survey_2020_01_05",
            file_path="/data/surveys/2020-01-05.las",
            filename="2020-01-05.las",
            date="2020-01-05",
            min_x=0.0, min_y=0.0, min_z=0.0,
            max_x=100.0, max_y=100.0, max_z=50.0,
            point_count=1000000,
            platform="mobile",
            location_name="del_mar"
        )

    def test_create_catalog(self, temp_db):
        """Test catalog creation."""
        assert len(temp_db) == 0

    def test_add_survey(self, temp_db, sample_metadata):
        """Test adding a survey."""
        assert temp_db.add_survey(sample_metadata) is True
        assert len(temp_db) == 1

    def test_add_duplicate_survey(self, temp_db, sample_metadata):
        """Test that duplicate survey_id fails without replace=True."""
        temp_db.add_survey(sample_metadata)
        assert temp_db.add_survey(sample_metadata) is False
        assert len(temp_db) == 1

    def test_add_duplicate_with_replace(self, temp_db, sample_metadata):
        """Test replacing existing survey."""
        temp_db.add_survey(sample_metadata)
        sample_metadata.point_count = 2000000
        assert temp_db.add_survey(sample_metadata, replace=True) is True
        assert len(temp_db) == 1

        retrieved = temp_db.get_survey(sample_metadata.survey_id)
        assert retrieved.point_count == 2000000

    def test_get_survey(self, temp_db, sample_metadata):
        """Test retrieving a survey by ID."""
        temp_db.add_survey(sample_metadata)
        retrieved = temp_db.get_survey(sample_metadata.survey_id)

        assert retrieved is not None
        assert retrieved.survey_id == sample_metadata.survey_id
        assert retrieved.date == sample_metadata.date

    def test_get_survey_not_found(self, temp_db):
        """Test retrieving non-existent survey."""
        assert temp_db.get_survey("nonexistent") is None

    def test_get_all_surveys(self, temp_db):
        """Test getting all surveys."""
        for i in range(5):
            meta = SurveyMetadata(
                survey_id=f"survey_{i}",
                file_path=f"/path/survey_{i}.las",
                filename=f"survey_{i}.las",
                date=f"2020-01-{i+1:02d}"
            )
            temp_db.add_survey(meta)

        surveys = temp_db.get_all_surveys()
        assert len(surveys) == 5

    def test_get_surveys_by_date_range(self, temp_db):
        """Test date range filtering."""
        dates = ["2020-01-05", "2020-01-15", "2020-02-01", "2020-03-01"]
        for i, date in enumerate(dates):
            meta = SurveyMetadata(
                survey_id=f"survey_{i}",
                file_path=f"/path/survey_{i}.las",
                filename=f"survey_{i}.las",
                date=date
            )
            temp_db.add_survey(meta)

        # Query January only
        january = temp_db.get_surveys_by_date_range("2020-01-01", "2020-01-31")
        assert len(january) == 2

        # Query all
        all_surveys = temp_db.get_surveys_by_date_range("2020-01-01", "2020-12-31")
        assert len(all_surveys) == 4

    def test_get_surveys_by_location(self, temp_db):
        """Test location filtering."""
        locations = ["del_mar", "del_mar", "solana_beach", "del_mar"]
        for i, loc in enumerate(locations):
            meta = SurveyMetadata(
                survey_id=f"survey_{i}",
                file_path=f"/path/survey_{i}.las",
                filename=f"survey_{i}.las",
                location_name=loc
            )
            temp_db.add_survey(meta)

        del_mar = temp_db.get_surveys_by_location("del_mar")
        assert len(del_mar) == 3

        solana = temp_db.get_surveys_by_location("solana_beach")
        assert len(solana) == 1

    def test_delete_survey(self, temp_db, sample_metadata):
        """Test deleting a survey."""
        temp_db.add_survey(sample_metadata)
        assert len(temp_db) == 1

        assert temp_db.delete_survey(sample_metadata.survey_id) is True
        assert len(temp_db) == 0

    def test_delete_nonexistent(self, temp_db):
        """Test deleting non-existent survey."""
        assert temp_db.delete_survey("nonexistent") is False


# =============================================================================
# Temporal Pair Tests
# =============================================================================

class TestTemporalPairs:
    """Tests for temporal pair finding."""

    @pytest.fixture
    def catalog_with_temporal_data(self, tmp_path):
        """Create catalog with temporal survey data."""
        db_path = tmp_path / "temporal_catalog.db"
        catalog = SurveyCatalog(str(db_path))

        # Add surveys at weekly intervals for 2 months
        base_date = datetime(2020, 1, 1)
        for i in range(8):
            date = base_date + timedelta(days=i * 7)
            meta = SurveyMetadata(
                survey_id=f"survey_{date.strftime('%Y%m%d')}",
                file_path=f"/path/{date.strftime('%Y-%m-%d')}.las",
                filename=f"{date.strftime('%Y-%m-%d')}.las",
                date=date.strftime('%Y-%m-%d'),
                min_x=0.0, min_y=0.0, min_z=0.0,
                max_x=100.0, max_y=100.0, max_z=50.0,
                location_name="del_mar"
            )
            catalog.add_survey(meta)

        return catalog

    def test_find_temporal_pairs(self, catalog_with_temporal_data):
        """Test finding temporal pairs."""
        pairs = catalog_with_temporal_data.get_temporal_pairs(
            min_days=7,
            max_days=14
        )

        # With 8 weekly surveys and 7-14 day constraint,
        # each survey pairs with next 1-2 surveys
        assert len(pairs) > 0

        # Check all pairs satisfy constraints
        for a, b in pairs:
            delta = (datetime.fromisoformat(b.date) - datetime.fromisoformat(a.date)).days
            assert 7 <= delta <= 14

    def test_temporal_pairs_same_location(self, catalog_with_temporal_data):
        """Test that pairs require same location by default."""
        # Add a survey at different location
        meta = SurveyMetadata(
            survey_id="survey_other_loc",
            file_path="/path/other.las",
            filename="other.las",
            date="2020-01-08",  # 7 days after first survey
            min_x=0.0, min_y=0.0, min_z=0.0,
            max_x=100.0, max_y=100.0, max_z=50.0,
            location_name="solana_beach"  # Different location
        )
        catalog_with_temporal_data.add_survey(meta)

        pairs = catalog_with_temporal_data.get_temporal_pairs(
            min_days=7,
            max_days=14,
            same_location=True
        )

        # No pairs should include the different location
        for a, b in pairs:
            assert a.location_name == b.location_name

    def test_overlap_calculation(self, tmp_path):
        """Test spatial overlap ratio calculation."""
        db_path = tmp_path / "overlap_catalog.db"
        catalog = SurveyCatalog(str(db_path))

        # Two overlapping surveys
        meta1 = SurveyMetadata(
            survey_id="survey_1",
            file_path="/path/1.las",
            filename="1.las",
            date="2020-01-01",
            min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0,
            location_name="test"
        )
        meta2 = SurveyMetadata(
            survey_id="survey_2",
            file_path="/path/2.las",
            filename="2.las",
            date="2020-01-08",
            min_x=50.0, min_y=50.0, max_x=150.0, max_y=150.0,  # 50% overlap
            location_name="test"
        )

        overlap = catalog._compute_overlap_ratio(meta1, meta2)
        # Overlap area = 50*50 = 2500
        # Union area = 100*100 + 100*100 - 2500 = 17500
        # Ratio = 2500/17500 ≈ 0.143
        assert 0.1 < overlap < 0.2


# =============================================================================
# Statistics and Export Tests
# =============================================================================

class TestStatisticsAndExport:
    """Tests for statistics and export functionality."""

    @pytest.fixture
    def populated_catalog(self, tmp_path):
        """Create a populated catalog for testing."""
        db_path = tmp_path / "stats_catalog.db"
        catalog = SurveyCatalog(str(db_path))

        for i in range(10):
            meta = SurveyMetadata(
                survey_id=f"survey_{i}",
                file_path=f"/path/survey_{i}.las",
                filename=f"survey_{i}.las",
                date=f"2020-01-{i+1:02d}",
                point_count=1000000 * (i + 1),
                platform="mobile" if i % 2 == 0 else "airborne",
                location_name="del_mar" if i < 7 else "solana_beach"
            )
            catalog.add_survey(meta)

        return catalog

    def test_get_statistics(self, populated_catalog):
        """Test statistics computation."""
        stats = populated_catalog.get_statistics()

        assert stats['total_surveys'] == 10
        assert stats['total_points'] == sum(1000000 * (i + 1) for i in range(10))
        assert stats['surveys_with_dates'] == 10
        assert 'mobile' in stats['by_platform']
        assert 'airborne' in stats['by_platform']

    def test_export_csv(self, populated_catalog, tmp_path):
        """Test CSV export."""
        csv_path = tmp_path / "export.csv"
        populated_catalog.export_to_csv(str(csv_path))

        assert csv_path.exists()

        # Read and verify
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 11  # Header + 10 surveys

    def test_export_json(self, populated_catalog, tmp_path):
        """Test JSON export."""
        import json

        json_path = tmp_path / "export.json"
        populated_catalog.export_to_json(str(json_path))

        assert json_path.exists()

        with open(json_path, 'r') as f:
            data = json.load(f)
        assert len(data) == 10


# =============================================================================
# Integration Tests with Real LAS Files (Optional)
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get('TEST_LAS_PATH'),
    reason="Set TEST_LAS_PATH environment variable to run LAS file tests"
)
class TestLASIntegration:
    """Integration tests with real LAS files."""

    def test_read_las_header(self):
        """Test reading real LAS file header."""
        las_path = os.environ['TEST_LAS_PATH']
        metadata = read_las_header(las_path)

        assert metadata is not None
        assert metadata.point_count > 0
        assert metadata.bounds is not None

    def test_add_from_file(self, tmp_path):
        """Test adding survey from real file."""
        las_path = os.environ['TEST_LAS_PATH']
        db_path = tmp_path / "real_catalog.db"
        catalog = SurveyCatalog(str(db_path))

        survey_id = catalog.add_survey_from_file(las_path)
        assert survey_id is not None

        retrieved = catalog.get_survey(survey_id)
        assert retrieved is not None
        assert retrieved.point_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
