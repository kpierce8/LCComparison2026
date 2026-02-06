"""Tests for GEE export module."""

import pytest
from unittest.mock import MagicMock, patch

from src.data.gee_export import GEEExporter, _EE_AVAILABLE


class TestGEEExporterNoAuth:
    """Tests that work without GEE authentication."""

    def test_setup_instructions(self):
        instructions = GEEExporter.get_setup_instructions()
        assert "earthengine" in instructions
        assert "authenticate" in instructions

    def test_unavailable_when_not_authenticated(self):
        exporter = GEEExporter()
        # Without auth, should handle gracefully
        if not exporter.is_available():
            assert not exporter._authenticated

    def test_config_defaults(self):
        exporter = GEEExporter(config={})
        assert exporter.strategy == "drive"
        assert exporter.drive_folder == "LCComparison2026_exports"
        assert exporter.max_pixels == int(1e9)

    def test_config_overrides(self):
        exporter = GEEExporter(config={
            "strategy": "cloud_storage",
            "cloud_bucket": "gs://my-bucket",
            "max_pixels": 5e8,
        })
        assert exporter.strategy == "cloud_storage"
        assert exporter.cloud_bucket == "gs://my-bucket"
        assert exporter.max_pixels == int(5e8)

    def test_check_export_status_unavailable(self):
        exporter = GEEExporter.__new__(GEEExporter)
        exporter._authenticated = False
        exporter.config = {}
        exporter._active_tasks = {}
        result = exporter.check_export_status("fake-task-id")
        assert result["status"] == "unavailable"

    def test_active_tasks_empty(self):
        exporter = GEEExporter.__new__(GEEExporter)
        exporter._authenticated = False
        exporter.config = {}
        exporter._active_tasks = {}
        results = exporter.check_all_tasks()
        assert results == []


class TestGEEExporterMocked:
    """Tests with mocked GEE."""

    @patch("src.data.gee_export._EE_AVAILABLE", True)
    @patch("src.data.gee_export.ee")
    def test_sentinel2_composite_build(self, mock_ee):
        """Test that S2 composite building calls correct GEE methods."""
        mock_ee.Initialize.return_value = None
        mock_ee.Geometry.Rectangle.return_value = "mock_region"
        mock_collection = MagicMock()
        mock_ee.ImageCollection.return_value = mock_collection
        mock_collection.filterBounds.return_value = mock_collection
        mock_collection.filterDate.return_value = mock_collection
        mock_collection.filter.return_value = mock_collection
        mock_collection.map.return_value = mock_collection
        mock_collection.median.return_value = mock_collection
        mock_collection.select.return_value = mock_collection
        mock_collection.clip.return_value = "mock_composite"

        mock_ee.Filter.lt.return_value = "cloud_filter"

        exporter = GEEExporter.__new__(GEEExporter)
        exporter._authenticated = True
        exporter.config = {}

        result = exporter._build_sentinel2_composite(
            bbox={"west": -122, "south": 47, "east": -121, "north": 48},
            date_range=("2024-06-01", "2024-09-30"),
            bands=["B2", "B3", "B4"],
            cloud_filter=20,
        )

        mock_ee.ImageCollection.assert_called_with("COPERNICUS/S2_SR_HARMONIZED")
        assert result == "mock_composite"

    @patch("src.data.gee_export._EE_AVAILABLE", True)
    @patch("src.data.gee_export.ee")
    def test_naip_mosaic_build(self, mock_ee):
        mock_ee.Initialize.return_value = None
        mock_ee.Geometry.Rectangle.return_value = "mock_region"
        mock_collection = MagicMock()
        mock_ee.ImageCollection.return_value = mock_collection
        mock_collection.filterBounds.return_value = mock_collection
        mock_collection.filterDate.return_value = mock_collection
        mock_collection.select.return_value = mock_collection
        mock_collection.mosaic.return_value = mock_collection
        mock_collection.clip.return_value = "mock_mosaic"

        exporter = GEEExporter.__new__(GEEExporter)
        exporter._authenticated = True
        exporter.config = {}

        result = exporter._build_naip_mosaic(
            bbox={"west": -122, "south": 47, "east": -121, "north": 48},
            year=2023,
        )

        mock_ee.ImageCollection.assert_called_with("USDA/NAIP/DOQQ")
        assert result == "mock_mosaic"

    def test_require_available_raises(self):
        exporter = GEEExporter.__new__(GEEExporter)
        exporter._authenticated = False
        exporter.config = {}
        with pytest.raises(RuntimeError, match="GEE is not available"):
            exporter._require_available()
