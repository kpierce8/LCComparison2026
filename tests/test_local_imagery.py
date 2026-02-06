"""Tests for local imagery handler."""

import numpy as np
import pytest
from pathlib import Path

from src.data.local_imagery import LocalImageryHandler

try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample GeoTIFF for testing."""
    if not _RASTERIO_AVAILABLE:
        pytest.skip("rasterio not available")

    path = tmp_path / "test_tile.tif"
    data = np.random.randint(0, 10000, (6, 64, 64), dtype=np.uint16)
    transform = from_bounds(500000, 5200000, 500640, 5200640, 64, 64)

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": 64,
        "height": 64,
        "count": 6,
        "crs": "EPSG:32610",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    return path


@pytest.fixture
def handler():
    return LocalImageryHandler()


@pytest.mark.skipif(not _RASTERIO_AVAILABLE, reason="rasterio not available")
class TestLocalImageryHandler:
    def test_list_files_empty_dir(self, tmp_path, handler):
        files = handler.list_files(tmp_path)
        assert files == []

    def test_list_files_finds_tifs(self, tmp_path, handler, sample_raster):
        files = handler.list_files(tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".tif"

    def test_read_raster(self, handler, sample_raster):
        result = handler.read_raster(sample_raster)
        assert result["data"].shape == (6, 64, 64)
        assert result["crs"] == "EPSG:32610"
        assert "bounds" in result
        assert "transform" in result

    def test_read_raster_band_selection(self, handler, sample_raster):
        result = handler.read_raster(sample_raster, bands=[1, 3])
        assert result["data"].shape == (2, 64, 64)

    def test_get_raster_info(self, handler, sample_raster):
        info = handler.get_raster_info(sample_raster)
        assert info["count"] == 6
        assert info["shape"] == (64, 64)
        assert info["crs"] == "EPSG:32610"
        assert info["dtype"] == "uint16"
        assert "bounds" in info

    def test_extract_tile(self, handler, sample_raster, tmp_path):
        output = tmp_path / "extracted.tif"
        bbox = {"west": 500000, "south": 5200000, "east": 500320, "north": 5200320}
        result = handler.extract_tile(
            raster_path=sample_raster,
            bbox=bbox,
            output_path=output,
            target_resolution=10.0,
            tile_size=32,
        )
        assert Path(result["path"]).exists()
        assert result["shape"][0] == 6  # all bands
        assert result["shape"][1] == 32
        assert result["shape"][2] == 32

    def test_reproject_raster(self, handler, sample_raster, tmp_path):
        output = tmp_path / "reprojected.tif"
        result = handler.reproject_raster(
            input_path=sample_raster,
            output_path=output,
            target_crs="EPSG:4326",
        )
        assert Path(result["path"]).exists()
        assert result["crs"] == "EPSG:4326"

    def test_config_defaults(self):
        h = LocalImageryHandler()
        assert h.file_pattern == "*.tif"

    def test_config_custom(self):
        h = LocalImageryHandler(config={
            "data_dir": "/custom/path",
            "file_pattern": "*.tiff",
        })
        assert str(h.data_dir) == "/custom/path"
        assert h.file_pattern == "*.tiff"
