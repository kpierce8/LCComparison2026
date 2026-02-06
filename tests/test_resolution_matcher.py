"""Tests for resolution matching and alignment."""

from pathlib import Path

import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


def _create_test_raster(path, data, bbox=(0, 0, 320, 320), crs="EPSG:32610"):
    h, w = data.shape[-2], data.shape[-1]
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    profile = {
        "driver": "GTiff", "dtype": str(data.dtype),
        "count": data.shape[0], "height": h, "width": w,
        "crs": crs, "transform": transform, "nodata": 255,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


@pytest.mark.skipif(not _RASTERIO_AVAILABLE, reason="rasterio required")
class TestResolutionMatcher:
    """Test resolution matching and alignment."""

    def test_get_raster_info(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        data = np.full((32, 32), 1, dtype=np.uint8)
        raster_path = tmp_path / "test.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        matcher = ResolutionMatcher()
        info = matcher.get_raster_info(raster_path)

        assert info["resolution"] == (10.0, 10.0)
        assert info["shape"] == (32, 32)
        assert info["count"] == 1
        assert "EPSG:32610" in info["crs"]

    def test_compute_common_extent_intersection(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        data = np.full((32, 32), 1, dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data, bbox=(0, 0, 320, 320))
        _create_test_raster(path_b, data, bbox=(160, 160, 480, 480))

        matcher = ResolutionMatcher()
        extent = matcher.compute_common_extent([path_a, path_b], mode="intersection")

        assert extent == (160, 160, 320, 320)

    def test_compute_common_extent_union(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        data = np.full((32, 32), 1, dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data, bbox=(0, 0, 320, 320))
        _create_test_raster(path_b, data, bbox=(160, 160, 480, 480))

        matcher = ResolutionMatcher()
        extent = matcher.compute_common_extent([path_a, path_b], mode="union")

        assert extent == (0, 0, 480, 480)

    def test_compute_common_extent_no_overlap(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        data = np.full((32, 32), 1, dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data, bbox=(0, 0, 100, 100))
        _create_test_raster(path_b, data, bbox=(200, 200, 300, 300))

        matcher = ResolutionMatcher()
        with pytest.raises(ValueError, match="do not overlap"):
            matcher.compute_common_extent([path_a, path_b], mode="intersection")

    def test_resample_to_target(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        # 32x32 at 10m resolution (320m extent)
        data = np.full((32, 32), 3, dtype=np.uint8)
        raster_path = tmp_path / "input.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        matcher = ResolutionMatcher()
        output = tmp_path / "resampled.tif"

        result = matcher.resample_to_target(
            raster_path, output,
            target_resolution=20.0,  # coarsen to 20m
        )

        assert output.exists()
        assert result["resolution"] == 20.0
        assert result["shape"] == (16, 16)  # 320/20 = 16

        with rasterio.open(output) as src:
            resampled = src.read(1)
            # Should be mostly class 3 (nearest neighbor)
            assert np.sum(resampled == 3) > 200

    def test_resample_with_bounds(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        data = np.full((32, 32), 1, dtype=np.uint8)
        raster_path = tmp_path / "input.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        matcher = ResolutionMatcher()
        output = tmp_path / "clipped.tif"

        result = matcher.resample_to_target(
            raster_path, output,
            target_resolution=10.0,
            target_bounds=(0, 0, 160, 160),  # half the extent
        )

        assert result["shape"] == (16, 16)

    def test_align_rasters_same_resolution(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        # Two rasters at same resolution but different extents
        data = np.full((32, 32), 1, dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data, bbox=(0, 0, 320, 320))

        data_b = np.full((32, 32), 2, dtype=np.uint8)
        _create_test_raster(path_b, data_b, bbox=(160, 0, 480, 320))

        matcher = ResolutionMatcher()
        results = matcher.align_rasters(
            {"model_a": path_a, "model_b": path_b},
            tmp_path / "aligned",
        )

        assert "model_a" in results
        assert "model_b" in results
        assert results["model_a"]["shape"] == results["model_b"]["shape"]

    def test_align_rasters_different_resolutions(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        # 10m resolution raster
        data_10m = np.full((32, 32), 1, dtype=np.uint8)
        path_10m = tmp_path / "10m.tif"
        _create_test_raster(path_10m, data_10m, bbox=(0, 0, 320, 320))

        # 5m resolution raster (64x64 for same extent)
        data_5m = np.full((64, 64), 2, dtype=np.uint8)
        path_5m = tmp_path / "5m.tif"
        _create_test_raster(path_5m, data_5m, bbox=(0, 0, 320, 320))

        matcher = ResolutionMatcher()
        results = matcher.align_rasters(
            {"sentinel2": path_10m, "naip": path_5m},
            tmp_path / "aligned",
        )

        # Should align to finest (5m)
        assert results["sentinel2"]["resolution"] == 5.0
        assert results["naip"]["resolution"] == 5.0
        assert results["sentinel2"]["shape"] == results["naip"]["shape"]

    def test_load_aligned_arrays(self, tmp_path):
        from src.processing.resolution_matcher import ResolutionMatcher

        data_a = np.full((16, 16), 1, dtype=np.uint8)
        data_b = np.full((16, 16), 2, dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data_a, bbox=(0, 0, 160, 160))
        _create_test_raster(path_b, data_b, bbox=(0, 0, 160, 160))

        matcher = ResolutionMatcher()
        arrays, metadata = matcher.load_aligned_arrays(
            {"a": path_a, "b": path_b},
        )

        assert "a" in arrays
        assert "b" in arrays
        assert arrays["a"].shape == (16, 16)
        assert np.all(arrays["a"] == 1)
        assert np.all(arrays["b"] == 2)
        assert "profile" in metadata
