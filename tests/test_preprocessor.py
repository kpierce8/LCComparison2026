"""Tests for the image preprocessing pipeline."""

import numpy as np
import pytest
from pathlib import Path

from src.data.preprocessor import (
    MODEL_NORMALIZATION,
    Preprocessor,
    S2_BAND_INDEX,
)

try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.fixture
def sample_tile(tmp_path):
    """Create a sample 6-band GeoTIFF tile."""
    if not _RASTERIO_AVAILABLE:
        pytest.skip("rasterio not available")

    path = tmp_path / "test_tile.tif"
    data = np.random.randint(100, 5000, (6, 128, 128), dtype=np.uint16)
    transform = from_bounds(500000, 5200000, 501280, 5201280, 128, 128)

    with rasterio.open(
        path, "w", driver="GTiff", dtype="uint16",
        width=128, height=128, count=6, crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


@pytest.fixture
def tile_with_nodata(tmp_path):
    """Create a tile with nodata values."""
    if not _RASTERIO_AVAILABLE:
        pytest.skip("rasterio not available")

    path = tmp_path / "nodata_tile.tif"
    data = np.random.randint(100, 5000, (6, 64, 64)).astype(np.float32)
    # Set some pixels to nodata
    data[:, :10, :10] = -9999.0
    transform = from_bounds(500000, 5200000, 500640, 5200640, 64, 64)

    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32",
        width=64, height=64, count=6, crs="EPSG:32610",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data)

    return path


class TestModelNormalization:
    def test_prithvi_config(self):
        cfg = MODEL_NORMALIZATION["prithvi"]
        assert cfg["input_size"] == 224
        assert len(cfg["bands"]) == 6
        assert len(cfg["mean"]) == 6
        assert len(cfg["std"]) == 6

    def test_satlas_config(self):
        cfg = MODEL_NORMALIZATION["satlas"]
        assert cfg["input_size"] == 256
        assert len(cfg["bands"]) == 3
        assert len(cfg["mean"]) == 3

    def test_ssl4eo_config(self):
        cfg = MODEL_NORMALIZATION["ssl4eo"]
        assert cfg["input_size"] == 224
        assert len(cfg["bands"]) == 6

    def test_s2_band_index(self):
        assert S2_BAND_INDEX["B2"] == 0
        assert S2_BAND_INDEX["B4"] == 2
        assert S2_BAND_INDEX["B8"] == 3


class TestPreprocessorInit:
    def test_prithvi_defaults(self):
        pp = Preprocessor("prithvi")
        assert pp.model_name == "prithvi"
        assert pp.input_size == 224
        assert len(pp.bands) == 6
        assert pp.mean.shape == (6,)

    def test_satlas_defaults(self):
        pp = Preprocessor("satlas")
        assert pp.input_size == 256
        assert len(pp.bands) == 3

    def test_config_overrides(self):
        pp = Preprocessor("prithvi", config={"input_size": 128, "mean": [0, 0, 0, 0, 0, 0]})
        assert pp.input_size == 128
        assert np.allclose(pp.mean, 0)

    def test_unknown_model_uses_prithvi_defaults(self):
        pp = Preprocessor("custom_model")
        assert pp.input_size == 224  # Falls back to prithvi defaults


class TestPreprocessorArrayOps:
    def test_normalize(self):
        pp = Preprocessor("prithvi")
        data = np.ones((6, 32, 32), dtype=np.float32) * 1000
        result = pp._normalize(data.copy())
        # Should be shifted and scaled
        assert not np.allclose(result, data)

    def test_resize_identity(self):
        pp = Preprocessor("prithvi")
        data = np.random.randn(6, 224, 224).astype(np.float32)
        result = pp._resize(data, 224)
        assert result.shape == (6, 224, 224)
        np.testing.assert_array_equal(result, data)

    def test_resize_smaller(self):
        pp = Preprocessor("prithvi")
        data = np.random.randn(6, 128, 128).astype(np.float32)
        result = pp._resize(data, 64)
        assert result.shape == (6, 64, 64)

    def test_resize_larger(self):
        pp = Preprocessor("prithvi")
        data = np.random.randn(6, 64, 64).astype(np.float32)
        result = pp._resize(data, 128)
        assert result.shape == (6, 128, 128)

    def test_handle_nodata(self):
        pp = Preprocessor("prithvi")
        data = np.ones((6, 32, 32), dtype=np.float32) * 500
        data[:, 0, 0] = -9999
        cleaned, mask = pp._handle_nodata(data, nodata=-9999)
        assert cleaned[:, 0, 0].sum() == 0  # Replaced with 0
        assert not mask[0, 0]  # Marked invalid
        assert mask[1, 1]  # Valid elsewhere

    def test_handle_nan(self):
        pp = Preprocessor("prithvi")
        data = np.ones((6, 32, 32), dtype=np.float32)
        data[0, 5, 5] = np.nan
        cleaned, mask = pp._handle_nodata(data, nodata=None)
        assert not mask[5, 5]
        assert np.isfinite(cleaned).all()

    def test_minimize_edge_effects(self):
        pp = Preprocessor("prithvi")
        data = np.ones((6, 32, 32), dtype=np.float32) * 100
        mask = np.ones((32, 32), dtype=bool)
        mask[:5, :] = False
        data[:, :5, :] = 0

        result = pp._minimize_edge_effects(data, mask)
        # Invalid pixels should be filled with valid mean
        assert result[:, 0, 0].mean() > 0

    def test_preprocess_array(self):
        pp = Preprocessor("prithvi")
        data = np.random.randint(100, 5000, (6, 128, 128)).astype(np.float32)
        result = pp.preprocess_array(data)

        assert result["data"].shape == (6, 224, 224)  # Resized
        assert result["valid_mask"].shape == (224, 224)

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
    def test_preprocess_array_returns_tensor(self):
        pp = Preprocessor("prithvi")
        data = np.random.randint(100, 5000, (6, 128, 128)).astype(np.float32)
        result = pp.preprocess_array(data)
        assert "tensor" in result
        assert result["tensor"].shape == (6, 224, 224)


@pytest.mark.skipif(not _RASTERIO_AVAILABLE, reason="rasterio not available")
class TestPreprocessorFile:
    def test_preprocess_file(self, sample_tile):
        pp = Preprocessor("prithvi")
        result = pp.preprocess_file(sample_tile)

        assert result["data"].shape == (6, 224, 224)
        assert result["valid_mask"].shape == (224, 224)
        assert result["metadata"]["model"] == "prithvi"

    def test_preprocess_file_save(self, sample_tile, tmp_path):
        pp = Preprocessor("prithvi")
        output = tmp_path / "preprocessed.npz"
        result = pp.preprocess_file(sample_tile, output_path=output)

        assert output.exists()
        loaded = np.load(output)
        assert "data" in loaded
        assert "valid_mask" in loaded

    def test_preprocess_with_nodata(self, tile_with_nodata):
        pp = Preprocessor("prithvi")
        result = pp.preprocess_file(tile_with_nodata)

        assert result["data"].shape == (6, 224, 224)
        assert np.isfinite(result["data"]).all()

    def test_satlas_preprocessing(self, sample_tile):
        pp = Preprocessor("satlas")
        result = pp.preprocess_file(sample_tile)

        assert result["data"].shape[0] == 3  # 3 bands for satlas
        assert result["data"].shape[1] == 256  # 256 input size
        assert result["data"].shape[2] == 256

    def test_preprocess_batch(self, sample_tile, tmp_path):
        # Create a second tile
        import shutil
        tile2 = tmp_path / "tile2.tif"
        shutil.copy(sample_tile, tile2)

        pp = Preprocessor("prithvi")
        result = pp.preprocess_batch([sample_tile, tile2])

        assert result["batch"].shape == (2, 6, 224, 224)
        assert result["metadata"]["count"] == 2
        assert result["metadata"]["failed"] == 0

    def test_preprocess_batch_handles_failures(self, tmp_path):
        pp = Preprocessor("prithvi")
        result = pp.preprocess_batch([tmp_path / "nonexistent.tif"])

        assert result["batch"] is None
        assert result["metadata"]["count"] == 0
