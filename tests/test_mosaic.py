"""Tests for tile mosaicking."""

from pathlib import Path

import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

from src.spatial.mosaic import TileMosaicker


@pytest.mark.skipif(not _RASTERIO_AVAILABLE, reason="rasterio not available")
class TestTileMosaicker:
    """Test tile mosaic creation."""

    def _make_bbox(self, col, row, size=2560.0):
        """Create a bbox for tile at grid position (col, row)."""
        return {
            "west": col * size,
            "south": row * size,
            "east": (col + 1) * size,
            "north": (row + 1) * size,
        }

    def test_init(self):
        m = TileMosaicker(tile_size=256, resolution=10.0)
        assert m.tile_size == 256
        assert m.resolution == 10.0

    def test_create_classification_tiles(self, tmp_path):
        m = TileMosaicker(tile_size=32, resolution=10.0)
        predictions = {"t0": 1, "t1": 3, "t2": 5}
        bboxes = {
            "t0": self._make_bbox(0, 0),
            "t1": self._make_bbox(1, 0),
            "t2": self._make_bbox(0, 1),
        }

        tiles = m.create_classification_tiles(predictions, bboxes, tmp_path / "class")
        assert len(tiles) == 3

        # Verify tile content
        with rasterio.open(tiles[0]) as src:
            data = src.read(1)
            assert data.shape == (32, 32)
            assert np.all(data == 1)
            assert src.crs is not None

    def test_create_probability_tiles(self, tmp_path):
        m = TileMosaicker(tile_size=16)
        probabilities = {
            "t0": np.array([0.1, 0.5, 0.0, 0.2, 0.0, 0.1, 0.1]),
            "t1": np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1]),
        }
        bboxes = {
            "t0": self._make_bbox(0, 0),
            "t1": self._make_bbox(1, 0),
        }

        tiles = m.create_probability_tiles(
            probabilities, bboxes, tmp_path / "prob", n_classes=7,
        )
        assert len(tiles) == 2

        with rasterio.open(tiles[0]) as src:
            assert src.count == 7
            data = src.read()
            assert data.shape == (7, 16, 16)
            # Band 1 (trees) should be 0.5
            assert np.allclose(data[1], 0.5)

    def test_create_confidence_tiles(self, tmp_path):
        m = TileMosaicker(tile_size=16)
        probabilities = {
            "t0": np.array([0.1, 0.7, 0.2]),
        }
        bboxes = {"t0": self._make_bbox(0, 0)}

        tiles = m.create_confidence_tiles(probabilities, bboxes, tmp_path / "conf")
        assert len(tiles) == 1

        with rasterio.open(tiles[0]) as src:
            data = src.read(1)
            assert np.allclose(data, 0.7)

    def test_mosaic_tiles(self, tmp_path):
        m = TileMosaicker(tile_size=32, crs="EPSG:32610")

        # Create two adjacent tiles
        tile_paths = []
        for i in range(2):
            bbox = self._make_bbox(i, 0, size=320.0)
            data = np.full((1, 32, 32), i, dtype=np.uint8)
            path = tmp_path / f"tile_{i}.tif"

            transform = from_bounds(
                bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                32, 32,
            )
            profile = {
                "driver": "GTiff", "dtype": "uint8", "count": 1,
                "height": 32, "width": 32, "crs": "EPSG:32610",
                "transform": transform,
            }
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data)
            tile_paths.append(path)

        output = tmp_path / "mosaic.tif"
        metadata = m.mosaic_tiles(tile_paths, output)

        assert output.exists()
        assert metadata["n_tiles"] == 2

        with rasterio.open(output) as src:
            data = src.read(1)
            # Mosaic should be wider than a single tile
            assert data.shape[1] >= 32

    def test_mosaic_empty_raises(self, tmp_path):
        m = TileMosaicker()
        with pytest.raises(ValueError, match="No tile paths"):
            m.mosaic_tiles([], tmp_path / "empty.tif")

    def test_generate_all_products(self, tmp_path):
        m = TileMosaicker(tile_size=16, crs="EPSG:32610")

        predictions = {"t0": 2, "t1": 4}
        probabilities = {
            "t0": np.array([0.0, 0.1, 0.6, 0.1, 0.0, 0.1, 0.1]),
            "t1": np.array([0.0, 0.0, 0.1, 0.0, 0.7, 0.1, 0.1]),
        }
        bboxes = {
            "t0": self._make_bbox(0, 0, 160.0),
            "t1": self._make_bbox(1, 0, 160.0),
        }

        products = m.generate_all_products(
            predictions, probabilities, bboxes,
            output_dir=tmp_path / "products",
            model_name="test_model",
            n_classes=7,
        )

        assert "classification" in products
        assert "confidence" in products
        assert "probability" in products
        assert products["tile_counts"]["classification"] == 2

        # Check output files exist
        assert (tmp_path / "products" / "test_model_classification.tif").exists()
        assert (tmp_path / "products" / "test_model_confidence.tif").exists()
        assert (tmp_path / "products" / "test_model_metadata.json").exists()

    def test_missing_bbox_skipped(self, tmp_path):
        m = TileMosaicker(tile_size=16)
        predictions = {"t0": 1, "t_missing": 2}
        bboxes = {"t0": self._make_bbox(0, 0)}

        tiles = m.create_classification_tiles(predictions, bboxes, tmp_path)
        assert len(tiles) == 1  # Only t0 created
