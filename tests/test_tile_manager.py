"""Tests for tile management system."""

import json
import pytest
from pathlib import Path

from src.data.tile_manager import TileInfo, TileManager, TileStatus
from src.utils.geo_utils import BBox


@pytest.fixture
def tmp_index(tmp_path):
    """Temporary tile index path."""
    return tmp_path / "tile_index.json"


@pytest.fixture
def tile_manager(tmp_index):
    """TileManager with temporary storage."""
    return TileManager(index_path=tmp_index)


@pytest.fixture
def sample_bbox():
    """Sample bounding box in projected coords (meters)."""
    return BBox(west=500000, south=5200000, east=510000, north=5210000)


class TestTileManager:
    def test_create_grid(self, tile_manager, sample_bbox):
        count = tile_manager.create_grid(
            bbox=sample_bbox,
            resolution=10.0,
            tile_size=256,
            overlap=0,
            source="sentinel2",
        )
        assert count > 0
        assert len(tile_manager.tiles) == count

    def test_create_grid_persists(self, tmp_index, sample_bbox):
        tm1 = TileManager(index_path=tmp_index)
        count = tm1.create_grid(bbox=sample_bbox, resolution=10.0, tile_size=256)

        # Load from disk
        tm2 = TileManager(index_path=tmp_index)
        assert len(tm2.tiles) == count

    def test_tile_ids_correct_format(self, tile_manager, sample_bbox):
        tile_manager.create_grid(bbox=sample_bbox, resolution=10.0, tile_size=256, source="s2")
        for tile_id in tile_manager.tiles:
            assert tile_id.startswith("s2_")
            parts = tile_id.split("_")
            assert len(parts) == 3

    def test_add_tile(self, tile_manager):
        tile = TileInfo(
            tile_id="test_0000_0000",
            row=0, col=0,
            bbox={"west": 0, "south": 0, "east": 100, "north": 100},
            crs="EPSG:32610",
            resolution=10.0,
            source="test",
        )
        tile_manager.add_tile(tile)
        assert "test_0000_0000" in tile_manager.tiles

    def test_update_status(self, tile_manager):
        tile = TileInfo(
            tile_id="test_0001_0001",
            row=1, col=1,
            bbox={"west": 0, "south": 0, "east": 100, "north": 100},
            crs="EPSG:32610",
            resolution=10.0,
            source="test",
        )
        tile_manager.add_tile(tile)

        tile_manager.update_status("test_0001_0001", TileStatus.EXPORTED)
        assert tile_manager.tiles["test_0001_0001"].status == "exported"

    def test_update_status_with_paths(self, tile_manager):
        tile = TileInfo(
            tile_id="test_0002_0002",
            row=2, col=2,
            bbox={"west": 0, "south": 0, "east": 100, "north": 100},
            crs="EPSG:32610",
            resolution=10.0,
            source="test",
        )
        tile_manager.add_tile(tile)
        tile_manager.update_status(
            "test_0002_0002",
            TileStatus.EXPORTED,
            file_paths={"image": "/path/to/image.tif"},
        )
        assert tile_manager.tiles["test_0002_0002"].file_paths["image"] == "/path/to/image.tif"

    def test_update_status_nonexistent_raises(self, tile_manager):
        with pytest.raises(KeyError):
            tile_manager.update_status("nonexistent", TileStatus.EXPORTED)

    def test_get_tiles_by_status(self, tile_manager, sample_bbox):
        tile_manager.create_grid(bbox=sample_bbox, resolution=10.0, tile_size=256)
        pending = tile_manager.get_tiles_by_status(TileStatus.PENDING)
        assert len(pending) == len(tile_manager.tiles)

        # Update one
        first_id = next(iter(tile_manager.tiles))
        tile_manager.update_status(first_id, TileStatus.EXPORTED)
        pending = tile_manager.get_tiles_by_status(TileStatus.PENDING)
        exported = tile_manager.get_tiles_by_status(TileStatus.EXPORTED)
        assert len(pending) == len(tile_manager.tiles) - 1
        assert len(exported) == 1

    def test_get_progress(self, tile_manager, sample_bbox):
        tile_manager.create_grid(bbox=sample_bbox, resolution=10.0, tile_size=256)
        progress = tile_manager.get_progress()
        assert progress["total"] == len(tile_manager.tiles)
        assert progress["pending"] == len(tile_manager.tiles)

    def test_persistence_roundtrip(self, tmp_index):
        tm = TileManager(index_path=tmp_index)
        tile = TileInfo(
            tile_id="rt_0000_0000",
            row=0, col=0,
            bbox={"west": 1, "south": 2, "east": 3, "north": 4},
            crs="EPSG:32610",
            resolution=10.0,
            source="test",
        )
        tm.add_tile(tile)
        tm.update_status("rt_0000_0000", TileStatus.EMBEDDED)

        tm2 = TileManager(index_path=tmp_index)
        assert "rt_0000_0000" in tm2.tiles
        assert tm2.tiles["rt_0000_0000"].status == "embedded"
        assert tm2.tiles["rt_0000_0000"].bbox == {"west": 1, "south": 2, "east": 3, "north": 4}

    def test_failed_status_with_error(self, tile_manager):
        tile = TileInfo(
            tile_id="fail_0000_0000",
            row=0, col=0,
            bbox={"west": 0, "south": 0, "east": 100, "north": 100},
            crs="EPSG:32610",
            resolution=10.0,
            source="test",
        )
        tile_manager.add_tile(tile)
        tile_manager.update_status(
            "fail_0000_0000", TileStatus.FAILED, error="Download timeout"
        )
        assert tile_manager.tiles["fail_0000_0000"].status == "failed"
        assert tile_manager.tiles["fail_0000_0000"].error == "Download timeout"


class TestTileInfo:
    def test_bbox_tuple(self):
        tile = TileInfo(
            tile_id="t_0_0", row=0, col=0,
            bbox={"west": 1.0, "south": 2.0, "east": 3.0, "north": 4.0},
            crs="EPSG:4326", resolution=10.0, source="test",
        )
        bt = tile.bbox_tuple
        assert bt.west == 1.0
        assert bt.north == 4.0

    def test_default_status(self):
        tile = TileInfo(
            tile_id="t_0_0", row=0, col=0,
            bbox={"west": 0, "south": 0, "east": 1, "north": 1},
            crs="EPSG:4326", resolution=10.0, source="test",
        )
        assert tile.status == "pending"
