"""Tests for spatial analysis: focus areas, boundaries, zonal statistics."""

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_bounds
    from shapely.geometry import Point, Polygon, box
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False


def _create_test_raster(path, data, bbox=(0, 0, 2560, 2560), crs="EPSG:32610"):
    """Create a test GeoTIFF raster."""
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


def _create_test_boundaries(path, features, id_field="NAME", crs="EPSG:32610"):
    """Create a test boundary shapefile/GeoJSON."""
    gdf = gpd.GeoDataFrame(features, crs=crs)
    gdf.to_file(path)


@pytest.mark.skipif(not _GEO_AVAILABLE, reason="geopandas/rasterio required")
class TestFocusAreaManager:
    """Test focus area layer loading and management."""

    def test_load_layer(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer

        # Create test boundaries
        features = [
            {"NAME": "County_A", "geometry": box(0, 0, 1000, 1000)},
            {"NAME": "County_B", "geometry": box(1000, 0, 2000, 1000)},
        ]
        boundary_path = tmp_path / "counties.geojson"
        _create_test_boundaries(boundary_path, features, "NAME")

        layer = FocusAreaLayer(name="county", path=boundary_path, id_field="NAME")
        layer.load()

        assert layer.is_loaded
        assert layer.count == 2
        assert "County_A" in layer.features
        assert "County_B" in layer.features

    def test_get_feature(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer

        features = [{"NAME": "Test", "geometry": box(0, 0, 100, 100)}]
        path = tmp_path / "test.geojson"
        _create_test_boundaries(path, features)

        layer = FocusAreaLayer(name="test", path=path, id_field="NAME")
        layer.load()

        result = layer.get_feature("Test")
        assert len(result) == 1

    def test_get_geometry(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer

        features = [{"NAME": "Zone1", "geometry": box(0, 0, 500, 500)}]
        path = tmp_path / "zones.geojson"
        _create_test_boundaries(path, features)

        layer = FocusAreaLayer(name="zone", path=path, id_field="NAME")
        layer.load()

        geom = layer.get_geometry("Zone1")
        assert geom is not None
        assert geom.area > 0

    def test_get_bounds(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer

        features = [{"NAME": "Area", "geometry": box(100, 200, 300, 400)}]
        path = tmp_path / "area.geojson"
        _create_test_boundaries(path, features)

        layer = FocusAreaLayer(name="area", path=path, id_field="NAME")
        layer.load()

        bounds = layer.get_bounds("Area")
        assert bounds[0] == pytest.approx(100)  # west
        assert bounds[1] == pytest.approx(200)  # south

    def test_missing_id_field(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer

        features = [{"NAME": "X", "geometry": box(0, 0, 10, 10)}]
        path = tmp_path / "bad.geojson"
        _create_test_boundaries(path, features)

        layer = FocusAreaLayer(name="bad", path=path, id_field="MISSING_FIELD")
        with pytest.raises(ValueError, match="not found"):
            layer.load()

    def test_file_not_found(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer

        layer = FocusAreaLayer(name="x", path=tmp_path / "nope.shp", id_field="ID")
        with pytest.raises(FileNotFoundError):
            layer.load()

    def test_manager(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaManager

        features = [{"ID": "1", "geometry": box(0, 0, 100, 100)}]
        path = tmp_path / "layer.geojson"
        _create_test_boundaries(path, features, "ID")

        manager = FocusAreaManager()
        manager.add_layer("test", path, "ID")
        assert "test" in manager.layers

        loaded = manager.load_layer("test")
        assert loaded.count == 1

    def test_manager_from_config(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaManager

        features = [{"FID": "a", "geometry": box(0, 0, 50, 50)}]
        path = tmp_path / "cfg.geojson"
        _create_test_boundaries(path, features, "FID")

        config = {
            "layers": [
                {"name": "test_layer", "path": str(path), "id_field": "FID", "enabled": True},
                {"name": "disabled", "path": "nope", "id_field": "X", "enabled": False},
            ]
        }
        manager = FocusAreaManager(config=config)
        assert "test_layer" in manager.layers
        assert "disabled" not in manager.layers

    def test_get_info(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer

        features = [{"NAME": "Z", "geometry": box(0, 0, 10, 10)}]
        path = tmp_path / "info.geojson"
        _create_test_boundaries(path, features)

        layer = FocusAreaLayer(name="info", path=path, id_field="NAME")
        layer.load()
        info = layer.get_info()
        assert info["name"] == "info"
        assert info["loaded"]
        assert info["count"] == 1


@pytest.mark.skipif(not _GEO_AVAILABLE, reason="geopandas/rasterio required")
class TestBoundaryProcessor:
    """Test raster clipping to boundaries."""

    def test_clip_raster(self, tmp_path):
        from src.spatial.boundary_processor import BoundaryProcessor

        # Create a 32x32 raster covering 0-320, 0-320
        data = np.full((32, 32), 3, dtype=np.uint8)
        raster_path = tmp_path / "input.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        # Clip to a sub-region
        geom = box(0, 0, 160, 160)
        output = tmp_path / "clipped.tif"

        processor = BoundaryProcessor(nodata=255)
        result = processor.clip_raster(raster_path, geom, output)

        assert output.exists()
        assert result["valid_pixels"] > 0

        with rasterio.open(output) as src:
            clipped = src.read(1)
            assert clipped.shape[0] <= 32
            assert clipped.shape[1] <= 32

    def test_clip_to_bbox(self, tmp_path):
        from src.spatial.boundary_processor import BoundaryProcessor

        data = np.full((32, 32), 1, dtype=np.uint8)
        raster_path = tmp_path / "raster.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        output = tmp_path / "bbox_clip.tif"
        processor = BoundaryProcessor()

        result = processor.clip_to_bbox(
            raster_path,
            {"west": 0, "south": 0, "east": 160, "north": 160},
            output,
        )
        assert output.exists()
        assert result["valid_pixels"] > 0

    def test_process_layer(self, tmp_path):
        from src.spatial.boundary_processor import BoundaryProcessor
        from src.spatial.focus_area_manager import FocusAreaLayer

        # Raster: 64x64, bbox 0-640
        data = np.full((64, 64), 2, dtype=np.uint8)
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 640, 640))

        # Boundaries: two regions
        features = [
            {"NAME": "Region_A", "geometry": box(0, 0, 320, 320)},
            {"NAME": "Region_B", "geometry": box(320, 320, 640, 640)},
        ]
        boundary_path = tmp_path / "regions.geojson"
        _create_test_boundaries(boundary_path, features)

        layer = FocusAreaLayer(name="region", path=boundary_path, id_field="NAME")
        layer.load()

        processor = BoundaryProcessor()
        results = processor.process_layer(raster_path, layer, tmp_path / "output")

        assert "Region_A" in results
        assert results["Region_A"]["status"] == "success"
        assert results["Region_B"]["status"] == "success"


@pytest.mark.skipif(not _GEO_AVAILABLE, reason="geopandas/rasterio required")
class TestZonalStatistics:
    """Test zonal statistics computation."""

    def test_compute_zonal_stats(self, tmp_path):
        from src.spatial.zonal_statistics import compute_zonal_stats

        # Create raster with known class distribution
        data = np.zeros((32, 32), dtype=np.uint8)
        data[:16, :] = 1  # trees (top half)
        data[16:, :] = 3  # grass (bottom half)

        raster_path = tmp_path / "classified.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        # Full raster zone
        geom = box(0, 0, 320, 320)
        stats = compute_zonal_stats(raster_path, geom, resolution=10.0)

        assert stats["valid_pixels"] == 32 * 32
        assert "classes" in stats
        assert "trees" in stats["classes"]
        assert "grass" in stats["classes"]
        assert stats["classes"]["trees"]["percentage"] == pytest.approx(50.0)
        assert stats["classes"]["grass"]["percentage"] == pytest.approx(50.0)

    def test_zonal_stats_partial_zone(self, tmp_path):
        from src.spatial.zonal_statistics import compute_zonal_stats

        data = np.full((32, 32), 5, dtype=np.uint8)  # all built
        raster_path = tmp_path / "built.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        # Zone covering only part
        geom = box(0, 0, 160, 160)
        stats = compute_zonal_stats(raster_path, geom, resolution=10.0)

        assert stats["valid_pixels"] > 0
        assert stats["valid_pixels"] < 32 * 32
        assert stats["dominant_class"] == "built"

    def test_compute_layer_stats(self, tmp_path):
        from src.spatial.focus_area_manager import FocusAreaLayer
        from src.spatial.zonal_statistics import compute_layer_stats

        data = np.full((32, 32), 0, dtype=np.uint8)  # all water
        raster_path = tmp_path / "water.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        features = [
            {"NAME": "Z1", "geometry": box(0, 0, 160, 160)},
            {"NAME": "Z2", "geometry": box(160, 160, 320, 320)},
        ]
        boundary_path = tmp_path / "zones.geojson"
        _create_test_boundaries(boundary_path, features)

        layer = FocusAreaLayer(name="zone", path=boundary_path, id_field="NAME")
        layer.load()

        csv_path = tmp_path / "stats.csv"
        stats = compute_layer_stats(raster_path, layer, output_path=csv_path)

        assert "Z1" in stats
        assert "Z2" in stats
        assert stats["Z1"]["dominant_class"] == "water"
        assert csv_path.exists()

    def test_compute_raster_stats(self, tmp_path):
        from src.spatial.zonal_statistics import compute_raster_stats

        data = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.uint8)
        raster_path = tmp_path / "multi.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 30, 20))

        stats = compute_raster_stats(raster_path, resolution=10.0)
        assert stats["valid_pixels"] == 6
        assert "trees" in stats["classes"]
        assert "shrub" in stats["classes"]
        assert "grass" in stats["classes"]

    def test_area_calculations(self, tmp_path):
        from src.spatial.zonal_statistics import compute_zonal_stats

        data = np.full((10, 10), 4, dtype=np.uint8)  # 100 pixels of crops
        raster_path = tmp_path / "crops.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 100, 100))

        geom = box(0, 0, 100, 100)
        stats = compute_zonal_stats(raster_path, geom, resolution=10.0)

        assert stats["classes"]["crops"]["pixel_count"] == 100
        assert stats["classes"]["crops"]["area_sq_m"] == 10000.0  # 100 * 10*10
        assert stats["classes"]["crops"]["area_hectares"] == 1.0
