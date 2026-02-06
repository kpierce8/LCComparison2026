"""Tests for label management, custom import, and class mapping."""

import json
import pytest
from pathlib import Path

import numpy as np

from src.config_schema import CLASS_SCHEMA
from src.data.label_generator import (
    DYNAMIC_WORLD_MAPPING,
    ESA_WORLDCOVER_MAPPING,
)

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False


@pytest.fixture
def sample_geojson(tmp_path):
    """Create a sample GeoJSON label file."""
    if not _GEO_AVAILABLE:
        pytest.skip("geopandas not available")

    points = [
        {"geometry": Point(-122.0, 47.5), "land_cover": "Water", "confidence": 0.9},
        {"geometry": Point(-122.1, 47.5), "land_cover": "Forest", "confidence": 0.8},
        {"geometry": Point(-122.2, 47.5), "land_cover": "Urban", "confidence": 0.7},
        {"geometry": Point(-122.3, 47.5), "land_cover": "Agriculture", "confidence": 0.85},
        {"geometry": Point(-122.4, 47.5), "land_cover": "Grassland", "confidence": 0.75},
        {"geometry": Point(-122.5, 47.5), "land_cover": "Shrubland", "confidence": 0.8},
        {"geometry": Point(-122.6, 47.5), "land_cover": "Bareland", "confidence": 0.9},
    ]
    gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
    path = tmp_path / "test_labels.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV label file."""
    if not _GEO_AVAILABLE:
        pytest.skip("geopandas not available")

    data = {
        "longitude": [-122.0, -122.1, -122.2, -122.3],
        "latitude": [47.5, 47.5, 47.5, 47.5],
        "land_cover": ["water", "trees", "built", "grass"],
    }
    path = tmp_path / "test_labels.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame with labels."""
    if not _GEO_AVAILABLE:
        pytest.skip("geopandas not available")

    records = []
    classes = list(CLASS_SCHEMA.keys())
    for i, cls in enumerate(classes):
        for j in range(25):  # 25 points per class
            records.append({
                "geometry": Point(-122.0 + i * 0.1, 47.5 + j * 0.01),
                "lc_class": CLASS_SCHEMA[cls],
                "lc_name": cls,
                "confidence": 0.8,
                "source": "test",
            })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


class TestDynamicWorldMapping:
    def test_all_dw_classes_mapped(self):
        for dw_class in range(9):
            assert dw_class in DYNAMIC_WORLD_MAPPING

    def test_targets_valid(self):
        valid = set(CLASS_SCHEMA.values())
        for target in DYNAMIC_WORLD_MAPPING.values():
            assert target in valid

    def test_water_maps_to_water(self):
        assert DYNAMIC_WORLD_MAPPING[0] == CLASS_SCHEMA["water"]

    def test_trees_maps_to_trees(self):
        assert DYNAMIC_WORLD_MAPPING[1] == CLASS_SCHEMA["trees"]

    def test_built_maps_to_built(self):
        assert DYNAMIC_WORLD_MAPPING[6] == CLASS_SCHEMA["built"]

    def test_bare_maps_to_bare(self):
        assert DYNAMIC_WORLD_MAPPING[7] == CLASS_SCHEMA["bare"]


class TestESAWorldCoverMapping:
    def test_key_mappings(self):
        assert ESA_WORLDCOVER_MAPPING[10] == CLASS_SCHEMA["trees"]
        assert ESA_WORLDCOVER_MAPPING[20] == CLASS_SCHEMA["shrub"]
        assert ESA_WORLDCOVER_MAPPING[30] == CLASS_SCHEMA["grass"]
        assert ESA_WORLDCOVER_MAPPING[40] == CLASS_SCHEMA["crops"]
        assert ESA_WORLDCOVER_MAPPING[50] == CLASS_SCHEMA["built"]
        assert ESA_WORLDCOVER_MAPPING[60] == CLASS_SCHEMA["bare"]
        assert ESA_WORLDCOVER_MAPPING[80] == CLASS_SCHEMA["water"]

    def test_targets_valid(self):
        valid = set(CLASS_SCHEMA.values())
        for target in ESA_WORLDCOVER_MAPPING.values():
            assert target in valid


@pytest.mark.skipif(not _GEO_AVAILABLE, reason="geopandas not available")
class TestCustomLabelImporter:
    def test_import_geojson(self, sample_geojson):
        from src.data.custom_labels import CustomLabelImporter

        importer = CustomLabelImporter()
        result = importer.import_labels(sample_geojson, class_field="land_cover")

        assert len(result["points"]) == 7
        assert "lc_class" in result["points"].columns
        assert "lc_name" in result["points"].columns
        assert result["summary"]["total_points"] == 7

    def test_import_csv(self, sample_csv):
        from src.data.custom_labels import CustomLabelImporter

        importer = CustomLabelImporter()
        result = importer.import_labels(sample_csv, class_field="land_cover")

        assert len(result["points"]) == 4
        assert all(result["points"]["lc_class"] >= 0)

    def test_class_alias_mapping(self, sample_geojson):
        from src.data.custom_labels import CustomLabelImporter

        importer = CustomLabelImporter()
        result = importer.import_labels(sample_geojson, class_field="land_cover")
        gdf = result["points"]

        # Check aliases resolved correctly
        classes = gdf["lc_name"].unique()
        assert "water" in classes
        assert "trees" in classes
        assert "built" in classes
        assert "crops" in classes

    def test_custom_class_mapping(self, tmp_path):
        from src.data.custom_labels import CustomLabelImporter

        points = [
            {"geometry": Point(-122.0, 47.5), "class": "Wetland"},
            {"geometry": Point(-122.1, 47.5), "class": "Impervious"},
        ]
        gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
        path = tmp_path / "custom.geojson"
        gdf.to_file(path, driver="GeoJSON")

        importer = CustomLabelImporter()
        result = importer.import_labels(
            path,
            class_field="class",
            class_mapping={"Wetland": "water", "Impervious": "built"},
        )
        assert len(result["points"]) == 2
        assert set(result["points"]["lc_name"]) == {"water", "built"}

    def test_validation_within_aoi(self, sample_geojson):
        from src.data.custom_labels import CustomLabelImporter

        # AOI that excludes some points
        importer = CustomLabelImporter(
            aoi_bbox={"west": -122.3, "south": 47.0, "east": -121.8, "north": 48.0}
        )
        result = importer.import_labels(sample_geojson, class_field="land_cover")
        # Points at -122.4, -122.5, -122.6 are outside
        assert any("outside the AOI" in issue for issue in result["issues"])

    def test_validation_min_points(self, tmp_path):
        from src.data.custom_labels import CustomLabelImporter

        # Only 5 points of one class
        points = [
            {"geometry": Point(-122.0 + i * 0.01, 47.5), "lc": "water"}
            for i in range(5)
        ]
        gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
        path = tmp_path / "few.geojson"
        gdf.to_file(path, driver="GeoJSON")

        importer = CustomLabelImporter(min_points_per_class=20)
        result = importer.import_labels(path, class_field="lc")
        assert any("only 5 points" in issue for issue in result["issues"])

    def test_missing_class_field_raises(self, sample_geojson):
        from src.data.custom_labels import CustomLabelImporter

        importer = CustomLabelImporter()
        with pytest.raises(ValueError, match="Class field"):
            importer.import_labels(sample_geojson, class_field="nonexistent")

    def test_file_not_found_raises(self):
        from src.data.custom_labels import CustomLabelImporter

        importer = CustomLabelImporter()
        with pytest.raises(FileNotFoundError):
            importer.import_labels("/nonexistent/file.geojson")

    def test_spatial_buffer(self, sample_geojson):
        from src.data.custom_labels import CustomLabelImporter

        importer = CustomLabelImporter()
        result = importer.import_labels(sample_geojson, class_field="land_cover")
        buffered = importer.apply_spatial_buffer(result["points"], buffer_m=10.0)
        # Buffered geometries should be polygons, not points
        assert all(g.geom_type == "Polygon" for g in buffered.geometry)


@pytest.mark.skipif(not _GEO_AVAILABLE, reason="geopandas not available")
class TestLabelManager:
    def test_add_and_get_combined(self, sample_gdf, tmp_path):
        from src.data.label_manager import LabelManager

        manager = LabelManager(labels_dir=tmp_path / "labels")
        count = manager.add_labels(sample_gdf, "test_source")
        assert count == len(sample_gdf)

        combined = manager.get_combined()
        assert len(combined) == len(sample_gdf)

    def test_train_val_split(self, sample_gdf, tmp_path):
        from src.data.label_manager import LabelManager

        manager = LabelManager(labels_dir=tmp_path / "labels")
        manager.add_labels(sample_gdf, "test_source")

        train, val = manager.get_train_val_split(validation_split=0.2)
        assert len(train) + len(val) == len(sample_gdf)
        assert len(val) > 0
        assert len(train) > len(val)

    def test_stratified_split(self, sample_gdf, tmp_path):
        from src.data.label_manager import LabelManager

        manager = LabelManager(labels_dir=tmp_path / "labels")
        manager.add_labels(sample_gdf, "test")

        train, val = manager.get_train_val_split(validation_split=0.2, stratified=True)
        # Each class should be represented in both sets
        train_classes = set(train["lc_name"].unique())
        val_classes = set(val["lc_name"].unique())
        assert train_classes == val_classes

    def test_get_summary(self, sample_gdf, tmp_path):
        from src.data.label_manager import LabelManager

        manager = LabelManager(labels_dir=tmp_path / "labels")
        manager.add_labels(sample_gdf, "test_source")

        summary = manager.get_summary()
        assert summary["total_points"] == len(sample_gdf)
        # Source names may be truncated after GeoJSON roundtrip; check prefix
        assert any("test" in s for s in summary["sources"])
        assert len(summary["per_class"]) == 7  # all 7 classes

    def test_save_splits(self, sample_gdf, tmp_path):
        from src.data.label_manager import LabelManager

        manager = LabelManager(labels_dir=tmp_path / "labels")
        manager.add_labels(sample_gdf, "test")

        train, val = manager.get_train_val_split()
        paths = manager.save_splits(train, val, output_dir=tmp_path / "splits")

        assert Path(paths["train"]).exists()
        assert Path(paths["val"]).exists()
        assert Path(paths["summary"]).exists()

        with open(paths["summary"]) as f:
            summary = json.load(f)
        assert summary["train_count"] > 0
        assert summary["val_count"] > 0

    def test_persistence(self, sample_gdf, tmp_path):
        from src.data.label_manager import LabelManager

        labels_dir = tmp_path / "labels"
        manager1 = LabelManager(labels_dir=labels_dir)
        manager1.add_labels(sample_gdf, "test")

        # New manager should load from disk
        manager2 = LabelManager(labels_dir=labels_dir)
        combined = manager2.get_combined()
        assert combined is not None
        assert len(combined) == len(sample_gdf)
