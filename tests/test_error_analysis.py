"""Tests for error analysis module."""

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_bounds
    from shapely.geometry import Point
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False


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


def _create_reference_points(path, points_data, crs="EPSG:32610"):
    geometry = [Point(x, y) for x, y, _ in points_data]
    classes = [c for _, _, c in points_data]
    gdf = gpd.GeoDataFrame(
        {"LC_CLASS": classes, "geometry": geometry},
        crs=crs,
    )
    gdf.to_file(path)


@pytest.mark.skipif(not _GEO_AVAILABLE, reason="geopandas/rasterio required")
class TestErrorMap:
    """Test error map generation."""

    def test_perfect_predictions(self, tmp_path):
        from src.validation.error_analysis import compute_error_map

        data = np.full((32, 32), 1, dtype=np.uint8)  # all trees
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data)

        ref_points = [
            (50, 50, "trees"), (100, 100, "trees"),
            (200, 200, "trees"), (250, 250, "trees"),
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        result = compute_error_map(raster_path, ref_path, "LC_CLASS")

        assert result["error_rate"] == 0
        assert result["n_correct"] == 4
        assert result["n_errors"] == 0
        assert result["n_points"] == 4

    def test_mixed_errors(self, tmp_path):
        from src.validation.error_analysis import compute_error_map

        data = np.full((32, 32), 1, dtype=np.uint8)  # all trees
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data)

        # 2 correct (trees), 2 wrong (water, shrub)
        ref_points = [
            (50, 50, "trees"), (100, 100, "trees"),
            (200, 200, "water"), (250, 250, "shrub"),
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        result = compute_error_map(raster_path, ref_path, "LC_CLASS")

        assert result["error_rate"] == 0.5
        assert result["n_errors"] == 2
        assert result["n_correct"] == 2
        assert len(result["top_confusions"]) > 0

    def test_confusion_pairs(self, tmp_path):
        from src.validation.error_analysis import compute_error_map

        data = np.full((32, 32), 0, dtype=np.uint8)  # all water
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data)

        # All errors: reference says trees, prediction is water
        ref_points = [
            (50, 50, "trees"), (100, 100, "trees"),
            (150, 150, "trees"), (200, 200, "trees"),
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        result = compute_error_map(raster_path, ref_path, "LC_CLASS")

        assert result["error_rate"] == 1.0
        # The top confusion should be trees -> water
        pair_key, count = result["top_confusions"][0]
        assert "trees" in pair_key
        assert "water" in pair_key
        assert count == 4

    def test_save_error_points(self, tmp_path):
        from src.validation.error_analysis import compute_error_map

        data = np.full((32, 32), 1, dtype=np.uint8)
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data)

        ref_points = [
            (50, 50, "trees"), (200, 200, "water"),
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        output_path = tmp_path / "errors.gpkg"
        result = compute_error_map(
            raster_path, ref_path, "LC_CLASS", output_path=output_path,
        )

        assert output_path.exists()

    def test_per_class_error_rates(self, tmp_path):
        from src.validation.error_analysis import compute_error_map

        data = np.full((32, 32), 1, dtype=np.uint8)  # all trees
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data)

        ref_points = [
            (50, 50, "trees"),      # correct
            (100, 100, "water"),    # error (water misclassified as trees)
            (150, 150, "water"),    # error
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        result = compute_error_map(raster_path, ref_path, "LC_CLASS")

        per_class = result["per_class_error_rates"]
        assert "water" in per_class
        assert per_class["water"] == 1.0  # 2/2 water points wrong


class TestClassConfusions:
    """Test confusion matrix analysis."""

    def test_analyze_perfect(self):
        from src.validation.error_analysis import analyze_class_confusions

        cm = [[10, 0], [0, 10]]
        labels = [0, 1]
        result = analyze_class_confusions(cm, labels)

        assert len(result["top_confusions"]) == 0
        assert result["per_class"]["water"]["omission_errors"] == 0
        assert result["per_class"]["water"]["commission_errors"] == 0

    def test_analyze_with_errors(self):
        from src.validation.error_analysis import analyze_class_confusions

        # water: 8 correct, 2 confused as trees
        # trees: 7 correct, 3 confused as water
        cm = [[8, 2], [3, 7]]
        labels = [0, 1]
        result = analyze_class_confusions(cm, labels)

        assert len(result["top_confusions"]) > 0
        # Most confused pair
        top = result["top_confusions"][0]
        assert top["count"] in [2, 3]

        # Omission/commission
        assert result["per_class"]["water"]["omission_errors"] == 2
        assert result["per_class"]["trees"]["commission_errors"] == 2

    def test_asymmetry(self):
        from src.validation.error_analysis import analyze_class_confusions

        # Very asymmetric: many water->trees errors but few trees->water
        cm = [[5, 10], [1, 20]]
        labels = [0, 1]
        result = analyze_class_confusions(cm, labels)

        assert len(result["asymmetry"]) > 0
        assert result["asymmetry"][0]["asymmetry_ratio"] > 1

    def test_three_classes(self):
        from src.validation.error_analysis import analyze_class_confusions

        cm = [[8, 1, 1], [2, 7, 1], [0, 2, 8]]
        labels = [0, 1, 2]
        result = analyze_class_confusions(cm, labels)

        assert "water" in result["per_class"]
        assert "trees" in result["per_class"]
        assert "shrub" in result["per_class"]


class TestSpatialErrorDensity:
    """Test spatial error density grid."""

    def test_uniform_errors(self):
        from src.validation.error_analysis import compute_spatial_error_density

        errors = [(10, 10), (90, 90)]
        correct = [(50, 50), (60, 60)]

        result = compute_spatial_error_density(
            errors, correct, grid_size=5, bounds=(0, 0, 100, 100),
        )

        assert result["grid_size"] == 5
        assert len(result["error_grid"]) == 5
        assert result["mean_error_rate"] >= 0

    def test_no_points(self):
        from src.validation.error_analysis import compute_spatial_error_density

        result = compute_spatial_error_density([], [], grid_size=5)
        assert result["n_cells"] == 0

    def test_hotspot_detection(self):
        from src.validation.error_analysis import compute_spatial_error_density

        # All errors in one corner, all correct in another
        errors = [(5, 5), (10, 10), (15, 15), (5, 10), (10, 5)]
        correct = [(90, 90), (85, 85), (80, 80), (90, 85), (85, 90)]

        result = compute_spatial_error_density(
            errors, correct, grid_size=4, bounds=(0, 0, 100, 100),
        )

        assert result["n_hotspots"] > 0
        # The hotspot should be in the low-coordinate area
        top_hotspot = result["hotspots"][0]
        assert top_hotspot["error_rate"] > 0.5

    def test_bounds_inference(self):
        from src.validation.error_analysis import compute_spatial_error_density

        errors = [(50, 50)]
        correct = [(100, 100)]

        result = compute_spatial_error_density(errors, correct, grid_size=3)
        assert result["bounds"][0] == 50
        assert result["bounds"][2] == 100


@pytest.mark.skipif(not _GEO_AVAILABLE, reason="geopandas/rasterio required")
class TestEdgeErrorRate:
    """Test edge vs interior error analysis."""

    def test_edge_analysis(self, tmp_path):
        from src.validation.error_analysis import compute_edge_error_rate

        # Raster with two classes side by side
        data = np.zeros((32, 32), dtype=np.uint8)
        data[:, 16:] = 1  # left=water, right=trees
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        # Points at edge (column ~16) and interior
        ref_points = [
            (50, 50, "water"),     # interior water - correct
            (250, 50, "trees"),    # interior trees - correct
            (160, 50, "trees"),    # near edge - may be water (error)
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        result = compute_edge_error_rate(raster_path, ref_path, "LC_CLASS")

        assert "edge_error_rate" in result
        assert "interior_error_rate" in result
        assert result["edge_points"] + result["interior_points"] > 0

    def test_uniform_raster_no_edges(self, tmp_path):
        from src.validation.error_analysis import compute_edge_error_rate

        # Uniform raster - no edges
        data = np.full((32, 32), 1, dtype=np.uint8)
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data)

        ref_points = [(50, 50, "trees"), (200, 200, "trees")]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        result = compute_edge_error_rate(raster_path, ref_path, "LC_CLASS")

        # All points should be interior (no edges in uniform raster)
        assert result["edge_points"] == 0
        assert result["interior_points"] == 2
