"""Tests for accuracy assessment and model comparison."""

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_bounds
    from shapely.geometry import Point, box
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


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
    """Create reference points GeoDataFrame and save.

    points_data: list of (x, y, class_label) tuples.
    """
    geometry = [Point(x, y) for x, y, _ in points_data]
    classes = [c for _, _, c in points_data]
    gdf = gpd.GeoDataFrame(
        {"LC_CLASS": classes, "geometry": geometry},
        crs=crs,
    )
    gdf.to_file(path)


@pytest.mark.skipif(
    not (_GEO_AVAILABLE and _SKLEARN_AVAILABLE),
    reason="geopandas, rasterio, sklearn required",
)
class TestAccuracyAssessor:
    """Test accuracy assessment."""

    def test_assess_perfect(self, tmp_path):
        from src.validation.accuracy_assessor import AccuracyAssessor

        # Create a 32x32 raster: uniform class=1 (trees)
        data = np.full((32, 32), 1, dtype=np.uint8)
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        # All reference points say "trees" - should be perfect
        ref_points = [
            (50, 50, "trees"),
            (100, 100, "trees"),
            (150, 150, "trees"),
            (200, 200, "trees"),
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        assessor = AccuracyAssessor()
        results = assessor.assess(
            raster_path, ref_path, "LC_CLASS",
            output_dir=tmp_path / "output",
        )

        assert results["overall_accuracy"] == 1.0
        assert results["n_points"] == 4
        assert (tmp_path / "output" / "accuracy_metrics.json").exists()
        assert (tmp_path / "output" / "confusion_matrix.csv").exists()

    def test_assess_imperfect(self, tmp_path):
        from src.validation.accuracy_assessor import AccuracyAssessor

        # All pixels = trees(1)
        data = np.full((32, 32), 1, dtype=np.uint8)
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        # Some points are correct, some wrong
        ref_points = [
            (50, 50, "trees"),    # correct
            (100, 100, "trees"),  # correct
            (150, 150, "water"),  # wrong: predicted trees, actual water
            (200, 200, "water"),  # wrong
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        assessor = AccuracyAssessor()
        results = assessor.assess(raster_path, ref_path, "LC_CLASS")

        assert results["overall_accuracy"] == 0.5
        assert results["n_points"] == 4

    def test_assess_with_integer_labels(self, tmp_path):
        from src.validation.accuracy_assessor import AccuracyAssessor

        data = np.full((32, 32), 3, dtype=np.uint8)  # all grass
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        # Reference with integer class indices
        geometry = [Point(50, 50), Point(100, 100)]
        gdf = gpd.GeoDataFrame(
            {"LC_CLASS": [3, 3], "geometry": geometry},
            crs="EPSG:32610",
        )
        ref_path = tmp_path / "ref_int.geojson"
        gdf.to_file(ref_path)

        assessor = AccuracyAssessor()
        results = assessor.assess(raster_path, ref_path, "LC_CLASS")
        assert results["overall_accuracy"] == 1.0

    def test_per_class_metrics(self, tmp_path):
        from src.validation.accuracy_assessor import AccuracyAssessor

        data = np.zeros((32, 32), dtype=np.uint8)
        data[:, :16] = 0  # water
        data[:, 16:] = 1  # trees
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        ref_points = [
            (50, 50, "water"), (80, 50, "water"),
            (200, 50, "trees"), (250, 50, "trees"),
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        assessor = AccuracyAssessor()
        results = assessor.assess(raster_path, ref_path, "LC_CLASS")

        assert "per_class" in results
        assert "water" in results["per_class"]
        assert "trees" in results["per_class"]

    def test_sample_size_warnings(self, tmp_path):
        from src.validation.accuracy_assessor import AccuracyAssessor

        data = np.full((32, 32), 0, dtype=np.uint8)
        raster_path = tmp_path / "pred.tif"
        _create_test_raster(raster_path, data, bbox=(0, 0, 320, 320))

        ref_points = [(50, 50, "water")]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        assessor = AccuracyAssessor(min_samples_per_class=30)
        results = assessor.assess(raster_path, ref_path, "LC_CLASS")

        assert len(results["warnings"]) > 0
        assert "water" in results["warnings"][0]

    def test_compare_models(self, tmp_path):
        from src.validation.accuracy_assessor import AccuracyAssessor

        # Model A: all trees
        data_a = np.full((32, 32), 1, dtype=np.uint8)
        path_a = tmp_path / "model_a.tif"
        _create_test_raster(path_a, data_a, bbox=(0, 0, 320, 320))

        # Model B: all water
        data_b = np.full((32, 32), 0, dtype=np.uint8)
        path_b = tmp_path / "model_b.tif"
        _create_test_raster(path_b, data_b, bbox=(0, 0, 320, 320))

        # Reference: half trees, half water
        ref_points = [
            (50, 50, "trees"), (100, 100, "trees"),
            (150, 150, "water"), (200, 200, "water"),
        ]
        ref_path = tmp_path / "ref.geojson"
        _create_reference_points(ref_path, ref_points)

        assessor = AccuracyAssessor()
        comparison = assessor.compare_models(
            {"model_a": path_a, "model_b": path_b},
            ref_path, "LC_CLASS",
            output_dir=tmp_path / "comp",
        )

        assert "models" in comparison
        assert "model_a" in comparison["models"]
        assert "model_b" in comparison["models"]
        assert "ranking" in comparison


@pytest.mark.skipif(
    not (_GEO_AVAILABLE and _SKLEARN_AVAILABLE),
    reason="geopandas, rasterio, sklearn required",
)
class TestComparisonMetrics:
    """Test spatial agreement and model comparison."""

    def test_spatial_agreement_identical(self, tmp_path):
        from src.validation.comparison_metrics import compute_spatial_agreement

        data = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data, bbox=(0, 0, 30, 20))
        _create_test_raster(path_b, data, bbox=(0, 0, 30, 20))

        result = compute_spatial_agreement(path_a, path_b)
        assert result["agreement_pct"] == 100.0
        assert result["valid_pixels"] == 6

    def test_spatial_agreement_partial(self, tmp_path):
        from src.validation.comparison_metrics import compute_spatial_agreement

        data_a = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
        data_b = np.array([[0, 1, 1, 1], [0, 0, 0, 1]], dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data_a, bbox=(0, 0, 40, 20))
        _create_test_raster(path_b, data_b, bbox=(0, 0, 40, 20))

        result = compute_spatial_agreement(path_a, path_b)
        assert 0 < result["agreement_pct"] < 100
        assert "per_class" in result
        assert "confusion_matrix" in result

    def test_create_agreement_map(self, tmp_path):
        from src.validation.comparison_metrics import create_agreement_map

        data_a = np.array([[1, 1], [2, 2]], dtype=np.uint8)
        data_b = np.array([[1, 2], [2, 1]], dtype=np.uint8)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        _create_test_raster(path_a, data_a, bbox=(0, 0, 20, 20))
        _create_test_raster(path_b, data_b, bbox=(0, 0, 20, 20))

        output = tmp_path / "agree.tif"
        result = create_agreement_map(path_a, path_b, output)

        assert output.exists()
        assert result["agreement_pct"] == 50.0

        with rasterio.open(output) as src:
            agree_data = src.read(1)
            # Diagonal: agree (1,1) and (2,2) -> 1; off: (1,2) and (2,1) -> 0
            assert agree_data[0, 0] == 1  # both predict 1
            assert agree_data[0, 1] == 0  # 1 vs 2
            assert agree_data[1, 0] == 1  # both predict 2
            assert agree_data[1, 1] == 0  # 2 vs 1

    def test_compare_with_existing(self, tmp_path):
        from src.validation.comparison_metrics import compare_with_existing

        data_a = np.full((16, 16), 3, dtype=np.uint8)
        data_b = np.full((16, 16), 3, dtype=np.uint8)
        data_b[:8, :] = 1  # Half different

        path_a = tmp_path / "new.tif"
        path_b = tmp_path / "existing.tif"
        _create_test_raster(path_a, data_a, bbox=(0, 0, 160, 160))
        _create_test_raster(path_b, data_b, bbox=(0, 0, 160, 160))

        report = compare_with_existing(
            path_a, path_b, tmp_path / "comp",
            model_name="prithvi", existing_name="segformer",
        )

        assert report["model_a"] == "prithvi"
        assert report["model_b"] == "segformer"
        assert 0 < report["agreement"]["agreement_pct"] < 100
        assert (tmp_path / "comp" / "agreement_map.tif").exists()
        assert (tmp_path / "comp" / "comparison_report.json").exists()
        assert (tmp_path / "comp" / "confusion_matrix.csv").exists()
