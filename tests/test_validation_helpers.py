"""Tests for input validation helpers."""

from pathlib import Path

import pytest


class TestFileValidation:
    """Test file existence validators."""

    def test_validate_file_exists_found(self, tmp_path):
        from src.utils.validation_helpers import validate_file_exists

        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert validate_file_exists(f) == []

    def test_validate_file_exists_missing(self, tmp_path):
        from src.utils.validation_helpers import validate_file_exists

        issues = validate_file_exists(tmp_path / "missing.txt")
        assert len(issues) == 1
        assert "not found" in issues[0]

    def test_validate_file_exists_directory(self, tmp_path):
        from src.utils.validation_helpers import validate_file_exists

        issues = validate_file_exists(tmp_path)
        assert len(issues) == 1
        assert "not a file" in issues[0]

    def test_validate_file_custom_label(self, tmp_path):
        from src.utils.validation_helpers import validate_file_exists

        issues = validate_file_exists(tmp_path / "x.tif", label="Prediction raster")
        assert "Prediction raster" in issues[0]


class TestRasterValidation:
    """Test raster path validation."""

    def test_valid_tif(self, tmp_path):
        from src.utils.validation_helpers import validate_raster_path

        f = tmp_path / "output.tif"
        f.write_text("fake")
        assert validate_raster_path(f) == []

    def test_valid_tiff(self, tmp_path):
        from src.utils.validation_helpers import validate_raster_path

        f = tmp_path / "output.tiff"
        f.write_text("fake")
        assert validate_raster_path(f) == []

    def test_wrong_extension(self, tmp_path):
        from src.utils.validation_helpers import validate_raster_path

        f = tmp_path / "output.png"
        f.write_text("fake")
        issues = validate_raster_path(f)
        assert len(issues) == 1
        assert ".png" in issues[0]

    def test_missing_raster(self, tmp_path):
        from src.utils.validation_helpers import validate_raster_path

        issues = validate_raster_path(tmp_path / "missing.tif")
        assert len(issues) == 1
        assert "not found" in issues[0]


class TestVectorValidation:
    """Test vector path validation."""

    def test_valid_geojson(self, tmp_path):
        from src.utils.validation_helpers import validate_vector_path

        f = tmp_path / "points.geojson"
        f.write_text("{}")
        assert validate_vector_path(f) == []

    def test_valid_gpkg(self, tmp_path):
        from src.utils.validation_helpers import validate_vector_path

        f = tmp_path / "data.gpkg"
        f.write_text("fake")
        assert validate_vector_path(f) == []

    def test_valid_shp(self, tmp_path):
        from src.utils.validation_helpers import validate_vector_path

        f = tmp_path / "zones.shp"
        f.write_text("fake")
        assert validate_vector_path(f) == []

    def test_wrong_extension(self, tmp_path):
        from src.utils.validation_helpers import validate_vector_path

        f = tmp_path / "data.xyz"
        f.write_text("fake")
        issues = validate_vector_path(f)
        assert len(issues) == 1
        assert ".xyz" in issues[0]


class TestModelValidation:
    """Test model name validation."""

    def test_valid_models(self):
        from src.utils.validation_helpers import validate_model_name

        assert validate_model_name("prithvi") == []
        assert validate_model_name("satlas") == []
        assert validate_model_name("ssl4eo") == []

    def test_invalid_model(self):
        from src.utils.validation_helpers import validate_model_name

        issues = validate_model_name("resnet50")
        assert len(issues) == 1
        assert "resnet50" in issues[0]
        assert "prithvi" in issues[0]


class TestClassifierValidation:
    """Test classifier method validation."""

    def test_valid_methods(self):
        from src.utils.validation_helpers import validate_classifier_method

        for method in ["xgboost", "random_forest", "mlp", "linear"]:
            assert validate_classifier_method(method) == []

    def test_invalid_method(self):
        from src.utils.validation_helpers import validate_classifier_method

        issues = validate_classifier_method("svm")
        assert len(issues) == 1
        assert "svm" in issues[0]


class TestEnsembleValidation:
    """Test ensemble strategy validation."""

    def test_valid_strategies(self):
        from src.utils.validation_helpers import validate_ensemble_strategy

        for s in ["majority_vote", "weighted_vote", "probability_average"]:
            assert validate_ensemble_strategy(s) == []

    def test_invalid_strategy(self):
        from src.utils.validation_helpers import validate_ensemble_strategy

        issues = validate_ensemble_strategy("stacking")
        assert len(issues) == 1
        assert "stacking" in issues[0]


class TestFusionValidation:
    """Test fusion strategy validation."""

    def test_valid_strategies(self):
        from src.utils.validation_helpers import validate_fusion_strategy

        assert validate_fusion_strategy("high_res_priority") == []
        assert validate_fusion_strategy("confidence_weighted") == []

    def test_invalid_strategy(self):
        from src.utils.validation_helpers import validate_fusion_strategy

        issues = validate_fusion_strategy("average")
        assert len(issues) == 1


class TestWeightsValidation:
    """Test ensemble weight validation."""

    def test_valid_weights(self):
        from src.utils.validation_helpers import validate_weights

        assert validate_weights([0.5, 0.3, 0.2], 3) == []

    def test_wrong_count(self):
        from src.utils.validation_helpers import validate_weights

        issues = validate_weights([0.5, 0.5], 3)
        assert len(issues) == 1
        assert "does not match" in issues[0]

    def test_negative_weights(self):
        from src.utils.validation_helpers import validate_weights

        issues = validate_weights([0.5, -0.1], 2)
        assert any("non-negative" in i for i in issues)

    def test_zero_sum(self):
        from src.utils.validation_helpers import validate_weights

        issues = validate_weights([0.0, 0.0], 2)
        assert any("positive" in i for i in issues)


class TestConfigValidation:
    """Test config section validation."""

    def test_valid_section(self):
        from src.utils.validation_helpers import validate_config_section

        config = {"processing": {"device": "cuda"}}
        assert validate_config_section(config, "processing") == []

    def test_missing_section(self):
        from src.utils.validation_helpers import validate_config_section

        config = {"processing": {"device": "cuda"}}
        issues = validate_config_section(config, "spatial")
        assert len(issues) == 1
        assert "spatial" in issues[0]

    def test_none_config(self):
        from src.utils.validation_helpers import validate_config_section

        issues = validate_config_section(None, "any")
        assert len(issues) == 1


class TestFormatIssues:
    """Test issue formatting."""

    def test_empty_issues(self):
        from src.utils.validation_helpers import format_issues

        assert format_issues([]) == ""

    def test_single_issue(self):
        from src.utils.validation_helpers import format_issues

        result = format_issues(["File not found"])
        assert result == "Error: File not found"

    def test_multiple_issues(self):
        from src.utils.validation_helpers import format_issues

        result = format_issues(["Issue 1", "Issue 2"])
        assert "Errors (2)" in result
        assert "Issue 1" in result
        assert "Issue 2" in result

    def test_custom_prefix(self):
        from src.utils.validation_helpers import format_issues

        result = format_issues(["bad input"], prefix="Warning")
        assert result == "Warning: bad input"


class TestCLIVersion:
    """Test CLI version flag."""

    def test_version_constant(self):
        from src.pipeline import __version__

        assert __version__ == "0.1.0"

    def test_validate_helpers_in_pipeline(self):
        """Check that pipeline has validation helper functions."""
        from src.pipeline import _validate_raster, _validate_vector, _validate_model_name

        # These are thin wrappers - just verify they exist and are callable
        assert callable(_validate_raster)
        assert callable(_validate_vector)
        assert callable(_validate_model_name)
