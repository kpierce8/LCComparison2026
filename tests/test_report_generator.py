"""Tests for report generation."""

from pathlib import Path

import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


@pytest.mark.skipif(not _MPL_AVAILABLE, reason="matplotlib required")
class TestPlotFunctions:
    """Test individual plot generation."""

    def test_confusion_matrix_plot_base64(self):
        from src.validation.report_generator import plot_confusion_matrix

        cm = [[8, 2], [1, 9]]
        labels = [0, 1]
        result = plot_confusion_matrix(cm, labels, "Test CM")

        assert result is not None
        assert len(result) > 100  # base64 string

    def test_confusion_matrix_plot_file(self, tmp_path):
        from src.validation.report_generator import plot_confusion_matrix

        cm = [[8, 2], [1, 9]]
        labels = [0, 1]
        output = tmp_path / "cm.png"
        result = plot_confusion_matrix(cm, labels, "Test CM", output_path=output)

        assert output.exists()
        assert result == "cm.png"

    def test_confusion_matrix_multiclass(self):
        from src.validation.report_generator import plot_confusion_matrix

        cm = [[7, 1, 2], [0, 8, 2], [1, 1, 8]]
        labels = [0, 1, 2]
        result = plot_confusion_matrix(cm, labels, "3-class CM")
        assert result is not None

    def test_per_class_accuracy_plot(self):
        from src.validation.report_generator import plot_per_class_accuracy

        per_class = {
            "water": {"producers_accuracy": 0.9, "users_accuracy": 0.85, "f1": 0.87},
            "trees": {"producers_accuracy": 0.8, "users_accuracy": 0.75, "f1": 0.77},
            "built": {"producers_accuracy": 0.7, "users_accuracy": 0.65, "f1": 0.67},
        }
        result = plot_per_class_accuracy(per_class, "Per-Class Test")
        assert result is not None
        assert len(result) > 100

    def test_per_class_accuracy_file(self, tmp_path):
        from src.validation.report_generator import plot_per_class_accuracy

        per_class = {
            "water": {"producers_accuracy": 0.9, "users_accuracy": 0.85, "f1": 0.87},
        }
        output = tmp_path / "pc.png"
        result = plot_per_class_accuracy(per_class, "Test", output_path=output)
        assert output.exists()

    def test_model_comparison_plot(self):
        from src.validation.report_generator import plot_model_comparison

        metrics = {
            "prithvi": {"overall_accuracy": 0.85, "kappa": 0.7, "f1_macro": 0.8},
            "satlas": {"overall_accuracy": 0.82, "kappa": 0.65, "f1_macro": 0.75},
            "segformer": {"overall_accuracy": 0.88, "kappa": 0.75, "f1_macro": 0.84},
        }
        result = plot_model_comparison(metrics, "Comparison Test")
        assert result is not None

    def test_error_density_plot(self):
        from src.validation.report_generator import plot_error_density

        rate_grid = [[0.5, 0.3, -1], [0.1, 0.0, 0.8], [0.2, 0.4, 0.6]]
        bounds = [0, 0, 100, 100]
        result = plot_error_density(rate_grid, bounds, "Error Density")
        assert result is not None


@pytest.mark.skipif(not _MPL_AVAILABLE, reason="matplotlib required")
class TestReportGeneration:
    """Test full HTML report generation."""

    def test_accuracy_report(self, tmp_path):
        from src.validation.report_generator import generate_accuracy_report

        results = {
            "overall_accuracy": 0.85,
            "kappa": 0.72,
            "f1_macro": 0.80,
            "f1_weighted": 0.83,
            "n_points": 100,
            "confusion_matrix": [[40, 5], [10, 45]],
            "labels": [0, 1],
            "per_class": {
                "water": {
                    "producers_accuracy": 0.89,
                    "users_accuracy": 0.80,
                    "f1": 0.84,
                    "support": 45,
                },
                "trees": {
                    "producers_accuracy": 0.82,
                    "users_accuracy": 0.90,
                    "f1": 0.86,
                    "support": 55,
                },
            },
            "warnings": ["water: only 5 samples (min recommended: 30)"],
        }

        output_path = tmp_path / "report.html"
        result = generate_accuracy_report(results, output_path, model_name="prithvi")

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert "prithvi" in content
        assert "0.85" in content  # overall accuracy
        assert "Confusion Matrix" in content
        assert "Per-Class" in content
        assert "WARNING" in content or "warning" in content.lower()

    def test_accuracy_report_with_error_analysis(self, tmp_path):
        from src.validation.report_generator import generate_accuracy_report

        results = {
            "overall_accuracy": 0.75,
            "kappa": 0.6,
            "f1_macro": 0.7,
            "f1_weighted": 0.72,
            "n_points": 50,
            "confusion_matrix": [[20, 5], [8, 17]],
            "labels": [0, 1],
            "per_class": {
                "water": {"producers_accuracy": 0.8, "users_accuracy": 0.71, "f1": 0.75, "support": 25},
                "trees": {"producers_accuracy": 0.68, "users_accuracy": 0.77, "f1": 0.72, "support": 25},
            },
            "warnings": [],
        }

        error_analysis = {
            "error_rate": 0.25,
            "n_errors": 13,
            "n_correct": 37,
            "top_confusions": [("water -> trees", 5), ("trees -> water", 8)],
        }

        error_density = {
            "rate_grid": [[0.5, 0.1], [0.3, 0.0]],
            "bounds": [0, 0, 100, 100],
        }

        output_path = tmp_path / "err_report.html"
        result = generate_accuracy_report(
            results, output_path, model_name="test",
            error_analysis=error_analysis, error_density=error_density,
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "Top Confusion Pairs" in content
        assert "Spatial Error" in content

    def test_comparison_report(self, tmp_path):
        from src.validation.report_generator import generate_comparison_report

        comparison = {
            "models": {
                "prithvi": {"overall_accuracy": 0.85, "kappa": 0.72, "f1_macro": 0.80, "n_points": 100},
                "satlas": {"overall_accuracy": 0.82, "kappa": 0.65, "f1_macro": 0.75, "n_points": 100},
            },
            "ranking": [
                {"model": "prithvi", "overall_accuracy": 0.85},
                {"model": "satlas", "overall_accuracy": 0.82},
            ],
            "best_model": "prithvi",
        }

        output_path = tmp_path / "comparison.html"
        result = generate_comparison_report(comparison, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "prithvi" in content
        assert "satlas" in content
        assert "Ranking" in content

    def test_comparison_report_with_per_model(self, tmp_path):
        from src.validation.report_generator import generate_comparison_report

        comparison = {
            "models": {
                "model_a": {"overall_accuracy": 0.80, "kappa": 0.6, "f1_macro": 0.75, "n_points": 50},
            },
            "ranking": [{"model": "model_a", "overall_accuracy": 0.80}],
            "best_model": "model_a",
        }

        per_model = {
            "model_a": {
                "confusion_matrix": [[10, 2], [3, 10]],
                "labels": [0, 1],
            },
        }

        output_path = tmp_path / "comp_detail.html"
        generate_comparison_report(comparison, output_path, per_model_results=per_model)

        assert output_path.exists()
        content = output_path.read_text()
        assert "model_a" in content

    def test_report_creates_parent_dirs(self, tmp_path):
        from src.validation.report_generator import generate_accuracy_report

        results = {
            "overall_accuracy": 0.9,
            "kappa": 0.8,
            "f1_macro": 0.85,
            "f1_weighted": 0.87,
            "n_points": 10,
            "per_class": {},
            "warnings": [],
        }

        output_path = tmp_path / "nested" / "deep" / "report.html"
        generate_accuracy_report(results, output_path)
        assert output_path.exists()
