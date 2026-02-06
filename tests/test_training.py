"""Tests for training pipeline and validation."""

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from sklearn.metrics import accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import xgboost
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def _make_data(n_samples=200, n_features=64, n_classes=7, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, n_samples)
    return X, y


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not available")
class TestValidation:
    """Test validation metrics and cross-validation."""

    def test_compute_metrics(self):
        from src.training.validation import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1])
        metrics = compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "iou_mean" in metrics
        assert "kappa" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["accuracy"] == pytest.approx(0.75)

    def test_compute_metrics_with_names(self):
        from src.training.validation import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])

        class_names = ["water", "trees", "shrub"]
        metrics = compute_metrics(y_true, y_pred, class_names)

        assert "per_class" in metrics
        assert "water" in metrics["per_class"]
        assert metrics["per_class"]["water"]["f1"] == 1.0

    def test_compute_metrics_perfect(self):
        from src.training.validation import compute_metrics

        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["kappa"] == 1.0
        assert metrics["iou_mean"] == 1.0

    def test_iou_per_class(self):
        from src.training.validation import compute_metrics

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])
        metrics = compute_metrics(y_true, y_pred)

        assert len(metrics["iou_per_class"]) == 3
        # Class 0: TP=2, FP=1, FN=0 -> IoU=2/3
        assert metrics["iou_per_class"][0] == pytest.approx(2 / 3, abs=1e-5)
        # Class 2: perfect -> IoU=1.0
        assert metrics["iou_per_class"][2] == pytest.approx(1.0)


@pytest.mark.skipif(
    not (_SKLEARN_AVAILABLE and _XGB_AVAILABLE),
    reason="sklearn and xgboost required",
)
class TestCrossValidation:
    """Test cross-validation functionality."""

    def test_stratified_cv(self):
        from src.training.validation import cross_validate
        from src.classification.classifier import LandCoverClassifier

        X, y = _make_data(300, n_classes=4)

        def factory():
            return LandCoverClassifier(
                method="xgboost", n_classes=4,
                config={"n_estimators": 10, "max_depth": 3},
            )

        results = cross_validate(factory, X, y, n_folds=3, stratified=True)

        assert "fold_metrics" in results
        assert len(results["fold_metrics"]) == 3
        assert "summary" in results
        assert results["summary"]["n_folds"] == 3
        assert results["summary"]["accuracy_mean"] > 0

    def test_spatial_cv(self):
        from src.training.validation import cross_validate
        from src.classification.classifier import LandCoverClassifier

        X, y = _make_data(200, n_classes=3)
        coords = np.random.randn(200, 2).astype(np.float32) * 10000

        def factory():
            return LandCoverClassifier(
                method="xgboost", n_classes=3,
                config={"n_estimators": 10, "max_depth": 3},
            )

        results = cross_validate(
            factory, X, y, n_folds=3,
            spatial_coords=coords, buffer_m=100.0,
        )

        assert len(results["fold_metrics"]) == 3
        assert results["summary"]["accuracy_mean"] > 0


@pytest.mark.skipif(
    not (_SKLEARN_AVAILABLE and _XGB_AVAILABLE),
    reason="sklearn and xgboost required",
)
class TestTrainer:
    """Test end-to-end training pipeline."""

    def test_prepare_training_data(self, tmp_path):
        from src.training.trainer import Trainer

        embeddings = {
            "tile_0": np.random.randn(64).astype(np.float32),
            "tile_1": np.random.randn(64).astype(np.float32),
            "tile_2": np.random.randn(64).astype(np.float32),
        }
        labels = {"tile_0": 0, "tile_1": 3, "tile_2": 5}

        trainer = Trainer(output_dir=tmp_path)
        X, y, ids = trainer.prepare_training_data(embeddings, labels)

        assert X.shape == (3, 64)
        assert y.shape == (3,)
        assert len(ids) == 3

    def test_prepare_no_match(self, tmp_path):
        from src.training.trainer import Trainer

        embeddings = {"a": np.zeros(10)}
        labels = {"b": 0}

        trainer = Trainer(output_dir=tmp_path)
        with pytest.raises(ValueError, match="No matching"):
            trainer.prepare_training_data(embeddings, labels)

    def test_train(self, tmp_path):
        from src.training.trainer import Trainer

        X, y = _make_data(100, n_features=32, n_classes=4)
        trainer = Trainer(
            method="xgboost", n_classes=4,
            classifier_config={"n_estimators": 10, "max_depth": 3},
            output_dir=tmp_path,
        )

        metrics = trainer.train(X, y, model_name="test")
        assert "train_accuracy" in metrics
        assert trainer.classifier is not None
        assert trainer.classifier.is_trained

        # Check saved files
        assert (tmp_path / "test_classifier.pkl").exists()
        assert (tmp_path / "test_metrics.json").exists()

    def test_train_with_val(self, tmp_path):
        from src.training.trainer import Trainer

        X, y = _make_data(200, n_features=32, n_classes=4)
        X_val, y_val = _make_data(50, n_features=32, n_classes=4, seed=99)

        trainer = Trainer(
            method="xgboost", n_classes=4,
            classifier_config={"n_estimators": 10},
            output_dir=tmp_path,
        )

        metrics = trainer.train(X, y, X_val, y_val, model_name="test_val")
        assert "val_accuracy" in metrics

    def test_cross_validate(self, tmp_path):
        from src.training.trainer import Trainer

        X, y = _make_data(200, n_features=32, n_classes=4)
        trainer = Trainer(
            method="xgboost", n_classes=4,
            classifier_config={"n_estimators": 10, "max_depth": 3},
            output_dir=tmp_path,
        )

        results = trainer.cross_validate(X, y, n_folds=3)
        assert results["summary"]["n_folds"] == 3
        assert (tmp_path / "cv_results.json").exists()

    def test_predict_tiles(self, tmp_path):
        from src.training.trainer import Trainer

        X, y = _make_data(100, n_features=32, n_classes=4)
        trainer = Trainer(
            method="xgboost", n_classes=4,
            classifier_config={"n_estimators": 10},
            output_dir=tmp_path,
        )
        trainer.train(X, y, model_name="pred_test")

        embeddings = {f"tile_{i}": np.random.randn(32).astype(np.float32) for i in range(10)}
        result = trainer.predict_tiles(embeddings, output_dir=tmp_path / "preds")

        assert result["summary"]["total_tiles"] == 10
        assert len(result["predictions"]) == 10
        assert "class_distribution" in result["summary"]
        assert (tmp_path / "preds" / "predictions.npz").exists()

    def test_load_classifier(self, tmp_path):
        from src.training.trainer import Trainer

        X, y = _make_data(100, n_features=32, n_classes=3)
        trainer1 = Trainer(
            method="xgboost", n_classes=3,
            classifier_config={"n_estimators": 10},
            output_dir=tmp_path,
        )
        trainer1.train(X, y, model_name="load_test")

        trainer2 = Trainer(output_dir=tmp_path)
        trainer2.load_classifier(tmp_path / "load_test_classifier")
        assert trainer2.classifier is not None
        assert trainer2.classifier.is_trained

    def test_get_results(self, tmp_path):
        from src.training.trainer import Trainer

        trainer = Trainer(output_dir=tmp_path)
        results = trainer.get_results()
        assert "train_metrics" in results
        assert "cv_results" in results
