"""Tests for land cover classifier."""

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import xgboost
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from src.classification.classifier import LandCoverClassifier, SUPPORTED_METHODS


def _make_data(n_samples=200, n_features=64, n_classes=7, seed=42):
    """Create synthetic classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, n_samples)
    return X, y


class TestClassifierInit:
    """Test classifier initialization."""

    def test_supported_methods(self):
        assert "xgboost" in SUPPORTED_METHODS
        assert "random_forest" in SUPPORTED_METHODS
        assert "mlp" in SUPPORTED_METHODS
        assert "linear" in SUPPORTED_METHODS

    def test_init_default(self):
        clf = LandCoverClassifier()
        assert clf.method == "xgboost"
        assert clf.n_classes == 7
        assert not clf.is_trained

    def test_init_with_method(self):
        clf = LandCoverClassifier(method="random_forest", n_classes=5)
        assert clf.method == "random_forest"
        assert clf.n_classes == 5

    def test_init_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            LandCoverClassifier(method="nonexistent")


@pytest.mark.skipif(not _XGB_AVAILABLE, reason="xgboost not available")
class TestXGBoostClassifier:
    """Test XGBoost classifier."""

    def test_train(self):
        X, y = _make_data(n_classes=4)
        clf = LandCoverClassifier(method="xgboost", n_classes=4, config={
            "n_estimators": 10, "max_depth": 3,
        })
        metrics = clf.train(X, y)
        assert clf.is_trained
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0

    def test_train_with_validation(self):
        X, y = _make_data(300, n_classes=4)
        X_val, y_val = _make_data(100, n_classes=4, seed=99)
        clf = LandCoverClassifier(method="xgboost", n_classes=4, config={
            "n_estimators": 20, "max_depth": 3, "early_stopping_rounds": 5,
        })
        metrics = clf.train(X, y, X_val, y_val)
        assert "val_accuracy" in metrics

    def test_predict(self):
        X, y = _make_data(n_classes=3)
        clf = LandCoverClassifier(method="xgboost", n_classes=3, config={
            "n_estimators": 10, "max_depth": 3,
        })
        clf.train(X, y)
        preds = clf.predict(X[:10])
        assert preds.shape == (10,)
        assert all(p in [0, 1, 2] for p in preds)

    def test_predict_proba(self):
        X, y = _make_data(n_classes=3)
        clf = LandCoverClassifier(method="xgboost", n_classes=3, config={
            "n_estimators": 10, "max_depth": 3,
        })
        clf.train(X, y)
        proba = clf.predict_proba(X[:10])
        assert proba.shape == (10, 3)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load(self, tmp_path):
        X, y = _make_data(n_classes=4)
        clf = LandCoverClassifier(method="xgboost", n_classes=4, config={
            "n_estimators": 10,
        })
        clf.train(X, y)

        save_path = tmp_path / "test_clf"
        clf.save(save_path)

        assert (tmp_path / "test_clf.pkl").exists()
        assert (tmp_path / "test_clf.json").exists()

        loaded = LandCoverClassifier.load(save_path)
        assert loaded.is_trained
        assert loaded.method == "xgboost"

        # Predictions should match
        orig_preds = clf.predict(X[:5])
        loaded_preds = loaded.predict(X[:5])
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_predict_not_trained(self):
        clf = LandCoverClassifier(method="xgboost")
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(np.zeros((5, 10)))

    def test_metadata(self):
        X, y = _make_data(n_classes=3)
        clf = LandCoverClassifier(method="xgboost", n_classes=3, config={
            "n_estimators": 10,
        })
        clf.train(X, y)
        meta = clf.get_metadata()
        assert meta["method"] == "xgboost"
        assert meta["n_classes"] == 3
        assert meta["n_samples"] == 200
        assert meta["n_features"] == 64


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not available")
class TestRandomForestClassifier:
    """Test Random Forest classifier."""

    def test_train_and_predict(self):
        X, y = _make_data(n_classes=4)
        clf = LandCoverClassifier(method="random_forest", n_classes=4, config={
            "n_estimators": 10, "max_depth": 5,
        })
        metrics = clf.train(X, y)
        assert clf.is_trained
        assert metrics["train_accuracy"] > 0

        preds = clf.predict(X[:10])
        assert preds.shape == (10,)

    def test_predict_proba(self):
        X, y = _make_data(n_classes=3)
        clf = LandCoverClassifier(method="random_forest", n_classes=3, config={
            "n_estimators": 10,
        })
        clf.train(X, y)
        proba = clf.predict_proba(X[:5])
        assert proba.shape == (5, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not available")
class TestMLPClassifier:
    """Test MLP classifier."""

    def test_train_and_predict(self):
        X, y = _make_data(100, n_classes=3)
        clf = LandCoverClassifier(method="mlp", n_classes=3, config={
            "hidden_dims": [32, 16], "epochs": 10,
        })
        metrics = clf.train(X, y)
        assert clf.is_trained

        preds = clf.predict(X[:5])
        assert preds.shape == (5,)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not available")
class TestLinearClassifier:
    """Test linear (logistic regression) classifier."""

    def test_train_and_predict(self):
        X, y = _make_data(n_classes=4)
        clf = LandCoverClassifier(method="linear", n_classes=4, config={
            "C": 1.0,
        })
        metrics = clf.train(X, y)
        assert clf.is_trained
        assert metrics["train_accuracy"] > 0

        preds = clf.predict(X[:10])
        assert preds.shape == (10,)

    def test_save_load(self, tmp_path):
        X, y = _make_data(n_classes=3)
        clf = LandCoverClassifier(method="linear", n_classes=3)
        clf.train(X, y)
        clf.save(tmp_path / "linear_clf")

        loaded = LandCoverClassifier.load(tmp_path / "linear_clf")
        assert loaded.method == "linear"
        preds = loaded.predict(X[:5])
        assert preds.shape == (5,)
