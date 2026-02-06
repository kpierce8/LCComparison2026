"""Classifier training and inference for embedding-based land cover classification.

Supports multiple classifier backends:
- XGBoost (primary, gradient boosted trees)
- Random Forest (sklearn)
- MLP (sklearn neural network)
- Linear (logistic regression)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.classifier")

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None
    _XGB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# Supported classifier methods
SUPPORTED_METHODS = ["xgboost", "random_forest", "mlp", "linear"]


class LandCoverClassifier:
    """Classifier that maps embedding vectors to land cover classes.

    Wraps multiple sklearn/xgboost backends with a unified interface
    for training, prediction, saving, and loading.
    """

    def __init__(
        self,
        method: str = "xgboost",
        n_classes: int = 7,
        config: dict[str, Any] | None = None,
    ):
        """
        Args:
            method: Classifier method (xgboost, random_forest, mlp, linear).
            n_classes: Number of output classes.
            config: Method-specific configuration parameters.
        """
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {method}. Supported: {SUPPORTED_METHODS}"
            )
        self.method = method
        self.n_classes = n_classes
        self.config = config or {}
        self._model: Any = None
        self._label_encoder: Any = None
        self._trained = False
        self._metadata: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Train the classifier.

        Args:
            X: Training features (N, embedding_dim).
            y: Training labels (N,) integer class indices.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.

        Returns:
            Dict with training metrics.
        """
        logger.info(
            f"Training {self.method} classifier: "
            f"{X.shape[0]} samples, {X.shape[1]} features, {self.n_classes} classes"
        )

        # Encode labels to 0..n_classes-1 if needed
        self._label_encoder = _build_label_encoder(y)
        y_enc = self._label_encoder.transform(y)
        y_val_enc = self._label_encoder.transform(y_val) if y_val is not None else None

        self._model = self._create_model()
        metrics = self._fit(X, y_enc, X_val, y_val_enc)

        self._trained = True
        self._metadata = {
            "method": self.method,
            "n_classes": self.n_classes,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "classes": self._label_encoder.classes_.tolist(),
            **metrics,
        }

        logger.info(f"Training complete. Accuracy: {metrics.get('train_accuracy', 'N/A')}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features (N, embedding_dim).

        Returns:
            Predicted class indices (N,).
        """
        if not self._trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        raw = self._predict_raw(X)
        return self._label_encoder.inverse_transform(raw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features (N, embedding_dim).

        Returns:
            Class probabilities (N, n_classes).
        """
        if not self._trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        if self.method == "xgboost":
            if not _XGB_AVAILABLE:
                raise ImportError("xgboost required")
            dmatrix = xgb.DMatrix(X)
            return self._model.predict(dmatrix)
        else:
            return self._model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Save trained classifier to disk.

        Args:
            path: Output file path (will create .pkl and .json).
        """
        if not self._trained:
            raise RuntimeError("Cannot save untrained classifier.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path.with_suffix(".pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "label_encoder": self._label_encoder,
                "method": self.method,
                "n_classes": self.n_classes,
                "config": self.config,
            }, f)

        # Save metadata
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        logger.info(f"Saved classifier to {model_path}")

    @classmethod
    def load(cls, path: str | Path) -> "LandCoverClassifier":
        """Load a trained classifier from disk.

        Args:
            path: Path to saved classifier (with or without .pkl extension).

        Returns:
            Loaded LandCoverClassifier instance.
        """
        path = Path(path)
        model_path = path.with_suffix(".pkl")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            method=data["method"],
            n_classes=data["n_classes"],
            config=data.get("config", {}),
        )
        instance._model = data["model"]
        instance._label_encoder = data["label_encoder"]
        instance._trained = True

        # Load metadata if available
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                instance._metadata = json.load(f)

        logger.info(f"Loaded {instance.method} classifier from {model_path}")
        return instance

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def _create_model(self) -> Any:
        """Create the underlying classifier model."""
        if self.method == "xgboost":
            return None  # XGBoost uses xgb.train API directly

        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for non-XGBoost classifiers")

        if self.method == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.get("n_estimators", 200),
                max_depth=self.config.get("max_depth", 25),
                class_weight=self.config.get("class_weight", "balanced"),
                n_jobs=-1,
                random_state=42,
            )
        elif self.method == "mlp":
            hidden = self.config.get("hidden_dims", [512, 256])
            return MLPClassifier(
                hidden_layer_sizes=tuple(hidden),
                alpha=self.config.get("dropout", 0.3) * 0.01,
                learning_rate_init=self.config.get("learning_rate", 0.001),
                max_iter=self.config.get("epochs", 100),
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
        elif self.method == "linear":
            return LogisticRegression(
                C=self.config.get("C", 1.0),
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> dict[str, Any]:
        """Fit the model and return metrics."""
        metrics: dict[str, Any] = {}

        if self.method == "xgboost":
            metrics = self._fit_xgboost(X, y, X_val, y_val)
        else:
            self._model.fit(X, y)
            train_pred = self._model.predict(X)
            metrics["train_accuracy"] = float((train_pred == y).mean())

            if X_val is not None and y_val is not None:
                val_pred = self._model.predict(X_val)
                metrics["val_accuracy"] = float((val_pred == y_val).mean())

        return metrics

    def _fit_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> dict[str, Any]:
        """Train XGBoost model."""
        if not _XGB_AVAILABLE:
            raise ImportError(
                "xgboost required. Install with: pip install xgboost"
            )

        params = {
            "objective": "multi:softprob",
            "num_class": self.n_classes,
            "max_depth": self.config.get("max_depth", 8),
            "learning_rate": self.config.get("learning_rate", 0.1),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
        }

        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        n_rounds = self.config.get("n_estimators", 500)
        early_stop = self.config.get("early_stopping_rounds", 50)

        callbacks = []
        if X_val is not None:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=early_stop, metric_name="mlogloss", data_name="val",
                )
            )

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            callbacks=callbacks if callbacks else None,
            verbose_eval=False,
        )

        # Compute metrics
        train_pred = self._model.predict(dtrain).argmax(axis=1)
        metrics = {"train_accuracy": float((train_pred == y).mean())}

        if X_val is not None and y_val is not None:
            val_pred = self._model.predict(dval).argmax(axis=1)
            metrics["val_accuracy"] = float((val_pred == y_val).mean())

        metrics["best_iteration"] = getattr(self._model, "best_iteration", n_rounds)
        return metrics

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict encoded class indices."""
        if self.method == "xgboost":
            if not _XGB_AVAILABLE:
                raise ImportError("xgboost required")
            dmatrix = xgb.DMatrix(X)
            proba = self._model.predict(dmatrix)
            return proba.argmax(axis=1)
        else:
            return self._model.predict(X)


def _build_label_encoder(y: np.ndarray) -> Any:
    """Build a label encoder that maps class indices to 0..n-1."""
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required")
    le = LabelEncoder()
    le.fit(y)
    return le
