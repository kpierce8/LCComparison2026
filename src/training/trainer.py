"""End-to-end training pipeline for embedding-based classification.

Orchestrates: loading embeddings + labels, training classifier,
cross-validation, prediction on full tile set, saving results.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.classification.classifier import LandCoverClassifier
from src.config_schema import CLASS_SCHEMA
from src.training.validation import compute_metrics, cross_validate

logger = logging.getLogger("lccomparison.trainer")


class Trainer:
    """End-to-end training pipeline.

    Connects embeddings, labels, and classifier into a complete
    training and prediction workflow.
    """

    def __init__(
        self,
        method: str = "xgboost",
        n_classes: int = 7,
        classifier_config: dict[str, Any] | None = None,
        output_dir: str | Path = "data/checkpoints",
    ):
        """
        Args:
            method: Classifier method.
            n_classes: Number of land cover classes.
            classifier_config: Method-specific config params.
            output_dir: Directory for saving trained models and metrics.
        """
        self.method = method
        self.n_classes = n_classes
        self.classifier_config = classifier_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._classifier: LandCoverClassifier | None = None
        self._train_metrics: dict[str, Any] = {}
        self._cv_results: dict[str, Any] | None = None

    @property
    def classifier(self) -> LandCoverClassifier | None:
        return self._classifier

    def prepare_training_data(
        self,
        embeddings: dict[str, np.ndarray],
        labels: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Match embeddings with labels to create training arrays.

        Args:
            embeddings: Dict of {tile_id: embedding_vector}.
            labels: Dict of {tile_id: class_index}.

        Returns:
            Tuple of (X, y, tile_ids) where X is (N, D), y is (N,).
        """
        matched_ids = []
        X_list = []
        y_list = []

        for tile_id, class_idx in labels.items():
            if tile_id in embeddings:
                X_list.append(embeddings[tile_id])
                y_list.append(class_idx)
                matched_ids.append(tile_id)

        if not X_list:
            raise ValueError(
                "No matching tile IDs between embeddings and labels. "
                f"Embeddings: {len(embeddings)} tiles, Labels: {len(labels)} tiles."
            )

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)

        logger.info(
            f"Prepared training data: {len(matched_ids)} matched samples "
            f"({len(embeddings)} embeddings, {len(labels)} labels)"
        )

        return X, y, matched_ids

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        model_name: str = "default",
    ) -> dict[str, Any]:
        """Train classifier on prepared data.

        Args:
            X: Training features (N, D).
            y: Training labels (N,).
            X_val: Optional validation features.
            y_val: Optional validation labels.
            model_name: Name for saving model artifacts.

        Returns:
            Training metrics dict.
        """
        start_time = time.time()

        self._classifier = LandCoverClassifier(
            method=self.method,
            n_classes=self.n_classes,
            config=self.classifier_config,
        )

        train_metrics = self._classifier.train(X, y, X_val, y_val)
        elapsed = time.time() - start_time

        # Compute detailed metrics on training set
        y_pred = self._classifier.predict(X)
        class_names = list(CLASS_SCHEMA.keys())
        detailed = compute_metrics(y, y_pred, class_names)

        self._train_metrics = {
            **train_metrics,
            "detailed": detailed,
            "elapsed_seconds": round(elapsed, 2),
            "model_name": model_name,
        }

        # Save model
        model_path = self.output_dir / f"{model_name}_classifier"
        self._classifier.save(model_path)

        # Save metrics
        metrics_path = self.output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._train_metrics, f, indent=2, default=str)

        logger.info(
            f"Training complete in {elapsed:.1f}s. "
            f"Saved to {model_path}"
        )

        return self._train_metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        stratified: bool = True,
        spatial_coords: np.ndarray | None = None,
        buffer_m: float = 500.0,
    ) -> dict[str, Any]:
        """Run cross-validation.

        Args:
            X: Features (N, D).
            y: Labels (N,).
            n_folds: Number of folds.
            stratified: Use stratified splits.
            spatial_coords: Optional spatial coordinates for spatial CV.
            buffer_m: Buffer distance for spatial CV.

        Returns:
            Cross-validation results.
        """
        class_names = list(CLASS_SCHEMA.keys())

        def classifier_factory():
            return LandCoverClassifier(
                method=self.method,
                n_classes=self.n_classes,
                config=self.classifier_config,
            )

        self._cv_results = cross_validate(
            classifier_factory=classifier_factory,
            X=X, y=y,
            n_folds=n_folds,
            stratified=stratified,
            spatial_coords=spatial_coords,
            buffer_m=buffer_m,
            class_names=class_names,
        )

        # Save CV results
        cv_path = self.output_dir / "cv_results.json"
        with open(cv_path, "w") as f:
            json.dump(self._cv_results, f, indent=2, default=str)

        summary = self._cv_results["summary"]
        logger.info(
            f"CV complete: accuracy={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f} "
            f"f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f} "
            f"iou={summary['iou_mean']:.4f}±{summary['iou_std']:.4f}"
        )

        return self._cv_results

    def predict_tiles(
        self,
        embeddings: dict[str, np.ndarray],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Predict class labels for all tiles.

        Args:
            embeddings: Dict of {tile_id: embedding_vector}.
            output_dir: Directory to save per-tile predictions.

        Returns:
            Dict with predictions, probabilities, and summary.
        """
        if self._classifier is None or not self._classifier.is_trained:
            raise RuntimeError("No trained classifier. Call train() first.")

        if output_dir is None:
            output_dir = self.output_dir / "predictions"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tile_ids = list(embeddings.keys())
        X = np.stack([embeddings[tid] for tid in tile_ids], axis=0)

        # Predict
        predictions = self._classifier.predict(X)
        probabilities = self._classifier.predict_proba(X)

        # Save per-tile predictions
        results: dict[str, Any] = {}
        for i, tid in enumerate(tile_ids):
            results[tid] = {
                "class": int(predictions[i]),
                "probabilities": probabilities[i].tolist(),
                "confidence": float(probabilities[i].max()),
            }

        # Save summary
        pred_path = output_dir / "tile_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save arrays for mosaicking
        np.savez_compressed(
            output_dir / "predictions.npz",
            tile_ids=np.array(tile_ids),
            predictions=predictions,
            probabilities=probabilities,
        )

        # Class distribution
        unique, counts = np.unique(predictions, return_counts=True)
        class_names = list(CLASS_SCHEMA.keys())
        distribution = {}
        for cls, cnt in zip(unique, counts):
            name = class_names[cls] if cls < len(class_names) else str(cls)
            distribution[name] = int(cnt)

        summary = {
            "total_tiles": len(tile_ids),
            "class_distribution": distribution,
            "mean_confidence": float(probabilities.max(axis=1).mean()),
        }

        logger.info(
            f"Predicted {len(tile_ids)} tiles. "
            f"Mean confidence: {summary['mean_confidence']:.4f}"
        )

        return {
            "predictions": dict(zip(tile_ids, predictions.tolist())),
            "results": results,
            "summary": summary,
        }

    def load_classifier(self, path: str | Path) -> None:
        """Load a previously trained classifier."""
        self._classifier = LandCoverClassifier.load(path)
        logger.info(f"Loaded classifier from {path}")

    def get_results(self) -> dict[str, Any]:
        """Get all training results."""
        return {
            "train_metrics": self._train_metrics,
            "cv_results": self._cv_results,
        }
