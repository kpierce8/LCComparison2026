"""Accuracy assessment for land cover classifications.

Evaluates classification accuracy against reference points,
producing metrics, confusion matrices, and error analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.accuracy_assessor")

try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    _GEO_AVAILABLE = True
except ImportError:
    gpd = None
    _GEO_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from src.config_schema import CLASS_SCHEMA


class AccuracyAssessor:
    """Assess classification accuracy against reference data."""

    def __init__(
        self,
        class_schema: dict[str, int] | None = None,
        min_samples_per_class: int = 30,
        buffer_m: float = 5.0,
    ):
        """
        Args:
            class_schema: Class name -> index mapping.
            min_samples_per_class: Minimum reference points per class.
            buffer_m: Buffer around points for extraction.
        """
        self.class_schema = class_schema or CLASS_SCHEMA
        self.min_samples = min_samples_per_class
        self.buffer_m = buffer_m
        self._idx_to_name = {v: k for k, v in self.class_schema.items()}

    def assess(
        self,
        prediction_path: str | Path,
        reference_path: str | Path,
        class_field: str = "LC_CLASS",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run accuracy assessment.

        Args:
            prediction_path: Path to classification raster.
            reference_path: Path to reference points (shapefile/GPKG/GeoJSON).
            class_field: Column name for reference class labels.
            output_dir: Directory for assessment outputs.

        Returns:
            Dict with metrics, confusion matrix, per-class accuracy.
        """
        if not all([_RASTERIO_AVAILABLE, _GEO_AVAILABLE, _SKLEARN_AVAILABLE]):
            raise ImportError("rasterio, geopandas, and scikit-learn required")

        # Load reference points
        ref_gdf = gpd.read_file(reference_path)
        if class_field not in ref_gdf.columns:
            raise ValueError(
                f"Class field '{class_field}' not found. "
                f"Available: {list(ref_gdf.columns)}"
            )

        # Extract predicted values at reference points
        y_true, y_pred, matched_gdf = self._extract_at_points(
            prediction_path, ref_gdf, class_field,
        )

        if len(y_true) == 0:
            return {"error": "No valid reference points matched predictions"}

        # Compute metrics
        results = self._compute_full_metrics(y_true, y_pred)
        results["n_points"] = len(y_true)

        # Warnings for low sample classes
        warnings = self._check_sample_sizes(y_true)
        results["warnings"] = warnings

        # Save outputs
        if output_dir is not None:
            self._save_outputs(results, matched_gdf, y_pred, output_dir)

        logger.info(
            f"Accuracy assessment: {results['overall_accuracy']:.4f} OA, "
            f"{results['kappa']:.4f} kappa on {len(y_true)} points"
        )

        return results

    def compare_models(
        self,
        prediction_paths: dict[str, str | Path],
        reference_path: str | Path,
        class_field: str = "LC_CLASS",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compare accuracy of multiple models.

        Args:
            prediction_paths: Dict of {model_name: raster_path}.
            reference_path: Reference points path.
            class_field: Class label column.
            output_dir: Output directory.

        Returns:
            Dict with per-model metrics and comparison summary.
        """
        model_results = {}
        for model_name, pred_path in prediction_paths.items():
            try:
                result = self.assess(pred_path, reference_path, class_field)
                model_results[model_name] = result
            except Exception as e:
                logger.warning(f"Assessment failed for {model_name}: {e}")
                model_results[model_name] = {"error": str(e)}

        # Build comparison table
        comparison = self._build_comparison(model_results)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "model_comparison.json", "w") as f:
                json.dump(comparison, f, indent=2, default=str)

        return comparison

    def _extract_at_points(
        self,
        raster_path: str | Path,
        ref_gdf: Any,
        class_field: str,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Extract raster values at reference point locations."""
        # Reproject reference points to raster CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            if ref_gdf.crs != raster_crs:
                ref_gdf = ref_gdf.to_crs(raster_crs)

            coords = [(geom.x, geom.y) for geom in ref_gdf.geometry]
            sampled = list(src.sample(coords))

        # Match reference labels to class indices
        y_true_list = []
        y_pred_list = []
        valid_indices = []

        for i, (ref_val, pred_val) in enumerate(zip(ref_gdf[class_field], sampled)):
            pred_idx = int(pred_val[0])

            # Convert reference label to index
            if isinstance(ref_val, (int, np.integer)):
                true_idx = int(ref_val)
            elif isinstance(ref_val, str):
                ref_lower = ref_val.lower().strip()
                true_idx = None
                for name, idx in self.class_schema.items():
                    if name.lower() == ref_lower:
                        true_idx = idx
                        break
                if true_idx is None:
                    continue
            else:
                continue

            # Skip nodata predictions
            if pred_idx == 255 or pred_idx < 0:
                continue

            y_true_list.append(true_idx)
            y_pred_list.append(pred_idx)
            valid_indices.append(i)

        matched_gdf = ref_gdf.iloc[valid_indices].copy()
        return (
            np.array(y_true_list, dtype=np.int64),
            np.array(y_pred_list, dtype=np.int64),
            matched_gdf,
        )

    def _compute_full_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Compute comprehensive accuracy metrics."""
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

        results = {
            "overall_accuracy": float(accuracy_score(y_true, y_pred)),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        results["confusion_matrix"] = cm.tolist()
        results["labels"] = [int(x) for x in labels]

        # Per-class metrics
        producers = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        users = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

        per_class = {}
        for i, label in enumerate(labels):
            name = self._idx_to_name.get(label, f"class_{label}")
            tp = cm[i, i]
            support = int(cm[i].sum())
            per_class[name] = {
                "class_index": int(label),
                "producers_accuracy": float(producers[i]),
                "users_accuracy": float(users[i]),
                "f1": float(f1_per[i]),
                "support": support,
                "correct": int(tp),
            }
        results["per_class"] = per_class

        return results

    def _check_sample_sizes(self, y_true: np.ndarray) -> list[str]:
        """Check for classes with insufficient samples."""
        warnings = []
        unique, counts = np.unique(y_true, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            name = self._idx_to_name.get(int(cls_idx), f"class_{cls_idx}")
            if count < self.min_samples:
                warnings.append(
                    f"{name}: only {count} samples (min recommended: {self.min_samples})"
                )
        return warnings

    def _save_outputs(
        self,
        results: dict[str, Any],
        matched_gdf: Any,
        y_pred: np.ndarray,
        output_dir: str | Path,
    ) -> None:
        """Save assessment outputs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        with open(output_dir / "accuracy_metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save confusion matrix CSV
        cm = results.get("confusion_matrix", [])
        labels = results.get("labels", [])
        if cm:
            header = "," + ",".join(
                self._idx_to_name.get(l, str(l)) for l in labels
            )
            lines = [header]
            for i, row in enumerate(cm):
                name = self._idx_to_name.get(labels[i], str(labels[i]))
                lines.append(f"{name}," + ",".join(str(x) for x in row))
            with open(output_dir / "confusion_matrix.csv", "w") as f:
                f.write("\n".join(lines) + "\n")

        # Save per-class accuracy CSV
        per_class = results.get("per_class", {})
        if per_class:
            lines = ["class,producers_accuracy,users_accuracy,f1,support"]
            for name, metrics in per_class.items():
                lines.append(
                    f"{name},{metrics['producers_accuracy']:.4f},"
                    f"{metrics['users_accuracy']:.4f},{metrics['f1']:.4f},"
                    f"{metrics['support']}"
                )
            with open(output_dir / "per_class_accuracy.csv", "w") as f:
                f.write("\n".join(lines) + "\n")

        # Save matched points with predictions
        if matched_gdf is not None and len(matched_gdf) > 0:
            matched_gdf = matched_gdf.copy()
            matched_gdf["predicted"] = y_pred
            matched_gdf["predicted_name"] = [
                self._idx_to_name.get(int(p), str(p)) for p in y_pred
            ]
            matched_gdf.to_file(output_dir / "points_with_predictions.gpkg")

        logger.info(f"Saved assessment outputs to {output_dir}")

    def _build_comparison(
        self,
        model_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Build comparison summary from multiple model assessments."""
        comparison = {"models": {}}

        for model_name, result in model_results.items():
            if "error" in result:
                comparison["models"][model_name] = {"error": result["error"]}
                continue

            comparison["models"][model_name] = {
                "overall_accuracy": result.get("overall_accuracy"),
                "kappa": result.get("kappa"),
                "f1_macro": result.get("f1_macro"),
                "n_points": result.get("n_points"),
            }

        # Rank by accuracy
        valid = {
            k: v for k, v in comparison["models"].items()
            if "error" not in v and v.get("overall_accuracy") is not None
        }
        if valid:
            ranked = sorted(valid.items(), key=lambda x: x[1]["overall_accuracy"], reverse=True)
            comparison["ranking"] = [
                {"model": name, "overall_accuracy": m["overall_accuracy"]}
                for name, m in ranked
            ]
            comparison["best_model"] = ranked[0][0]

        return comparison
