"""Ensemble and fusion methods for combining model predictions.

Supports majority voting, weighted voting, probability averaging,
hierarchical fusion, and agreement/uncertainty product generation.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.ensemble")

try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False

from src.config_schema import CLASS_SCHEMA


class EnsembleClassifier:
    """Combine predictions from multiple models or resolutions.

    Strategies:
    - majority_vote: Each model votes, majority wins.
    - weighted_vote: Weighted majority vote.
    - probability_average: Average class probabilities.
    - weighted_probability: Weighted average of probabilities.
    """

    def __init__(
        self,
        n_classes: int = 7,
        nodata: int = 255,
    ):
        self.n_classes = n_classes
        self.nodata = nodata
        self._idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}

    def majority_vote(
        self,
        predictions: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Combine predictions by majority vote.

        Args:
            predictions: Dict of {model_name: class_array (H, W)}.

        Returns:
            Dict with ensemble classification, agreement map, metadata.
        """
        names = list(predictions.keys())
        arrays = list(predictions.values())
        n_models = len(arrays)

        if n_models == 0:
            raise ValueError("No predictions provided")

        shape = arrays[0].shape
        stacked = np.stack(arrays, axis=0)  # (n_models, H, W)

        # Valid mask: all models have data
        valid = np.all(stacked != self.nodata, axis=0)  # (H, W)

        # Count votes per class
        vote_counts = np.zeros((self.n_classes, *shape), dtype=np.int32)
        for cls in range(self.n_classes):
            vote_counts[cls] = np.sum(stacked == cls, axis=0)

        # Winner = class with most votes
        ensemble = np.full(shape, self.nodata, dtype=np.uint8)
        ensemble[valid] = np.argmax(vote_counts[:, valid], axis=0).astype(np.uint8)

        # Agreement: fraction of models that agree with the winner
        max_votes = np.max(vote_counts, axis=0)
        agreement = np.zeros(shape, dtype=np.float32)
        agreement[valid] = max_votes[valid] / n_models

        return {
            "classification": ensemble,
            "agreement": agreement,
            "n_models": n_models,
            "models": names,
            "valid_pixels": int(valid.sum()),
            "mean_agreement": float(agreement[valid].mean()) if valid.any() else 0,
        }

    def weighted_vote(
        self,
        predictions: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """Combine predictions by weighted vote.

        Args:
            predictions: Dict of {model_name: class_array (H, W)}.
            weights: Dict of {model_name: weight}.

        Returns:
            Dict with ensemble classification, confidence, metadata.
        """
        names = list(predictions.keys())
        arrays = list(predictions.values())
        n_models = len(arrays)

        if n_models == 0:
            raise ValueError("No predictions provided")

        shape = arrays[0].shape
        stacked = np.stack(arrays, axis=0)
        valid = np.all(stacked != self.nodata, axis=0)

        # Weighted vote counts
        weighted_counts = np.zeros((self.n_classes, *shape), dtype=np.float64)
        for i, name in enumerate(names):
            w = weights.get(name, 1.0)
            for cls in range(self.n_classes):
                weighted_counts[cls] += (stacked[i] == cls) * w

        # Winner = class with highest weighted vote
        ensemble = np.full(shape, self.nodata, dtype=np.uint8)
        ensemble[valid] = np.argmax(weighted_counts[:, valid], axis=0).astype(np.uint8)

        # Confidence: weighted fraction for winning class
        total_weight = sum(weights.get(n, 1.0) for n in names)
        max_weighted = np.max(weighted_counts, axis=0)
        confidence = np.zeros(shape, dtype=np.float32)
        confidence[valid] = max_weighted[valid] / total_weight

        return {
            "classification": ensemble,
            "confidence": confidence,
            "n_models": n_models,
            "models": names,
            "weights": {n: weights.get(n, 1.0) for n in names},
            "valid_pixels": int(valid.sum()),
            "mean_confidence": float(confidence[valid].mean()) if valid.any() else 0,
        }

    def probability_average(
        self,
        probabilities: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Average class probabilities across models.

        Args:
            probabilities: Dict of {model_name: probability_array (n_classes, H, W)}.
            weights: Optional weights per model.

        Returns:
            Dict with ensemble classification, probabilities, uncertainty.
        """
        names = list(probabilities.keys())
        n_models = len(names)

        if n_models == 0:
            raise ValueError("No probabilities provided")

        first = probabilities[names[0]]
        n_cls = first.shape[0]
        shape = first.shape[1:]

        if weights is None:
            weights = {n: 1.0 / n_models for n in names}

        # Normalize weights
        total_w = sum(weights.get(n, 1.0 / n_models) for n in names)
        norm_weights = {n: weights.get(n, 1.0 / n_models) / total_w for n in names}

        # Weighted average
        avg_prob = np.zeros((n_cls, *shape), dtype=np.float64)
        for name in names:
            avg_prob += probabilities[name] * norm_weights[name]

        # Classification from averaged probabilities
        ensemble = np.argmax(avg_prob, axis=0).astype(np.uint8)

        # Confidence: max probability
        confidence = np.max(avg_prob, axis=0).astype(np.float32)

        # Uncertainty: entropy of averaged probabilities
        with np.errstate(divide="ignore", invalid="ignore"):
            log_prob = np.where(avg_prob > 0, np.log2(avg_prob), 0)
        entropy = -np.sum(avg_prob * log_prob, axis=0)
        max_entropy = np.log2(n_cls) if n_cls > 1 else 1
        uncertainty = entropy / max_entropy  # normalize to 0-1

        return {
            "classification": ensemble,
            "probabilities": avg_prob.astype(np.float32),
            "confidence": confidence,
            "uncertainty": uncertainty.astype(np.float32),
            "n_models": n_models,
            "models": names,
            "valid_pixels": int(np.prod(shape)),
            "mean_confidence": float(confidence.mean()),
            "mean_uncertainty": float(uncertainty.mean()),
        }

    def compute_agreement_map(
        self,
        predictions: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Compute per-pixel agreement across multiple models.

        Args:
            predictions: Dict of {model_name: class_array (H, W)}.

        Returns:
            Dict with agreement map, disagreement locations, statistics.
        """
        names = list(predictions.keys())
        arrays = list(predictions.values())
        n_models = len(arrays)

        if n_models < 2:
            raise ValueError("Need at least 2 models for agreement analysis")

        shape = arrays[0].shape
        stacked = np.stack(arrays, axis=0)
        valid = np.all(stacked != self.nodata, axis=0)

        # Count most common class per pixel
        vote_counts = np.zeros((self.n_classes, *shape), dtype=np.int32)
        for cls in range(self.n_classes):
            vote_counts[cls] = np.sum(stacked == cls, axis=0)

        max_votes = np.max(vote_counts, axis=0)

        # Agreement fraction (0 to 1)
        agreement = np.zeros(shape, dtype=np.float32)
        agreement[valid] = max_votes[valid] / n_models

        # Full agreement mask
        full_agree = agreement == 1.0
        # Full disagreement: all models predict differently
        n_unique = np.zeros(shape, dtype=np.int32)
        for r in range(shape[0]):
            for c in range(shape[1]):
                if valid[r, c]:
                    n_unique[r, c] = len(set(stacked[:, r, c]))

        # Pairwise agreement matrix
        pairwise = {}
        for i in range(n_models):
            for j in range(i + 1, n_models):
                agree_pct = float(
                    np.mean(
                        (arrays[i][valid] == arrays[j][valid])
                    ) * 100
                ) if valid.any() else 0
                pairwise[f"{names[i]}_vs_{names[j]}"] = round(agree_pct, 2)

        # Per-class agreement
        per_class = {}
        idx_to_name = self._idx_to_name
        for cls in range(self.n_classes):
            cls_name = idx_to_name.get(cls, f"class_{cls}")
            # Pixels where any model predicts this class
            any_predicts = np.any(stacked == cls, axis=0) & valid
            if any_predicts.any():
                # Fraction of models that agree on this class where any predicts it
                cls_agreement = vote_counts[cls][any_predicts] / n_models
                per_class[cls_name] = {
                    "mean_agreement": round(float(cls_agreement.mean()), 4),
                    "pixels_any": int(any_predicts.sum()),
                    "pixels_all": int(np.all(stacked == cls, axis=0).sum()),
                }

        return {
            "agreement_map": agreement,
            "n_models": n_models,
            "models": names,
            "valid_pixels": int(valid.sum()),
            "full_agreement_pct": round(float(full_agree[valid].mean() * 100), 2) if valid.any() else 0,
            "mean_agreement": round(float(agreement[valid].mean()), 4) if valid.any() else 0,
            "pairwise_agreement": pairwise,
            "per_class": per_class,
        }


class HierarchicalFusion:
    """Hierarchical multi-resolution fusion.

    Uses a base resolution classification and refines with higher-resolution
    data where available.
    """

    def __init__(self, nodata: int = 255, n_classes: int = 7):
        self.nodata = nodata
        self.n_classes = n_classes

    def fuse(
        self,
        base: np.ndarray,
        refinement: np.ndarray,
        refinement_confidence: np.ndarray | None = None,
        confidence_threshold: float = 0.5,
        strategy: str = "high_res_priority",
    ) -> dict[str, Any]:
        """Fuse base and refinement classifications.

        Args:
            base: Base resolution classification (H, W).
            refinement: Higher-resolution classification (H, W), same grid.
            refinement_confidence: Optional confidence scores for refinement.
            confidence_threshold: Min confidence to use refinement.
            strategy: "high_res_priority" or "confidence_weighted".

        Returns:
            Dict with fused classification, source map, statistics.
        """
        if base.shape != refinement.shape:
            raise ValueError(
                f"Shape mismatch: base={base.shape}, refinement={refinement.shape}. "
                "Rasters must be aligned to the same grid first."
            )

        base_valid = base != self.nodata
        ref_valid = refinement != self.nodata

        fused = np.full_like(base, self.nodata)
        # Source map: 0=nodata, 1=base, 2=refinement
        source_map = np.zeros_like(base, dtype=np.uint8)

        if strategy == "high_res_priority":
            # Use refinement where available, base elsewhere
            fused[base_valid] = base[base_valid]
            source_map[base_valid] = 1

            fused[ref_valid] = refinement[ref_valid]
            source_map[ref_valid] = 2

            # If confidence threshold is set and confidence is available, only
            # use refinement where confident enough
            if refinement_confidence is not None:
                low_conf = ref_valid & (refinement_confidence < confidence_threshold)
                # Revert to base where refinement is low confidence
                revert = low_conf & base_valid
                fused[revert] = base[revert]
                source_map[revert] = 1

        elif strategy == "confidence_weighted":
            if refinement_confidence is None:
                raise ValueError("confidence_weighted requires refinement_confidence")

            # Where both available, use the one with higher confidence
            both_valid = base_valid & ref_valid
            fused[base_valid & ~ref_valid] = base[base_valid & ~ref_valid]
            source_map[base_valid & ~ref_valid] = 1
            fused[ref_valid & ~base_valid] = refinement[ref_valid & ~base_valid]
            source_map[ref_valid & ~base_valid] = 2

            # Where both valid, threshold determines source
            use_ref = both_valid & (refinement_confidence >= confidence_threshold)
            use_base = both_valid & ~use_ref
            fused[use_ref] = refinement[use_ref]
            source_map[use_ref] = 2
            fused[use_base] = base[use_base]
            source_map[use_base] = 1

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        valid = fused != self.nodata
        base_used = int((source_map == 1).sum())
        ref_used = int((source_map == 2).sum())

        return {
            "classification": fused,
            "source_map": source_map,
            "valid_pixels": int(valid.sum()),
            "base_pixels": base_used,
            "refinement_pixels": ref_used,
            "base_pct": round(base_used / max(valid.sum(), 1) * 100, 1),
            "refinement_pct": round(ref_used / max(valid.sum(), 1) * 100, 1),
            "strategy": strategy,
        }


def save_ensemble_raster(
    data: np.ndarray,
    output_path: str | Path,
    profile: dict,
    nodata: int = 255,
) -> Path:
    """Save an ensemble result as a GeoTIFF.

    Args:
        data: 2D or 3D array.
        output_path: Output path.
        profile: Rasterio profile dict (from reference raster).
        nodata: Nodata value.

    Returns:
        Output path.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio required")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    out_profile = profile.copy()
    out_profile.update({
        "count": data.shape[0],
        "dtype": str(data.dtype),
        "nodata": nodata,
        "compress": "lzw",
    })

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(data)

    return output_path


def save_ensemble_outputs(
    ensemble_result: dict[str, Any],
    output_dir: str | Path,
    profile: dict,
    name_prefix: str = "ensemble",
    nodata: int = 255,
) -> dict[str, str]:
    """Save all ensemble products (classification, confidence, uncertainty, agreement).

    Args:
        ensemble_result: Result dict from ensemble methods.
        output_dir: Output directory.
        profile: Rasterio profile for georeferencing.
        name_prefix: Filename prefix.
        nodata: Nodata value.

    Returns:
        Dict of {product_name: output_path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Classification
    if "classification" in ensemble_result:
        path = save_ensemble_raster(
            ensemble_result["classification"], output_dir / f"{name_prefix}_classification.tif",
            profile, nodata=nodata,
        )
        outputs["classification"] = str(path)

    # Confidence
    if "confidence" in ensemble_result:
        conf_profile = profile.copy()
        conf_profile["dtype"] = "float32"
        path = save_ensemble_raster(
            ensemble_result["confidence"],
            output_dir / f"{name_prefix}_confidence.tif",
            conf_profile, nodata=-1,
        )
        outputs["confidence"] = str(path)

    # Uncertainty
    if "uncertainty" in ensemble_result:
        unc_profile = profile.copy()
        unc_profile["dtype"] = "float32"
        path = save_ensemble_raster(
            ensemble_result["uncertainty"],
            output_dir / f"{name_prefix}_uncertainty.tif",
            unc_profile, nodata=-1,
        )
        outputs["uncertainty"] = str(path)

    # Agreement map
    if "agreement_map" in ensemble_result:
        ag_profile = profile.copy()
        ag_profile["dtype"] = "float32"
        path = save_ensemble_raster(
            ensemble_result["agreement_map"],
            output_dir / f"{name_prefix}_agreement.tif",
            ag_profile, nodata=-1,
        )
        outputs["agreement"] = str(path)

    # Source map (from hierarchical fusion)
    if "source_map" in ensemble_result:
        path = save_ensemble_raster(
            ensemble_result["source_map"],
            output_dir / f"{name_prefix}_source.tif",
            profile, nodata=0,
        )
        outputs["source_map"] = str(path)

    # Probabilities (multi-band)
    if "probabilities" in ensemble_result:
        prob_profile = profile.copy()
        prob_profile["dtype"] = "float32"
        path = save_ensemble_raster(
            ensemble_result["probabilities"],
            output_dir / f"{name_prefix}_probabilities.tif",
            prob_profile, nodata=-1,
        )
        outputs["probabilities"] = str(path)

    # Metadata JSON
    meta = {k: v for k, v in ensemble_result.items()
            if not isinstance(v, np.ndarray)}
    meta_path = output_dir / f"{name_prefix}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    outputs["metadata"] = str(meta_path)

    logger.info(f"Saved {len(outputs)} ensemble products to {output_dir}")
    return outputs
