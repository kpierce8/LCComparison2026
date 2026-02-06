"""Cross-validation and metrics for land cover classification.

Supports spatial CV with buffer zones, stratified k-fold,
and comprehensive metric calculation.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.validation")

try:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        class_names: Optional class name list for per-class reporting.

    Returns:
        Dict with accuracy, f1, iou, kappa, per-class metrics, confusion matrix.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for metrics")

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    # IoU (Jaccard) per class
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    iou_per_class = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        iou_per_class.append(float(iou))

    metrics["iou_mean"] = float(np.mean(iou_per_class))
    metrics["iou_per_class"] = iou_per_class

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics["f1_per_class"] = [float(x) for x in f1_per]

    # Confusion matrix
    metrics["confusion_matrix"] = cm.tolist()
    metrics["labels"] = [int(x) for x in labels]

    # Named per-class if provided
    if class_names is not None:
        per_class = {}
        for i, label in enumerate(labels):
            name = class_names[label] if label < len(class_names) else str(label)
            per_class[name] = {
                "f1": metrics["f1_per_class"][i] if i < len(metrics["f1_per_class"]) else 0,
                "iou": iou_per_class[i] if i < len(iou_per_class) else 0,
                "support": int(cm[i].sum()),
            }
        metrics["per_class"] = per_class

    return metrics


def cross_validate(
    classifier_factory: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    stratified: bool = True,
    spatial_coords: np.ndarray | None = None,
    buffer_m: float = 500.0,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run cross-validation on embeddings.

    Args:
        classifier_factory: Callable that returns a new LandCoverClassifier.
        X: Feature matrix (N, D).
        y: Labels (N,).
        n_folds: Number of CV folds.
        stratified: Use stratified splits.
        spatial_coords: Optional (N, 2) coordinates for spatial CV.
        buffer_m: Buffer distance for spatial CV (meters).
        class_names: Optional class names for per-class reporting.

    Returns:
        Dict with per-fold and aggregated metrics.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for cross-validation")

    if spatial_coords is not None:
        folds = _spatial_cv_splits(X, y, spatial_coords, n_folds, buffer_m)
    elif stratified:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(skf.split(X, y))
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(kf.split(X))

    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = classifier_factory()
        clf.train(X_train, y_train, X_val, y_val)

        y_pred = clf.predict(X_val)
        metrics = compute_metrics(y_val, y_pred, class_names)
        metrics["fold"] = fold_idx
        metrics["train_size"] = len(train_idx)
        metrics["val_size"] = len(val_idx)
        fold_metrics.append(metrics)

        all_y_true.append(y_val)
        all_y_pred.append(y_pred)

        logger.info(
            f"Fold {fold_idx}: accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1_macro']:.4f} iou={metrics['iou_mean']:.4f}"
        )

    # Aggregate metrics
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    aggregate = compute_metrics(all_y_true, all_y_pred, class_names)

    # Mean and std across folds
    summary = {
        "n_folds": n_folds,
        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics])),
        "f1_mean": float(np.mean([m["f1_macro"] for m in fold_metrics])),
        "f1_std": float(np.std([m["f1_macro"] for m in fold_metrics])),
        "iou_mean": float(np.mean([m["iou_mean"] for m in fold_metrics])),
        "iou_std": float(np.std([m["iou_mean"] for m in fold_metrics])),
        "kappa_mean": float(np.mean([m["kappa"] for m in fold_metrics])),
        "kappa_std": float(np.std([m["kappa"] for m in fold_metrics])),
    }

    return {
        "fold_metrics": fold_metrics,
        "aggregate": aggregate,
        "summary": summary,
    }


def _spatial_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    n_folds: int,
    buffer_m: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create spatial CV splits with buffer zones.

    Assigns samples to folds based on spatial clustering, then
    removes samples within buffer_m of fold boundaries from training.

    Args:
        X: Features (N, D).
        y: Labels (N,).
        coords: Coordinates (N, 2) in projected CRS (meters).
        n_folds: Number of folds.
        buffer_m: Buffer distance in meters.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    from sklearn.cluster import KMeans

    # Cluster spatial coordinates into n_folds groups
    kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init=10)
    fold_assignments = kmeans.fit_predict(coords)

    folds = []
    for fold_idx in range(n_folds):
        val_mask = fold_assignments == fold_idx
        val_indices = np.where(val_mask)[0]
        val_coords = coords[val_mask]

        # For training: exclude samples within buffer of val set
        train_mask = ~val_mask
        if buffer_m > 0 and len(val_coords) > 0:
            train_candidates = np.where(train_mask)[0]
            for tc in train_candidates:
                dists = np.sqrt(((coords[tc] - val_coords) ** 2).sum(axis=1))
                if dists.min() < buffer_m:
                    train_mask[tc] = False

        train_indices = np.where(train_mask)[0]

        if len(train_indices) > 0 and len(val_indices) > 0:
            folds.append((train_indices, val_indices))

    # If buffer removed too many, fall back to no buffer
    if len(folds) < n_folds:
        logger.warning(
            f"Spatial CV: only {len(folds)} valid folds "
            f"(buffer may be too large). Using all {n_folds} without buffer."
        )
        folds = []
        for fold_idx in range(n_folds):
            val_mask = fold_assignments == fold_idx
            train_indices = np.where(~val_mask)[0]
            val_indices = np.where(val_mask)[0]
            folds.append((train_indices, val_indices))

    return folds
