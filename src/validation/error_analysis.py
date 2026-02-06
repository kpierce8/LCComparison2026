"""Error pattern analysis for land cover classifications.

Analyzes spatial patterns of classification errors, class confusion
hotspots, and edge effects.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.error_analysis")

try:
    import rasterio
    from rasterio.transform import rowcol
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

from src.config_schema import CLASS_SCHEMA


def compute_error_map(
    prediction_path: str | Path,
    reference_path: str | Path,
    class_field: str = "LC_CLASS",
    output_path: str | Path | None = None,
    nodata: int = 255,
) -> dict[str, Any]:
    """Create a spatial error map from prediction raster and reference points.

    Assigns error codes at each reference point location:
    - 0: correct prediction
    - 1: incorrect prediction
    - nodata: no reference data

    Args:
        prediction_path: Path to classification raster.
        reference_path: Path to reference points.
        class_field: Column name with reference class labels.
        output_path: Optional path to save error point layer.
        nodata: Nodata value.

    Returns:
        Dict with error summary and spatial patterns.
    """
    if not (_RASTERIO_AVAILABLE and _GEO_AVAILABLE):
        raise ImportError("rasterio and geopandas required")

    name_to_idx = {k.lower(): v for k, v in CLASS_SCHEMA.items()}

    ref_gdf = gpd.read_file(reference_path)

    with rasterio.open(prediction_path) as src:
        raster_crs = src.crs
        if ref_gdf.crs != raster_crs:
            ref_gdf = ref_gdf.to_crs(raster_crs)

        coords = [(geom.x, geom.y) for geom in ref_gdf.geometry]
        sampled = list(src.sample(coords))

    correct = []
    errors = []
    error_types = []  # (true_class, predicted_class)

    for i, (ref_val, pred_val) in enumerate(zip(ref_gdf[class_field], sampled)):
        pred_idx = int(pred_val[0])
        if pred_idx == nodata or pred_idx < 0:
            continue

        # Convert reference to index
        if isinstance(ref_val, (int, np.integer)):
            true_idx = int(ref_val)
        elif isinstance(ref_val, str):
            true_idx = name_to_idx.get(ref_val.lower().strip())
            if true_idx is None:
                continue
        else:
            continue

        if true_idx == pred_idx:
            correct.append(i)
        else:
            errors.append(i)
            error_types.append((true_idx, pred_idx))

    total = len(correct) + len(errors)
    if total == 0:
        return {"error_rate": 0, "n_points": 0}

    # Build confusion pairs
    idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}
    confusion_pairs = {}
    for true_idx, pred_idx in error_types:
        true_name = idx_to_name.get(true_idx, f"class_{true_idx}")
        pred_name = idx_to_name.get(pred_idx, f"class_{pred_idx}")
        pair_key = f"{true_name} -> {pred_name}"
        confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1

    # Sort by frequency
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    # Per-class error rates
    per_class_errors = {}
    for true_idx, pred_idx in error_types:
        true_name = idx_to_name.get(true_idx, f"class_{true_idx}")
        if true_name not in per_class_errors:
            per_class_errors[true_name] = 0
        per_class_errors[true_name] += 1

    per_class_totals = {}
    for i in correct:
        ref_val = ref_gdf[class_field].iloc[i]
        if isinstance(ref_val, (int, np.integer)):
            name = idx_to_name.get(int(ref_val), str(ref_val))
        else:
            name = str(ref_val).lower().strip()
        per_class_totals[name] = per_class_totals.get(name, 0) + 1
    for i in errors:
        ref_val = ref_gdf[class_field].iloc[i]
        if isinstance(ref_val, (int, np.integer)):
            name = idx_to_name.get(int(ref_val), str(ref_val))
        else:
            name = str(ref_val).lower().strip()
        per_class_totals[name] = per_class_totals.get(name, 0) + 1

    per_class_error_rates = {}
    for cls_name, err_count in per_class_errors.items():
        total_cls = per_class_totals.get(cls_name, err_count)
        per_class_error_rates[cls_name] = round(err_count / total_cls, 4) if total_cls > 0 else 0

    # Save error points if requested
    if output_path is not None and len(ref_gdf) > 0:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        error_gdf = ref_gdf.copy()
        error_gdf["is_error"] = 0
        error_gdf.loc[error_gdf.index.isin(ref_gdf.index[errors]), "is_error"] = 1
        error_gdf.to_file(output_path)

    result = {
        "error_rate": round(len(errors) / total, 4),
        "n_correct": len(correct),
        "n_errors": len(errors),
        "n_points": total,
        "top_confusions": top_confusions[:10],
        "per_class_error_rates": per_class_error_rates,
    }

    logger.info(f"Error analysis: {len(errors)}/{total} errors ({result['error_rate']:.2%})")
    return result


def analyze_class_confusions(
    confusion_matrix: list[list[int]],
    labels: list[int],
) -> dict[str, Any]:
    """Analyze patterns in a confusion matrix.

    Args:
        confusion_matrix: NxN confusion matrix.
        labels: Class indices corresponding to rows/columns.

    Returns:
        Dict with confusion patterns, most confused pairs, asymmetry analysis.
    """
    cm = np.array(confusion_matrix)
    n_classes = len(labels)
    idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}

    if cm.shape[0] != n_classes or cm.shape[1] != n_classes:
        raise ValueError(f"Confusion matrix shape {cm.shape} doesn't match {n_classes} labels")

    total = cm.sum()
    if total == 0:
        return {"pairs": [], "asymmetry": []}

    # Find most confused pairs (off-diagonal)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            if cm[i, j] > 0:
                true_name = idx_to_name.get(labels[i], f"class_{labels[i]}")
                pred_name = idx_to_name.get(labels[j], f"class_{labels[j]}")
                pairs.append({
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                    "pct_of_true": round(cm[i, j] / cm[i].sum() * 100, 1) if cm[i].sum() > 0 else 0,
                })

    pairs.sort(key=lambda x: x["count"], reverse=True)

    # Asymmetry analysis: where A->B != B->A
    asymmetry = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            a_to_b = int(cm[i, j])
            b_to_a = int(cm[j, i])
            if a_to_b + b_to_a == 0:
                continue
            ratio = max(a_to_b, b_to_a) / (min(a_to_b, b_to_a) + 1e-6)
            name_i = idx_to_name.get(labels[i], f"class_{labels[i]}")
            name_j = idx_to_name.get(labels[j], f"class_{labels[j]}")
            asymmetry.append({
                "class_a": name_i,
                "class_b": name_j,
                f"{name_i}_to_{name_j}": a_to_b,
                f"{name_j}_to_{name_i}": b_to_a,
                "asymmetry_ratio": round(ratio, 2),
            })

    asymmetry.sort(key=lambda x: x["asymmetry_ratio"], reverse=True)

    # Per-class commission/omission summary
    per_class = {}
    for i, label in enumerate(labels):
        name = idx_to_name.get(label, f"class_{label}")
        row_sum = cm[i].sum()
        col_sum = cm[:, i].sum()
        tp = cm[i, i]
        omission = int(row_sum - tp)
        commission = int(col_sum - tp)
        per_class[name] = {
            "true_positives": int(tp),
            "omission_errors": omission,
            "commission_errors": commission,
            "omission_rate": round(omission / row_sum, 4) if row_sum > 0 else 0,
            "commission_rate": round(commission / col_sum, 4) if col_sum > 0 else 0,
        }

    return {
        "top_confusions": pairs[:10],
        "asymmetry": asymmetry[:5],
        "per_class": per_class,
    }


def compute_spatial_error_density(
    error_points: list[tuple[float, float]],
    correct_points: list[tuple[float, float]],
    grid_size: int = 10,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Compute spatial density of errors on a grid.

    Divides the study area into a grid and computes the error rate
    in each cell to identify error hotspots.

    Args:
        error_points: List of (x, y) coordinates of errors.
        correct_points: List of (x, y) coordinates of correct predictions.
        grid_size: Number of cells per side.
        bounds: (west, south, east, north) or inferred from points.

    Returns:
        Dict with grid error rates, hotspot cells, and summary.
    """
    all_points = error_points + correct_points
    if len(all_points) == 0:
        return {"grid": [], "hotspots": [], "n_cells": 0}

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    if bounds is None:
        bounds = (min(xs), min(ys), max(xs), max(ys))

    west, south, east, north = bounds
    dx = (east - west) / grid_size if east != west else 1
    dy = (north - south) / grid_size if north != south else 1

    # Count errors and totals per cell
    error_grid = np.zeros((grid_size, grid_size), dtype=int)
    total_grid = np.zeros((grid_size, grid_size), dtype=int)

    for x, y in error_points:
        col = min(int((x - west) / dx), grid_size - 1)
        row = min(int((y - south) / dy), grid_size - 1)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            error_grid[row, col] += 1
            total_grid[row, col] += 1

    for x, y in correct_points:
        col = min(int((x - west) / dx), grid_size - 1)
        row = min(int((y - south) / dy), grid_size - 1)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            total_grid[row, col] += 1

    # Error rate grid
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_grid = np.where(total_grid > 0, error_grid / total_grid, -1)

    # Identify hotspots (cells with above-average error rate and enough samples)
    valid_mask = total_grid >= 2
    if valid_mask.any():
        valid_rates = rate_grid[valid_mask]
        mean_rate = valid_rates[valid_rates >= 0].mean() if (valid_rates >= 0).any() else 0
    else:
        mean_rate = 0

    hotspots = []
    for r in range(grid_size):
        for c in range(grid_size):
            if total_grid[r, c] >= 2 and rate_grid[r, c] > mean_rate:
                hotspots.append({
                    "row": r,
                    "col": c,
                    "error_rate": round(float(rate_grid[r, c]), 4),
                    "n_errors": int(error_grid[r, c]),
                    "n_total": int(total_grid[r, c]),
                    "center_x": west + (c + 0.5) * dx,
                    "center_y": south + (r + 0.5) * dy,
                })

    hotspots.sort(key=lambda x: x["error_rate"], reverse=True)

    return {
        "grid_size": grid_size,
        "bounds": list(bounds),
        "error_grid": error_grid.tolist(),
        "total_grid": total_grid.tolist(),
        "rate_grid": np.where(rate_grid >= 0, np.round(rate_grid, 4), -1).tolist(),
        "mean_error_rate": round(float(mean_rate), 4),
        "n_hotspots": len(hotspots),
        "hotspots": hotspots[:10],
    }


def compute_edge_error_rate(
    prediction_path: str | Path,
    reference_path: str | Path,
    class_field: str = "LC_CLASS",
    edge_kernel: int = 3,
    nodata: int = 255,
) -> dict[str, Any]:
    """Analyze whether errors concentrate at class boundaries (edges).

    Compares error rates at edge pixels vs interior pixels to detect
    whether misclassification is driven by mixed pixels at boundaries.

    Args:
        prediction_path: Classification raster path.
        reference_path: Reference points path.
        class_field: Class label column.
        edge_kernel: Kernel size for edge detection.
        nodata: Nodata value.

    Returns:
        Dict with edge vs interior error rates.
    """
    if not (_RASTERIO_AVAILABLE and _GEO_AVAILABLE):
        raise ImportError("rasterio and geopandas required")

    name_to_idx = {k.lower(): v for k, v in CLASS_SCHEMA.items()}
    idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}

    ref_gdf = gpd.read_file(reference_path)

    with rasterio.open(prediction_path) as src:
        data = src.read(1)
        transform = src.transform
        raster_crs = src.crs

        if ref_gdf.crs != raster_crs:
            ref_gdf = ref_gdf.to_crs(raster_crs)

    # Detect edges: pixels adjacent to a different class
    h, w = data.shape
    pad = edge_kernel // 2
    is_edge = np.zeros_like(data, dtype=bool)

    for dr in range(-pad, pad + 1):
        for dc in range(-pad, pad + 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.full_like(data, nodata)
            r_start = max(0, dr)
            r_end = min(h, h + dr)
            c_start = max(0, dc)
            c_end = min(w, w + dc)
            src_r_start = max(0, -dr)
            src_r_end = min(h, h - dr)
            src_c_start = max(0, -dc)
            src_c_end = min(w, w - dc)
            shifted[r_start:r_end, c_start:c_end] = data[src_r_start:src_r_end, src_c_start:src_c_end]
            is_edge |= (data != shifted) & (data != nodata) & (shifted != nodata)

    # Classify each reference point as edge or interior
    edge_correct = 0
    edge_error = 0
    interior_correct = 0
    interior_error = 0

    for i, (ref_val, geom) in enumerate(zip(ref_gdf[class_field], ref_gdf.geometry)):
        try:
            row, col = rowcol(transform, geom.x, geom.y)
        except Exception:
            continue

        if not (0 <= row < h and 0 <= col < w):
            continue

        pred_idx = int(data[row, col])
        if pred_idx == nodata or pred_idx < 0:
            continue

        if isinstance(ref_val, (int, np.integer)):
            true_idx = int(ref_val)
        elif isinstance(ref_val, str):
            true_idx = name_to_idx.get(ref_val.lower().strip())
            if true_idx is None:
                continue
        else:
            continue

        at_edge = bool(is_edge[row, col])
        is_correct = (true_idx == pred_idx)

        if at_edge:
            if is_correct:
                edge_correct += 1
            else:
                edge_error += 1
        else:
            if is_correct:
                interior_correct += 1
            else:
                interior_error += 1

    edge_total = edge_correct + edge_error
    interior_total = interior_correct + interior_error

    return {
        "edge_error_rate": round(edge_error / edge_total, 4) if edge_total > 0 else 0,
        "interior_error_rate": round(interior_error / interior_total, 4) if interior_total > 0 else 0,
        "edge_points": edge_total,
        "interior_points": interior_total,
        "edge_errors": edge_error,
        "interior_errors": interior_error,
        "edge_effect_ratio": (
            round((edge_error / edge_total) / (interior_error / interior_total + 1e-9), 2)
            if edge_total > 0 and interior_total > 0
            else 0
        ),
    }
