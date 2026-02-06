"""Model comparison and spatial agreement metrics.

Computes spatial agreement between multiple land cover maps,
generating agreement maps and comparison statistics.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.comparison_metrics")

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False

from src.config_schema import CLASS_SCHEMA


def compute_spatial_agreement(
    raster_a_path: str | Path,
    raster_b_path: str | Path,
    nodata: int = 255,
) -> dict[str, Any]:
    """Compute pixel-wise agreement between two classification rasters.

    Args:
        raster_a_path: Path to first classification raster.
        raster_b_path: Path to second classification raster.
        nodata: Nodata value.

    Returns:
        Dict with agreement percentage, per-class agreement, confusion matrix.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio required")

    with rasterio.open(raster_a_path) as src_a:
        data_a = src_a.read(1)
        profile_a = src_a.profile

    with rasterio.open(raster_b_path) as src_b:
        data_b = src_b.read(1)

    # Ensure same shape
    if data_a.shape != data_b.shape:
        min_h = min(data_a.shape[0], data_b.shape[0])
        min_w = min(data_a.shape[1], data_b.shape[1])
        data_a = data_a[:min_h, :min_w]
        data_b = data_b[:min_h, :min_w]

    # Valid pixels (both not nodata)
    valid = (data_a != nodata) & (data_b != nodata)
    if valid.sum() == 0:
        return {"agreement_pct": 0, "valid_pixels": 0}

    a_valid = data_a[valid]
    b_valid = data_b[valid]

    agree = a_valid == b_valid
    agreement_pct = float(agree.mean() * 100)

    # Per-class agreement
    idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}
    all_classes = sorted(set(np.unique(a_valid)) | set(np.unique(b_valid)))

    per_class = {}
    for cls in all_classes:
        cls_name = idx_to_name.get(int(cls), f"class_{cls}")
        mask_a = a_valid == cls
        mask_b = b_valid == cls
        # Agreement: both predict this class at same pixels
        both = mask_a & mask_b
        either = mask_a | mask_b
        iou = float(both.sum() / either.sum()) if either.sum() > 0 else 0
        per_class[cls_name] = {
            "class_index": int(cls),
            "count_a": int(mask_a.sum()),
            "count_b": int(mask_b.sum()),
            "agreement_iou": round(iou, 4),
        }

    # Confusion matrix between the two maps
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    cm = sk_confusion_matrix(a_valid, b_valid, labels=all_classes)

    return {
        "agreement_pct": round(agreement_pct, 2),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(data_a.size),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": [int(x) for x in all_classes],
    }


def create_agreement_map(
    raster_a_path: str | Path,
    raster_b_path: str | Path,
    output_path: str | Path,
    nodata: int = 255,
) -> dict[str, Any]:
    """Create a binary agreement map between two classifications.

    Output raster values:
    - 1: agree (same class)
    - 0: disagree (different class)
    - nodata: either input is nodata

    Args:
        raster_a_path: First raster.
        raster_b_path: Second raster.
        output_path: Output agreement raster path.
        nodata: Nodata value.

    Returns:
        Dict with output metadata.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio required")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_a_path) as src_a:
        data_a = src_a.read(1)
        profile = src_a.profile.copy()

    with rasterio.open(raster_b_path) as src_b:
        data_b = src_b.read(1)

    # Handle shape mismatch
    if data_a.shape != data_b.shape:
        min_h = min(data_a.shape[0], data_b.shape[0])
        min_w = min(data_a.shape[1], data_b.shape[1])
        data_a = data_a[:min_h, :min_w]
        data_b = data_b[:min_h, :min_w]
        profile.update({"height": min_h, "width": min_w})

    # Build agreement map
    valid = (data_a != nodata) & (data_b != nodata)
    agreement = np.full(data_a.shape, nodata, dtype=np.uint8)
    agreement[valid] = (data_a[valid] == data_b[valid]).astype(np.uint8)

    profile.update({
        "dtype": "uint8",
        "count": 1,
        "nodata": nodata,
        "compress": "lzw",
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(agreement[np.newaxis, :, :])

    agree_pct = float(agreement[valid].mean() * 100) if valid.any() else 0

    return {
        "output": str(output_path),
        "agreement_pct": round(agree_pct, 2),
        "shape": list(agreement.shape),
    }


def compare_with_existing(
    new_raster_path: str | Path,
    existing_raster_path: str | Path,
    output_dir: str | Path,
    model_name: str = "new",
    existing_name: str = "existing",
) -> dict[str, Any]:
    """Full comparison between a new model and an existing LCAnalysis2026 model.

    Args:
        new_raster_path: New model's classification raster.
        existing_raster_path: Existing model's raster.
        output_dir: Output directory for comparison artifacts.
        model_name: Name for the new model.
        existing_name: Name for the existing model.

    Returns:
        Dict with agreement metrics and output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spatial agreement
    agreement = compute_spatial_agreement(new_raster_path, existing_raster_path)

    # Agreement map
    agree_map = create_agreement_map(
        new_raster_path, existing_raster_path,
        output_dir / "agreement_map.tif",
    )

    # Save comparison report
    report = {
        "model_a": model_name,
        "model_b": existing_name,
        "agreement": agreement,
        "agreement_map": agree_map,
    }

    with open(output_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save confusion matrix CSV
    if "confusion_matrix" in agreement:
        idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}
        labels = agreement["labels"]
        cm = agreement["confusion_matrix"]
        header = f"{model_name}\\{existing_name}," + ",".join(
            idx_to_name.get(l, str(l)) for l in labels
        )
        lines = [header]
        for i, row in enumerate(cm):
            name = idx_to_name.get(labels[i], str(labels[i]))
            lines.append(f"{name}," + ",".join(str(x) for x in row))
        with open(output_dir / "confusion_matrix.csv", "w") as f:
            f.write("\n".join(lines) + "\n")

    logger.info(
        f"Comparison {model_name} vs {existing_name}: "
        f"{agreement['agreement_pct']}% agreement"
    )

    return report
