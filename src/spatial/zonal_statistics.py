"""Zonal statistics for land cover analysis by spatial zones.

Calculates per-zone class area, percentages, and summary statistics
from classified rasters.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.zonal_statistics")

try:
    import rasterio
    from rasterio.features import geometry_mask
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


def compute_zonal_stats(
    raster_path: str | Path,
    geometry: Any,
    resolution: float = 10.0,
    nodata: int = 255,
    class_names: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute land cover statistics for a single zone.

    Args:
        raster_path: Path to classification raster.
        geometry: Shapely geometry defining the zone.
        resolution: Pixel resolution in meters (for area calculation).
        nodata: Nodata value.
        class_names: Class name to index mapping. Defaults to CLASS_SCHEMA.

    Returns:
        Dict with per-class pixel counts, area (sq m), and percentages.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio required for zonal statistics")

    if class_names is None:
        class_names = CLASS_SCHEMA

    from shapely.geometry import mapping
    geom_json = [mapping(geometry)]

    with rasterio.open(raster_path) as src:
        # Create mask for the zone
        mask = geometry_mask(
            geom_json, transform=src.transform,
            invert=True,  # True inside geometry
            out_shape=(src.height, src.width),
        )
        data = src.read(1)

    # Extract pixels within zone
    zone_pixels = data[mask]
    valid_pixels = zone_pixels[zone_pixels != nodata]

    if len(valid_pixels) == 0:
        return {
            "total_pixels": 0,
            "valid_pixels": 0,
            "classes": {},
            "area_sq_m": 0,
        }

    pixel_area = resolution * resolution
    name_to_idx = class_names
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    # Count pixels per class
    unique, counts = np.unique(valid_pixels, return_counts=True)
    class_stats = {}
    for cls_idx, count in zip(unique, counts):
        cls_name = idx_to_name.get(int(cls_idx), f"class_{cls_idx}")
        area = float(count) * pixel_area
        pct = float(count) / len(valid_pixels) * 100
        class_stats[cls_name] = {
            "class_index": int(cls_idx),
            "pixel_count": int(count),
            "area_sq_m": round(area, 1),
            "area_hectares": round(area / 10000, 2),
            "area_acres": round(area / 4046.86, 2),
            "percentage": round(pct, 2),
        }

    total_area = float(len(valid_pixels)) * pixel_area

    return {
        "total_pixels": int(len(zone_pixels)),
        "valid_pixels": int(len(valid_pixels)),
        "nodata_pixels": int(len(zone_pixels) - len(valid_pixels)),
        "total_area_sq_m": round(total_area, 1),
        "total_area_hectares": round(total_area / 10000, 2),
        "total_area_acres": round(total_area / 4046.86, 2),
        "classes": class_stats,
        "dominant_class": max(class_stats, key=lambda k: class_stats[k]["pixel_count"])
        if class_stats else None,
    }


def compute_layer_stats(
    raster_path: str | Path,
    layer: Any,
    resolution: float = 10.0,
    nodata: int = 255,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute zonal statistics for all features in a boundary layer.

    Args:
        raster_path: Path to classification raster.
        layer: FocusAreaLayer with loaded boundaries.
        resolution: Pixel resolution in meters.
        nodata: Nodata value.
        output_path: Optional path to save CSV summary.

    Returns:
        Dict of {feature_id: zonal_stats}.
    """
    results = {}

    for feature_id in layer.features:
        try:
            geometry = layer.get_geometry(feature_id)
            stats = compute_zonal_stats(
                raster_path, geometry,
                resolution=resolution, nodata=nodata,
            )
            results[feature_id] = stats
            logger.info(
                f"{layer.name}/{feature_id}: "
                f"{stats['valid_pixels']} pixels, "
                f"dominant={stats.get('dominant_class', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"Failed stats for {layer.name}/{feature_id}: {e}")
            results[feature_id] = {"error": str(e)}

    # Save CSV summary if requested
    if output_path is not None:
        _save_stats_csv(results, layer.name, output_path)

    return results


def compute_raster_stats(
    raster_path: str | Path,
    nodata: int = 255,
    resolution: float = 10.0,
) -> dict[str, Any]:
    """Compute statistics for an entire raster (no spatial zones).

    Args:
        raster_path: Path to classification raster.
        nodata: Nodata value.
        resolution: Pixel resolution.

    Returns:
        Raster-wide statistics dict.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio required")

    with rasterio.open(raster_path) as src:
        data = src.read(1)

    valid = data[data != nodata]
    if len(valid) == 0:
        return {"total_pixels": int(data.size), "valid_pixels": 0, "classes": {}}

    pixel_area = resolution * resolution
    idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}

    unique, counts = np.unique(valid, return_counts=True)
    class_stats = {}
    for cls_idx, count in zip(unique, counts):
        cls_name = idx_to_name.get(int(cls_idx), f"class_{cls_idx}")
        area = float(count) * pixel_area
        class_stats[cls_name] = {
            "class_index": int(cls_idx),
            "pixel_count": int(count),
            "area_hectares": round(area / 10000, 2),
            "percentage": round(float(count) / len(valid) * 100, 2),
        }

    return {
        "total_pixels": int(data.size),
        "valid_pixels": int(len(valid)),
        "total_area_hectares": round(float(len(valid)) * pixel_area / 10000, 2),
        "classes": class_stats,
        "dominant_class": max(class_stats, key=lambda k: class_stats[k]["pixel_count"])
        if class_stats else None,
    }


def _save_stats_csv(
    results: dict[str, Any],
    layer_name: str,
    output_path: str | Path,
) -> None:
    """Save zonal statistics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all class names
    all_classes = set()
    for stats in results.values():
        if "classes" in stats:
            all_classes.update(stats["classes"].keys())
    all_classes = sorted(all_classes)

    lines = []
    header = [f"{layer_name}_id", "valid_pixels", "total_area_hectares", "dominant_class"]
    for cls in all_classes:
        header.extend([f"{cls}_pct", f"{cls}_hectares"])
    lines.append(",".join(header))

    for feature_id, stats in results.items():
        if "error" in stats:
            continue
        row = [
            str(feature_id),
            str(stats.get("valid_pixels", 0)),
            str(stats.get("total_area_hectares", 0)),
            str(stats.get("dominant_class", "")),
        ]
        for cls in all_classes:
            cls_data = stats.get("classes", {}).get(cls, {})
            row.append(str(cls_data.get("percentage", 0)))
            row.append(str(cls_data.get("area_hectares", 0)))
        lines.append(",".join(row))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Saved stats CSV: {output_path}")
