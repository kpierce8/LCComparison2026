"""Boundary processor for clipping rasters to spatial features.

Clips prediction rasters to county/WRIA/RMZ boundaries and
organizes outputs by spatial unit.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.boundary_processor")

try:
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from rasterio.features import geometry_mask
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import mapping
    _GEO_AVAILABLE = True
except ImportError:
    gpd = None
    _GEO_AVAILABLE = False


class BoundaryProcessor:
    """Clip rasters to boundary features and organize outputs."""

    def __init__(self, nodata: int = 255):
        """
        Args:
            nodata: Nodata value for clipped outputs.
        """
        self.nodata = nodata

    def clip_raster(
        self,
        raster_path: str | Path,
        geometry: Any,
        output_path: str | Path,
        crop: bool = True,
    ) -> dict[str, Any]:
        """Clip a raster to a geometry boundary.

        Args:
            raster_path: Path to input GeoTIFF.
            geometry: Shapely geometry to clip to.
            output_path: Path for clipped output.
            crop: Whether to crop to the geometry extent.

        Returns:
            Dict with clip metadata (shape, bounds, valid_pixels).
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required for boundary processing")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        geom_json = [mapping(geometry)]

        with rasterio.open(raster_path) as src:
            clipped, transform = rasterio_mask(
                src, geom_json, crop=crop, nodata=self.nodata,
            )
            profile = src.profile.copy()

        profile.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform,
            "nodata": self.nodata,
            "compress": "lzw",
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)

        # Compute stats
        valid_mask = clipped[0] != self.nodata
        metadata = {
            "output": str(output_path),
            "shape": list(clipped.shape),
            "valid_pixels": int(valid_mask.sum()),
            "total_pixels": int(valid_mask.size),
            "coverage_pct": float(valid_mask.sum() / valid_mask.size * 100),
        }

        return metadata

    def process_layer(
        self,
        raster_path: str | Path,
        layer: Any,
        output_dir: str | Path,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Clip a raster to all features in a boundary layer.

        Args:
            raster_path: Path to prediction raster.
            layer: FocusAreaLayer instance.
            output_dir: Base output directory.
            model_name: Model name for file naming.

        Returns:
            Dict with per-feature clip results.
        """
        output_dir = Path(output_dir) / layer.name
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for feature_id in layer.features:
            try:
                geometry = layer.get_geometry(feature_id)

                # Sanitize feature ID for filename
                safe_id = str(feature_id).replace(" ", "_").replace("/", "-")
                out_path = output_dir / safe_id / f"{model_name}_classification.tif"

                clip_meta = self.clip_raster(raster_path, geometry, out_path)
                results[feature_id] = {
                    "status": "success",
                    **clip_meta,
                }

                logger.info(
                    f"Clipped {layer.name}/{feature_id}: "
                    f"{clip_meta['valid_pixels']} valid pixels"
                )

            except Exception as e:
                logger.warning(f"Failed to clip {layer.name}/{feature_id}: {e}")
                results[feature_id] = {
                    "status": "failed",
                    "error": str(e),
                }

        logger.info(
            f"Processed {layer.name}: "
            f"{sum(1 for r in results.values() if r['status'] == 'success')}"
            f"/{len(results)} features"
        )

        return results

    def clip_to_bbox(
        self,
        raster_path: str | Path,
        bbox: dict[str, float],
        output_path: str | Path,
    ) -> dict[str, Any]:
        """Clip a raster to a bounding box.

        Args:
            raster_path: Input raster path.
            bbox: Dict with west, south, east, north.
            output_path: Output path.

        Returns:
            Clip metadata.
        """
        if not _GEO_AVAILABLE:
            raise ImportError("geopandas/shapely required")

        from shapely.geometry import box as shapely_box
        geometry = shapely_box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
        return self.clip_raster(raster_path, geometry, output_path)
