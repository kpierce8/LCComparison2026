"""Tile mosaicking for creating final land cover raster outputs.

Assembles per-tile predictions into seamless GeoTIFF rasters:
- Hard classification map (single band, class indices)
- Probability map (multi-band, per-class probabilities)
- Confidence map (single band, max probability)
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.mosaic")

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False


class TileMosaicker:
    """Mosaic tile predictions into final raster products."""

    def __init__(
        self,
        tile_size: int = 256,
        resolution: float = 10.0,
        crs: str = "EPSG:32610",
        nodata: int = 255,
    ):
        """
        Args:
            tile_size: Tile size in pixels.
            resolution: Pixel resolution in meters.
            crs: Coordinate reference system.
            nodata: Nodata value for output rasters.
        """
        self.tile_size = tile_size
        self.resolution = resolution
        self.crs = crs
        self.nodata = nodata

    def create_classification_tiles(
        self,
        predictions: dict[str, int],
        tile_bboxes: dict[str, dict[str, float]],
        output_dir: str | Path,
    ) -> list[Path]:
        """Create per-tile classification GeoTIFFs from predictions.

        Args:
            predictions: Dict of {tile_id: class_index}.
            tile_bboxes: Dict of {tile_id: {west, south, east, north}}.
            output_dir: Directory for output tiles.

        Returns:
            List of created tile paths.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required for mosaicking")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        created = []

        for tile_id, class_idx in predictions.items():
            if tile_id not in tile_bboxes:
                logger.warning(f"No bbox for tile {tile_id}, skipping")
                continue

            bbox = tile_bboxes[tile_id]
            data = np.full(
                (1, self.tile_size, self.tile_size),
                class_idx, dtype=np.uint8,
            )

            out_path = output_dir / f"{tile_id}_class.tif"
            transform = from_bounds(
                bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                self.tile_size, self.tile_size,
            )

            profile = {
                "driver": "GTiff",
                "dtype": "uint8",
                "count": 1,
                "height": self.tile_size,
                "width": self.tile_size,
                "crs": self.crs,
                "transform": transform,
                "nodata": self.nodata,
                "compress": "lzw",
            }

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

            created.append(out_path)

        logger.info(f"Created {len(created)} classification tiles")
        return created

    def create_probability_tiles(
        self,
        probabilities: dict[str, np.ndarray],
        tile_bboxes: dict[str, dict[str, float]],
        output_dir: str | Path,
        n_classes: int = 7,
    ) -> list[Path]:
        """Create per-tile probability GeoTIFFs.

        Args:
            probabilities: Dict of {tile_id: probability_array (n_classes,)}.
            tile_bboxes: Dict of {tile_id: bbox_dict}.
            output_dir: Output directory.
            n_classes: Number of classes.

        Returns:
            List of created tile paths.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required for mosaicking")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        created = []

        for tile_id, proba in probabilities.items():
            if tile_id not in tile_bboxes:
                continue

            bbox = tile_bboxes[tile_id]

            # Expand probability vector to tile-sized raster
            data = np.zeros(
                (n_classes, self.tile_size, self.tile_size),
                dtype=np.float32,
            )
            for c in range(min(len(proba), n_classes)):
                data[c, :, :] = proba[c]

            out_path = output_dir / f"{tile_id}_prob.tif"
            transform = from_bounds(
                bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                self.tile_size, self.tile_size,
            )

            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": n_classes,
                "height": self.tile_size,
                "width": self.tile_size,
                "crs": self.crs,
                "transform": transform,
                "compress": "lzw",
            }

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

            created.append(out_path)

        logger.info(f"Created {len(created)} probability tiles")
        return created

    def mosaic_tiles(
        self,
        tile_paths: list[str | Path],
        output_path: str | Path,
    ) -> dict[str, Any]:
        """Merge tile GeoTIFFs into a single mosaic.

        Args:
            tile_paths: List of tile GeoTIFF paths.
            output_path: Output mosaic path.

        Returns:
            Dict with mosaic metadata.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required for mosaicking")

        if not tile_paths:
            raise ValueError("No tile paths provided for mosaicking")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open all tiles
        datasets = []
        for tp in tile_paths:
            datasets.append(rasterio.open(tp))

        try:
            mosaic_data, mosaic_transform = merge(datasets)
        finally:
            for ds in datasets:
                ds.close()

        # Write mosaic
        profile = datasets[0].profile.copy()
        profile.update({
            "height": mosaic_data.shape[1],
            "width": mosaic_data.shape[2],
            "transform": mosaic_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic_data)

        metadata = {
            "output": str(output_path),
            "n_tiles": len(tile_paths),
            "shape": list(mosaic_data.shape),
            "dtype": str(mosaic_data.dtype),
            "crs": str(profile.get("crs")),
        }

        logger.info(
            f"Mosaicked {len(tile_paths)} tiles -> "
            f"{output_path} ({mosaic_data.shape})"
        )

        return metadata

    def create_confidence_tiles(
        self,
        probabilities: dict[str, np.ndarray],
        tile_bboxes: dict[str, dict[str, float]],
        output_dir: str | Path,
    ) -> list[Path]:
        """Create per-tile confidence GeoTIFFs (max probability).

        Args:
            probabilities: Dict of {tile_id: probability_array}.
            tile_bboxes: Dict of {tile_id: bbox_dict}.
            output_dir: Output directory.

        Returns:
            List of created tile paths.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required for mosaicking")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        created = []

        for tile_id, proba in probabilities.items():
            if tile_id not in tile_bboxes:
                continue

            bbox = tile_bboxes[tile_id]
            confidence = float(np.max(proba))
            data = np.full(
                (1, self.tile_size, self.tile_size),
                confidence, dtype=np.float32,
            )

            out_path = output_dir / f"{tile_id}_conf.tif"
            transform = from_bounds(
                bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                self.tile_size, self.tile_size,
            )

            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "height": self.tile_size,
                "width": self.tile_size,
                "crs": self.crs,
                "transform": transform,
                "compress": "lzw",
            }

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

            created.append(out_path)

        logger.info(f"Created {len(created)} confidence tiles")
        return created

    def generate_all_products(
        self,
        predictions: dict[str, int],
        probabilities: dict[str, np.ndarray],
        tile_bboxes: dict[str, dict[str, float]],
        output_dir: str | Path,
        model_name: str = "model",
        n_classes: int = 7,
    ) -> dict[str, Any]:
        """Generate all output products: classification, probability, confidence.

        Args:
            predictions: {tile_id: class_index}.
            probabilities: {tile_id: probability_array}.
            tile_bboxes: {tile_id: bbox_dict}.
            output_dir: Base output directory.
            model_name: Model name for file naming.
            n_classes: Number of classes.

        Returns:
            Dict with paths to all generated products.
        """
        output_dir = Path(output_dir)
        tiles_dir = output_dir / "tiles"

        # Create per-tile rasters
        class_tiles = self.create_classification_tiles(
            predictions, tile_bboxes, tiles_dir / "classification",
        )
        prob_tiles = self.create_probability_tiles(
            probabilities, tile_bboxes, tiles_dir / "probability", n_classes,
        )
        conf_tiles = self.create_confidence_tiles(
            probabilities, tile_bboxes, tiles_dir / "confidence",
        )

        products = {"tile_counts": {
            "classification": len(class_tiles),
            "probability": len(prob_tiles),
            "confidence": len(conf_tiles),
        }}

        # Mosaic each product
        if class_tiles:
            class_path = output_dir / f"{model_name}_classification.tif"
            products["classification"] = self.mosaic_tiles(class_tiles, class_path)

        if conf_tiles:
            conf_path = output_dir / f"{model_name}_confidence.tif"
            products["confidence"] = self.mosaic_tiles(conf_tiles, conf_path)

        # Probability mosaic (large, optional)
        if prob_tiles:
            prob_path = output_dir / f"{model_name}_probability.tif"
            products["probability"] = self.mosaic_tiles(prob_tiles, prob_path)

        # Save metadata
        meta_path = output_dir / f"{model_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(products, f, indent=2, default=str)

        logger.info(f"Generated all products for {model_name} in {output_dir}")
        return products
