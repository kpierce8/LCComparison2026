"""Handle local rasters and custom orthomosaics.

Reads local GeoTIFF files, handles CRS reprojection, band selection,
and tiling into the standard tile grid.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.local_imagery")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.windows import from_bounds as window_from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False


class LocalImageryHandler:
    """Handle local raster imagery for the comparison pipeline.

    Reads GeoTIFF files, extracts tiles matching the tile grid,
    handles CRS reprojection and band selection.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.data_dir = Path(self.config.get("data_dir", "data/tiles/custom"))
        self.file_pattern = self.config.get("file_pattern", "*.tif")

    def list_files(self, data_dir: str | Path | None = None) -> list[Path]:
        """List raster files matching the configured pattern.

        Args:
            data_dir: Directory to scan. Defaults to configured data_dir.

        Returns:
            Sorted list of matching file paths.
        """
        search_dir = Path(data_dir) if data_dir else self.data_dir
        if not search_dir.exists():
            logger.warning(f"Data directory not found: {search_dir}")
            return []

        files = sorted(search_dir.glob(self.file_pattern))
        # Also check for common GeoTIFF extensions
        if not files:
            for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
                files.extend(search_dir.glob(ext))
            files = sorted(set(files))

        logger.info(f"Found {len(files)} raster files in {search_dir}")
        return files

    def read_raster(
        self,
        path: str | Path,
        bands: list[int] | None = None,
    ) -> dict[str, Any]:
        """Read a raster file and return data with metadata.

        Args:
            path: Path to raster file.
            bands: 1-based band indices to read. None = all bands.

        Returns:
            Dict with keys: data (ndarray), profile, bounds, crs, transform.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for local imagery handling")

        path = Path(path)
        with rasterio.open(path) as src:
            if bands is None:
                bands = list(range(1, src.count + 1))

            data = src.read(bands)
            profile = src.profile.copy()
            return {
                "data": data,
                "profile": profile,
                "bounds": src.bounds,
                "crs": str(src.crs),
                "transform": src.transform,
                "shape": data.shape,
                "nodata": src.nodata,
                "path": str(path),
            }

    def get_raster_info(self, path: str | Path) -> dict[str, Any]:
        """Get metadata about a raster file without reading pixel data.

        Args:
            path: Path to raster file.

        Returns:
            Dict with metadata.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for local imagery handling")

        path = Path(path)
        with rasterio.open(path) as src:
            return {
                "path": str(path),
                "crs": str(src.crs),
                "bounds": {
                    "west": src.bounds.left,
                    "south": src.bounds.bottom,
                    "east": src.bounds.right,
                    "north": src.bounds.top,
                },
                "resolution": (src.res[0], src.res[1]),
                "shape": (src.height, src.width),
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "driver": src.driver,
            }

    def extract_tile(
        self,
        raster_path: str | Path,
        bbox: dict[str, float],
        output_path: str | Path,
        target_crs: str | None = None,
        target_resolution: float | None = None,
        bands: list[int] | None = None,
        tile_size: int = 256,
    ) -> dict[str, Any]:
        """Extract a tile from a raster file.

        Handles CRS reprojection and resampling if needed.

        Args:
            raster_path: Source raster path.
            bbox: Target tile bbox (west, south, east, north).
            output_path: Where to save the tile.
            target_crs: Target CRS. None = use source CRS.
            target_resolution: Target resolution in CRS units. None = native.
            bands: 1-based band indices. None = all.
            tile_size: Output tile size in pixels.

        Returns:
            Dict with tile metadata.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for local imagery handling")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(raster_path) as src:
            if bands is None:
                bands = list(range(1, src.count + 1))

            src_crs = src.crs
            dst_crs = rasterio.crs.CRS.from_string(target_crs) if target_crs else src_crs

            if target_resolution:
                dst_transform = from_bounds(
                    bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                    tile_size, tile_size,
                )
                dst_width = tile_size
                dst_height = tile_size
            else:
                # Calculate from bbox and source resolution
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, dst_crs,
                    src.width, src.height,
                    left=bbox["west"], bottom=bbox["south"],
                    right=bbox["east"], top=bbox["north"],
                )

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                crs=dst_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                count=len(bands),
                compress="lzw",
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                for i, band_idx in enumerate(bands, start=1):
                    src_data = src.read(band_idx)
                    dst_data = np.empty((dst_height, dst_width), dtype=src_data.dtype)

                    reproject(
                        source=src_data,
                        destination=dst_data,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
                    dst.write(dst_data, i)

        logger.info(f"Extracted tile: {output_path}")
        return {
            "path": str(output_path),
            "crs": str(dst_crs),
            "bounds": bbox,
            "shape": (len(bands), dst_height, dst_width),
        }

    def reproject_raster(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_crs: str,
        target_resolution: float | None = None,
    ) -> dict[str, Any]:
        """Reproject a raster to a new CRS.

        Args:
            input_path: Source raster path.
            output_path: Output raster path.
            target_crs: Target CRS string.
            target_resolution: Target resolution. None = auto-compute.

        Returns:
            Dict with output metadata.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for local imagery handling")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            dst_crs = rasterio.crs.CRS.from_string(target_crs)

            if target_resolution:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height,
                    *src.bounds, resolution=target_resolution,
                )
            else:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds,
                )

            profile = src.profile.copy()
            profile.update(
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                compress="lzw",
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )

        logger.info(f"Reprojected: {input_path} -> {output_path} ({target_crs})")
        return self.get_raster_info(output_path)
