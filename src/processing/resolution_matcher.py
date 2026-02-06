"""Resolution matching and alignment for multi-resolution rasters.

Resamples and aligns rasters of different resolutions to a common grid,
handling CRS differences and extent alignment.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.resolution_matcher")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False


class ResolutionMatcher:
    """Align rasters of different resolutions to a common grid.

    Supports resampling classification rasters (nearest neighbor)
    and probability rasters (bilinear/cubic) to a target resolution.
    """

    def __init__(self, nodata: int = 255):
        self.nodata = nodata

    def get_raster_info(self, path: str | Path) -> dict[str, Any]:
        """Get resolution, CRS, and extent info for a raster.

        Args:
            path: Path to raster file.

        Returns:
            Dict with resolution, crs, bounds, shape, transform.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required")

        with rasterio.open(path) as src:
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            return {
                "path": str(path),
                "resolution": (res_x, res_y),
                "crs": str(src.crs),
                "bounds": src.bounds,
                "shape": (src.height, src.width),
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "transform": src.transform,
            }

    def compute_common_extent(
        self,
        raster_paths: list[str | Path],
        mode: str = "intersection",
    ) -> tuple[float, float, float, float]:
        """Compute the common spatial extent of multiple rasters.

        Args:
            raster_paths: List of raster file paths.
            mode: "intersection" for overlap area, "union" for full extent.

        Returns:
            Tuple of (west, south, east, north).
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required")

        bounds_list = []
        for path in raster_paths:
            with rasterio.open(path) as src:
                bounds_list.append(src.bounds)

        if mode == "intersection":
            west = max(b.left for b in bounds_list)
            south = max(b.bottom for b in bounds_list)
            east = min(b.right for b in bounds_list)
            north = min(b.top for b in bounds_list)
            if west >= east or south >= north:
                raise ValueError("Rasters do not overlap")
        else:  # union
            west = min(b.left for b in bounds_list)
            south = min(b.bottom for b in bounds_list)
            east = max(b.right for b in bounds_list)
            north = max(b.top for b in bounds_list)

        return (west, south, east, north)

    def resample_to_target(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_resolution: float,
        target_crs: str | None = None,
        target_bounds: tuple[float, float, float, float] | None = None,
        resampling: str = "nearest",
    ) -> dict[str, Any]:
        """Resample a raster to a target resolution.

        Args:
            input_path: Input raster path.
            output_path: Output raster path.
            target_resolution: Target pixel size in map units.
            target_crs: Target CRS (default: same as input).
            target_bounds: Target extent (west, south, east, north).
            resampling: Resampling method ("nearest", "bilinear", "cubic").

        Returns:
            Dict with output metadata.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resample_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        resample_method = resample_map.get(resampling, Resampling.nearest)

        with rasterio.open(input_path) as src:
            src_crs = src.crs
            dst_crs = rasterio.crs.CRS.from_string(target_crs) if target_crs else src_crs

            if target_bounds is not None:
                west, south, east, north = target_bounds
            else:
                if str(src_crs) != str(dst_crs):
                    from rasterio.warp import transform_bounds
                    west, south, east, north = transform_bounds(
                        src_crs, dst_crs,
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top,
                    )
                else:
                    west, south, east, north = (
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top,
                    )

            # Compute target dimensions
            dst_width = max(1, int(round((east - west) / target_resolution)))
            dst_height = max(1, int(round((north - south) / target_resolution)))
            dst_transform = from_bounds(west, south, east, north, dst_width, dst_height)

            dst_profile = src.profile.copy()
            dst_profile.update({
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "nodata": self.nodata,
            })

            with rasterio.open(output_path, "w", **dst_profile) as dst:
                for band_idx in range(1, src.count + 1):
                    src_data = src.read(band_idx)
                    dst_data = np.full((dst_height, dst_width), self.nodata, dtype=src_data.dtype)
                    reproject(
                        source=src_data,
                        destination=dst_data,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resample_method,
                        src_nodata=self.nodata,
                        dst_nodata=self.nodata,
                    )
                    dst.write(dst_data, band_idx)

        return {
            "output": str(output_path),
            "resolution": target_resolution,
            "shape": (dst_height, dst_width),
            "bounds": (west, south, east, north),
            "crs": str(dst_crs),
        }

    def align_rasters(
        self,
        raster_paths: dict[str, str | Path],
        output_dir: str | Path,
        target_resolution: float | None = None,
        target_crs: str | None = None,
        extent_mode: str = "intersection",
        resampling: str = "nearest",
    ) -> dict[str, dict[str, Any]]:
        """Align multiple rasters to a common grid.

        Args:
            raster_paths: Dict of {name: path} for input rasters.
            output_dir: Directory for aligned outputs.
            target_resolution: Target resolution (default: finest input).
            target_crs: Target CRS (default: first raster's CRS).
            extent_mode: "intersection" or "union".
            resampling: Resampling method.

        Returns:
            Dict of {name: {output metadata}}.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get info for all rasters
        infos = {}
        for name, path in raster_paths.items():
            infos[name] = self.get_raster_info(path)

        # Determine target CRS
        if target_crs is None:
            first_name = next(iter(infos))
            target_crs = infos[first_name]["crs"]

        # Determine target resolution (finest by default)
        if target_resolution is None:
            target_resolution = min(
                min(info["resolution"]) for info in infos.values()
            )

        # Compute common extent
        paths_list = list(raster_paths.values())
        target_bounds = self.compute_common_extent(paths_list, mode=extent_mode)

        # Resample each raster
        results = {}
        for name, path in raster_paths.items():
            output_path = output_dir / f"{name}_aligned.tif"
            result = self.resample_to_target(
                path, output_path,
                target_resolution=target_resolution,
                target_crs=target_crs,
                target_bounds=target_bounds,
                resampling=resampling,
            )
            result["source"] = str(path)
            result["source_resolution"] = infos[name]["resolution"]
            results[name] = result

        logger.info(
            f"Aligned {len(results)} rasters to {target_resolution}m, "
            f"extent={extent_mode}"
        )
        return results

    def load_aligned_arrays(
        self,
        aligned_paths: dict[str, str | Path],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Load aligned raster data into numpy arrays.

        Args:
            aligned_paths: Dict of {name: aligned_raster_path}.

        Returns:
            Tuple of (data dict, metadata dict).
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required")

        arrays = {}
        metadata = {}

        for name, path in aligned_paths.items():
            with rasterio.open(path) as src:
                arrays[name] = src.read(1)
                if not metadata:
                    metadata = {
                        "transform": src.transform,
                        "crs": src.crs,
                        "shape": (src.height, src.width),
                        "nodata": src.nodata or self.nodata,
                        "profile": src.profile.copy(),
                    }

        return arrays, metadata
