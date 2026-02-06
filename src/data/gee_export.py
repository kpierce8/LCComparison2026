"""Google Earth Engine export with cloud/drive strategy and graceful degradation.

Supports Sentinel-2 and NAIP export with batch task management,
retry logic, and tile manager integration.
"""

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("lccomparison.gee_export")

# Graceful import
try:
    import ee
    _EE_AVAILABLE = True
except ImportError:
    ee = None
    _EE_AVAILABLE = False

try:
    import geemap
    _GEEMAP_AVAILABLE = True
except ImportError:
    geemap = None
    _GEEMAP_AVAILABLE = False


class GEEExporter:
    """Google Earth Engine imagery exporter.

    Supports both Drive and Cloud Storage export strategies with
    batch task management and graceful degradation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._authenticated = False
        self._check_availability()

        # Export settings from config
        self.strategy = self.config.get("strategy", "drive")
        self.drive_folder = self.config.get("drive_folder", "LCComparison2026_exports")
        self.cloud_bucket = self.config.get("cloud_bucket")
        self.max_pixels = int(self.config.get("max_pixels", 1e9))
        self.file_format = self.config.get("file_format", "GeoTIFF")

        # Task tracking
        self._active_tasks: dict[str, dict[str, Any]] = {}

    def _check_availability(self) -> None:
        """Check if GEE is available and authenticated."""
        if not _EE_AVAILABLE:
            logger.warning("earthengine-api not installed. GEE export unavailable.")
            return

        try:
            ee.Initialize()
            self._authenticated = True
            logger.info("GEE authenticated successfully.")
        except Exception as e:
            logger.warning(f"GEE authentication failed: {e}")
            self._authenticated = False

    def is_available(self) -> bool:
        """Check if GEE export is available (installed + authenticated)."""
        return _EE_AVAILABLE and self._authenticated

    def _require_available(self) -> None:
        """Raise if GEE is not available."""
        if not self.is_available():
            raise RuntimeError(
                "GEE is not available. " + self.get_setup_instructions()
            )

    def _build_sentinel2_composite(
        self,
        bbox: dict[str, float],
        date_range: tuple[str, str],
        bands: list[str] | None = None,
        cloud_filter: int = 20,
    ) -> Any:
        """Build a cloud-free Sentinel-2 composite.

        Args:
            bbox: Dict with west, south, east, north.
            date_range: (start_date, end_date) as YYYY-MM-DD strings.
            bands: Band names. Defaults to B2,B3,B4,B8,B11,B12.
            cloud_filter: Max cloud cover percentage.

        Returns:
            ee.Image composite.
        """
        if bands is None:
            bands = ["B2", "B3", "B4", "B8", "B11", "B12"]

        region = ee.Geometry.Rectangle(
            [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
        )

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(date_range[0], date_range[1])
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
        )

        # Cloud masking using SCL band
        def mask_clouds(image):
            scl = image.select("SCL")
            # Keep: vegetation(4), bare(5), water(6), unclassified(7)
            mask = scl.gte(4).And(scl.lte(7))
            return image.updateMask(mask)

        composite = collection.map(mask_clouds).median().select(bands)
        return composite.clip(region)

    def _build_naip_mosaic(
        self,
        bbox: dict[str, float],
        year: int = 2023,
        bands: list[str] | None = None,
    ) -> Any:
        """Build a NAIP mosaic for a bounding box.

        Args:
            bbox: Dict with west, south, east, north.
            year: NAIP collection year.
            bands: Band names. Defaults to R,G,B,N.

        Returns:
            ee.Image mosaic.
        """
        if bands is None:
            bands = ["R", "G", "B", "N"]

        region = ee.Geometry.Rectangle(
            [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
        )

        collection = (
            ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(region)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select(bands)
        )

        return collection.mosaic().clip(region)

    def export_sentinel2(
        self,
        bbox: dict[str, float],
        date_range: tuple[str, str],
        output_path: str,
        scale: int = 10,
        bands: list[str] | None = None,
        cloud_filter: int = 20,
        description: str | None = None,
        crs: str = "EPSG:32610",
    ) -> dict[str, Any]:
        """Export Sentinel-2 composite for a bounding box.

        Args:
            bbox: Dict with west, south, east, north.
            date_range: (start_date, end_date) strings.
            output_path: Output file path or Drive/GCS prefix.
            scale: Export resolution in meters.
            bands: Band names to export.
            cloud_filter: Max cloud cover percentage.
            description: Task description.
            crs: Target coordinate reference system.

        Returns:
            Dict with task info including task_id.
        """
        self._require_available()

        composite = self._build_sentinel2_composite(bbox, date_range, bands, cloud_filter)
        region = ee.Geometry.Rectangle(
            [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
        )

        if description is None:
            description = f"S2_{Path(output_path).stem}"

        return self._start_export(
            image=composite,
            region=region,
            description=description,
            output_path=output_path,
            scale=scale,
            crs=crs,
        )

    def export_naip(
        self,
        bbox: dict[str, float],
        year: int = 2023,
        output_path: str = "",
        scale: int = 1,
        bands: list[str] | None = None,
        description: str | None = None,
        crs: str = "EPSG:32610",
    ) -> dict[str, Any]:
        """Export NAIP imagery for a bounding box.

        Args:
            bbox: Dict with west, south, east, north.
            year: NAIP year.
            output_path: Output file path.
            scale: Export resolution in meters.
            bands: Band names.
            description: Task description.
            crs: Target CRS.

        Returns:
            Dict with task info.
        """
        self._require_available()

        mosaic = self._build_naip_mosaic(bbox, year, bands)
        region = ee.Geometry.Rectangle(
            [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
        )

        if description is None:
            description = f"NAIP_{Path(output_path).stem}"

        return self._start_export(
            image=mosaic,
            region=region,
            description=description,
            output_path=output_path,
            scale=scale,
            crs=crs,
        )

    def _start_export(
        self,
        image: Any,
        region: Any,
        description: str,
        output_path: str,
        scale: int,
        crs: str,
    ) -> dict[str, Any]:
        """Start a GEE export task using configured strategy.

        Returns:
            Dict with task_id, status, and metadata.
        """
        # Sanitize description for GEE (alphanumeric and underscores only)
        safe_desc = "".join(c if c.isalnum() or c == "_" else "_" for c in description)[:100]

        if self.strategy == "cloud_storage" and self.cloud_bucket:
            task = ee.batch.Export.image.toCloudStorage(
                image=image,
                description=safe_desc,
                bucket=self.cloud_bucket.replace("gs://", "").split("/")[0],
                fileNamePrefix=output_path,
                region=region,
                scale=scale,
                crs=crs,
                maxPixels=self.max_pixels,
                fileFormat=self.file_format,
            )
        else:
            # Default to Drive export
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=safe_desc,
                folder=self.drive_folder,
                fileNamePrefix=Path(output_path).stem,
                region=region,
                scale=scale,
                crs=crs,
                maxPixels=self.max_pixels,
                fileFormat=self.file_format,
            )

        task.start()
        task_id = task.status()["id"]

        task_info = {
            "task_id": task_id,
            "description": safe_desc,
            "status": "SUBMITTED",
            "strategy": self.strategy,
            "output_path": output_path,
            "scale": scale,
            "crs": crs,
        }
        self._active_tasks[task_id] = task_info
        logger.info(f"Export started: {safe_desc} (task_id={task_id})")
        return task_info

    def check_export_status(self, task_id: str) -> dict[str, Any]:
        """Check the status of a GEE export task.

        Args:
            task_id: GEE task identifier.

        Returns:
            Dict with status info.
        """
        if not self.is_available():
            return {"status": "unavailable", "task_id": task_id}

        try:
            status = ee.data.getTaskStatus(task_id)
            if status:
                state = status[0].get("state", "UNKNOWN")
                result = {
                    "task_id": task_id,
                    "status": state,
                    "description": status[0].get("description", ""),
                }
                if state == "FAILED":
                    result["error"] = status[0].get("error_message", "Unknown error")
                return result
        except Exception as e:
            logger.warning(f"Error checking task {task_id}: {e}")
            return {"task_id": task_id, "status": "ERROR", "error": str(e)}

        return {"task_id": task_id, "status": "UNKNOWN"}

    def check_all_tasks(self) -> list[dict[str, Any]]:
        """Check status of all tracked export tasks.

        Returns:
            List of task status dicts.
        """
        results = []
        for task_id in list(self._active_tasks):
            status = self.check_export_status(task_id)
            self._active_tasks[task_id].update(status)
            results.append(self._active_tasks[task_id])
        return results

    def wait_for_tasks(
        self,
        task_ids: list[str] | None = None,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> list[dict[str, Any]]:
        """Wait for export tasks to complete.

        Args:
            task_ids: Specific tasks to wait for. None = all active tasks.
            poll_interval: Seconds between status checks.
            timeout: Max seconds to wait.

        Returns:
            Final status for each task.
        """
        if task_ids is None:
            task_ids = list(self._active_tasks.keys())

        if not task_ids:
            return []

        start = time.time()
        terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}

        while time.time() - start < timeout:
            all_done = True
            for tid in task_ids:
                status = self.check_export_status(tid)
                if status["status"] not in terminal_states:
                    all_done = False

            if all_done:
                break

            remaining = len([
                tid for tid in task_ids
                if self.check_export_status(tid)["status"] not in terminal_states
            ])
            logger.info(f"Waiting for {remaining} task(s)... ({int(time.time() - start)}s elapsed)")
            time.sleep(poll_interval)

        return [self.check_export_status(tid) for tid in task_ids]

    def export_tiles(
        self,
        tiles: list[dict[str, Any]],
        source: str = "sentinel2",
        date_range: tuple[str, str] = ("2024-06-01", "2024-09-30"),
        scale: int = 10,
        crs: str = "EPSG:32610",
        output_dir: str = "data/tiles",
        max_concurrent: int = 20,
    ) -> list[dict[str, Any]]:
        """Export a batch of tiles.

        Args:
            tiles: List of tile dicts with tile_id and bbox keys.
            source: Imagery source (sentinel2 or naip).
            date_range: Date range for composite.
            scale: Export resolution.
            crs: Target CRS.
            output_dir: Output directory prefix.
            max_concurrent: Max concurrent GEE tasks.

        Returns:
            List of task info dicts.
        """
        self._require_available()
        results = []

        for tile in tiles:
            bbox = tile["bbox"]
            tile_id = tile["tile_id"]
            output_path = f"{output_dir}/{source}/{tile_id}"

            if source == "naip":
                year = int(date_range[0][:4])
                info = self.export_naip(
                    bbox=bbox, year=year, output_path=output_path,
                    scale=scale, description=tile_id, crs=crs,
                )
            else:
                info = self.export_sentinel2(
                    bbox=bbox, date_range=date_range, output_path=output_path,
                    scale=scale, description=tile_id, crs=crs,
                )

            info["tile_id"] = tile_id
            results.append(info)

            # Respect concurrent task limit
            active_count = len([
                t for t in self._active_tasks.values()
                if t.get("status") not in ("COMPLETED", "FAILED", "CANCELLED")
            ])
            if active_count >= max_concurrent:
                logger.info(f"Concurrent limit ({max_concurrent}) reached, waiting...")
                self.wait_for_tasks(poll_interval=15, timeout=600)

        return results

    @staticmethod
    def get_setup_instructions() -> str:
        """Get instructions for setting up GEE access."""
        return (
            "To set up Google Earth Engine:\n"
            "1. Install: pip install earthengine-api geemap\n"
            "2. Authenticate: earthengine authenticate\n"
            "3. Or use a service account with ee.ServiceAccountCredentials\n"
            "4. Verify: python -c \"import ee; ee.Initialize(); print('OK')\"\n"
        )
