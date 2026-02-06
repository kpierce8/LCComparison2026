"""Tile management with checkpoint/resume support for LCComparison2026."""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.utils.geo_utils import BBox, compute_tile_grid

logger = logging.getLogger("lccomparison.tile_manager")


class TileStatus(str, Enum):
    """Processing status for a tile."""
    PENDING = "pending"
    EXPORTING = "exporting"
    EXPORTED = "exported"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    PREDICTING = "predicting"
    PREDICTED = "predicted"
    FAILED = "failed"


@dataclass
class TileInfo:
    """Information about a single tile."""
    tile_id: str
    row: int
    col: int
    bbox: dict[str, float]  # {west, south, east, north}
    crs: str
    resolution: float
    source: str
    status: str = TileStatus.PENDING.value
    file_paths: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def bbox_tuple(self) -> BBox:
        return BBox(
            west=self.bbox["west"],
            south=self.bbox["south"],
            east=self.bbox["east"],
            north=self.bbox["north"],
        )


class TileManager:
    """Manages tile grid with JSON persistence for checkpoint/resume.

    Persists tile state to a JSON file so processing can be resumed
    after interruption.
    """

    def __init__(self, index_path: str | Path = "data/checkpoints/tile_index.json"):
        self.index_path = Path(index_path)
        self.tiles: dict[str, TileInfo] = {}
        self.metadata: dict[str, Any] = {}

        if self.index_path.exists():
            self._load()

    def create_grid(
        self,
        bbox: BBox,
        resolution: float = 10.0,
        tile_size: int = 256,
        overlap: int = 0,
        source: str = "sentinel2",
        crs: str = "EPSG:32610",
    ) -> int:
        """Generate a tile grid covering the bounding box.

        Args:
            bbox: Bounding box in projected coordinates.
            resolution: Pixel resolution in meters.
            tile_size: Tile size in pixels.
            overlap: Overlap in pixels.
            source: Imagery source name.
            crs: Coordinate reference system.

        Returns:
            Number of tiles created.
        """
        grid = compute_tile_grid(bbox, tile_size, resolution, overlap)

        self.metadata = {
            "bbox": {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north},
            "resolution": resolution,
            "tile_size": tile_size,
            "overlap": overlap,
            "source": source,
            "crs": crs,
            "rows": grid.rows,
            "cols": grid.cols,
        }

        for i, tile_bbox in enumerate(grid.tiles):
            row = i // grid.cols
            col = i % grid.cols
            tile_id = f"{source}_{row:04d}_{col:04d}"

            self.tiles[tile_id] = TileInfo(
                tile_id=tile_id,
                row=row,
                col=col,
                bbox={
                    "west": tile_bbox.west,
                    "south": tile_bbox.south,
                    "east": tile_bbox.east,
                    "north": tile_bbox.north,
                },
                crs=crs,
                resolution=resolution,
                source=source,
            )

        self.save()
        logger.info(f"Created grid: {grid.rows}x{grid.cols} = {len(self.tiles)} tiles")
        return len(self.tiles)

    def add_tile(self, tile: TileInfo) -> None:
        """Add a single tile to the index."""
        self.tiles[tile.tile_id] = tile
        self.save()

    def update_status(
        self,
        tile_id: str,
        status: TileStatus,
        file_paths: dict[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        """Update the status of a tile.

        Args:
            tile_id: Tile identifier.
            status: New status.
            file_paths: Optional dict of output file paths to merge.
            error: Optional error message (for FAILED status).
        """
        if tile_id not in self.tiles:
            raise KeyError(f"Tile not found: {tile_id}")

        self.tiles[tile_id].status = status.value
        if file_paths:
            self.tiles[tile_id].file_paths.update(file_paths)
        if error is not None:
            self.tiles[tile_id].error = error
        self.save()

    def get_tiles_by_status(self, status: TileStatus) -> list[TileInfo]:
        """Get all tiles with a given status."""
        return [t for t in self.tiles.values() if t.status == status.value]

    def get_progress(self) -> dict[str, int]:
        """Get count of tiles in each status."""
        progress = {s.value: 0 for s in TileStatus}
        for tile in self.tiles.values():
            progress[tile.status] = progress.get(tile.status, 0) + 1
        progress["total"] = len(self.tiles)
        return progress

    def save(self) -> None:
        """Persist tile index to JSON."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": self.metadata,
            "tiles": {tid: asdict(t) for tid, t in self.tiles.items()},
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load tile index from JSON."""
        with open(self.index_path) as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.tiles = {}
        for tid, tdata in data.get("tiles", {}).items():
            self.tiles[tid] = TileInfo(**tdata)

        logger.info(f"Loaded {len(self.tiles)} tiles from {self.index_path}")
