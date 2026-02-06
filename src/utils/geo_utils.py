"""Geospatial utility functions for LCComparison2026."""

import math
from typing import NamedTuple


class BBox(NamedTuple):
    """Bounding box in (west, south, east, north) order."""
    west: float
    south: float
    east: float
    north: float


class TileGrid(NamedTuple):
    """Grid of tile bounding boxes."""
    tiles: list[BBox]
    rows: int
    cols: int
    tile_size_meters: float
    resolution: float


def get_utm_zone(lon: float, lat: float) -> str:
    """Get EPSG code for UTM zone containing the given point.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.

    Returns:
        EPSG code string, e.g. "EPSG:32610".
    """
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone_number:02d}"
    else:
        return f"EPSG:327{zone_number:02d}"


def compute_tile_grid(
    bbox: BBox,
    tile_size: int = 256,
    resolution: float = 10.0,
    overlap: int = 0,
) -> TileGrid:
    """Compute a grid of tiles covering a bounding box.

    Assumes the bbox is in a projected CRS (meters). Tiles are generated
    left-to-right, top-to-bottom.

    Args:
        bbox: Bounding box in projected coordinates (meters).
        tile_size: Tile size in pixels.
        resolution: Pixel resolution in meters.
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        TileGrid with list of tile bounding boxes.
    """
    tile_meters = tile_size * resolution
    step_meters = (tile_size - overlap) * resolution

    width = bbox.east - bbox.west
    height = bbox.north - bbox.south

    cols = max(1, math.ceil(width / step_meters))
    rows = max(1, math.ceil(height / step_meters))

    tiles = []
    for row in range(rows):
        for col in range(cols):
            west = bbox.west + col * step_meters
            north = bbox.north - row * step_meters
            east = west + tile_meters
            south = north - tile_meters
            tiles.append(BBox(west=west, south=south, east=east, north=north))

    return TileGrid(
        tiles=tiles,
        rows=rows,
        cols=cols,
        tile_size_meters=tile_meters,
        resolution=resolution,
    )


def bbox_area_acres(bbox: BBox, is_degrees: bool = True) -> float:
    """Compute approximate area of a bounding box in acres.

    Args:
        bbox: Bounding box.
        is_degrees: If True, bbox is in lat/lon degrees. If False, meters.

    Returns:
        Area in acres.
    """
    if is_degrees:
        # Approximate conversion at mid-latitude
        mid_lat = (bbox.south + bbox.north) / 2
        lat_m = 111_320  # meters per degree latitude
        lon_m = 111_320 * math.cos(math.radians(mid_lat))
        width_m = abs(bbox.east - bbox.west) * lon_m
        height_m = abs(bbox.north - bbox.south) * lat_m
    else:
        width_m = abs(bbox.east - bbox.west)
        height_m = abs(bbox.north - bbox.south)

    area_sq_m = width_m * height_m
    return area_sq_m / 4046.86  # sq meters to acres
