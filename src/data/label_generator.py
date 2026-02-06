"""Generate training labels from Dynamic World and ESA WorldCover via GEE.

Creates stratified random training points from existing land cover products,
with confidence thresholding and minimum distance constraints.
"""

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.label_generator")

try:
    import ee
    _EE_AVAILABLE = True
except ImportError:
    ee = None
    _EE_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    _GEO_AVAILABLE = True
except ImportError:
    gpd = None
    _GEO_AVAILABLE = False


# Class mapping from Dynamic World to LCComparison 7-class
DYNAMIC_WORLD_MAPPING = {
    0: 0,  # water -> water
    1: 1,  # trees -> trees
    2: 3,  # grass -> grass
    3: 4,  # flooded_vegetation -> crops (approximate)
    4: 4,  # crops -> crops
    5: 2,  # shrub_and_scrub -> shrub
    6: 5,  # built -> built
    7: 6,  # bare -> bare
    8: 0,  # snow_and_ice -> water (approximate)
}

# Class mapping from ESA WorldCover to LCComparison 7-class
ESA_WORLDCOVER_MAPPING = {
    10: 1,  # tree_cover -> trees
    20: 2,  # shrubland -> shrub
    30: 3,  # grassland -> grass
    40: 4,  # cropland -> crops
    50: 5,  # built_up -> built
    60: 6,  # bare -> bare
    70: 0,  # snow_and_ice -> water
    80: 0,  # permanent_water -> water
    90: 3,  # herbaceous_wetland -> grass
    95: 1,  # mangroves -> trees
    100: 3,  # moss_and_lichen -> grass
}

# Reverse lookup: target class -> class name
from src.config_schema import CLASS_NAMES


class LabelGenerator:
    """Generate training labels from GEE land cover products.

    Supports Dynamic World and ESA WorldCover as label sources,
    with stratified sampling and confidence thresholding.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._check_availability()

    def _check_availability(self) -> None:
        if not _EE_AVAILABLE:
            logger.warning("earthengine-api not installed. Label generation unavailable.")
            self._available = False
            return
        try:
            ee.Initialize()
            self._available = True
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return _EE_AVAILABLE and self._available

    def generate_dynamic_world_labels(
        self,
        bbox: dict[str, float],
        date_range: tuple[str, str] = ("2024-01-01", "2024-12-31"),
        samples_per_class: int = 500,
        confidence_threshold: float = 0.6,
        min_distance_m: float = 100.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Generate training labels from Dynamic World.

        Args:
            bbox: Bounding box (west, south, east, north).
            date_range: Date range for composite.
            samples_per_class: Target samples per LCComparison class.
            confidence_threshold: Min confidence for label inclusion.
            min_distance_m: Min distance between sample points in meters.
            seed: Random seed.

        Returns:
            Dict with 'points' (GeoDataFrame) and 'summary' metadata.
        """
        if not self.is_available():
            raise RuntimeError("GEE not available for label generation")

        region = ee.Geometry.Rectangle(
            [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
        )

        # Get Dynamic World mode composite
        dw = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(region)
            .filterDate(date_range[0], date_range[1])
        )

        # Get the mode (most common) label and max probability
        label_image = dw.select("label").mode()
        # Get average probability for confidence
        prob_bands = [
            "water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice",
        ]
        prob_composite = dw.select(prob_bands).mean()
        max_prob = prob_composite.reduce(ee.Reducer.max()).rename("confidence")

        # Combine label and confidence
        combined = label_image.addBands(max_prob)

        # Sample points stratified by class
        samples = combined.stratifiedSample(
            numPoints=samples_per_class,
            classBand="label",
            region=region,
            scale=10,
            seed=seed,
            geometries=True,
        )

        # Filter by confidence
        samples = samples.filter(ee.Filter.gte("confidence", confidence_threshold))

        # Convert to local GeoDataFrame
        return self._ee_fc_to_geodataframe(
            samples, "dynamic_world", DYNAMIC_WORLD_MAPPING, min_distance_m
        )

    def generate_esa_worldcover_labels(
        self,
        bbox: dict[str, float],
        samples_per_class: int = 300,
        min_distance_m: float = 100.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Generate training labels from ESA WorldCover v200.

        Args:
            bbox: Bounding box (west, south, east, north).
            samples_per_class: Target samples per LCComparison class.
            min_distance_m: Min distance between sample points.
            seed: Random seed.

        Returns:
            Dict with 'points' (GeoDataFrame) and 'summary' metadata.
        """
        if not self.is_available():
            raise RuntimeError("GEE not available for label generation")

        region = ee.Geometry.Rectangle(
            [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]
        )

        esa = ee.Image("ESA/WorldCover/v200").select("Map")

        samples = esa.stratifiedSample(
            numPoints=samples_per_class,
            classBand="Map",
            region=region,
            scale=10,
            seed=seed,
            geometries=True,
        )

        return self._ee_fc_to_geodataframe(
            samples, "esa_worldcover", ESA_WORLDCOVER_MAPPING, min_distance_m,
            source_band="Map",
        )

    def _ee_fc_to_geodataframe(
        self,
        feature_collection: Any,
        source_name: str,
        class_mapping: dict[int, int],
        min_distance_m: float,
        source_band: str = "label",
    ) -> dict[str, Any]:
        """Convert EE FeatureCollection to GeoDataFrame with class mapping.

        Returns:
            Dict with 'points' (GeoDataFrame) and 'summary'.
        """
        if not _GEO_AVAILABLE:
            raise ImportError("geopandas is required for label operations")

        # Fetch from GEE
        features = feature_collection.getInfo()["features"]
        logger.info(f"Fetched {len(features)} raw samples from {source_name}")

        records = []
        for feat in features:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]
            source_class = int(props.get(source_band, -1))
            target_class = class_mapping.get(source_class)
            if target_class is None:
                continue

            records.append({
                "geometry": Point(coords[0], coords[1]),
                "source_class": source_class,
                "lc_class": target_class,
                "lc_name": CLASS_NAMES.get(target_class, "unknown"),
                "confidence": props.get("confidence", 1.0),
                "source": source_name,
            })

        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

        # Enforce minimum distance
        if min_distance_m > 0 and len(gdf) > 0:
            gdf = self._enforce_min_distance(gdf, min_distance_m)

        # Summary
        summary = {
            "source": source_name,
            "total_points": len(gdf),
            "per_class": gdf["lc_name"].value_counts().to_dict() if len(gdf) > 0 else {},
            "min_distance_m": min_distance_m,
        }

        logger.info(f"Generated {len(gdf)} labels from {source_name}")
        return {"points": gdf, "summary": summary}

    def _enforce_min_distance(
        self, gdf: Any, min_distance_m: float,
    ) -> Any:
        """Remove points that are too close together.

        Projects to UTM for distance calculation, then filters.
        """
        if len(gdf) <= 1:
            return gdf

        # Project to UTM for accurate distance
        centroid = gdf.geometry.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_epsg = f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"
        gdf_proj = gdf.to_crs(utm_epsg)

        # Greedy distance filtering
        keep = [True] * len(gdf_proj)
        coords = np.array([(g.x, g.y) for g in gdf_proj.geometry])

        for i in range(len(coords)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(coords)):
                if not keep[j]:
                    continue
                dist = np.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                               (coords[i][1] - coords[j][1]) ** 2)
                if dist < min_distance_m:
                    keep[j] = False

        filtered = gdf[keep].copy()
        logger.info(
            f"Distance filter: {len(gdf)} -> {len(filtered)} "
            f"(min_distance={min_distance_m}m)"
        )
        return filtered

    def generate_labels(
        self,
        bbox: dict[str, float],
        sources: list[str] | None = None,
        date_range: tuple[str, str] = ("2024-01-01", "2024-12-31"),
        samples_per_class: int = 500,
        confidence_threshold: float = 0.6,
        min_distance_m: float = 100.0,
        validation_split: float = 0.2,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Generate labels from multiple sources and split train/val.

        Args:
            bbox: Bounding box.
            sources: List of sources to use. Default: ['dynamic_world', 'esa_worldcover'].
            date_range: Date range for Dynamic World.
            samples_per_class: Samples per class per source.
            confidence_threshold: Min confidence for DW labels.
            min_distance_m: Min distance between points.
            validation_split: Fraction for validation set.
            seed: Random seed.

        Returns:
            Dict with 'train', 'val' GeoDataFrames and 'summary'.
        """
        if sources is None:
            sources = ["dynamic_world", "esa_worldcover"]

        all_points = []

        if "dynamic_world" in sources:
            dw_result = self.generate_dynamic_world_labels(
                bbox, date_range, samples_per_class,
                confidence_threshold, min_distance_m, seed,
            )
            all_points.append(dw_result["points"])

        if "esa_worldcover" in sources:
            esa_result = self.generate_esa_worldcover_labels(
                bbox, samples_per_class, min_distance_m, seed + 1,
            )
            all_points.append(esa_result["points"])

        if not all_points:
            return {"train": None, "val": None, "summary": {"total": 0}}

        import pandas as pd
        combined = gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True), crs="EPSG:4326")

        # Stratified train/val split
        random.seed(seed)
        train_indices = []
        val_indices = []

        for cls in combined["lc_class"].unique():
            cls_indices = combined[combined["lc_class"] == cls].index.tolist()
            random.shuffle(cls_indices)
            split_idx = max(1, int(len(cls_indices) * (1 - validation_split)))
            train_indices.extend(cls_indices[:split_idx])
            val_indices.extend(cls_indices[split_idx:])

        train_gdf = combined.loc[train_indices].copy()
        val_gdf = combined.loc[val_indices].copy()

        summary = {
            "total_points": len(combined),
            "train_points": len(train_gdf),
            "val_points": len(val_gdf),
            "sources": sources,
            "train_per_class": train_gdf["lc_name"].value_counts().to_dict(),
            "val_per_class": val_gdf["lc_name"].value_counts().to_dict(),
        }

        logger.info(
            f"Label generation complete: {len(train_gdf)} train, "
            f"{len(val_gdf)} val from {sources}"
        )
        return {"train": train_gdf, "val": val_gdf, "summary": summary}
