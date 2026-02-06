"""Import and validate user-provided training labels.

Supports GeoJSON, Shapefile, CSV, and GeoPackage formats with
comprehensive validation: CRS, AOI bounds, class names, duplicates,
and minimum sample counts.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.custom_labels")

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, box
    _GEO_AVAILABLE = True
except ImportError:
    gpd = None
    pd = None
    _GEO_AVAILABLE = False

from src.config_schema import CLASS_SCHEMA


class CustomLabelImporter:
    """Import and validate user-provided training labels.

    Handles multiple file formats with comprehensive validation.
    """

    # Common class name aliases
    CLASS_ALIASES = {
        "Water": "water", "WATER": "water", "water_body": "water",
        "Forest": "trees", "FOREST": "trees", "tree": "trees",
        "Trees": "trees", "TREES": "trees", "tree_cover": "trees",
        "Shrub": "shrub", "SHRUB": "shrub", "Shrubland": "shrub",
        "shrubs": "shrub", "shrub_and_scrub": "shrub",
        "Grass": "grass", "GRASS": "grass", "Grassland": "grass",
        "herbaceous": "grass", "Herbaceous": "grass",
        "Crops": "crops", "CROPS": "crops", "Agriculture": "crops",
        "cropland": "crops", "Cropland": "crops",
        "Built": "built", "BUILT": "built", "Urban": "built",
        "built_up": "built", "developed": "built", "Developed": "built",
        "Bare": "bare", "BARE": "bare", "Bareland": "bare",
        "bare_ground": "bare", "barren": "bare", "Barren": "bare",
    }

    def __init__(
        self,
        aoi_bbox: dict[str, float] | None = None,
        duplicate_tolerance_m: float = 5.0,
        min_points_per_class: int = 20,
    ):
        """
        Args:
            aoi_bbox: Area of interest bbox for within-AOI checking.
            duplicate_tolerance_m: Distance for duplicate detection.
            min_points_per_class: Minimum points required per class.
        """
        self.aoi_bbox = aoi_bbox
        self.duplicate_tolerance_m = duplicate_tolerance_m
        self.min_points_per_class = min_points_per_class

    def import_labels(
        self,
        path: str | Path,
        class_field: str = "land_cover",
        confidence_field: str | None = None,
        class_mapping: dict[str, str] | None = None,
        target_crs: str = "EPSG:4326",
    ) -> dict[str, Any]:
        """Import labels from a file.

        Args:
            path: Path to label file (GeoJSON, Shapefile, CSV, GeoPackage).
            class_field: Column name containing land cover class.
            confidence_field: Optional column with confidence values.
            class_mapping: Optional dict mapping source class names to 7-class names.
            target_crs: Target CRS for output.

        Returns:
            Dict with 'points' (GeoDataFrame), 'issues' (list), and 'summary'.
        """
        if not _GEO_AVAILABLE:
            raise ImportError("geopandas is required for label import")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Label file not found: {path}")

        # Read the file
        gdf = self._read_file(path)
        logger.info(f"Read {len(gdf)} features from {path}")

        # Map classes
        gdf = self._map_classes(gdf, class_field, class_mapping)

        # Reproject if needed
        if gdf.crs and str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Extract confidence if available
        if confidence_field and confidence_field in gdf.columns:
            gdf["confidence"] = gdf[confidence_field].astype(float)
        else:
            gdf["confidence"] = 1.0

        # Add source info
        gdf["source"] = f"custom:{path.name}"

        # Validate
        issues = self.validate(gdf)

        summary = {
            "file": str(path),
            "total_points": len(gdf),
            "crs": target_crs,
            "per_class": gdf["lc_name"].value_counts().to_dict() if "lc_name" in gdf.columns else {},
            "issues_count": len(issues),
        }

        return {"points": gdf, "issues": issues, "summary": summary}

    def _read_file(self, path: Path) -> Any:
        """Read a geospatial file into a GeoDataFrame."""
        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
            # Try common coordinate column names
            lon_cols = [c for c in df.columns if c.lower() in ("lon", "longitude", "x", "lng")]
            lat_cols = [c for c in df.columns if c.lower() in ("lat", "latitude", "y")]
            if not lon_cols or not lat_cols:
                raise ValueError(
                    "CSV must have longitude/latitude columns "
                    "(lon/longitude/x and lat/latitude/y)"
                )
            geometry = [Point(x, y) for x, y in zip(df[lon_cols[0]], df[lat_cols[0]])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        elif suffix in (".geojson", ".json"):
            gdf = gpd.read_file(path, driver="GeoJSON")
        elif suffix in (".shp",):
            gdf = gpd.read_file(path)
        elif suffix in (".gpkg",):
            gdf = gpd.read_file(path)
        else:
            # Try generic read
            gdf = gpd.read_file(path)

        return gdf

    def _map_classes(
        self,
        gdf: Any,
        class_field: str,
        custom_mapping: dict[str, str] | None = None,
    ) -> Any:
        """Map source class names to 7-class schema."""
        if class_field not in gdf.columns:
            raise ValueError(f"Class field '{class_field}' not found in data. "
                             f"Available columns: {list(gdf.columns)}")

        # Build effective mapping: custom overrides + aliases
        mapping = dict(self.CLASS_ALIASES)
        if custom_mapping:
            mapping.update(custom_mapping)

        # Map each value
        lc_classes = []
        lc_names = []
        unmapped = set()

        for val in gdf[class_field]:
            str_val = str(val).strip()
            # Try direct lookup
            mapped = mapping.get(str_val)
            if mapped is None:
                # Try case-insensitive
                mapped = mapping.get(str_val.lower())
            if mapped is None:
                # Try as-is if it's already a valid class name
                if str_val.lower() in CLASS_SCHEMA:
                    mapped = str_val.lower()

            if mapped and mapped in CLASS_SCHEMA:
                lc_names.append(mapped)
                lc_classes.append(CLASS_SCHEMA[mapped])
            else:
                unmapped.add(str_val)
                lc_names.append("unknown")
                lc_classes.append(-1)

        gdf["lc_class"] = lc_classes
        gdf["lc_name"] = lc_names

        if unmapped:
            logger.warning(f"Unmapped class values: {unmapped}")

        # Remove unmapped
        before = len(gdf)
        gdf = gdf[gdf["lc_class"] >= 0].copy()
        if len(gdf) < before:
            logger.warning(f"Dropped {before - len(gdf)} unmapped points")

        return gdf

    def validate(self, gdf: Any) -> list[str]:
        """Validate a label GeoDataFrame.

        Checks:
        - CRS is set
        - Points are within AOI (if configured)
        - Class names are valid
        - No duplicates (within tolerance)
        - Minimum points per class

        Args:
            gdf: GeoDataFrame with lc_class, lc_name, geometry columns.

        Returns:
            List of validation issue strings.
        """
        issues = []

        if len(gdf) == 0:
            issues.append("No valid label points after class mapping")
            return issues

        # CRS check
        if gdf.crs is None:
            issues.append("No CRS defined for label data")

        # Within AOI check
        if self.aoi_bbox:
            aoi_geom = box(
                self.aoi_bbox["west"], self.aoi_bbox["south"],
                self.aoi_bbox["east"], self.aoi_bbox["north"],
            )
            outside = ~gdf.geometry.within(aoi_geom)
            n_outside = outside.sum()
            if n_outside > 0:
                issues.append(f"{n_outside} points are outside the AOI")

        # Class name check
        invalid_classes = gdf[gdf["lc_class"] < 0]
        if len(invalid_classes) > 0:
            issues.append(f"{len(invalid_classes)} points have invalid class assignments")

        # Duplicate check
        n_duplicates = self._count_duplicates(gdf)
        if n_duplicates > 0:
            issues.append(
                f"{n_duplicates} potential duplicates "
                f"(within {self.duplicate_tolerance_m}m)"
            )

        # Min points per class
        class_counts = gdf["lc_name"].value_counts()
        for cls_name in CLASS_SCHEMA:
            count = class_counts.get(cls_name, 0)
            if 0 < count < self.min_points_per_class:
                issues.append(
                    f"Class '{cls_name}' has only {count} points "
                    f"(min: {self.min_points_per_class})"
                )

        return issues

    def _count_duplicates(self, gdf: Any) -> int:
        """Count points that are within duplicate_tolerance_m of each other."""
        if len(gdf) <= 1 or self.duplicate_tolerance_m <= 0:
            return 0

        # Project to UTM for distance
        centroid = gdf.geometry.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_epsg = f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"

        try:
            gdf_proj = gdf.to_crs(utm_epsg)
        except Exception:
            return 0

        coords = np.array([(g.x, g.y) for g in gdf_proj.geometry])
        n_dup = 0
        # Check pairwise distances (sample if too many points)
        n = len(coords)
        if n > 5000:
            # Sample check for large datasets
            indices = np.random.choice(n, size=5000, replace=False)
            coords = coords[indices]
            n = len(coords)

        for i in range(n):
            for j in range(i + 1, min(n, i + 100)):  # Limit to nearby indices
                dist = np.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                               (coords[i][1] - coords[j][1]) ** 2)
                if dist < self.duplicate_tolerance_m:
                    n_dup += 1

        return n_dup

    def apply_spatial_buffer(
        self,
        gdf: Any,
        buffer_m: float = 5.0,
    ) -> Any:
        """Apply spatial buffer around label points.

        Projects to UTM, buffers, then reprojects back.

        Args:
            gdf: GeoDataFrame of label points.
            buffer_m: Buffer distance in meters.

        Returns:
            GeoDataFrame with buffered geometries.
        """
        if not _GEO_AVAILABLE:
            raise ImportError("geopandas required")

        original_crs = gdf.crs

        # Project to UTM
        centroid = gdf.geometry.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_epsg = f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"

        buffered = gdf.to_crs(utm_epsg)
        buffered["geometry"] = buffered.geometry.buffer(buffer_m)
        buffered = buffered.to_crs(original_crs)

        logger.info(f"Applied {buffer_m}m buffer to {len(buffered)} features")
        return buffered
