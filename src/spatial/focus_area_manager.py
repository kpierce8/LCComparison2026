"""Focus area management for county/WRIA/RMZ boundaries.

Loads and manages spatial boundary layers for clipping predictions
and computing per-feature statistics.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("lccomparison.focus_area_manager")

try:
    import geopandas as gpd
    from shapely.geometry import box
    _GEO_AVAILABLE = True
except ImportError:
    gpd = None
    _GEO_AVAILABLE = False


class FocusAreaLayer:
    """A single spatial boundary layer (e.g., counties, WRIAs)."""

    def __init__(
        self,
        name: str,
        path: str | Path,
        id_field: str,
        target_crs: str = "EPSG:32610",
    ):
        """
        Args:
            name: Layer name (e.g., 'county', 'wria', 'rmz').
            path: Path to shapefile/GeoPackage/GeoJSON.
            id_field: Column name for feature identifiers.
            target_crs: CRS to reproject boundaries to.
        """
        self.name = name
        self.path = Path(path)
        self.id_field = id_field
        self.target_crs = target_crs
        self._gdf: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._gdf is not None

    @property
    def features(self) -> list[str]:
        """Get list of feature IDs."""
        if not self.is_loaded:
            return []
        return self._gdf[self.id_field].tolist()

    @property
    def count(self) -> int:
        return len(self._gdf) if self.is_loaded else 0

    def load(self) -> None:
        """Load the boundary layer from file."""
        if not _GEO_AVAILABLE:
            raise ImportError("geopandas required for spatial operations")

        if not self.path.exists():
            raise FileNotFoundError(f"Boundary file not found: {self.path}")

        self._gdf = gpd.read_file(self.path)

        if self.id_field not in self._gdf.columns:
            raise ValueError(
                f"ID field '{self.id_field}' not found in {self.path}. "
                f"Available: {list(self._gdf.columns)}"
            )

        # Reproject if needed
        if self._gdf.crs is not None and str(self._gdf.crs) != self.target_crs:
            self._gdf = self._gdf.to_crs(self.target_crs)

        logger.info(
            f"Loaded {self.name} layer: {len(self._gdf)} features "
            f"from {self.path}"
        )

    def get_feature(self, feature_id: str) -> Any:
        """Get a single feature's geometry by ID.

        Returns:
            GeoDataFrame with one row.
        """
        if not self.is_loaded:
            raise RuntimeError(f"Layer '{self.name}' not loaded. Call load() first.")

        mask = self._gdf[self.id_field] == feature_id
        result = self._gdf[mask]

        if len(result) == 0:
            raise KeyError(f"Feature '{feature_id}' not found in {self.name}")

        return result

    def get_geometry(self, feature_id: str) -> Any:
        """Get just the geometry for a feature."""
        feature = self.get_feature(feature_id)
        return feature.geometry.union_all()

    def get_bounds(self, feature_id: str) -> tuple[float, float, float, float]:
        """Get bounding box (west, south, east, north) for a feature."""
        geom = self.get_geometry(feature_id)
        return geom.bounds  # (minx, miny, maxx, maxy)

    def get_all_geometries(self) -> dict[str, Any]:
        """Get all feature geometries as a dict."""
        if not self.is_loaded:
            raise RuntimeError(f"Layer '{self.name}' not loaded.")

        result = {}
        for _, row in self._gdf.iterrows():
            fid = row[self.id_field]
            result[fid] = row.geometry
        return result

    def get_geodataframe(self) -> Any:
        """Get the underlying GeoDataFrame."""
        if not self.is_loaded:
            raise RuntimeError(f"Layer '{self.name}' not loaded.")
        return self._gdf.copy()

    def get_info(self) -> dict[str, Any]:
        """Get layer summary info."""
        return {
            "name": self.name,
            "path": str(self.path),
            "id_field": self.id_field,
            "loaded": self.is_loaded,
            "count": self.count,
            "features": self.features[:10],  # First 10
            "crs": str(self._gdf.crs) if self.is_loaded else None,
        }


class FocusAreaManager:
    """Manages multiple spatial boundary layers."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: spatial_focus config section with layers list.
        """
        self.layers: dict[str, FocusAreaLayer] = {}
        self._config = config or {}

        if "layers" in self._config:
            for layer_cfg in self._config["layers"]:
                if layer_cfg.get("enabled", True):
                    self.add_layer(
                        name=layer_cfg["name"],
                        path=layer_cfg["path"],
                        id_field=layer_cfg["id_field"],
                    )

    def add_layer(
        self,
        name: str,
        path: str | Path,
        id_field: str,
        target_crs: str = "EPSG:32610",
    ) -> None:
        """Register a boundary layer."""
        self.layers[name] = FocusAreaLayer(
            name=name, path=path, id_field=id_field, target_crs=target_crs,
        )

    def load_layer(self, name: str) -> FocusAreaLayer:
        """Load a specific layer."""
        if name not in self.layers:
            raise KeyError(f"Unknown layer: {name}. Available: {list(self.layers.keys())}")
        self.layers[name].load()
        return self.layers[name]

    def load_all(self) -> dict[str, int]:
        """Load all registered layers. Returns {name: feature_count}."""
        results = {}
        for name, layer in self.layers.items():
            try:
                layer.load()
                results[name] = layer.count
            except Exception as e:
                logger.warning(f"Failed to load layer '{name}': {e}")
                results[name] = 0
        return results

    def get_layer(self, name: str) -> FocusAreaLayer:
        """Get a loaded layer by name."""
        if name not in self.layers:
            raise KeyError(f"Unknown layer: {name}. Available: {list(self.layers.keys())}")
        return self.layers[name]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all layers."""
        return {
            name: layer.get_info()
            for name, layer in self.layers.items()
        }
