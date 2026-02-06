"""Unified label management combining generated and custom labels.

Manages training label lifecycle: generation, import, validation,
persistence, and train/val splitting.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("lccomparison.label_manager")

try:
    import geopandas as gpd
    import pandas as pd
    _GEO_AVAILABLE = True
except ImportError:
    gpd = None
    pd = None
    _GEO_AVAILABLE = False

from src.config_schema import CLASS_NAMES, CLASS_SCHEMA


class LabelManager:
    """Unified label management for training data.

    Combines labels from multiple sources (generated, custom imports)
    and provides train/val splitting, persistence, and summary statistics.
    """

    def __init__(
        self,
        labels_dir: str | Path = "data/labels",
        generated_dir: str | Path | None = None,
        custom_dir: str | Path | None = None,
    ):
        self.labels_dir = Path(labels_dir)
        self.generated_dir = Path(generated_dir) if generated_dir else self.labels_dir / "generated"
        self.custom_dir = Path(custom_dir) if custom_dir else self.labels_dir / "custom"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.custom_dir.mkdir(parents=True, exist_ok=True)

        self._labels: list[Any] = []  # List of GeoDataFrames
        self._metadata: dict[str, Any] = {}

    def add_labels(
        self,
        gdf: Any,
        source_name: str,
        save: bool = True,
    ) -> int:
        """Add a GeoDataFrame of labels.

        Args:
            gdf: GeoDataFrame with lc_class, lc_name, geometry, confidence, source columns.
            source_name: Name for this label source.
            save: Whether to save to disk immediately.

        Returns:
            Number of labels added.
        """
        if not _GEO_AVAILABLE:
            raise ImportError("geopandas required for label management")

        # Ensure required columns
        required = {"geometry", "lc_class", "lc_name"}
        missing = required - set(gdf.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "confidence" not in gdf.columns:
            gdf["confidence"] = 1.0
        if "source" not in gdf.columns:
            gdf["source"] = source_name

        self._labels.append(gdf)

        if save:
            output_path = self._get_source_path(source_name)
            gdf.to_file(output_path, driver="GeoJSON")
            logger.info(f"Saved {len(gdf)} labels to {output_path}")

        return len(gdf)

    def get_combined(self) -> Any:
        """Get all labels combined into a single GeoDataFrame.

        Returns:
            Combined GeoDataFrame, or None if no labels.
        """
        if not _GEO_AVAILABLE:
            raise ImportError("geopandas required")

        if not self._labels:
            self._load_existing()

        if not self._labels:
            return None

        combined = gpd.GeoDataFrame(
            pd.concat(self._labels, ignore_index=True),
            crs=self._labels[0].crs,
        )
        return combined

    def get_train_val_split(
        self,
        validation_split: float = 0.2,
        stratified: bool = True,
        seed: int = 42,
    ) -> tuple[Any, Any]:
        """Split labels into training and validation sets.

        Args:
            validation_split: Fraction for validation.
            stratified: Whether to stratify by class.
            seed: Random seed.

        Returns:
            Tuple of (train_gdf, val_gdf).
        """
        import random

        combined = self.get_combined()
        if combined is None or len(combined) == 0:
            return None, None

        random.seed(seed)
        train_indices = []
        val_indices = []

        if stratified:
            for cls in combined["lc_class"].unique():
                cls_idx = combined[combined["lc_class"] == cls].index.tolist()
                random.shuffle(cls_idx)
                split = max(1, int(len(cls_idx) * (1 - validation_split)))
                train_indices.extend(cls_idx[:split])
                val_indices.extend(cls_idx[split:])
        else:
            all_idx = combined.index.tolist()
            random.shuffle(all_idx)
            split = int(len(all_idx) * (1 - validation_split))
            train_indices = all_idx[:split]
            val_indices = all_idx[split:]

        train_gdf = combined.loc[train_indices].copy()
        val_gdf = combined.loc[val_indices].copy()

        return train_gdf, val_gdf

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all labels.

        Returns:
            Dict with label counts, sources, and per-class breakdown.
        """
        combined = self.get_combined()
        if combined is None or len(combined) == 0:
            return {"total_points": 0, "sources": [], "per_class": {}}

        return {
            "total_points": len(combined),
            "sources": combined["source"].unique().tolist(),
            "per_class": combined["lc_name"].value_counts().to_dict(),
            "per_source": combined["source"].value_counts().to_dict(),
            "crs": str(combined.crs),
            "bounds": {
                "west": combined.total_bounds[0],
                "south": combined.total_bounds[1],
                "east": combined.total_bounds[2],
                "north": combined.total_bounds[3],
            },
        }

    def save_splits(
        self,
        train_gdf: Any,
        val_gdf: Any,
        output_dir: str | Path | None = None,
    ) -> dict[str, str]:
        """Save train/val split to disk.

        Args:
            train_gdf: Training GeoDataFrame.
            val_gdf: Validation GeoDataFrame.
            output_dir: Output directory. Defaults to labels_dir.

        Returns:
            Dict with paths to saved files.
        """
        out_dir = Path(output_dir) if output_dir else self.labels_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train_labels.geojson"
        val_path = out_dir / "val_labels.geojson"
        summary_path = out_dir / "label_summary.json"

        if train_gdf is not None and len(train_gdf) > 0:
            train_gdf.to_file(train_path, driver="GeoJSON")
        if val_gdf is not None and len(val_gdf) > 0:
            val_gdf.to_file(val_path, driver="GeoJSON")

        # Save summary
        summary = {
            "train_count": len(train_gdf) if train_gdf is not None else 0,
            "val_count": len(val_gdf) if val_gdf is not None else 0,
            "train_per_class": train_gdf["lc_name"].value_counts().to_dict() if train_gdf is not None and len(train_gdf) > 0 else {},
            "val_per_class": val_gdf["lc_name"].value_counts().to_dict() if val_gdf is not None and len(val_gdf) > 0 else {},
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved splits: train={train_path}, val={val_path}")
        return {
            "train": str(train_path),
            "val": str(val_path),
            "summary": str(summary_path),
        }

    def _load_existing(self) -> None:
        """Load existing label files from disk."""
        if not _GEO_AVAILABLE:
            return

        for geojson in sorted(self.generated_dir.glob("*.geojson")):
            try:
                gdf = gpd.read_file(geojson)
                self._labels.append(gdf)
                logger.info(f"Loaded {len(gdf)} labels from {geojson}")
            except Exception as e:
                logger.warning(f"Error loading {geojson}: {e}")

        for geojson in sorted(self.custom_dir.glob("*.geojson")):
            try:
                gdf = gpd.read_file(geojson)
                self._labels.append(gdf)
                logger.info(f"Loaded {len(gdf)} labels from {geojson}")
            except Exception as e:
                logger.warning(f"Error loading {geojson}: {e}")

    def _get_source_path(self, source_name: str) -> Path:
        """Get the file path for a label source."""
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in source_name)
        if source_name.startswith("custom:"):
            return self.custom_dir / f"{safe_name}.geojson"
        return self.generated_dir / f"{safe_name}.geojson"
