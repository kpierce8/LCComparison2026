"""Configuration loading, validation, and class schema for LCComparison2026."""

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

# 7-class schema for LCComparison2026
CLASS_SCHEMA = {
    "water": 0,
    "trees": 1,
    "shrub": 2,
    "grass": 3,
    "crops": 4,
    "built": 5,
    "bare": 6,
}

CLASS_NAMES = {v: k for k, v in CLASS_SCHEMA.items()}

CLASS_COLORS = {
    0: "#0077BE",  # water - blue
    1: "#228B22",  # trees - forest green
    2: "#90EE90",  # shrub - light green
    3: "#ADFF2F",  # grass - green-yellow
    4: "#FFD700",  # crops - gold
    5: "#DC143C",  # built - crimson
    6: "#D2B48C",  # bare - tan
}

# Mapping from LCAnalysis2026 8-class to LCComparison2026 7-class
# LCAnalysis2026: background=0, trees=1, shrubs=2, herbaceous=3, ground=4, water=5, gravel=6, built=7
# LCComparison2026: water=0, trees=1, shrub=2, grass=3, crops=4, built=5, bare=6
LCANALYSIS_TO_LCCOMPARISON = {
    0: 6,  # background -> bare
    1: 1,  # trees -> trees
    2: 2,  # shrubs -> shrub
    3: 3,  # herbaceous -> grass
    4: 6,  # ground -> bare
    5: 0,  # water -> water
    6: 6,  # gravel -> bare
    7: 5,  # built -> built
}

LCANALYSIS_CLASS_NAMES = [
    "background", "trees", "shrubs", "herbaceous",
    "ground", "water", "gravel", "built",
]


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    """Load configuration from YAML file with optional CLI overrides.

    Args:
        path: Path to YAML config file. Defaults to config/config.yaml.
        overrides: List of dotlist overrides, e.g. ["processing.device=cpu"].

    Returns:
        Merged OmegaConf DictConfig.
    """
    if path is None:
        path = Path(__file__).parent.parent / "config" / "config.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def validate_config(config: DictConfig) -> list[str]:
    """Validate configuration and return list of issues.

    Args:
        config: OmegaConf DictConfig to validate.

    Returns:
        List of validation issue strings. Empty list means valid.
    """
    issues = []

    # Check required top-level keys
    required_keys = ["project", "existing_models", "processing"]
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required config section: {key}")

    # Validate existing_models.base_path
    if "existing_models" in config:
        base_path = config.existing_models.get("base_path")
        if base_path and not Path(base_path).exists():
            issues.append(f"LCAnalysis2026 base_path not found: {base_path}")

    # Validate processing device
    if "processing" in config:
        device = config.processing.get("device", "cpu")
        if device not in ("cpu", "cuda", "mps", "auto"):
            issues.append(f"Invalid processing device: {device}")

    # Validate class schema
    if "class_schema" in config:
        schema = OmegaConf.to_container(config.class_schema, resolve=True)
        if isinstance(schema, dict):
            expected_classes = set(CLASS_SCHEMA.keys())
            actual_classes = set(schema.get("classes", {}).keys())
            if actual_classes and actual_classes != expected_classes:
                issues.append(
                    f"Class schema mismatch. Expected {expected_classes}, got {actual_classes}"
                )

    # Validate GEE export settings
    if "gee_export" in config:
        strategy = config.gee_export.get("strategy", "drive")
        if strategy not in ("drive", "cloud_storage", "asset"):
            issues.append(f"Invalid GEE export strategy: {strategy}")

    # Validate tile settings
    if "tiles" in config:
        tile_size = config.tiles.get("size")
        if tile_size is not None and tile_size <= 0:
            issues.append(f"Tile size must be positive, got {tile_size}")
        overlap = config.tiles.get("overlap", 0)
        if overlap < 0:
            issues.append(f"Tile overlap must be non-negative, got {overlap}")

    return issues
