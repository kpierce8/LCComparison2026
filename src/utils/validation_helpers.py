"""Input validation and error handling utilities.

Reusable validators for common pipeline inputs: rasters, vectors,
model names, and directory structures.
"""

from pathlib import Path
from typing import Any

VALID_RASTER_EXTS = {".tif", ".tiff", ".geotiff"}
VALID_VECTOR_EXTS = {".geojson", ".gpkg", ".shp", ".csv", ".json"}
VALID_MODELS = {"prithvi", "satlas", "ssl4eo"}
VALID_CLASSIFIERS = {"xgboost", "random_forest", "mlp", "linear"}
VALID_ENSEMBLE_STRATEGIES = {"majority_vote", "weighted_vote", "probability_average"}
VALID_FUSION_STRATEGIES = {"high_res_priority", "confidence_weighted"}


def validate_file_exists(path: str | Path, label: str = "File") -> list[str]:
    """Check that a file exists. Returns list of issues (empty = valid)."""
    p = Path(path)
    if not p.exists():
        return [f"{label} not found: {path}"]
    if not p.is_file():
        return [f"{label} is not a file: {path}"]
    return []


def validate_raster_path(path: str | Path, label: str = "Raster") -> list[str]:
    """Validate a raster file path. Returns list of issues."""
    issues = validate_file_exists(path, label)
    if issues:
        return issues
    p = Path(path)
    if p.suffix.lower() not in VALID_RASTER_EXTS:
        issues.append(
            f"{label} '{p.name}' has extension '{p.suffix}', "
            f"expected one of: {', '.join(sorted(VALID_RASTER_EXTS))}"
        )
    return issues


def validate_vector_path(path: str | Path, label: str = "Vector file") -> list[str]:
    """Validate a vector/reference file path. Returns list of issues."""
    issues = validate_file_exists(path, label)
    if issues:
        return issues
    p = Path(path)
    if p.suffix.lower() not in VALID_VECTOR_EXTS:
        issues.append(
            f"{label} '{p.name}' has extension '{p.suffix}', "
            f"expected one of: {', '.join(sorted(VALID_VECTOR_EXTS))}"
        )
    return issues


def validate_model_name(model: str) -> list[str]:
    """Validate that a model name is recognized. Returns list of issues."""
    if model not in VALID_MODELS:
        return [f"Unknown model '{model}'. Choose from: {', '.join(sorted(VALID_MODELS))}"]
    return []


def validate_classifier_method(method: str) -> list[str]:
    """Validate that a classifier method is recognized."""
    if method not in VALID_CLASSIFIERS:
        return [
            f"Unknown classifier '{method}'. "
            f"Choose from: {', '.join(sorted(VALID_CLASSIFIERS))}"
        ]
    return []


def validate_ensemble_strategy(strategy: str) -> list[str]:
    """Validate an ensemble strategy name."""
    if strategy not in VALID_ENSEMBLE_STRATEGIES:
        return [
            f"Unknown ensemble strategy '{strategy}'. "
            f"Choose from: {', '.join(sorted(VALID_ENSEMBLE_STRATEGIES))}"
        ]
    return []


def validate_fusion_strategy(strategy: str) -> list[str]:
    """Validate a fusion strategy name."""
    if strategy not in VALID_FUSION_STRATEGIES:
        return [
            f"Unknown fusion strategy '{strategy}'. "
            f"Choose from: {', '.join(sorted(VALID_FUSION_STRATEGIES))}"
        ]
    return []


def validate_weights(weights: list[float], n_models: int) -> list[str]:
    """Validate ensemble weights."""
    issues = []
    if len(weights) != n_models:
        issues.append(
            f"Number of weights ({len(weights)}) does not match "
            f"number of models ({n_models})."
        )
    if any(w < 0 for w in weights):
        issues.append("Weights must be non-negative.")
    if sum(weights) <= 0:
        issues.append("Weights must sum to a positive value.")
    return issues


def validate_config_section(config: Any, section: str) -> list[str]:
    """Check that a config section exists and is non-empty."""
    if config is None:
        return [f"Config is None"]
    value = config.get(section) if hasattr(config, "get") else None
    if value is None:
        return [f"Missing config section: '{section}'"]
    return []


def format_issues(issues: list[str], prefix: str = "Error") -> str:
    """Format a list of issues into a user-friendly string."""
    if not issues:
        return ""
    if len(issues) == 1:
        return f"{prefix}: {issues[0]}"
    lines = [f"{prefix}s ({len(issues)}):"]
    for issue in issues:
        lines.append(f"  - {issue}")
    return "\n".join(lines)
