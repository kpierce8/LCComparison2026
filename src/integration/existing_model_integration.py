"""Discover and catalog existing LCAnalysis2026 experiments.

Scans the LCAnalysis2026 experiments directory to find completed training runs,
extract their configurations, metrics, and checkpoint paths, and generate
a class mapping from the 8-class to 7-class schema.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import yaml

from src.config_schema import (
    CLASS_NAMES,
    CLASS_SCHEMA,
    LCANALYSIS_CLASS_NAMES,
    LCANALYSIS_TO_LCCOMPARISON,
)

logger = logging.getLogger("lccomparison.discovery")


def discover_experiments(
    base_path: str | Path,
    experiments_dir: str = "experiments",
    config_filename: str = "config.yaml",
    metrics_filename: str = "final_metrics.yaml",
    metadata_filename: str = "experiment_metadata.yaml",
) -> dict[str, Any]:
    """Scan LCAnalysis2026 experiments directory and catalog findings.

    Args:
        base_path: Root path to LCAnalysis2026 project.
        experiments_dir: Name of experiments subdirectory.
        config_filename: Config file name within each experiment.
        metrics_filename: Final metrics file name.
        metadata_filename: Experiment metadata file name.

    Returns:
        Dict with discovered experiments, class mapping, and summary.
    """
    base_path = Path(base_path)
    exp_root = base_path / experiments_dir

    if not exp_root.exists():
        logger.error(f"Experiments directory not found: {exp_root}")
        return {"experiments": [], "error": f"Not found: {exp_root}"}

    experiments = []
    experiment_log = _parse_experiment_log(exp_root / "experiment_log.csv")

    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        config_path = exp_dir / config_filename
        if not config_path.exists():
            continue

        experiment = _parse_experiment(
            exp_dir, config_filename, metrics_filename, metadata_filename
        )
        if experiment:
            # Enrich with experiment log data if available
            exp_id = experiment.get("experiment_id", exp_dir.name)
            if exp_id in experiment_log:
                experiment["log_summary"] = experiment_log[exp_id]
            experiments.append(experiment)

    # Read the default class schema
    default_config = _read_default_config(base_path / "configs" / "default.yaml")

    result = {
        "source_project": str(base_path),
        "num_experiments": len(experiments),
        "experiments": experiments,
        "complete_experiments": [
            e for e in experiments if e.get("has_metrics")
        ],
        "class_mapping": _build_class_mapping(default_config),
        "lcanalysis_schema": {
            "num_classes": 8,
            "class_names": LCANALYSIS_CLASS_NAMES,
        },
        "lccomparison_schema": {
            "num_classes": 7,
            "class_names": {v: k for k, v in CLASS_SCHEMA.items()},
        },
    }

    logger.info(
        f"Discovered {len(experiments)} experiments "
        f"({len(result['complete_experiments'])} complete)"
    )
    return result


def _parse_experiment(
    exp_dir: Path,
    config_filename: str,
    metrics_filename: str,
    metadata_filename: str,
) -> dict[str, Any] | None:
    """Parse a single experiment directory."""
    try:
        config = _load_yaml(exp_dir / config_filename)
        if config is None:
            return None

        experiment = {
            "experiment_id": exp_dir.name,
            "path": str(exp_dir),
            "config": config,
        }

        # Extract key info from config
        experiment["model_name"] = config.get("model", {}).get("name", "unknown")
        experiment["encoder_name"] = config.get("model", {}).get("encoder_name", "unknown")
        experiment["dataset"] = _extract_dataset_name(config)
        experiment["num_classes"] = config.get("data", {}).get("num_classes", 8)
        experiment["class_names"] = config.get("data", {}).get("class_names", [])

        # Parse naming convention: {dataset}_{model}_{timestamp}
        parts = exp_dir.name.split("_")
        experiment["naming_convention"] = {
            "raw_name": exp_dir.name,
            "detected_pattern": "{dataset}_{model}_{timestamp}",
        }

        # Metadata
        metadata = _load_yaml(exp_dir / metadata_filename)
        if metadata:
            experiment["metadata"] = metadata
            experiment["model_short"] = metadata.get("model_short", experiment["model_name"])

        # Final metrics
        metrics = _load_yaml(exp_dir / metrics_filename)
        experiment["has_metrics"] = metrics is not None
        if metrics:
            experiment["metrics"] = metrics

        # Checkpoints
        ckpt_dir = exp_dir / "checkpoints"
        if ckpt_dir.exists():
            checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
            experiment["checkpoints"] = [str(c) for c in checkpoints]
            if checkpoints:
                experiment["best_checkpoint"] = str(checkpoints[-1])
        else:
            experiment["checkpoints"] = []

        # Data source paths
        data_csv = config.get("data", {}).get("data_csv")
        if data_csv:
            experiment["data_source"] = data_csv

        return experiment

    except Exception as e:
        logger.warning(f"Error parsing experiment {exp_dir.name}: {e}")
        return None


def _extract_dataset_name(config: dict) -> str:
    """Extract dataset name from experiment config."""
    data_csv = config.get("data", {}).get("data_csv", "")
    if data_csv:
        # Extract from path like /path/to/lctest_20000_jan26/lctest_20000_jan26_linux.csv
        csv_path = Path(data_csv)
        return csv_path.parent.name
    return "unknown"


def _parse_experiment_log(log_path: Path) -> dict[str, dict]:
    """Parse experiment_log.csv into a dict keyed by experiment_id."""
    if not log_path.exists():
        return {}

    result = {}
    try:
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_id = row.get("experiment_id", "")
                if exp_id:
                    result[exp_id] = {
                        "model_name": row.get("model_name", ""),
                        "dataset_name": row.get("dataset_name", ""),
                        "duration_minutes": _safe_float(row.get("duration_minutes")),
                        "num_epochs": _safe_int(row.get("num_epochs")),
                        "val_loss": _safe_float(row.get("val_loss")),
                        "val_iou": _safe_float(row.get("val_iou")),
                        "val_f1": _safe_float(row.get("val_f1")),
                        "machine_name": row.get("machine_name", ""),
                    }
    except Exception as e:
        logger.warning(f"Error parsing experiment log: {e}")

    return result


def _safe_float(value: str | None, default: float = 0.0) -> float:
    """Convert string to float, returning default on failure."""
    if not value:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: str | None, default: int = 0) -> int:
    """Convert string to int, returning default on failure."""
    if not value:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _build_class_mapping(default_config: dict | None) -> dict[str, Any]:
    """Build the 8-class to 7-class mapping."""
    mapping = {
        "description": "LCAnalysis2026 8-class to LCComparison2026 7-class mapping",
        "source_classes": {
            i: name for i, name in enumerate(LCANALYSIS_CLASS_NAMES)
        },
        "target_classes": {v: k for k, v in CLASS_SCHEMA.items()},
        "index_mapping": LCANALYSIS_TO_LCCOMPARISON,
        "name_mapping": {
            LCANALYSIS_CLASS_NAMES[src]: CLASS_NAMES[dst]
            for src, dst in LCANALYSIS_TO_LCCOMPARISON.items()
        },
        "merge_details": {
            "bare": "Merges background (0), ground (4), gravel (6) into bare (6)",
            "grass": "Renames herbaceous (3) to grass (3)",
            "shrub": "Renames shrubs (2) to shrub (2)",
        },
    }

    if default_config:
        mapping["source_raw_mapping"] = default_config.get("data", {}).get("class_mapping", {})

    return mapping


def _read_default_config(config_path: Path) -> dict | None:
    """Read the LCAnalysis2026 default config."""
    return _load_yaml(config_path)


def _load_yaml(path: Path) -> dict | None:
    """Safely load a YAML file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None


def save_discovery_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Save discovery results to YAML.

    Args:
        results: Discovery results dict.
        output_path: Output YAML file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip full configs to keep output manageable
    slim_results = {
        "source_project": results["source_project"],
        "num_experiments": results["num_experiments"],
        "class_mapping": results["class_mapping"],
        "lcanalysis_schema": results["lcanalysis_schema"],
        "lccomparison_schema": results["lccomparison_schema"],
        "experiments": [],
    }

    for exp in results.get("complete_experiments", []):
        slim_exp = {
            "experiment_id": exp["experiment_id"],
            "model_name": exp["model_name"],
            "encoder_name": exp.get("encoder_name", "unknown"),
            "dataset": exp.get("dataset", "unknown"),
            "path": exp["path"],
            "checkpoints": exp.get("checkpoints", []),
            "best_checkpoint": exp.get("best_checkpoint"),
        }
        if exp.get("metrics"):
            slim_exp["metrics"] = {
                k: v for k, v in exp["metrics"].items()
                if not k.startswith("test/") or "/" not in k.split("test/", 1)[1]
            }
            # Include top-level test metrics
            slim_exp["metrics"] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in exp["metrics"].items()
            }
        if exp.get("log_summary"):
            slim_exp["log_summary"] = exp["log_summary"]
        if exp.get("data_source"):
            slim_exp["data_source"] = exp["data_source"]

        slim_results["experiments"].append(slim_exp)

    with open(output_path, "w") as f:
        yaml.dump(slim_results, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Discovery results saved to {output_path}")
