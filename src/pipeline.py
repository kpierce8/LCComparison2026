"""CLI entry point for LCComparison2026.

Usage:
    lccompare --help
    python -m src.pipeline --help
"""

import logging
import sys
from pathlib import Path

import click
from omegaconf import OmegaConf

from src.config_schema import CLASS_SCHEMA, load_config, validate_config
from src.utils.logging_utils import setup_logging

PROJECT_ROOT = Path(__file__).parent.parent


def _get_config(config_path: str | None, overrides: tuple[str, ...] | None = None) -> OmegaConf:
    """Load config with optional overrides."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    override_list = list(overrides) if overrides else None
    return load_config(path, override_list)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config YAML")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config_path, verbose):
    """LCComparison2026 - Land Cover Model Comparison Framework."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, log_file=PROJECT_ROOT / "logs" / "lccomparison.log")


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize project: validate config and create directories."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    issues = validate_config(config)
    if issues:
        logger.warning("Configuration issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Configuration is valid.")

    # Create data directories
    dirs = [
        "data/tiles", "data/labels", "data/spatial", "data/validation",
        "data/embeddings", "data/predictions", "data/outputs", "data/checkpoints",
        "models", "logs",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Directory: {d}")

    logger.info("Project initialized successfully.")


@cli.command("validate-config")
@click.option("--override", "-o", multiple=True, help="Config overrides (dotlist)")
@click.pass_context
def validate_config_cmd(ctx, override):
    """Validate the configuration file."""
    logger = logging.getLogger("lccomparison")

    try:
        config = _get_config(ctx.obj.get("config_path"), override)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    issues = validate_config(config)
    if issues:
        logger.error(f"Found {len(issues)} issue(s):")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)
    else:
        logger.info("Configuration is valid.")
        click.echo(f"Classes: {list(CLASS_SCHEMA.keys())}")


@cli.command("discover-existing")
@click.pass_context
def discover_existing(ctx):
    """Scan LCAnalysis2026 and generate existing model config."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.integration.existing_model_integration import (
        discover_experiments,
        save_discovery_results,
    )

    base_path = config.existing_models.get("base_path", "/media/ken/data/LCAnalysis2026")
    logger.info(f"Scanning: {base_path}")

    results = discover_experiments(
        base_path=base_path,
        experiments_dir=config.existing_models.get("experiments_dir", "experiments"),
        config_filename=config.existing_models.get("config_filename", "config.yaml"),
        metrics_filename=config.existing_models.get("metrics_filename", "final_metrics.yaml"),
        metadata_filename=config.existing_models.get("metadata_filename", "experiment_metadata.yaml"),
    )

    output_path = PROJECT_ROOT / config.existing_models.get(
        "discovery_output", "config/existing_model_config.yaml"
    )
    save_discovery_results(results, output_path)

    # Print summary
    click.echo(f"\nDiscovered {results['num_experiments']} experiments")
    click.echo(f"Complete (with metrics): {len(results['complete_experiments'])}")
    for exp in results["complete_experiments"]:
        metrics = exp.get("metrics", {})
        iou = metrics.get("test/iou", "N/A")
        f1 = metrics.get("test/f1", "N/A")
        if isinstance(iou, float):
            iou = f"{iou:.4f}"
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"
        click.echo(f"  - {exp['model_name']} ({exp.get('encoder_name', '?')}): IoU={iou} F1={f1}")

    click.echo(f"\nClass mapping (8 -> 7):")
    for src, dst in results["class_mapping"]["name_mapping"].items():
        click.echo(f"  {src} -> {dst}")
    click.echo(f"\nSaved to: {output_path}")


@cli.command("export-imagery")
@click.option("--source", default="sentinel2", help="Imagery source")
@click.pass_context
def export_imagery(ctx, source):
    """Export imagery from Google Earth Engine (stub)."""
    logger = logging.getLogger("lccomparison")

    from src.data.gee_export import GEEExporter

    exporter = GEEExporter()
    if exporter.is_available():
        logger.info("GEE is available. Export would proceed here.")
        click.echo("GEE export stub - full implementation in Phase 2")
    else:
        click.echo("GEE is not available.")
        click.echo(GEEExporter.get_setup_instructions())


@cli.command()
@click.pass_context
def status(ctx):
    """Show processing progress."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.data.tile_manager import TileManager

    index_path = PROJECT_ROOT / "data" / "checkpoints" / "tile_index.json"
    if index_path.exists():
        tm = TileManager(index_path)
        progress = tm.get_progress()
        click.echo("Tile Processing Status:")
        for status_name, count in progress.items():
            click.echo(f"  {status_name}: {count}")
    else:
        click.echo("No tile index found. Run 'lccompare init' first.")

    # Check for existing model discovery
    discovery_path = PROJECT_ROOT / config.existing_models.get(
        "discovery_output", "config/existing_model_config.yaml"
    )
    if discovery_path.exists():
        click.echo(f"\nExisting model config: {discovery_path}")
    else:
        click.echo("\nExisting models not yet discovered. Run 'lccompare discover-existing'.")


@cli.command("download-models")
@click.option("--model", default=None, help="Specific model to download")
@click.pass_context
def download_models(ctx, model):
    """Download foundation model weights (stub for Phase 3)."""
    click.echo("Model download stub - will be implemented in Phase 3")
    click.echo("Models to download: prithvi, satlas, ssl4eo")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
