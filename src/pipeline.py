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
        "data/tiles", "data/labels", "data/labels/generated", "data/labels/custom",
        "data/spatial", "data/validation",
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


# ---- Phase 2: Data Pipeline Commands ----


@cli.command("export-imagery")
@click.option("--source", default="sentinel2", help="Imagery source (sentinel2, naip)")
@click.option("--bbox", default=None, help="Override bbox as 'west,south,east,north'")
@click.pass_context
def export_imagery(ctx, source, bbox):
    """Export imagery from Google Earth Engine."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.data.gee_export import GEEExporter

    gee_config = OmegaConf.to_container(config.get("gee_export", {}), resolve=True)
    exporter = GEEExporter(config=gee_config)

    if not exporter.is_available():
        click.echo("GEE is not available.")
        click.echo(GEEExporter.get_setup_instructions())
        return

    # Parse bbox
    if bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        export_bbox = {"west": parts[0], "south": parts[1], "east": parts[2], "north": parts[3]}
    else:
        sa = config.get("study_area", {}).get("bbox", {})
        export_bbox = OmegaConf.to_container(sa, resolve=True)

    # Get date range from config
    collections = config.get("gee_export", {}).get("collections", {})
    source_config = OmegaConf.to_container(collections.get(source, {}), resolve=True)

    if source == "naip":
        year = source_config.get("year", 2023)
        scale = source_config.get("scale", 1)
        result = exporter.export_naip(
            bbox=export_bbox, year=year,
            output_path=f"data/tiles/naip_{year}",
            scale=scale,
        )
    else:
        dr = source_config.get("date_range", {})
        date_range = (dr.get("start", "2024-06-01"), dr.get("end", "2024-09-30"))
        scale = config.get("gee_export", {}).get("scale", 10)
        result = exporter.export_sentinel2(
            bbox=export_bbox, date_range=date_range,
            output_path=f"data/tiles/{source}",
            scale=scale,
            bands=source_config.get("bands"),
            cloud_filter=source_config.get("cloud_filter", 20),
        )

    click.echo(f"Export started: {result.get('task_id', 'N/A')}")
    click.echo(f"Strategy: {result.get('strategy', 'drive')}")
    click.echo("Use 'lccompare check-exports' to monitor progress.")


@cli.command("check-exports")
@click.pass_context
def check_exports(ctx):
    """Check status of GEE export tasks."""
    logger = logging.getLogger("lccomparison")

    from src.data.gee_export import GEEExporter

    exporter = GEEExporter()
    if not exporter.is_available():
        click.echo("GEE not available. Cannot check export status.")
        return

    results = exporter.check_all_tasks()
    if not results:
        click.echo("No active export tasks tracked in this session.")
        click.echo("(Task tracking is per-session; use GEE console for historical tasks)")
    else:
        for task in results:
            click.echo(
                f"  {task.get('description', '?')}: "
                f"{task.get('status', 'UNKNOWN')}"
            )


@cli.command("generate-labels")
@click.option("--source", default="all", help="Label source (dynamic_world, esa_worldcover, all)")
@click.option("--samples", default=500, help="Samples per class per source")
@click.option("--confidence", default=0.6, help="Min confidence threshold")
@click.option("--min-distance", default=100.0, help="Min distance between points (m)")
@click.option("--val-split", default=0.2, help="Validation split fraction")
@click.pass_context
def generate_labels(ctx, source, samples, confidence, min_distance, val_split):
    """Generate training labels from Dynamic World / ESA WorldCover."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.data.label_generator import LabelGenerator
    from src.data.label_manager import LabelManager

    generator = LabelGenerator()
    if not generator.is_available():
        click.echo("GEE not available. Cannot generate labels.")
        click.echo("Use 'lccompare import-labels' for local label files.")
        return

    # Get bbox from config
    sa = config.get("study_area", {}).get("bbox", {})
    bbox = OmegaConf.to_container(sa, resolve=True)
    dr = config.get("gee_export", {}).get("collections", {}).get("sentinel2", {}).get("date_range", {})
    date_range = (
        OmegaConf.to_container(dr, resolve=True).get("start", "2024-01-01"),
        OmegaConf.to_container(dr, resolve=True).get("end", "2024-12-31"),
    )

    sources = None
    if source != "all":
        sources = [source]

    click.echo(f"Generating labels (source={source}, samples/class={samples})...")
    result = generator.generate_labels(
        bbox=bbox,
        sources=sources,
        date_range=date_range,
        samples_per_class=samples,
        confidence_threshold=confidence,
        min_distance_m=min_distance,
        validation_split=val_split,
    )

    # Save via LabelManager
    manager = LabelManager(labels_dir=PROJECT_ROOT / "data" / "labels")
    if result["train"] is not None:
        manager.add_labels(result["train"], "generated_train")
    if result["val"] is not None:
        manager.add_labels(result["val"], "generated_val")

    paths = manager.save_splits(result["train"], result["val"])

    # Print summary
    summary = result["summary"]
    click.echo(f"\nLabel Generation Summary:")
    click.echo(f"  Total: {summary.get('total_points', 0)}")
    click.echo(f"  Train: {summary.get('train_points', 0)}")
    click.echo(f"  Val:   {summary.get('val_points', 0)}")
    click.echo(f"  Sources: {summary.get('sources', [])}")
    click.echo(f"\nTrain per class:")
    for cls, count in sorted(summary.get("train_per_class", {}).items()):
        click.echo(f"    {cls}: {count}")
    click.echo(f"\nSaved to: {paths.get('train', 'N/A')}")


@cli.command("import-labels")
@click.argument("path")
@click.option("--class-field", default="land_cover", help="Column with class names")
@click.option("--confidence-field", default=None, help="Optional confidence column")
@click.option("--name", default=None, help="Name for this label source")
@click.pass_context
def import_labels(ctx, path, class_field, confidence_field, name):
    """Import custom training labels from a file (GeoJSON, Shapefile, CSV, GPKG)."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.data.custom_labels import CustomLabelImporter
    from src.data.label_manager import LabelManager

    # Get AOI bbox
    sa = config.get("study_area", {}).get("bbox", {})
    aoi_bbox = OmegaConf.to_container(sa, resolve=True)

    importer = CustomLabelImporter(
        aoi_bbox=aoi_bbox,
        min_points_per_class=20,
    )

    click.echo(f"Importing labels from: {path}")
    result = importer.import_labels(
        path=path,
        class_field=class_field,
        confidence_field=confidence_field,
    )

    # Report issues
    if result["issues"]:
        click.echo(f"\nValidation issues ({len(result['issues'])}):")
        for issue in result["issues"]:
            click.echo(f"  - {issue}")

    # Save to label manager
    source_name = name or f"custom:{Path(path).stem}"
    manager = LabelManager(labels_dir=PROJECT_ROOT / "data" / "labels")
    count = manager.add_labels(result["points"], source_name)

    summary = result["summary"]
    click.echo(f"\nImported {count} labels")
    click.echo(f"Per class:")
    for cls, cnt in sorted(summary.get("per_class", {}).items()):
        click.echo(f"  {cls}: {cnt}")


@cli.command("preprocess")
@click.option("--model", default="prithvi", help="Model to preprocess for (prithvi, satlas, ssl4eo)")
@click.option("--input-dir", default=None, help="Input tiles directory")
@click.option("--output-dir", default=None, help="Output preprocessed directory")
@click.pass_context
def preprocess(ctx, model, input_dir, output_dir):
    """Preprocess imagery tiles for a specific embedding model."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.data.preprocessor import Preprocessor

    # Defaults from config
    if input_dir is None:
        input_dir = str(PROJECT_ROOT / "data" / "tiles")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "embeddings" / model / "preprocessed")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get model-specific config
    model_cfg = config.get("foundation_models", {}).get(model, {})
    proc_config = OmegaConf.to_container(model_cfg, resolve=True) if model_cfg else None

    preprocessor = Preprocessor(model_name=model, config=proc_config)

    # Find tiles
    from pathlib import Path as P
    tiles = sorted(P(input_dir).glob("**/*.tif"))
    if not tiles:
        click.echo(f"No .tif files found in {input_dir}")
        return

    click.echo(f"Preprocessing {len(tiles)} tiles for {model} (input_size={preprocessor.input_size})")
    success = 0
    failed = 0
    for tile_path in tiles:
        try:
            out_path = P(output_dir) / f"{tile_path.stem}.npz"
            preprocessor.preprocess_file(tile_path, output_path=out_path)
            success += 1
        except Exception as e:
            logger.warning(f"Failed: {tile_path}: {e}")
            failed += 1

    click.echo(f"\nPreprocessed: {success} success, {failed} failed")
    click.echo(f"Output: {output_dir}")


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

    # Check for labels
    labels_dir = PROJECT_ROOT / "data" / "labels"
    label_files = list(labels_dir.glob("**/*.geojson"))
    if label_files:
        click.echo(f"\nLabel files: {len(label_files)}")
        for lf in label_files:
            click.echo(f"  {lf.relative_to(PROJECT_ROOT)}")
    else:
        click.echo("\nNo labels found. Run 'lccompare generate-labels' or 'lccompare import-labels'.")


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
