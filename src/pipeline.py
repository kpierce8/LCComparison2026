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

    # Check for model weights
    models_dir = PROJECT_ROOT / config.get("paths", {}).get("models_dir", "models")
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if model_dirs:
            click.echo("\nModel Weights:")
            for md in model_dirs:
                weight_files = list(md.glob("*.pt")) + list(md.glob("*.pth"))
                status_str = f"{len(weight_files)} file(s)" if weight_files else "not downloaded"
                click.echo(f"  {md.name}: {status_str}")

    # Check for embeddings
    emb_dir = PROJECT_ROOT / "data" / "embeddings"
    if emb_dir.exists():
        for model_dir in sorted(emb_dir.iterdir()):
            if model_dir.is_dir():
                npz_files = list(model_dir.glob("*.npz"))
                cp_file = model_dir / f"{model_dir.name}_checkpoint.json"
                if npz_files or cp_file.exists():
                    click.echo(f"\nEmbeddings ({model_dir.name}): {len(npz_files)} cached")
                    if cp_file.exists():
                        import json as _json
                        with open(cp_file) as _f:
                            cp = _json.load(_f)
                        click.echo(f"  Completed: {len(cp.get('completed_tiles', []))}")
                        click.echo(f"  Failed: {len(cp.get('failed_tiles', []))}")

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
@click.option("--model", default=None, help="Specific model to download (prithvi, satlas, ssl4eo)")
@click.option("--force", is_flag=True, help="Re-download even if cached")
@click.pass_context
def download_models(ctx, model, force):
    """Download foundation model weights from HuggingFace."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.models.model_downloader import ModelDownloader, MODEL_REGISTRY

    models_dir = str(PROJECT_ROOT / config.get("paths", {}).get("models_dir", "models"))
    downloader = ModelDownloader(models_dir=models_dir)

    targets = [model] if model else list(MODEL_REGISTRY.keys())

    for name in targets:
        if name not in MODEL_REGISTRY:
            click.echo(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
            continue

        click.echo(f"Downloading {name}...")
        try:
            path = downloader.download(name, force=force)
            click.echo(f"  Saved to: {path}")
        except Exception as e:
            click.echo(f"  Failed: {e}")
            logger.error(f"Download failed for {name}: {e}")

    # Show status
    status = downloader.get_status()
    click.echo("\nModel Status:")
    for name, info in status.items():
        dl = "downloaded" if info["downloaded"] else "not downloaded"
        size = f" ({info['size_mb']:.1f} MB)" if info.get("size_mb") else ""
        click.echo(f"  {name}: {dl}{size}")


# ---- Phase 3: Model Integration Commands ----


@cli.command("generate-embeddings")
@click.option("--model", default="prithvi", help="Model name (prithvi, satlas, ssl4eo)")
@click.option("--input-dir", default=None, help="Input preprocessed tiles directory")
@click.option("--output-dir", default=None, help="Output embeddings directory")
@click.option("--batch-size", default=8, help="Batch size for inference")
@click.option("--device", default=None, help="Device (cpu, cuda)")
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoint")
@click.option("--auto-download", is_flag=True, help="Auto-download weights if missing")
@click.pass_context
def generate_embeddings(ctx, model, input_dir, output_dir, batch_size, device, resume, auto_download):
    """Generate embeddings for preprocessed tiles using a foundation model."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.models.model_registry import load_model
    from src.data.preprocessor import Preprocessor
    from src.processing.batch_processor import BatchProcessor

    # Resolve directories
    if input_dir is None:
        input_dir = str(PROJECT_ROOT / "data" / "embeddings" / model / "preprocessed")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "embeddings" / model)
    if device is None:
        device = config.get("processing", {}).get("device", "cpu")

    # Find input tiles
    input_path = Path(input_dir)
    tile_files = sorted(input_path.glob("*.npz")) + sorted(input_path.glob("*.tif"))
    if not tile_files:
        click.echo(f"No tiles found in {input_dir}")
        click.echo("Run 'lccompare preprocess' first to prepare tiles.")
        return

    tile_paths = {p.stem: p for p in tile_files}
    click.echo(f"Found {len(tile_paths)} tiles for {model}")

    # Load model
    click.echo(f"Loading {model} on {device}...")
    try:
        models_dir = str(PROJECT_ROOT / config.get("paths", {}).get("models_dir", "models"))
        embedding_model = load_model(
            model, device=device, auto_download=auto_download, models_dir=models_dir,
        )
    except Exception as e:
        click.echo(f"Failed to load model: {e}")
        click.echo("Try 'lccompare download-models' first.")
        return

    # Setup processor
    proc_config = config.get("processing", {})
    checkpoint_freq = proc_config.get("checkpoint_frequency", 100)
    cache = proc_config.get("cache_embeddings", True)
    max_mem = proc_config.get("max_memory_gb", 16)

    preprocessor = Preprocessor(model_name=model)
    processor = BatchProcessor(
        model=embedding_model,
        preprocessor=preprocessor,
        output_dir=output_dir,
        batch_size=batch_size,
        checkpoint_frequency=checkpoint_freq,
        cache_embeddings=cache,
        max_memory_gb=max_mem,
    )

    # Process
    click.echo(f"Generating embeddings (batch_size={batch_size}, resume={resume})...")
    result = processor.process_tiles(tile_paths, resume=resume)

    # Print summary
    summary = result["summary"]
    click.echo(f"\nEmbedding Generation Summary:")
    click.echo(f"  Model: {summary['model']}")
    click.echo(f"  Processed: {summary['processed']}/{summary['total_tiles']}")
    click.echo(f"  Failed: {summary['failed']}")
    click.echo(f"  Time: {summary['elapsed_seconds']}s ({summary['tiles_per_second']} tiles/sec)")
    click.echo(f"  Output: {output_dir}")


# ---- Phase 4: Classification & Training Commands ----


@cli.command("train-classifier")
@click.option("--model", default="prithvi", help="Embedding model (prithvi, satlas, ssl4eo)")
@click.option("--method", default=None, help="Classifier method (xgboost, random_forest, mlp, linear)")
@click.option("--labels", default=None, help="Path to labels file (GeoJSON/CSV with tile_id + class)")
@click.option("--embeddings-dir", default=None, help="Directory with cached embeddings")
@click.option("--cv/--no-cv", default=True, help="Run cross-validation")
@click.option("--n-folds", default=5, help="Number of CV folds")
@click.pass_context
def train_classifier(ctx, model, method, labels, embeddings_dir, cv, n_folds):
    """Train a land cover classifier on embeddings."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.training.trainer import Trainer
    from src.processing.batch_processor import BatchProcessor
    from src.config_schema import CLASS_SCHEMA

    # Resolve classifier config
    cls_config = OmegaConf.to_container(config.get("classification", {}), resolve=True)
    if method is None:
        method = cls_config.get("method", "xgboost")
    method_config = cls_config.get(method, {})

    # Load embeddings
    if embeddings_dir is None:
        embeddings_dir = str(PROJECT_ROOT / "data" / "embeddings" / model)

    emb_dir = Path(embeddings_dir)
    npz_files = sorted(emb_dir.glob("*.npz"))
    if not npz_files:
        click.echo(f"No embeddings found in {embeddings_dir}")
        click.echo("Run 'lccompare generate-embeddings' first.")
        return

    click.echo(f"Loading embeddings from {embeddings_dir}...")
    embeddings = {}
    for f in npz_files:
        try:
            data = np.load(f)
            if "embedding" in data:
                embeddings[f.stem] = data["embedding"]
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    click.echo(f"Loaded {len(embeddings)} embeddings")

    # Load labels
    if labels is None:
        labels_path = PROJECT_ROOT / "data" / "labels" / "train.geojson"
        if not labels_path.exists():
            # Try any geojson
            label_files = list((PROJECT_ROOT / "data" / "labels").glob("**/*.geojson"))
            if label_files:
                labels_path = label_files[0]
            else:
                click.echo("No labels found. Run 'lccompare generate-labels' or 'lccompare import-labels'.")
                return
    else:
        labels_path = Path(labels)

    click.echo(f"Loading labels from {labels_path}...")
    import geopandas as gpd
    gdf = gpd.read_file(labels_path)

    # Build label dict: tile_id -> class_index
    class_names = list(CLASS_SCHEMA.keys())
    label_dict = {}
    if "tile_id" in gdf.columns and "class_index" in gdf.columns:
        for _, row in gdf.iterrows():
            label_dict[row["tile_id"]] = int(row["class_index"])
    elif "tile_id" in gdf.columns and "land_cover" in gdf.columns:
        for _, row in gdf.iterrows():
            cls_name = row["land_cover"]
            if cls_name in CLASS_SCHEMA:
                label_dict[row["tile_id"]] = CLASS_SCHEMA[cls_name]

    if not label_dict:
        click.echo("Could not extract tile_id/class pairs from labels file.")
        click.echo("Expected columns: tile_id + (class_index or land_cover)")
        return

    click.echo(f"Matched labels: {len(label_dict)}")

    # Setup trainer
    output_dir = PROJECT_ROOT / "data" / "checkpoints"
    trainer = Trainer(
        method=method,
        n_classes=len(class_names),
        classifier_config=method_config,
        output_dir=output_dir,
    )

    # Prepare data
    try:
        X, y, matched_ids = trainer.prepare_training_data(embeddings, label_dict)
    except ValueError as e:
        click.echo(str(e))
        return

    click.echo(f"Training {method} classifier on {len(matched_ids)} samples...")

    # Split for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    # Train
    metrics = trainer.train(X_train, y_train, X_val, y_val, model_name=model)

    click.echo(f"\nTraining Results:")
    click.echo(f"  Method: {method}")
    click.echo(f"  Train accuracy: {metrics.get('train_accuracy', 'N/A'):.4f}")
    if "val_accuracy" in metrics:
        click.echo(f"  Val accuracy: {metrics['val_accuracy']:.4f}")
    click.echo(f"  Time: {metrics.get('elapsed_seconds', 'N/A')}s")

    # Cross-validation
    if cv:
        click.echo(f"\nRunning {n_folds}-fold cross-validation...")
        val_config = OmegaConf.to_container(config.get("validation", {}), resolve=True)
        cv_stratified = val_config.get("cross_validation", {}).get("stratified", True)

        cv_results = trainer.cross_validate(
            X, y, n_folds=n_folds, stratified=cv_stratified,
        )

        summary = cv_results["summary"]
        click.echo(f"  Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        click.echo(f"  F1 (macro): {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        click.echo(f"  IoU (mean): {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
        click.echo(f"  Kappa: {summary['kappa_mean']:.4f} ± {summary['kappa_std']:.4f}")

    click.echo(f"\nSaved to: {output_dir}")


@cli.command("predict")
@click.option("--model", default="prithvi", help="Embedding model name")
@click.option("--classifier-path", default=None, help="Path to trained classifier")
@click.option("--embeddings-dir", default=None, help="Directory with cached embeddings")
@click.option("--output-dir", default=None, help="Output predictions directory")
@click.pass_context
def predict(ctx, model, classifier_path, embeddings_dir, output_dir):
    """Predict land cover classes for all embedded tiles."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.training.trainer import Trainer

    # Resolve paths
    if embeddings_dir is None:
        embeddings_dir = str(PROJECT_ROOT / "data" / "embeddings" / model)
    if classifier_path is None:
        classifier_path = str(PROJECT_ROOT / "data" / "checkpoints" / f"{model}_classifier")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "predictions" / model)

    # Load embeddings
    emb_dir = Path(embeddings_dir)
    npz_files = sorted(emb_dir.glob("*.npz"))
    if not npz_files:
        click.echo(f"No embeddings found in {embeddings_dir}")
        return

    click.echo(f"Loading {len(npz_files)} embeddings...")
    embeddings = {}
    for f in npz_files:
        try:
            data = np.load(f)
            if "embedding" in data:
                embeddings[f.stem] = data["embedding"]
        except Exception:
            pass

    # Load classifier
    click.echo(f"Loading classifier from {classifier_path}...")
    trainer = Trainer(output_dir=output_dir)
    try:
        trainer.load_classifier(classifier_path)
    except FileNotFoundError:
        click.echo(f"Classifier not found at {classifier_path}")
        click.echo("Run 'lccompare train-classifier' first.")
        return

    # Predict
    click.echo(f"Predicting {len(embeddings)} tiles...")
    result = trainer.predict_tiles(embeddings, output_dir=output_dir)

    summary = result["summary"]
    click.echo(f"\nPrediction Summary:")
    click.echo(f"  Tiles: {summary['total_tiles']}")
    click.echo(f"  Mean confidence: {summary['mean_confidence']:.4f}")
    click.echo(f"  Class distribution:")
    for cls, cnt in sorted(summary["class_distribution"].items()):
        click.echo(f"    {cls}: {cnt}")
    click.echo(f"  Output: {output_dir}")


@cli.command("mosaic-tiles")
@click.option("--model", default="prithvi", help="Model name for file naming")
@click.option("--predictions-dir", default=None, help="Directory with tile predictions")
@click.option("--output", default=None, help="Output mosaic path")
@click.option("--product", default="classification", help="Product type (classification, confidence, probability)")
@click.pass_context
def mosaic_tiles(ctx, model, predictions_dir, output, product):
    """Mosaic tile predictions into final raster output."""
    logger = logging.getLogger("lccomparison")
    config = _get_config(ctx.obj.get("config_path"))

    from src.spatial.mosaic import TileMosaicker

    if predictions_dir is None:
        predictions_dir = str(PROJECT_ROOT / "data" / "predictions" / model / "tiles" / product)
    if output is None:
        output = str(PROJECT_ROOT / "data" / "outputs" / f"landcover_{model}.tif")

    pred_dir = Path(predictions_dir)
    tile_files = sorted(pred_dir.glob("*.tif"))
    if not tile_files:
        click.echo(f"No tile GeoTIFFs found in {predictions_dir}")
        click.echo("Run 'lccompare predict' first to generate tile predictions.")
        return

    click.echo(f"Mosaicking {len(tile_files)} tiles...")
    mosaicker = TileMosaicker()

    try:
        metadata = mosaicker.mosaic_tiles(tile_files, output)
        click.echo(f"\nMosaic created: {output}")
        click.echo(f"  Shape: {metadata['shape']}")
        click.echo(f"  Tiles: {metadata['n_tiles']}")
    except Exception as e:
        click.echo(f"Mosaicking failed: {e}")
        logger.error(f"Mosaic error: {e}")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
