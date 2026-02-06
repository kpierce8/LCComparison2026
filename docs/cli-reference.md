# CLI Reference

All commands are run via:

```bash
python -m src.pipeline [OPTIONS] COMMAND [ARGS]
```

Or if installed as a package:

```bash
lccompare [OPTIONS] COMMAND [ARGS]
```

## Global Options

| Option | Description |
|--------|-------------|
| `--config TEXT` | Path to config YAML (default: `config/config.yaml`) |
| `-v, --verbose` | Enable debug logging |
| `--help` | Show help message |

---

## Project Setup

### `init`

Initialize project: validate config and create data directories.

```bash
python -m src.pipeline init
```

Creates: `data/tiles`, `data/labels`, `data/embeddings`, `data/predictions`, `data/outputs`, `data/checkpoints`, `models/`, `logs/`.

### `validate-config`

Validate the configuration file and report any issues.

```bash
python -m src.pipeline validate-config
python -m src.pipeline validate-config -o "processing.device=cpu"
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --override` | - | Config overrides in dotlist format (repeatable) |

### `discover-existing`

Scan LCAnalysis2026 experiments directory and generate `config/existing_model_config.yaml`.

```bash
python -m src.pipeline discover-existing
```

Outputs discovered experiments with metrics, class mappings, and checkpoint paths.

### `status`

Show processing progress across all pipeline stages.

```bash
python -m src.pipeline status
```

Displays: tile index status, model weights, embeddings progress, and label files.

---

## Data Pipeline

### `export-imagery`

Export imagery from Google Earth Engine (requires GEE authentication).

```bash
python -m src.pipeline export-imagery
python -m src.pipeline export-imagery --source naip --bbox "-122.5,47.0,-122.0,47.5"
```

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `sentinel2` | Imagery source (`sentinel2`, `naip`) |
| `--bbox` | config value | Override bounding box as `west,south,east,north` |

### `check-exports`

Check status of active GEE export tasks.

```bash
python -m src.pipeline check-exports
```

### `generate-labels`

Generate training labels from Dynamic World and/or ESA WorldCover via GEE.

```bash
python -m src.pipeline generate-labels
python -m src.pipeline generate-labels --source dynamic_world --samples 1000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `all` | Label source (`dynamic_world`, `esa_worldcover`, `all`) |
| `--samples` | `500` | Samples per class per source |
| `--confidence` | `0.6` | Minimum confidence threshold |
| `--min-distance` | `100.0` | Minimum distance between points (meters) |
| `--val-split` | `0.2` | Validation split fraction |

### `import-labels`

Import custom training labels from a local file.

```bash
python -m src.pipeline import-labels path/to/labels.geojson
python -m src.pipeline import-labels data.csv --class-field land_cover --name my_labels
```

| Argument/Option | Default | Description |
|-----------------|---------|-------------|
| `PATH` (required) | - | Path to label file (GeoJSON, Shapefile, CSV, GPKG) |
| `--class-field` | `land_cover` | Column name with class labels |
| `--confidence-field` | - | Optional confidence column |
| `--name` | - | Name for this label source |

### `preprocess`

Preprocess imagery tiles for a specific embedding model.

```bash
python -m src.pipeline preprocess --model prithvi
python -m src.pipeline preprocess --model satlas --input-dir data/tiles/naip
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `prithvi` | Model to preprocess for (`prithvi`, `satlas`, `ssl4eo`) |
| `--input-dir` | `data/tiles` | Input tiles directory |
| `--output-dir` | `data/embeddings/<model>/preprocessed` | Output directory |

---

## Model Integration

### `download-models`

Download foundation model weights from HuggingFace Hub.

```bash
python -m src.pipeline download-models
python -m src.pipeline download-models --model prithvi --force
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | all models | Specific model to download (`prithvi`, `satlas`, `ssl4eo`) |
| `--force` | false | Re-download even if cached |

### `generate-embeddings`

Generate embeddings for preprocessed tiles using a foundation model.

```bash
python -m src.pipeline generate-embeddings --model prithvi
python -m src.pipeline generate-embeddings --model satlas --batch-size 16 --device cuda
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `prithvi` | Model name (`prithvi`, `satlas`, `ssl4eo`) |
| `--input-dir` | `data/embeddings/<model>/preprocessed` | Input preprocessed tiles |
| `--output-dir` | `data/embeddings/<model>` | Output embeddings directory |
| `--batch-size` | `8` | Batch size for inference |
| `--device` | config value | Device (`cpu`, `cuda`) |
| `--resume/--no-resume` | `--resume` | Resume from checkpoint |
| `--auto-download` | false | Auto-download weights if missing |

---

## Classification & Training

### `train-classifier`

Train a land cover classifier on embeddings.

```bash
python -m src.pipeline train-classifier --model prithvi
python -m src.pipeline train-classifier --model satlas --method random_forest --no-cv
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `prithvi` | Embedding model name |
| `--method` | config value | Classifier method (`xgboost`, `random_forest`, `mlp`, `linear`) |
| `--labels` | `data/labels/train.geojson` | Path to labels file |
| `--embeddings-dir` | `data/embeddings/<model>` | Directory with cached embeddings |
| `--cv/--no-cv` | `--cv` | Run cross-validation |
| `--n-folds` | `5` | Number of CV folds |

### `predict`

Predict land cover classes for all embedded tiles.

```bash
python -m src.pipeline predict --model prithvi
python -m src.pipeline predict --model satlas --classifier-path checkpoints/satlas_classifier
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `prithvi` | Embedding model name |
| `--classifier-path` | `data/checkpoints/<model>_classifier` | Path to trained classifier |
| `--embeddings-dir` | `data/embeddings/<model>` | Directory with cached embeddings |
| `--output-dir` | `data/predictions/<model>` | Output predictions directory |

### `mosaic-tiles`

Mosaic tile predictions into a final raster output.

```bash
python -m src.pipeline mosaic-tiles --model prithvi
python -m src.pipeline mosaic-tiles --model satlas --product confidence
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `prithvi` | Model name for file naming |
| `--predictions-dir` | `data/predictions/<model>/tiles/<product>` | Tile predictions directory |
| `--output` | `data/outputs/landcover_<model>.tif` | Output mosaic path |
| `--product` | `classification` | Product type (`classification`, `confidence`, `probability`) |

---

## Spatial Analysis & Validation

### `process-by-focus`

Clip predictions to spatial boundaries and compute per-zone statistics.

```bash
python -m src.pipeline process-by-focus --model prithvi
python -m src.pipeline process-by-focus --model satlas --layer county,wria
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `prithvi` | Model name |
| `--layer` | all layers | Layer name or comma-separated list |
| `--prediction` | `data/outputs/landcover_<model>.tif` | Classification raster path |

### `zonal-stats`

Calculate land cover statistics by spatial zones.

```bash
python -m src.pipeline zonal-stats --prediction output.tif --boundaries zones.shp
python -m src.pipeline zonal-stats --prediction output.tif --boundaries zones.gpkg --id-field FID
```

| Option | Default | Description |
|--------|---------|-------------|
| `--prediction` (required) | - | Path to classification raster |
| `--boundaries` (required) | - | Path to boundary file (Shapefile/GPKG/GeoJSON) |
| `--id-field` | `NAME` | Feature ID field name |
| `--output` | `data/outputs/<layer>_stats.csv` | Output CSV path |
| `--resolution` | `10.0` | Pixel resolution in meters |

### `assess-accuracy`

Run accuracy assessment against reference points.

```bash
python -m src.pipeline assess-accuracy --prediction output.tif --reference points.gpkg
python -m src.pipeline assess-accuracy --prediction output.tif --reference ref.geojson --class-field class_name
```

| Option | Default | Description |
|--------|---------|-------------|
| `--prediction` (required) | - | Path to classification raster |
| `--reference` (required) | - | Path to reference points file |
| `--class-field` | `LC_CLASS` | Class label column in reference data |
| `--output` | `data/outputs/accuracy` | Output directory for results |

Outputs: `accuracy_metrics.json`, `confusion_matrix.csv`, `per_class_accuracy.csv`, `points_with_predictions.gpkg`.

### `compare-models`

Compare two land cover classification maps pixel-by-pixel.

```bash
python -m src.pipeline compare-models --raster-a prithvi.tif --raster-b segformer.tif \
    --name-a prithvi --name-b segformer
```

| Option | Default | Description |
|--------|---------|-------------|
| `--raster-a` (required) | - | First classification raster |
| `--raster-b` (required) | - | Second classification raster |
| `--name-a` | `model_a` | Name for first model |
| `--name-b` | `model_b` | Name for second model |
| `--output` | `data/outputs/comparisons/<a>_vs_<b>` | Output directory |

Outputs: `agreement_map.tif`, `comparison_report.json`, `confusion_matrix.csv`.
