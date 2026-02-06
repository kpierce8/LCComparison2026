# Configuration Reference

The main configuration file is `config/config.yaml`. It uses YAML format and is loaded with OmegaConf, which supports variable interpolation and dotlist overrides.

## Overriding Configuration

From the CLI:

```bash
python -m src.pipeline validate-config -o "processing.device=cpu" -o "tiles.size=512"
```

Or use a custom config file:

```bash
python -m src.pipeline --config path/to/custom.yaml init
```

---

## Sections

### `project`

General project metadata.

```yaml
project:
  name: "LCComparison2026"
  description: "Land Cover Model Comparison Framework"
  region: "Washington State"
  version: "0.1.0"
```

### `existing_models`

Configuration for discovering LCAnalysis2026 experiments.

```yaml
existing_models:
  base_path: "/media/ken/data/LCAnalysis2026"
  experiments_dir: "experiments"
  config_filename: "config.yaml"
  metrics_filename: "final_metrics.yaml"
  metadata_filename: "experiment_metadata.yaml"
  checkpoint_dir: "checkpoints"
  discovery_output: "config/existing_model_config.yaml"
```

| Field | Description |
|-------|-------------|
| `base_path` | Root directory of LCAnalysis2026 |
| `experiments_dir` | Subdirectory containing experiment folders |
| `config_filename` | Per-experiment config file name |
| `metrics_filename` | Per-experiment metrics file name |
| `discovery_output` | Where to save auto-generated discovery YAML |

### `class_schema`

The 7-class land cover schema and LCAnalysis2026 mapping.

```yaml
class_schema:
  num_classes: 7
  classes:
    water: 0
    trees: 1
    shrub: 2
    grass: 3
    crops: 4
    built: 5
    bare: 6
  colors:
    water: "#0077BE"
    trees: "#228B22"
    shrub: "#90EE90"
    grass: "#ADFF2F"
    crops: "#FFD700"
    built: "#DC143C"
    bare: "#D2B48C"
  lcanalysis_mapping:
    0: 6   # background -> bare
    1: 1   # trees -> trees
    2: 2   # shrubs -> shrub
    3: 3   # herbaceous -> grass
    4: 6   # ground -> bare
    5: 0   # water -> water
    6: 6   # gravel -> bare
    7: 5   # built -> built
```

The `lcanalysis_mapping` maps LCAnalysis2026 8-class indices to this project's 7-class indices.

### `study_area`

Geographic bounds of the study area.

```yaml
study_area:
  name: "Washington State"
  bbox:
    west: -124.85
    south: 45.54
    east: -116.92
    north: 49.00
  crs: "EPSG:4326"
  target_crs: "EPSG:32610"  # UTM Zone 10N
```

### `tiles`

Tile grid configuration.

```yaml
tiles:
  size: 256        # Pixels per side
  resolution: 10.0 # Meters per pixel
  overlap: 0       # Pixel overlap between tiles
  source: "sentinel2"
```

### `gee_export`

Google Earth Engine export settings.

```yaml
gee_export:
  strategy: "drive"          # "drive" or "cloud"
  drive_folder: "LCComparison2026_exports"
  cloud_bucket: null
  collections:
    sentinel2:
      collection_id: "COPERNICUS/S2_SR_HARMONIZED"
      bands: ["B2", "B3", "B4", "B8", "B11", "B12"]
      cloud_filter: 20
      date_range:
        start: "2024-06-01"
        end: "2024-09-30"
    naip:
      collection_id: "USDA/NAIP/DOQQ"
      bands: ["R", "G", "B", "N"]
      year: 2023
  scale: 10
  max_pixels: 1e9
```

| Field | Description |
|-------|-------------|
| `strategy` | Export destination: `drive` (Google Drive) or `cloud` (GCS) |
| `collections` | Per-source imagery collection configuration |
| `scale` | Output pixel resolution in meters |

### `processing`

Hardware and batch processing settings.

```yaml
processing:
  device: "cuda"       # "cuda" or "cpu"
  batch_size: 32
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
```

### `foundation_models`

Foundation model definitions.

```yaml
foundation_models:
  prithvi:
    enabled: true
    model_id: "ibm-nasa-geospatial/Prithvi-100M"
    source: "huggingface"
    weights_dir: "models/prithvi"
    input_bands: ["B2", "B3", "B4", "B8", "B11", "B12"]
    input_size: 224
    embedding_dim: 768
  satlas:
    enabled: true
    model_id: "allenai/satlas-pretrain"
    source: "huggingface"
    weights_dir: "models/satlas"
    input_bands: ["B2", "B3", "B4"]
    input_size: 256
    embedding_dim: 1024
  ssl4eo:
    enabled: true
    model_id: "ssl4eo-s12"
    source: "manual"
    weights_dir: "models/ssl4eo"
    input_bands: ["B2", "B3", "B4", "B8", "B11", "B12"]
    input_size: 224
    embedding_dim: 768
```

| Model | Architecture | Bands | Input Size | Embedding Dim |
|-------|-------------|-------|------------|---------------|
| Prithvi | ViT | 6 (B2,B3,B4,B8,B11,B12) | 224x224 | 768 |
| SatLas | Swin Transformer | 3 (B2,B3,B4) | 256x256 | 1024 |
| SSL4EO | ResNet-50 | 6 (B2,B3,B4,B8,B11,B12) | 224x224 | 768 |

### `embeddings`

Embedding extraction settings.

```yaml
embeddings:
  output_dir: "data/embeddings"
  batch_size: 64
  save_format: "npz"
  normalize: true
```

### `classification`

Classifier head configuration.

```yaml
classification:
  method: "xgboost"   # xgboost, random_forest, mlp, linear
  xgboost:
    n_estimators: 500
    max_depth: 8
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    early_stopping_rounds: 50
  mlp:
    hidden_dims: [512, 256]
    dropout: 0.3
    learning_rate: 0.001
    epochs: 100
  linear:
    regularization: "l2"
    C: 1.0
```

Available classifier methods:

| Method | Description |
|--------|-------------|
| `xgboost` | Gradient boosted trees (primary, best performance) |
| `random_forest` | Random forest ensemble |
| `mlp` | Multi-layer perceptron neural network |
| `linear` | Logistic regression |

### `validation`

Validation and cross-validation settings.

```yaml
validation:
  metrics: ["accuracy", "f1", "iou", "kappa"]
  per_class: true
  confusion_matrix: true
  spatial_metrics: true
  cross_validation:
    enabled: true
    n_folds: 5
    stratified: true
```

### `outputs`

Output file configuration.

```yaml
outputs:
  predictions_dir: "data/predictions"
  reports_dir: "data/outputs"
  figures_dir: "data/outputs/figures"
  format: "geotiff"
  comparison_report: true
```

### `logging`

Logging configuration.

```yaml
logging:
  level: "INFO"        # DEBUG, INFO, WARNING, ERROR
  file: "logs/lccomparison.log"
  console: true
  rich_formatting: true
```

### `paths`

Base directory paths.

```yaml
paths:
  data_dir: "data"
  models_dir: "models"
  config_dir: "config"
  logs_dir: "logs"
```
