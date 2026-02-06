# LCComparison2026

Land Cover Model Comparison Framework. Compares foundation models (Prithvi, SatLas, SSL4EO) against existing LCAnalysis2026 segmentation models for Washington State land cover classification.

## Approach

Uses **linear probing**: extract embeddings from pretrained satellite image foundation models, then train lightweight classifiers (XGBoost, Random Forest, MLP, Logistic Regression) on top. This avoids fine-tuning, making comparison fast and reproducible.

## Features

- **Multi-model embedding extraction** from Prithvi (ViT), SatLas (Swin), and SSL4EO (ResNet-50)
- **Flexible data pipeline** with GEE export, local imagery, and custom label import
- **Classifier training** with cross-validation and spatial CV support
- **Spatial analysis** with boundary clipping, zonal statistics, and focus area processing
- **Accuracy assessment** with confusion matrices, per-class metrics, and HTML reports
- **Model comparison** with spatial agreement maps and multi-model ranking
- **Ensemble methods** including majority vote, weighted vote, and probability averaging
- **Hierarchical fusion** of multi-resolution predictions with source tracking
- **Checkpoint/resume** support throughout the pipeline

## Pipeline

```
GEE / Local Imagery
        |
   preprocess (per model)
        |
   generate-embeddings
        |
   +----+----+
   |         |
train    predict
   |         |
   |    mosaic-tiles
   |         |
   +----+----+----+----+----+
        |    |    |    |    |
     assess  compare  zonal  ensemble
     accuracy models  stats
        |                   |
     reports          fuse-predictions
```

## Foundation Models

| Model | Architecture | Bands | Input Size | Embedding Dim |
|-------|-------------|-------|------------|---------------|
| Prithvi | Vision Transformer | 6 (B2,B3,B4,B8,B11,B12) | 224x224 | 768 |
| SatLas | Swin Transformer | 3 (RGB) | 256x256 | 1024 |
| SSL4EO | ResNet-50 | 6 (B2,B3,B4,B8,B11,B12) | 224x224 | 768 |

## 7-Class Schema

| Index | Class | Color |
|-------|-------|-------|
| 0 | Water | #0077BE |
| 1 | Trees | #228B22 |
| 2 | Shrub | #90EE90 |
| 3 | Grass | #ADFF2F |
| 4 | Crops | #FFD700 |
| 5 | Built | #DC143C |
| 6 | Bare | #D2B48C |

LCAnalysis2026 8-class mapping: background/ground/gravel -> bare, herbaceous -> grass, water/trees/shrubs/built -> direct match.

## Installation

```bash
conda activate lcanalysis
pip install -e .
```

Verify:

```bash
python -m src.pipeline --help
python -m src.pipeline --version
```

## Quick Start

```bash
# Initialize project directories
python -m src.pipeline init

# Discover existing LCAnalysis2026 experiments
python -m src.pipeline discover-existing

# Download foundation model weights
python -m src.pipeline download-models

# Check pipeline status
python -m src.pipeline status
```

## CLI Commands

| Category | Command | Description |
|----------|---------|-------------|
| Setup | `init` | Initialize project and create directories |
| Setup | `validate-config` | Validate configuration file |
| Setup | `discover-existing` | Scan LCAnalysis2026 experiments |
| Setup | `status` | Show pipeline progress |
| Data | `export-imagery` | Export imagery from Google Earth Engine |
| Data | `check-exports` | Check GEE export task status |
| Data | `generate-labels` | Generate labels from Dynamic World / ESA WorldCover |
| Data | `import-labels` | Import custom training labels |
| Data | `preprocess` | Preprocess tiles for a model |
| Models | `download-models` | Download model weights from HuggingFace |
| Models | `generate-embeddings` | Extract embeddings from tiles |
| Training | `train-classifier` | Train classifier on embeddings |
| Training | `predict` | Generate land cover predictions |
| Spatial | `mosaic-tiles` | Mosaic tile predictions into raster |
| Spatial | `process-by-focus` | Clip predictions to boundaries |
| Spatial | `zonal-stats` | Calculate land cover statistics by zone |
| Validation | `assess-accuracy` | Accuracy assessment with HTML reports |
| Validation | `compare-models` | Pixel-by-pixel model comparison |
| Validation | `compare-accuracy` | Multi-model accuracy comparison |
| Ensemble | `ensemble` | Combine predictions from multiple models |
| Ensemble | `fuse-predictions` | Hierarchical multi-resolution fusion |

See [docs/cli-reference.md](docs/cli-reference.md) for full option details.

## Project Structure

```
config/                  Configuration files
  config.yaml            Main configuration
  model_configs/         Per-model configs (prithvi, satlas, ssl4eo)
  imagery_sources/       Imagery source configs
src/                     Source code
  pipeline.py            CLI entry point (21 commands)
  config_schema.py       Config loading and validation
  data/                  Data pipeline (GEE, labels, preprocessing)
  models/                Foundation model wrappers
  processing/            Batch processing, resolution matching
  classification/        Classifiers, ensemble methods
  training/              Training pipeline, cross-validation
  spatial/               Mosaicking, boundaries, zonal stats
  validation/            Accuracy, comparison, error analysis, reports
  integration/           LCAnalysis2026 discovery
  utils/                 Geo utilities, logging, validation helpers
data/                    Data artifacts (gitignored)
models/                  Downloaded model weights (gitignored)
notebooks/               Example notebooks
tests/                   Test suite (320 tests)
docs/                    Documentation
```

## Testing

```bash
python -m pytest tests/ -v
```

The test suite has 320 tests covering all modules.

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and setup
- [CLI Reference](docs/cli-reference.md) - Full command documentation
- [Configuration](docs/configuration.md) - Config file reference
- [Architecture](docs/architecture.md) - Module design and data flow
- [Workflow Guide](docs/workflow.md) - End-to-end usage walkthrough

## Related

- [LCAnalysis2026](https://github.com/kpierce8/LCAnalysis2026) - The existing segmentation model framework being compared against
