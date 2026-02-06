# Getting Started

## Overview

LCComparison2026 is a framework for comparing foundation models (Prithvi, SatLas, SSL4EO) against existing LCAnalysis2026 segmentation models for land cover classification. It uses a linear probing approach: extract embeddings from pretrained models, then train lightweight classifiers on top.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda (for environment management)
- Google Earth Engine account (optional, for imagery export)

## Installation

### 1. Set up the conda environment

The project uses the `lcanalysis` conda environment:

```bash
conda activate lcanalysis
```

### 2. Install dependencies

```bash
pip install -e .
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -m src.pipeline --help
```

You should see the list of available commands.

## Quick Start

### Initialize the project

```bash
python -m src.pipeline init
```

This validates the configuration and creates the required data directories.

### Discover existing LCAnalysis2026 models

```bash
python -m src.pipeline discover-existing
```

Scans the LCAnalysis2026 experiments directory and generates `config/existing_model_config.yaml` with model details, metrics, and class mappings.

### Download foundation model weights

```bash
python -m src.pipeline download-models
```

Downloads Prithvi, SatLas, and SSL4EO weights from HuggingFace Hub.

### Check status

```bash
python -m src.pipeline status
```

Shows tile processing progress, model weights status, embeddings, and labels.

## Class Schema

The project uses a 7-class land cover schema:

| Class | Index | Color |
|-------|-------|-------|
| water | 0 | #0077BE |
| trees | 1 | #228B22 |
| shrub | 2 | #90EE90 |
| grass | 3 | #ADFF2F |
| crops | 4 | #FFD700 |
| built | 5 | #DC143C |
| bare  | 6 | #D2B48C |

LCAnalysis2026 uses 8 classes that map to this schema:
- background, ground, gravel -> bare
- herbaceous -> grass
- water, trees, shrubs, built -> direct match (different indices)

## Directory Structure

```
LCComparison2026/
  config/                  # Configuration files
    config.yaml            # Main configuration
    model_configs/         # Per-model configs
    imagery_sources/       # Imagery source configs
    existing_model_config.yaml  # Auto-generated discovery output
  src/                     # Source code
    pipeline.py            # CLI entry point
    config_schema.py       # Config loading and validation
    data/                  # Data pipeline modules
    models/                # Foundation model wrappers
    processing/            # Batch processing
    classification/        # Classifier implementations
    training/              # Training and validation
    spatial/               # Spatial analysis
    validation/            # Accuracy assessment
    integration/           # LCAnalysis2026 integration
    utils/                 # Utilities
  data/                    # Data files (gitignored)
    tiles/                 # Raw imagery tiles
    labels/                # Training labels
    embeddings/            # Model embeddings
    predictions/           # Classification outputs
    outputs/               # Final products
    checkpoints/           # Training checkpoints
  models/                  # Model weights (gitignored)
  tests/                   # Test suite
  docs/                    # Documentation
  logs/                    # Log files
```

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers all modules with 287 tests.

## Next Steps

- [CLI Reference](cli-reference.md) - Full command documentation
- [Configuration](configuration.md) - Config file reference
- [Architecture](architecture.md) - Module design and data flow
- [Workflow Guide](workflow.md) - End-to-end usage walkthrough
