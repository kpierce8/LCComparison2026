# Architecture

## Design Approach

LCComparison2026 uses a **linear probing** approach to evaluate foundation models for land cover classification:

1. Extract dense feature embeddings from pretrained satellite image models
2. Train lightweight classifiers (XGBoost, RF, MLP, logistic regression) on the embeddings
3. Compare classification quality against existing LCAnalysis2026 models (SAM2, SegFormer)

This approach avoids fine-tuning the foundation models, making it fast and computationally efficient.

## Module Overview

```
src/
  pipeline.py              # Click CLI - all commands
  config_schema.py         # Config loading, validation, CLASS_SCHEMA

  data/                    # Data Pipeline
    tile_manager.py        # Tile grid, status tracking, JSON checkpoint
    gee_export.py          # Google Earth Engine imagery export
    local_imagery.py       # Local raster handling with CRS reprojection
    label_generator.py     # Dynamic World / ESA WorldCover labels via GEE
    custom_labels.py       # Custom label import + validation
    label_manager.py       # Unified label management + train/val splits
    preprocessor.py        # Per-model tile preprocessing

  models/                  # Foundation Models
    embedding_base.py      # Abstract EmbeddingModel base class
    prithvi_model.py       # Prithvi ViT wrapper
    satlas_model.py        # SatLas Swin Transformer wrapper
    ssl4eo_model.py        # SSL4EO ResNet-50 wrapper
    model_registry.py      # Lazy model registry + create/load helpers
    model_downloader.py    # HuggingFace Hub auto-download + caching

  processing/              # Batch Processing
    batch_processor.py     # Batch embedding with checkpoint/resume
    resolution_matcher.py  # Multi-resolution alignment and resampling

  classification/          # Classifiers
    classifier.py          # Unified XGBoost/RF/MLP/Linear classifier
    ensemble.py            # Ensemble methods and hierarchical fusion

  training/                # Training Pipeline
    trainer.py             # End-to-end training orchestration
    validation.py          # Metrics, spatial CV, stratified k-fold

  spatial/                 # Spatial Analysis
    mosaic.py              # Tile mosaicking to GeoTIFF rasters
    focus_area_manager.py  # Boundary layer loading and management
    boundary_processor.py  # Clip rasters to boundary geometries
    zonal_statistics.py    # Per-zone land cover statistics

  validation/              # Accuracy Assessment
    accuracy_assessor.py   # Accuracy vs reference points
    comparison_metrics.py  # Spatial agreement between models
    error_analysis.py      # Spatial error maps, confusion analysis, edge effects
    report_generator.py    # HTML reports with embedded matplotlib plots

  integration/             # External Integration
    existing_model_integration.py  # LCAnalysis2026 experiment discovery

  utils/                   # Utilities
    geo_utils.py           # UTM zones, tile grids, bbox calculations
    logging_utils.py       # Logging setup (file + console)
```

## Data Flow

```
                         Google Earth Engine
                                |
                          export-imagery
                                |
                         data/tiles/*.tif
                                |
                           preprocess
                                |
                    data/embeddings/<model>/preprocessed/
                                |
                       generate-embeddings
                                |
                    data/embeddings/<model>/*.npz
                                |
         +----------------------+----------------------+
         |                                             |
   train-classifier                              predict
         |                                             |
   data/checkpoints/                    data/predictions/<model>/
   <model>_classifier.pkl                              |
                                              mosaic-tiles
                                                       |
                                       data/outputs/landcover_<model>.tif
                                                       |
                      +--------+--------+--------+--------+--------+
                      |        |        |        |        |
              process-by   zonal   assess    compare   ensemble
              -focus       -stats  -accuracy  -models
                      |        |        |        |        |
                by_<layer>/ *_stats  accuracy/ comparisons/  |
                clipped     .csv    metrics   agreement     |
                TIFFs               .json     _map.tif      |
                                    report                  |
                                    .html          fuse-predictions
                                                            |
                                                   fused_classification.tif
                                                   source_map.tif
```

## Key Design Patterns

### Checkpoint/Resume

Both the TileManager and BatchProcessor use JSON-based checkpointing:

- **TileManager** (`data/checkpoints/tile_index.json`): Tracks tile status through the pipeline (pending -> exporting -> exported -> embedding -> embedded -> predicting -> predicted).
- **BatchProcessor** (`data/embeddings/<model>/<model>_checkpoint.json`): Tracks completed/failed tiles during embedding generation. Supports resuming interrupted runs.

### Model Registry

Models are registered lazily in `model_registry.py`. The `load_model()` function handles:
- Looking up model configuration
- Auto-downloading weights via `ModelDownloader` if requested
- Instantiating the correct model class
- Moving to the specified device

### Classifier Interface

`LandCoverClassifier` provides a unified interface across all classifier types:
- `train(X, y, X_val, y_val)` - Train with optional validation set
- `predict(X)` - Return class indices
- `predict_proba(X)` - Return class probabilities
- `save(path)` / `load(path)` - Pickle + JSON metadata serialization

### Spatial Analysis

Spatial operations use rasterio for raster I/O and geopandas for vector data:
- `FocusAreaLayer` wraps a boundary shapefile with feature lookup by ID
- `BoundaryProcessor` clips rasters to geometries using `rasterio.mask`
- `compute_zonal_stats()` calculates per-class pixel counts, areas, and percentages within a zone

### Accuracy Assessment

`AccuracyAssessor` samples raster values at reference point locations:
- Supports both string class names and integer indices in reference data
- Reprojects reference points to raster CRS automatically
- Computes producer's/user's accuracy, kappa, F1, confusion matrix
- `compare_models()` ranks multiple models by overall accuracy

### Error Analysis

`error_analysis.py` provides spatial error diagnostics:
- `compute_error_map()` - Spatial error map from predictions vs reference, top confusion pairs, per-class error rates
- `analyze_class_confusions()` - Confusion matrix pattern analysis with asymmetry detection and omission/commission rates
- `compute_spatial_error_density()` - Grid-based error density with hotspot identification
- `compute_edge_error_rate()` - Edge vs interior error comparison using kernel-based edge detection

### Report Generation

`report_generator.py` produces HTML reports with embedded matplotlib plots:
- Confusion matrix heatmaps (row-normalized with cell annotations)
- Per-class accuracy bar charts (producer's accuracy, user's accuracy, F1)
- Multi-model comparison charts
- Spatial error density maps
- All plots encoded as base64 PNG embedded directly in self-contained HTML files

### Resolution Matching

`ResolutionMatcher` aligns rasters of different resolutions to a common grid:
- Auto-detects finest resolution across inputs
- Computes intersection or union extents
- Reprojects across CRS boundaries using `rasterio.warp`
- Supports nearest neighbor (for classifications) and bilinear/cubic (for probabilities) resampling

### Ensemble Methods

`EnsembleClassifier` combines multiple model predictions:
- **Majority vote** - Per-pixel mode with agreement map
- **Weighted vote** - Model-weighted voting with confidence output
- **Probability average** - Weighted probability averaging with entropy-based uncertainty

`HierarchicalFusion` merges base and refinement predictions:
- **High-res priority** - Refinement overrides base where valid
- **Confidence weighted** - Uses confidence raster to select between base and refinement
- Source tracking map shows which pixels came from each input

## Foundation Models

| Model | Architecture | Input | Output | Source |
|-------|-------------|-------|--------|--------|
| **Prithvi** | Vision Transformer (ViT) | 6-band, 224x224 | 768-dim embedding | IBM/NASA via HuggingFace |
| **SatLas** | Swin Transformer | 3-band (RGB), 256x256 | 1024-dim embedding | Allen AI via HuggingFace |
| **SSL4EO** | ResNet-50 | 6-band, 224x224 | 768-dim embedding | Manual download |

All models implement the `EmbeddingModel` abstract base class:
- `extract_embeddings(batch)` - Process a batch of tiles, return embeddings
- `input_size`, `num_bands`, `embedding_dim` - Model properties

## Dependencies

Core:
- **rasterio** - Raster I/O, CRS transformations
- **geopandas** - Vector data, spatial operations
- **shapely** - Geometry objects
- **PyTorch** - Foundation model inference
- **scikit-learn** - Metrics, cross-validation, logistic regression
- **XGBoost** - Gradient boosted classifier
- **OmegaConf** - Configuration management
- **Click** - CLI framework

Optional:
- **earthengine-api** / **geemap** - GEE imagery export (graceful degradation if unavailable)
- **google-cloud-storage** - GCS export strategy
- **matplotlib** - Report generation (confusion matrices, accuracy charts, error density maps)
- **plotly** - Interactive visualizations
