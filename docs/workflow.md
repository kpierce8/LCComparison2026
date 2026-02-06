# Workflow Guide

This guide walks through the complete pipeline from project setup to model comparison.

## 1. Initialize

```bash
python -m src.pipeline init
```

This validates `config/config.yaml` and creates the data directory structure.

## 2. Discover Existing Models

```bash
python -m src.pipeline discover-existing
```

Scans `/media/ken/data/LCAnalysis2026/experiments/` and writes `config/existing_model_config.yaml` with:
- Experiment names, model architectures, encoder types
- Final metrics (IoU, F1, accuracy) for complete experiments
- Class schema mapping (8-class to 7-class)
- Checkpoint and data source paths

## 3. Prepare Imagery

### Option A: Export from Google Earth Engine

```bash
# Export Sentinel-2 imagery
python -m src.pipeline export-imagery --source sentinel2

# Monitor export progress
python -m src.pipeline check-exports
```

### Option B: Use Local Imagery

Place GeoTIFF tiles in `data/tiles/`. Each tile should be a multi-band raster in a projected CRS (e.g., EPSG:32610).

## 4. Prepare Labels

### Option A: Generate from Global Products

```bash
# Generate labels from Dynamic World and ESA WorldCover
python -m src.pipeline generate-labels --samples 1000 --confidence 0.7
```

### Option B: Import Custom Labels

```bash
# Import from a GeoJSON/Shapefile/CSV
python -m src.pipeline import-labels my_reference_data.geojson --class-field land_cover
```

Labels should include a class field with values matching the 7-class schema (water, trees, shrub, grass, crops, built, bare) or integer indices (0-6).

## 5. Download Model Weights

```bash
# Download all foundation model weights
python -m src.pipeline download-models

# Or download a specific model
python -m src.pipeline download-models --model prithvi
```

## 6. Preprocess Tiles

Prepare tiles for each model's input requirements (band selection, resizing, normalization):

```bash
python -m src.pipeline preprocess --model prithvi
python -m src.pipeline preprocess --model satlas
python -m src.pipeline preprocess --model ssl4eo
```

## 7. Generate Embeddings

Extract feature embeddings from each foundation model:

```bash
python -m src.pipeline generate-embeddings --model prithvi --device cuda
python -m src.pipeline generate-embeddings --model satlas --device cuda
python -m src.pipeline generate-embeddings --model ssl4eo --device cuda
```

Embeddings are cached as `.npz` files with checkpoint/resume support. If interrupted, re-running with `--resume` (the default) continues from where it left off.

## 8. Train Classifiers

Train a classifier on each model's embeddings:

```bash
# Train with XGBoost (default) and 5-fold cross-validation
python -m src.pipeline train-classifier --model prithvi

# Train with a different method
python -m src.pipeline train-classifier --model satlas --method random_forest

# Skip cross-validation for faster iteration
python -m src.pipeline train-classifier --model ssl4eo --no-cv
```

Training outputs:
- Classifier model: `data/checkpoints/<model>_classifier.pkl`
- Metadata: `data/checkpoints/<model>_classifier_meta.json`
- Cross-validation results logged to console

## 9. Predict

Generate land cover predictions for all tiles:

```bash
python -m src.pipeline predict --model prithvi
python -m src.pipeline predict --model satlas
python -m src.pipeline predict --model ssl4eo
```

## 10. Mosaic

Combine tile predictions into full-extent rasters:

```bash
python -m src.pipeline mosaic-tiles --model prithvi
python -m src.pipeline mosaic-tiles --model satlas
python -m src.pipeline mosaic-tiles --model ssl4eo

# Also create confidence maps
python -m src.pipeline mosaic-tiles --model prithvi --product confidence
```

## 11. Spatial Analysis

### Clip to boundaries and compute zonal statistics

```bash
# Process all configured spatial layers
python -m src.pipeline process-by-focus --model prithvi

# Or compute stats for a specific boundary file
python -m src.pipeline zonal-stats \
    --prediction data/outputs/landcover_prithvi.tif \
    --boundaries data/spatial/counties.shp \
    --id-field NAME
```

## 12. Accuracy Assessment

```bash
python -m src.pipeline assess-accuracy \
    --prediction data/outputs/landcover_prithvi.tif \
    --reference data/validation/reference_points.gpkg \
    --class-field LC_CLASS
```

Outputs in `data/outputs/accuracy/`:
- `accuracy_metrics.json` - Overall accuracy, kappa, F1
- `confusion_matrix.csv` - Full confusion matrix
- `per_class_accuracy.csv` - Producer's/user's accuracy per class
- `points_with_predictions.gpkg` - Reference points with predicted values

## 13. Compare Models

Compare foundation model outputs against each other or against LCAnalysis2026 models:

```bash
# Compare Prithvi vs SegFormer
python -m src.pipeline compare-models \
    --raster-a data/outputs/landcover_prithvi.tif \
    --raster-b /media/ken/data/LCAnalysis2026/experiments/segformer/prediction.tif \
    --name-a prithvi --name-b segformer

# Compare Prithvi vs SatLas
python -m src.pipeline compare-models \
    --raster-a data/outputs/landcover_prithvi.tif \
    --raster-b data/outputs/landcover_satlas.tif \
    --name-a prithvi --name-b satlas
```

Outputs:
- `agreement_map.tif` - Binary raster (1=agree, 0=disagree)
- `comparison_report.json` - Agreement percentage, per-class IoU
- `confusion_matrix.csv` - Cross-tabulation between models

## Monitoring Progress

At any point, check the pipeline status:

```bash
python -m src.pipeline status
```

This shows:
- Tile processing status (pending/exported/embedded/predicted)
- Model weights download status
- Embedding generation progress
- Available label files

## Tips

- Use `--verbose` (or `-v`) on any command for debug logging
- The `--config` global option lets you use alternative config files for different experiments
- Override individual config values with `-o key=value` (on `validate-config`)
- All GEE-dependent operations gracefully degrade if GEE is not authenticated
- Embedding generation supports resume by default; safe to interrupt and restart
