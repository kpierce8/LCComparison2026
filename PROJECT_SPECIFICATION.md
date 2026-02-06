# Multi-Resolution Land Cover Mapping System
## Comprehensive Project Specification for Claude Code

---

## Executive Summary

Build a Python-based system that combines Google Earth Engine satellite imagery with multiple publicly available foundation model embeddings to generate and compare land cover classifications at multiple resolutions (10m to 1m). The system will process areas ranging from 100,000 to 1,500,000 acres, integrate with existing land cover models in `/media/ken/data/LCAnalysis2026`, and support custom training data and spatial analysis by administrative boundaries.

**Key Capabilities**:
- Multi-resolution imagery (Sentinel-2 10m, NAIP 1m, custom high-res)
- Multiple embedding models (Prithvi, SatLas, SSL4EO) with auto-download from HuggingFace
- Custom training label import and validation
- Spatial analysis by county, WRIA, and RMZ boundaries
- Comprehensive accuracy assessment with reference points
- Direct comparison with existing LCAnalysis2026 models
- Large-scale processing with checkpointing and resumability

---

## Project Structure

```
gee-embedding-landcover/
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── config.yaml                    # Main configuration
│   ├── model_configs/                 # Model-specific configs
│   │   ├── prithvi.yaml              # 10m multi-spectral
│   │   ├── satlas.yaml               # 1m RGB/NIR
│   │   └── ssl4eo.yaml               # Multi-resolution
│   ├── imagery_sources/               # Imagery source configs
│   │   ├── sentinel2.yaml
│   │   ├── naip.yaml
│   │   ├── planet.yaml
│   │   └── custom.yaml
│   └── existing_model_config.yaml     # Auto-generated from LCAnalysis2026
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                    # Main orchestration & CLI
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── gee_export.py             # GEE image export with cloud strategy
│   │   ├── naip_downloader.py        # NAIP-specific handler
│   │   ├── planet_api.py             # Planet API integration
│   │   ├── local_imagery.py          # Handle local rasters/orthomosaics
│   │   ├── resolution_matcher.py     # Multi-resolution alignment
│   │   ├── tile_manager.py           # Track tiles through pipeline
│   │   ├── preprocessor.py           # Image preprocessing for models
│   │   ├── label_manager.py          # Training label management
│   │   ├── label_generator.py        # Generate from Dynamic World/ESA
│   │   └── custom_labels.py          # Import user-provided labels
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding_base.py         # Abstract base class
│   │   ├── prithvi_model.py          # IBM/NASA Prithvi-100M
│   │   ├── satlas_model.py           # AllenAI SatLas
│   │   ├── ssl4eo_model.py           # SSL4EO models
│   │   ├── finetuning.py             # Fine-tune on custom labels
│   │   ├── model_downloader.py       # HuggingFace auto-download
│   │   └── model_registry.py         # Model loading/management
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── classifier.py             # Classifier training/inference
│   │   ├── ensemble.py               # Multi-model ensemble
│   │   └── active_learning.py        # Suggest points to label
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── batch_processor.py        # Large-scale tile processing
│   │   ├── sampling_strategy.py      # Smart sampling for high-res
│   │   ├── worker.py                 # Worker for parallel jobs
│   │   └── progress_tracker.py       # Checkpointing & progress
│   │
│   ├── spatial/
│   │   ├── __init__.py
│   │   ├── focus_area_manager.py     # County/WRIA/RMZ processing
│   │   ├── zonal_statistics.py       # Stats by spatial zones
│   │   └── boundary_processor.py     # Clip/process by boundaries
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── accuracy_assessor.py      # Accuracy assessment
│   │   ├── error_analysis.py         # Error pattern analysis
│   │   └── comparison_metrics.py     # Model comparison metrics
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   └── existing_model_integration.py  # Interface with LCAnalysis2026
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # End-to-end model training
│   │   ├── augmentation.py           # Data augmentation
│   │   └── validation.py             # K-fold, spatial CV
│   │
│   └── utils/
│       ├── __init__.py
│       ├── geo_utils.py              # CRS handling, tiling
│       ├── visualization.py          # Plotting utilities
│       └── cloud_utils.py            # GCS/Cloud operations
│
├── data/
│   ├── tiles/                        # Imagery tiles by source/resolution
│   │   ├── sentinel2_10m/
│   │   ├── naip_1m/
│   │   ├── planet_3m/
│   │   └── custom_highres/
│   ├── labels/
│   │   ├── generated/                # From Dynamic World/ESA
│   │   └── custom/                   # User-provided
│   ├── spatial/                      # Focus area boundaries
│   │   ├── counties.shp
│   │   ├── wria.shp
│   │   └── rmz.shp
│   ├── validation/
│   │   └── accuracy_points.shp       # User-provided reference points
│   ├── embeddings/                   # Cached embeddings by model
│   │   ├── prithvi/
│   │   └── satlas/
│   ├── predictions/                  # Per-tile predictions
│   ├── outputs/                      # Final mosaicked products
│   │   ├── models/                   # Per-model outputs
│   │   ├── comparisons/              # Model comparisons
│   │   ├── by_county/                # County-level outputs
│   │   ├── by_wria/                  # WRIA-level outputs
│   │   └── by_rmz/                   # RMZ-level outputs
│   └── checkpoints/                  # Processing state
│
├── models/                           # Auto-downloaded model weights
│   ├── prithvi/
│   ├── satlas/
│   └── ssl4eo/
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_results_analysis.ipynb
│
└── tests/
    ├── __init__.py
    ├── test_gee_export.py
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_accuracy.py
```

---

## Configuration Schema

### Main Configuration (`config/config.yaml`)

```yaml
# Project metadata
project:
  name: "multiresolution_landcover_2024"
  description: "Multi-resolution land cover mapping with foundation models"
  version: "1.0.0"
  
# Area of Interest
aoi:
  # Can be: GeoJSON, shapefile, or bounding box
  geometry_path: "data/spatial/aoi.geojson"
  # Or direct bbox: [xmin, ymin, xmax, ymax]
  bbox: null
  crs: "EPSG:4326"
  total_acres: 500000  # For planning/validation
  
# Multi-resolution imagery sources
imagery_sources:
  # Medium resolution (10m) - full coverage
  - name: "sentinel2"
    enabled: true
    resolution_m: 10
    type: "gee"
    coverage: "full"  # Process entire AOI
    config:
      collection: "COPERNICUS/S2_SR_HARMONIZED"
      start_date: "2023-01-01"
      end_date: "2023-12-31"
      bands: ["B2", "B3", "B4", "B8", "B11", "B12"]
      cloud_cover_max: 20
      reducer: "median"
      
  # High resolution (1m) - sampled coverage
  - name: "naip"
    enabled: true
    resolution_m: 1
    type: "gee"
    coverage: "sampled"  # "full", "sampled", or "roi"
    sampling_strategy:
      method: "systematic_grid"  # or "stratified", "targeted"
      spacing_m: 5000  # Sample every 5km
      tile_size_m: 512  # 512m x 512m tiles at 1m res
    config:
      collection: "USDA/NAIP/DOQQ"
      start_date: "2022-01-01"
      end_date: "2023-12-31"
      bands: ["R", "G", "B", "N"]
      
  # Very high resolution - custom orthomosaics
  - name: "custom_orthomosaic"
    enabled: false
    resolution_m: 0.5
    type: "local"
    coverage: "roi"
    config:
      data_dir: "data/custom_imagery/orthomosaics"
      file_pattern: "*.tif"
      crs: "EPSG:32610"
      bands: ["R", "G", "B", "NIR"]

# GEE Export Configuration
gee_export:
  strategy: "cloud"  # "cloud" or "drive" (auto-select based on area)
  bucket: "gs://your-gcs-bucket/gee-exports"
  max_pixels: 1e8  # Per export task
  file_format: "GeoTIFF"
  crs: "auto"  # Auto-select appropriate UTM zone
  scale: 10  # meters
  
# Tiling strategy (resolution-adaptive)
tiling:
  strategy: "resolution_adaptive"
  
  # For medium res (10m)
  medium_res:
    tile_size_px: 512  # 512px at 10m = 5.12km
    overlap_px: 64
    
  # For high res (1m)
  high_res:
    tile_size_px: 512  # 512px at 1m = 512m
    overlap_px: 64
    max_tiles_per_source: 10000
    
  # Very high res (0.5m)
  very_high_res:
    tile_size_px: 512  # 512px at 0.5m = 256m
    overlap_px: 32
    max_tiles_per_source: 1000

# Label Management
labels:
  # Auto-generated from existing products
  generated:
    sources:
      - name: "dynamic_world"
        enabled: true
        gee_collection: "GOOGLE/DYNAMICWORLD/V1"
        confidence_threshold: 0.6
        samples_per_class: 500
        
      - name: "esa_worldcover"
        enabled: true
        gee_collection: "ESA/WorldCover/v200"
        samples_per_class: 300
        
    strategy: "stratified_random"
    min_distance_m: 100
    validation_split: 0.2
    
  # User-provided custom labels
  custom:
    enabled: true
    sources:
      - name: "field_survey_2024"
        path: "data/labels/custom/field_points.geojson"
        format: "geojson"  # or "shapefile", "csv", "gpkg"
        class_field: "land_cover"
        confidence_field: "confidence"  # Optional
        date_field: "survey_date"  # Optional
        coordinate_system: "EPSG:4326"
        
      - name: "photo_interpreted"
        path: "data/labels/custom/interpreted_polygons.shp"
        format: "shapefile"
        class_field: "lc_class"
        rasterize: true
        
    validation:
      check_crs: true
      check_within_aoi: true
      check_class_names: true
      flag_duplicates: true  # Within 5m
      min_points_per_class: 20
      
    augmentation:
      spatial_buffer_m: 5
      temporal_matching: true
      
  # Class schema
  classes:
    water: 0
    trees: 1
    shrub: 2
    grass: 3
    crops: 4
    built: 5
    bare: 6
    
  # Class mapping from different sources
  class_mapping:
    dynamic_world:
      water: [0]
      trees: [1]
      grass: [2]
      shrub: [2]
      crops: [4]
      built: [6]
      bare: [8]
      
    esa_worldcover:
      water: [80]
      trees: [10]
      shrub: [20]
      grass: [30]
      crops: [40]
      built: [50]
      bare: [60]
      
    custom:
      "Water": "water"
      "Forest": "trees"
      "Agriculture": "crops"
      "Urban": "built"
      "Grassland": "grass"
      "Bareland": "bare"
      "Shrubland": "shrub"

# Embedding Models
embedding_models:
  # Medium-res specialist (10m, multi-spectral)
  - name: "prithvi"
    enabled: true
    resolution_target: "medium"  # 10-30m
    input_bands: ["blue", "green", "red", "nir", "swir1", "swir2"]
    input_size: [224, 224]
    hf_repo: "ibm-nasa-geospatial/Prithvi-100M"
    hf_file: "Prithvi_100M.pt"
    embedding_dim: 512
    
  # High-res specialist (0.5-3m, RGB/NIR)
  - name: "satlas"
    enabled: true
    resolution_target: "high"  # 0.5-5m
    input_bands: ["red", "green", "blue", "nir"]
    input_size: [512, 512]
    hf_repo: "allenai/satlas-pretrain"
    hf_file: "swinb_satlas_pretrain.pth"
    embedding_dim: 1024
    
  # Multi-resolution model
  - name: "ssl4eo"
    enabled: false
    resolution_target: "multi"
    input_bands: ["red", "green", "blue"]
    input_size: [256, 256]
    hf_repo: "your-org/ssl4eo-s12"
    hf_file: "ssl4eo_moco.pth"
    embedding_dim: 768

# Classification
classification:
  # Approach 1: Pre-trained embeddings + lightweight classifier
  pretrained_embedding:
    enabled: true
    algorithm: "random_forest"
    n_estimators: 200
    max_depth: 25
    class_weight: "balanced"
    
  # Approach 2: Fine-tune embedding model
  finetuned_model:
    enabled: false
    base_model: "satlas"
    freeze_backbone: false
    learning_rate: 1e-4
    epochs: 50
    batch_size: 32
    augmentation:
      - random_flip
      - random_rotation
      - color_jitter
      - random_crop
    early_stopping:
      patience: 10
      monitor: "val_loss"
      
  # Approach 3: Train from scratch
  from_scratch:
    enabled: false
    architecture: "unet"  # or "deeplabv3", "segformer"
    backbone: "resnet50"
    pretrained_backbone: true
    
  validation:
    method: "spatial_cv"  # or "random_cv", "temporal_cv"
    n_folds: 5
    buffer_m: 500

# Multi-resolution fusion
multi_resolution_fusion:
  enabled: true
  strategy: "hierarchical"  # or "ensemble", "weighted_average"
  
  hierarchical:
    base_resolution: "sentinel2"
    refinement_resolutions: ["naip"]
    aggregation: "high_res_priority"
    
  ensemble:
    weights:
      sentinel2: 0.4
      naip: 0.6
    consensus_threshold: 0.7

# Spatial focus areas
spatial_focus:
  enabled: true
  
  layers:
    - name: "county"
      path: "data/spatial/counties.shp"
      id_field: "COUNTY_NAME"
      enabled: true
      
    - name: "wria"
      path: "data/spatial/wria.shp"
      id_field: "WRIA_ID"
      enabled: true
      
    - name: "rmz"
      path: "data/spatial/rmz.shp"
      id_field: "RMZ_ID"
      enabled: true
      
  processing:
    clip_to_boundaries: true
    separate_outputs: true
    summary_statistics: true
    
  outputs:
    by_feature: true
    naming_pattern: "{model}_{year}_{layer}_{feature_id}"

# Accuracy Assessment
accuracy_assessment:
  enabled: true
  
  reference_data:
    path: "data/validation/accuracy_points.shp"
    class_field: "LC_CLASS"
    confidence_field: "CONFIDENCE"
    date_field: "DATE"
    coordinate_system: "EPSG:4326"
    
  assessment:
    stratified: true
    min_samples_per_class: 30
    buffer_m: 5
    exclude_edge_pixels: true
    
  metrics:
    - overall_accuracy
    - producers_accuracy
    - users_accuracy
    - kappa_coefficient
    - f1_score
    - confusion_matrix
    
  comparison:
    compare_to_existing: true
    existing_models_dir: "/media/ken/data/LCAnalysis2026"
    
  reports:
    confusion_matrix_plot: true
    accuracy_by_class_plot: true
    spatial_error_map: true
    error_summary_table: true
    format: ["html", "pdf", "csv"]

# Integration with existing models
existing_models:
  base_path: "/media/ken/data/LCAnalysis2026"
  
  discovery:
    auto_discover: true  # Scan directory on first run
    
  comparison:
    enabled: true
    output_dir: "data/outputs/comparisons"
    
    metrics:
      - overall_accuracy
      - kappa_coefficient
      - per_class_accuracy
      - confusion_matrix
      - spatial_agreement
      
  output_compatibility:
    match_crs: true
    match_resolution: true
    match_extent: true
    match_class_schema: true
    
  naming_convention:
    # Will be auto-detected from existing models
    pattern: "auto"

# Processing Configuration
processing:
  batch_size: 8
  num_workers: 4
  device: "cuda"  # or "cpu"
  mixed_precision: true
  cache_embeddings: true
  max_memory_gb: 16
  checkpoint_frequency: 100  # tiles
  
  high_res_sampling:
    enabled: true
    method: "active_learning"
    initial_sample_size: 1000
    refinement_iterations: 3
    samples_per_iteration: 500

# Output Configuration
outputs:
  base_dir: "data/outputs"
  
  formats:
    - geotiff
    - cloud_optimized_geotiff
    
  resolution: "native"  # or specify target
  
  products:
    - name: "classification"
      description: "Hard classification map"
      
    - name: "probability"
      description: "Class probabilities (multi-band)"
      
    - name: "confidence"
      description: "Classification confidence"
      
    - name: "multi_model_agreement"
      description: "Agreement between models"
      enabled: true
      
  naming:
    pattern: "{project}_{model}_{resolution}_{year}_{product}.tif"
    
  metadata:
    include_json: true
    include_xml: false
    
# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "logs/pipeline.log"
  console: true
```

---

## Core Dependencies

```txt
# requirements.txt

# Core scientific
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Geospatial
rasterio>=1.3.0
geopandas>=0.13.0
shapely>=2.0.0
fiona>=1.9.0
pyproj>=3.5.0
rasterstats>=0.19.0

# Google Earth Engine
earthengine-api>=0.1.350
geemap>=0.30.0

# Deep learning
torch>=2.0.0
torchvision>=0.15.0

# HuggingFace
huggingface-hub>=0.16.0
transformers>=4.30.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Cloud storage
google-cloud-storage>=2.10.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
click>=8.1.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

---

## Implementation Phases

### Phase 1: Foundation & Discovery (Days 1-2)

**Tasks**:
1. Set up project structure and dependencies
2. Implement configuration system
3. Discover existing LCAnalysis2026 structure
   - Scan directory and extract metadata
   - Document naming conventions
   - Identify class schemas
   - Locate validation data
4. Implement GEE authentication and basic export
5. Create tile management system

**Deliverables**:
- Working project skeleton
- `config/existing_model_config.yaml` auto-generated
- Basic GEE export functionality
- Tile tracking database

**Testing**:
- Small test AOI (10 km²)
- Export 100 tiles from Sentinel-2

### Phase 2: Data Pipeline (Days 3-5)

**Tasks**:
1. Complete GEE export with cloud strategy
2. Implement NAIP downloader
3. Local imagery handler for custom orthomosaics
4. Label generation from Dynamic World/ESA
5. Custom label import and validation
6. Preprocessing pipeline for embedding models

**Deliverables**:
- Full imagery export functionality
- Label generation and import working
- Preprocessing for all model types
- Data validation checks

**Testing**:
- Export medium-res for 100km² area
- Sample high-res for subset
- Generate 1000 training labels
- Import custom shapefile labels

### Phase 3: Model Integration (Days 6-8)

**Tasks**:
1. Implement HuggingFace auto-download
2. Create embedding base class
3. Implement Prithvi model wrapper
4. Implement SatLas model wrapper
5. Batch processing with checkpointing
6. Embedding caching system

**Deliverables**:
- Auto-download working for all models
- Embedding generation pipeline
- Memory-efficient batch processing
- Resume from checkpoint capability

**Testing**:
- Download and load Prithvi weights
- Generate embeddings for 1000 tiles
- Verify checkpoint/resume works

### Phase 4: Classification & Training (Days 9-11)

**Tasks**:
1. Random Forest classifier implementation
2. Training pipeline with cross-validation
3. Prediction on full tile set
4. Mosaic tiles to final raster
5. Optional: Fine-tuning implementation
6. Optional: Active learning

**Deliverables**:
- Working classification pipeline
- Mosaicking functionality
- Model persistence and loading
- Prediction export in standard formats

**Testing**:
- Train on 2000 labels
- Predict full test area
- Create final land cover map

### Phase 5: Spatial Analysis (Days 12-13)

**Tasks**:
1. Focus area manager (county/WRIA/RMZ)
2. Clip predictions to boundaries
3. Zonal statistics calculation
4. Per-feature output organization

**Deliverables**:
- Boundary processing working
- Statistics per county/WRIA/RMZ
- Organized outputs by spatial unit

**Testing**:
- Process by all counties
- Calculate area statistics
- Verify outputs match boundaries

### Phase 6: Validation & Comparison (Days 14-16)

**Tasks**:
1. Accuracy assessment implementation
2. Extract predictions at reference points
3. Calculate all accuracy metrics
4. Visualization and reporting
5. Model comparison framework
6. Integration with existing LCAnalysis2026 outputs

**Deliverables**:
- Complete accuracy assessment
- Comparison reports (HTML/PDF)
- Error maps and statistics
- Existing model comparison

**Testing**:
- Run accuracy assessment on all models
- Compare with existing LCAnalysis2026
- Generate comparison reports

### Phase 7: Multi-Resolution & Ensemble (Days 17-18)

**Tasks**:
1. Resolution matching and alignment
2. Hierarchical fusion implementation
3. Ensemble predictions
4. Agreement/disagreement mapping
5. Multi-source visualization

**Deliverables**:
- Multi-resolution fusion working
- Ensemble land cover maps
- Uncertainty/agreement products

**Testing**:
- Fuse 10m and 1m predictions
- Create ensemble from 2+ models
- Validate improved accuracy

### Phase 8: Polish & Documentation (Days 19-20)

**Tasks**:
1. CLI refinement and help text
2. Error handling and validation
3. Comprehensive README
4. Example notebooks
5. User guide documentation
6. Performance optimization

**Deliverables**:
- Production-ready CLI
- Complete documentation
- Example workflows
- Performance benchmarks

---

## Command-Line Interface

### Setup & Discovery

```bash
# Initialize project
python -m src.pipeline init --config config/config.yaml

# Auto-download models from HuggingFace
python -m src.pipeline download-models

# Discover existing LCAnalysis2026 structure
python -m src.pipeline discover-existing \
    --path /media/ken/data/LCAnalysis2026 \
    --output config/existing_model_config.yaml

# Validate configuration
python -m src.pipeline validate-config --config config/config.yaml
```

### Data Preparation

```bash
# Export imagery from GEE
python -m src.pipeline export-imagery \
    --config config/config.yaml \
    --sources sentinel2,naip

# Check export status
python -m src.pipeline check-exports --config config/config.yaml

# Generate training labels from existing products
python -m src.pipeline generate-labels \
    --config config/config.yaml \
    --output data/labels/generated/

# Import custom labels
python -m src.pipeline import-custom-labels \
    --config config/config.yaml \
    --source data/labels/custom/field_survey.geojson \
    --validate

# Validate all labels
python -m src.pipeline validate-labels --config config/config.yaml
```

### Model Processing

```bash
# Generate embeddings (with auto-resume)
python -m src.pipeline generate-embeddings \
    --config config/config.yaml \
    --model prithvi \
    --source sentinel2 \
    --resume

# Train classifier
python -m src.pipeline train-classifier \
    --config config/config.yaml \
    --model prithvi \
    --labels custom

# Predict on all tiles
python -m src.pipeline predict \
    --config config/config.yaml \
    --model prithvi \
    --resume

# Mosaic tiles to final output
python -m src.pipeline mosaic-tiles \
    --config config/config.yaml \
    --model prithvi \
    --output data/outputs/landcover_prithvi_10m_2023.tif
```

### Fine-tuning (Optional)

```bash
# Fine-tune model on custom labels
python -m src.pipeline finetune-model \
    --config config/config.yaml \
    --base-model satlas \
    --labels custom \
    --epochs 50
```

### Multi-Resolution & Ensemble

```bash
# Fuse multi-resolution predictions
python -m src.pipeline fuse-predictions \
    --config config/config.yaml \
    --sources sentinel2,naip \
    --strategy hierarchical

# Create ensemble from multiple models
python -m src.pipeline ensemble \
    --config config/config.yaml \
    --models prithvi,satlas \
    --strategy weighted
```

### Spatial Analysis

```bash
# Process by county
python -m src.pipeline process-by-focus \
    --config config/config.yaml \
    --layer county \
    --model prithvi

# Calculate zonal statistics
python -m src.pipeline zonal-stats \
    --prediction data/outputs/landcover_prithvi_10m_2023.tif \
    --layer county \
    --output data/outputs/stats/county_stats.csv

# Process all focus areas
python -m src.pipeline process-by-focus \
    --config config/config.yaml \
    --layers county,wria,rmz \
    --model prithvi
```

### Accuracy Assessment

```bash
# Run accuracy assessment
python -m src.pipeline assess-accuracy \
    --prediction data/outputs/landcover_prithvi_10m_2023.tif \
    --reference data/validation/accuracy_points.shp \
    --output data/outputs/accuracy/prithvi_assessment.html

# Compare multiple models
python -m src.pipeline compare-accuracy \
    --models data/outputs/landcover_prithvi_10m_2023.tif,data/outputs/landcover_satlas_1m_2023.tif \
    --reference data/validation/accuracy_points.shp \
    --output data/outputs/accuracy/model_comparison.html

# Compare with existing LCAnalysis2026
python -m src.pipeline compare-models \
    --new-model data/outputs/landcover_prithvi_10m_2023.tif \
    --existing-model /media/ken/data/LCAnalysis2026/model_2024.tif \
    --reference data/validation/accuracy_points.shp \
    --output data/outputs/comparisons/prithvi_vs_existing.html
```

### Active Learning

```bash
# Suggest most informative samples to label
python -m src.pipeline suggest-labels \
    --config config/config.yaml \
    --strategy uncertainty \
    --n-samples 200 \
    --output data/labels/to_label_next.geojson
```

### Utilities

```bash
# Show processing status
python -m src.pipeline status

# Resume interrupted processing
python -m src.pipeline resume --stage embeddings

# Clean cache
python -m src.pipeline clean-cache --older-than 7d

# Export results summary
python -m src.pipeline export-summary \
    --output results_summary.json
```

---

## Key Technical Specifications

### GEE Export Strategy (Large Areas)

For 100k-1.5M acre areas:
- Auto-calculate tile grid with overlap
- Batch export tasks (max 3000 concurrent)
- Export to Google Cloud Storage
- Organized folder structure: `{bucket}/{project}/{source}/tile_{row}_{col}.tif`
- Checkpoint system tracks completed exports
- Automatic retry for failed tasks

### Tile Management

Each tile tracked with:
- `tile_id`, `row`, `col`
- Bounding box and geometry
- Status per stage: `export`, `embedding`, `prediction`
- File paths for all intermediate products
- Metadata: resolution, CRS, bands

Stored in: `data/checkpoints/tile_index.json`

### Processing Estimates

For 1.5M acres at 10m (Sentinel-2):
- Total tiles: ~60,000 (512x512 px)
- GEE Export: 12-24 hours
- Embedding generation: 60-95 hours (4 GPUs)
- Classification: 2-4 hours
- Total: 3-5 days

For high-res sampling (1m NAIP):
- Sample every 5km: ~400 tiles
- Embedding: 4-6 hours (1 GPU)

### Memory Management

- Process tiles in batches
- Cache embeddings to disk immediately
- Free GPU memory after each batch
- Checkpoint every 100 tiles
- Resume from checkpoint on failure

### Output Formats

All rasters as Cloud-Optimized GeoTIFF:
- Compression: LZW or DEFLATE
- Tiling: 256x256 or 512x512
- Overviews: automatically generated
- Metadata: embedded in GeoTIFF tags

---

## Data Specifications

### Class Schema

Standard 7-class schema:
```python
CLASSES = {
    'water': 0,
    'trees': 1,
    'shrub': 2,
    'grass': 3,
    'crops': 4,
    'built': 5,
    'bare': 6
}
```

### Reference Data Requirements

Accuracy assessment points shapefile must include:
- Geometry: Point locations
- `LC_CLASS`: Class label (string or integer)
- Optional: `CONFIDENCE`, `DATE`, `INTERPRETER`
- Minimum 30 points per class recommended
- CRS can be any standard projection (will be reprojected)

### Spatial Focus Layers

County/WRIA/RMZ shapefiles must include:
- Geometry: Polygon boundaries
- ID field specified in config
- Optional: NAME, AREA_ACRES
- CRS can be any standard projection

---

## Quality Assurance

### Validation Checks

**Data Validation**:
- All points within AOI
- Valid class labels
- No duplicate points (within tolerance)
- Sufficient samples per class
- CRS compatibility

**Processing Validation**:
- Tile alignment verification
- Nodata handling
- Edge effects minimization
- Checkpoint integrity

**Output Validation**:
- CRS matches specification
- Resolution correct
- Extent covers AOI
- No data gaps
- Class values valid

### Testing Strategy

**Unit Tests**:
- Geo utilities (CRS transforms, tiling)
- Preprocessing functions
- Model loading and inference
- Accuracy calculations

**Integration Tests**:
- End-to-end on small test area (10km²)
- Mock GEE exports for CI/CD
- Checkpoint/resume functionality

**Validation**:
- Visual inspection of outputs
- Comparison with existing products
- Accuracy metrics on independent validation set

---

## Expected Outputs

### Per-Model Outputs

```
data/outputs/models/prithvi_10m/
├── landcover_2023.tif              # Hard classification
├── probability_2023.tif            # Class probabilities (7-band)
├── confidence_2023.tif             # Classification confidence
├── metadata.json                   # Model info, parameters, stats
└── training_report.html            # Training metrics, confusion matrix
```

### Comparison Outputs

```
data/outputs/comparisons/prithvi_vs_existing/
├── agreement_map.tif               # Binary agreement/disagreement
├── confusion_matrix.csv            # Class-by-class confusion
├── comparison_report.html          # Visual report with plots
├── side_by_side.png                # Map comparison
└── statistics.json                 # Agreement metrics
```

### Spatial Focus Outputs

```
data/outputs/by_county/
├── King/
│   ├── landcover_prithvi_2023.tif
│   └── statistics.csv
├── Pierce/
│   ├── landcover_prithvi_2023.tif
│   └── statistics.csv
└── summary_all_counties.csv
```

### Accuracy Assessment Outputs

```
data/outputs/accuracy/
├── prithvi_assessment/
│   ├── accuracy_report.html        # Full report with plots
│   ├── confusion_matrix.csv
│   ├── per_class_accuracy.csv
│   ├── error_map.png               # Spatial error distribution
│   └── points_with_predictions.gpkg
└── model_comparison.csv            # Compare all models
```

---

## Success Criteria

1. **Data Pipeline**: Successfully export and process 100k+ acres
2. **Model Performance**: Overall accuracy >80% on validation set
3. **Integration**: Direct comparison with LCAnalysis2026 models
4. **Scalability**: Process 1.5M acres in <5 days
5. **Resumability**: Checkpoint/resume works reliably
6. **Multi-Resolution**: Successfully fuse 10m and 1m predictions
7. **Spatial Analysis**: Generate statistics for all counties/WRIAs/RMZs
8. **Documentation**: Complete user guide and examples

---

## Open Items for Implementation

**Will be determined by examining existing LCAnalysis2026**:
1. Exact file naming convention
2. Metadata format (JSON, XML, or other)
3. Specific accuracy metrics used
4. Class schema compatibility
5. Output organization structure

**Will be determined based on user-provided data**:
1. Exact spatial extent of AOI
2. County/WRIA/RMZ boundary details
3. Accuracy assessment point characteristics
4. Preferred output resolution(s)

---

## Notes for Claude Code

### Priority Tasks on Startup

1. **Examine `/media/ken/data/LCAnalysis2026`**:
   - Run discovery script
   - Document structure in `existing_model_config.yaml`
   - Extract naming patterns
   - Identify validation data

2. **Set up development environment**:
   - Create virtual environment
   - Install dependencies
   - Configure GEE authentication
   - Test basic imports

3. **Create test dataset**:
   - Small AOI for development (~10km²)
   - Export test tiles
   - Generate test labels
   - Verify end-to-end pipeline

4. **Implement core pipeline first**:
   - Focus on GEE export → embeddings → classification
   - Get basic workflow running
   - Then add spatial analysis, multi-resolution, etc.

### Development Approach

- **Modular**: Each component should work independently
- **Testable**: Write tests alongside code
- **Configurable**: Use YAML configs, avoid hardcoding
- **Resumable**: Checkpoint frequently, allow resume
- **Documented**: Docstrings and inline comments
- **Validated**: Input validation, error messages

### Communication

- Use logging extensively (INFO level by default)
- Progress bars for long-running operations
- Clear error messages with suggestions
- Summary statistics at end of each stage
