# LCComparison2026

Land Cover Model Comparison Framework. Compares foundation models (Prithvi, SatlasPretrain, SSL4EO) against existing LCAnalysis2026 segmentation models for Washington State land cover classification.

## Setup

```bash
conda activate lcanalysis
pip install -e .
```

## Usage

```bash
# Validate configuration
lccompare validate-config

# Initialize project directories
lccompare init

# Discover existing LCAnalysis2026 experiments
lccompare discover-existing

# Check processing status
lccompare status

# Export imagery from GEE (requires authentication)
lccompare export-imagery
```

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

## Project Structure

```
config/          - YAML configuration files
src/             - Source code
data/            - Data artifacts (tiles, predictions, etc.)
models/          - Downloaded model weights
notebooks/       - Analysis notebooks
tests/           - Test suite
```
