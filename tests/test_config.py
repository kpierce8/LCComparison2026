"""Tests for configuration loading and validation."""

import pytest
from pathlib import Path
from omegaconf import OmegaConf

from src.config_schema import (
    CLASS_SCHEMA,
    CLASS_COLORS,
    CLASS_NAMES,
    LCANALYSIS_TO_LCCOMPARISON,
    LCANALYSIS_CLASS_NAMES,
    load_config,
    validate_config,
)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestClassSchema:
    def test_seven_classes(self):
        assert len(CLASS_SCHEMA) == 7

    def test_class_indices(self):
        assert CLASS_SCHEMA["water"] == 0
        assert CLASS_SCHEMA["trees"] == 1
        assert CLASS_SCHEMA["shrub"] == 2
        assert CLASS_SCHEMA["grass"] == 3
        assert CLASS_SCHEMA["crops"] == 4
        assert CLASS_SCHEMA["built"] == 5
        assert CLASS_SCHEMA["bare"] == 6

    def test_class_names_inverse(self):
        for name, idx in CLASS_SCHEMA.items():
            assert CLASS_NAMES[idx] == name

    def test_class_colors_complete(self):
        for idx in range(7):
            assert idx in CLASS_COLORS
            assert CLASS_COLORS[idx].startswith("#")

    def test_lcanalysis_mapping_covers_all_source_classes(self):
        assert len(LCANALYSIS_TO_LCCOMPARISON) == 8
        for i in range(8):
            assert i in LCANALYSIS_TO_LCCOMPARISON

    def test_lcanalysis_mapping_targets_valid(self):
        valid_targets = set(CLASS_SCHEMA.values())
        for target in LCANALYSIS_TO_LCCOMPARISON.values():
            assert target in valid_targets

    def test_lcanalysis_class_names(self):
        assert len(LCANALYSIS_CLASS_NAMES) == 8
        assert LCANALYSIS_CLASS_NAMES[0] == "background"
        assert LCANALYSIS_CLASS_NAMES[7] == "built"

    def test_bare_merges_three_classes(self):
        bare_sources = [
            src for src, dst in LCANALYSIS_TO_LCCOMPARISON.items()
            if dst == CLASS_SCHEMA["bare"]
        ]
        assert len(bare_sources) == 3  # background, ground, gravel


class TestLoadConfig:
    def test_load_default(self):
        config = load_config(CONFIG_PATH)
        assert config is not None
        assert "project" in config

    def test_load_with_overrides(self):
        config = load_config(CONFIG_PATH, overrides=["processing.device=cpu"])
        assert config.processing.device == "cpu"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_config_has_required_sections(self):
        config = load_config(CONFIG_PATH)
        assert "existing_models" in config
        assert "processing" in config
        assert "class_schema" in config
        assert "tiles" in config


class TestValidateConfig:
    def test_valid_config(self):
        config = load_config(CONFIG_PATH)
        issues = validate_config(config)
        # May have issues if LCAnalysis2026 path doesn't exist on CI,
        # but shouldn't have structural issues
        structural_issues = [i for i in issues if "Missing required" in i]
        assert len(structural_issues) == 0

    def test_missing_section(self):
        config = OmegaConf.create({"foo": "bar"})
        issues = validate_config(config)
        assert any("Missing required" in i for i in issues)

    def test_invalid_device(self):
        config = OmegaConf.create({
            "project": {},
            "existing_models": {},
            "processing": {"device": "tpu_invalid"},
        })
        issues = validate_config(config)
        assert any("Invalid processing device" in i for i in issues)

    def test_invalid_gee_strategy(self):
        config = OmegaConf.create({
            "project": {},
            "existing_models": {},
            "processing": {"device": "cpu"},
            "gee_export": {"strategy": "ftp"},
        })
        issues = validate_config(config)
        assert any("Invalid GEE export strategy" in i for i in issues)
