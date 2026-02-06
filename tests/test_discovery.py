"""Tests for LCAnalysis2026 experiment discovery."""

import pytest
from pathlib import Path

from src.config_schema import CLASS_SCHEMA, LCANALYSIS_TO_LCCOMPARISON
from src.integration.existing_model_integration import (
    discover_experiments,
    save_discovery_results,
)

LCANALYSIS_PATH = Path("/media/ken/data/LCAnalysis2026")


@pytest.mark.skipif(
    not LCANALYSIS_PATH.exists(),
    reason="LCAnalysis2026 not available",
)
class TestDiscovery:
    def test_discovers_experiments(self):
        results = discover_experiments(LCANALYSIS_PATH)
        assert results["num_experiments"] > 0

    def test_finds_complete_experiments(self):
        results = discover_experiments(LCANALYSIS_PATH)
        complete = results["complete_experiments"]
        assert len(complete) >= 2  # segformer, sam2 at minimum

    def test_complete_experiments_have_metrics(self):
        results = discover_experiments(LCANALYSIS_PATH)
        for exp in results["complete_experiments"]:
            assert exp["has_metrics"]
            assert "metrics" in exp
            assert "test/iou" in exp["metrics"]
            assert "test/f1" in exp["metrics"]

    def test_complete_experiments_have_checkpoints(self):
        results = discover_experiments(LCANALYSIS_PATH)
        for exp in results["complete_experiments"]:
            assert len(exp["checkpoints"]) > 0
            assert exp.get("best_checkpoint") is not None

    def test_experiments_have_model_info(self):
        results = discover_experiments(LCANALYSIS_PATH)
        for exp in results["complete_experiments"]:
            assert exp["model_name"] != "unknown"
            assert exp["num_classes"] == 8

    def test_class_mapping_present(self):
        results = discover_experiments(LCANALYSIS_PATH)
        mapping = results["class_mapping"]
        assert "index_mapping" in mapping
        assert "name_mapping" in mapping
        assert len(mapping["index_mapping"]) == 8

    def test_class_mapping_correct(self):
        results = discover_experiments(LCANALYSIS_PATH)
        mapping = results["class_mapping"]["index_mapping"]
        # background -> bare
        assert mapping[0] == CLASS_SCHEMA["bare"]
        # trees -> trees
        assert mapping[1] == CLASS_SCHEMA["trees"]
        # herbaceous -> grass
        assert mapping[3] == CLASS_SCHEMA["grass"]
        # built -> built
        assert mapping[7] == CLASS_SCHEMA["built"]

    def test_schema_info(self):
        results = discover_experiments(LCANALYSIS_PATH)
        assert results["lcanalysis_schema"]["num_classes"] == 8
        assert results["lccomparison_schema"]["num_classes"] == 7

    def test_save_discovery_results(self, tmp_path):
        results = discover_experiments(LCANALYSIS_PATH)
        output_path = tmp_path / "existing_model_config.yaml"
        save_discovery_results(results, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_experiment_has_dataset_info(self):
        results = discover_experiments(LCANALYSIS_PATH)
        for exp in results["complete_experiments"]:
            assert exp.get("dataset") is not None
            assert exp["dataset"] != "unknown"


class TestClassMapping:
    def test_all_source_classes_mapped(self):
        for i in range(8):
            assert i in LCANALYSIS_TO_LCCOMPARISON

    def test_all_targets_valid(self):
        valid = set(CLASS_SCHEMA.values())
        for target in LCANALYSIS_TO_LCCOMPARISON.values():
            assert target in valid

    def test_specific_mappings(self):
        assert LCANALYSIS_TO_LCCOMPARISON[0] == 6  # background -> bare
        assert LCANALYSIS_TO_LCCOMPARISON[1] == 1  # trees -> trees
        assert LCANALYSIS_TO_LCCOMPARISON[2] == 2  # shrubs -> shrub
        assert LCANALYSIS_TO_LCCOMPARISON[3] == 3  # herbaceous -> grass
        assert LCANALYSIS_TO_LCCOMPARISON[4] == 6  # ground -> bare
        assert LCANALYSIS_TO_LCCOMPARISON[5] == 0  # water -> water
        assert LCANALYSIS_TO_LCCOMPARISON[6] == 6  # gravel -> bare
        assert LCANALYSIS_TO_LCCOMPARISON[7] == 5  # built -> built

    def test_bare_merges_three(self):
        bare_idx = CLASS_SCHEMA["bare"]
        sources = [s for s, t in LCANALYSIS_TO_LCCOMPARISON.items() if t == bare_idx]
        assert set(sources) == {0, 4, 6}  # background, ground, gravel
