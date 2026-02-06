"""Tests for ensemble and fusion methods."""

from pathlib import Path

import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


class TestMajorityVote:
    """Test majority vote ensemble."""

    def test_unanimous(self):
        from src.classification.ensemble import EnsembleClassifier

        # All models agree
        pred_a = np.full((10, 10), 1, dtype=np.uint8)
        pred_b = np.full((10, 10), 1, dtype=np.uint8)
        pred_c = np.full((10, 10), 1, dtype=np.uint8)

        ec = EnsembleClassifier()
        result = ec.majority_vote({"a": pred_a, "b": pred_b, "c": pred_c})

        assert np.all(result["classification"] == 1)
        assert np.all(result["agreement"] == 1.0)
        assert result["mean_agreement"] == 1.0

    def test_majority_wins(self):
        from src.classification.ensemble import EnsembleClassifier

        pred_a = np.full((10, 10), 1, dtype=np.uint8)
        pred_b = np.full((10, 10), 1, dtype=np.uint8)
        pred_c = np.full((10, 10), 2, dtype=np.uint8)  # minority

        ec = EnsembleClassifier()
        result = ec.majority_vote({"a": pred_a, "b": pred_b, "c": pred_c})

        assert np.all(result["classification"] == 1)
        assert result["n_models"] == 3
        # Agreement should be 2/3
        assert result["mean_agreement"] == pytest.approx(2 / 3, abs=0.01)

    def test_handles_nodata(self):
        from src.classification.ensemble import EnsembleClassifier

        pred_a = np.full((10, 10), 1, dtype=np.uint8)
        pred_b = np.full((10, 10), 255, dtype=np.uint8)  # nodata

        ec = EnsembleClassifier()
        result = ec.majority_vote({"a": pred_a, "b": pred_b})

        # No valid pixels (both must be valid)
        assert result["valid_pixels"] == 0

    def test_empty_raises(self):
        from src.classification.ensemble import EnsembleClassifier

        ec = EnsembleClassifier()
        with pytest.raises(ValueError, match="No predictions"):
            ec.majority_vote({})

    def test_mixed_classes(self):
        from src.classification.ensemble import EnsembleClassifier

        # Left=water, right=trees for all models
        pred_a = np.zeros((10, 10), dtype=np.uint8)
        pred_a[:, 5:] = 1
        pred_b = pred_a.copy()
        pred_c = pred_a.copy()
        # One model disagrees on one column
        pred_c[:, 4] = 1  # instead of 0

        ec = EnsembleClassifier()
        result = ec.majority_vote({"a": pred_a, "b": pred_b, "c": pred_c})

        # Column 4 should be 0 (water) by 2:1 vote
        assert np.all(result["classification"][:, 4] == 0)
        # Column 5 should be 1 (trees) unanimously
        assert np.all(result["classification"][:, 5] == 1)


class TestWeightedVote:
    """Test weighted vote ensemble."""

    def test_equal_weights(self):
        from src.classification.ensemble import EnsembleClassifier

        pred_a = np.full((10, 10), 1, dtype=np.uint8)
        pred_b = np.full((10, 10), 1, dtype=np.uint8)

        ec = EnsembleClassifier()
        result = ec.weighted_vote(
            {"a": pred_a, "b": pred_b},
            {"a": 1.0, "b": 1.0},
        )

        assert np.all(result["classification"] == 1)
        assert result["mean_confidence"] == 1.0

    def test_weight_determines_winner(self):
        from src.classification.ensemble import EnsembleClassifier

        pred_a = np.full((10, 10), 1, dtype=np.uint8)  # weight 0.7
        pred_b = np.full((10, 10), 2, dtype=np.uint8)  # weight 0.3

        ec = EnsembleClassifier()
        result = ec.weighted_vote(
            {"a": pred_a, "b": pred_b},
            {"a": 0.7, "b": 0.3},
        )

        assert np.all(result["classification"] == 1)
        assert result["weights"]["a"] == 0.7
        assert result["mean_confidence"] == pytest.approx(0.7, abs=0.01)

    def test_high_weight_minority_wins(self):
        from src.classification.ensemble import EnsembleClassifier

        # 2 models say trees (weight=0.2 each), 1 says water (weight=0.8)
        pred_a = np.full((10, 10), 1, dtype=np.uint8)
        pred_b = np.full((10, 10), 1, dtype=np.uint8)
        pred_c = np.full((10, 10), 0, dtype=np.uint8)

        ec = EnsembleClassifier()
        result = ec.weighted_vote(
            {"a": pred_a, "b": pred_b, "c": pred_c},
            {"a": 0.2, "b": 0.2, "c": 0.8},
        )

        # Water (0.8) outweighs trees (0.4)
        assert np.all(result["classification"] == 0)


class TestProbabilityAverage:
    """Test probability averaging."""

    def test_uniform_probabilities(self):
        from src.classification.ensemble import EnsembleClassifier

        n_classes = 3
        shape = (5, 5)
        # Both models confident: class 0
        prob_a = np.zeros((n_classes, *shape), dtype=np.float32)
        prob_a[0] = 0.9
        prob_a[1] = 0.05
        prob_a[2] = 0.05
        prob_b = prob_a.copy()

        ec = EnsembleClassifier(n_classes=n_classes)
        result = ec.probability_average({"a": prob_a, "b": prob_b})

        assert np.all(result["classification"] == 0)
        assert result["mean_confidence"] > 0.8

    def test_conflicting_probabilities(self):
        from src.classification.ensemble import EnsembleClassifier

        n_classes = 2
        shape = (5, 5)
        # Model A says class 0, model B says class 1
        prob_a = np.zeros((n_classes, *shape), dtype=np.float32)
        prob_a[0] = 0.9
        prob_a[1] = 0.1
        prob_b = np.zeros((n_classes, *shape), dtype=np.float32)
        prob_b[0] = 0.1
        prob_b[1] = 0.9

        ec = EnsembleClassifier(n_classes=n_classes)
        result = ec.probability_average({"a": prob_a, "b": prob_b})

        # Average should be 0.5/0.5, uncertainty should be high
        assert result["mean_confidence"] == pytest.approx(0.5, abs=0.01)
        assert result["mean_uncertainty"] > 0.9

    def test_weighted_probabilities(self):
        from src.classification.ensemble import EnsembleClassifier

        n_classes = 2
        shape = (5, 5)
        prob_a = np.zeros((n_classes, *shape), dtype=np.float32)
        prob_a[0] = 1.0
        prob_b = np.zeros((n_classes, *shape), dtype=np.float32)
        prob_b[1] = 1.0

        ec = EnsembleClassifier(n_classes=n_classes)
        result = ec.probability_average(
            {"a": prob_a, "b": prob_b},
            weights={"a": 0.8, "b": 0.2},
        )

        # Weighted average: class 0 = 0.8, class 1 = 0.2
        assert np.all(result["classification"] == 0)
        assert result["mean_confidence"] == pytest.approx(0.8, abs=0.01)

    def test_uncertainty_output(self):
        from src.classification.ensemble import EnsembleClassifier

        n_classes = 3
        shape = (5, 5)
        # Model very confident
        prob_a = np.zeros((n_classes, *shape), dtype=np.float32)
        prob_a[0] = 1.0

        ec = EnsembleClassifier(n_classes=n_classes)
        result = ec.probability_average({"a": prob_a})

        assert "uncertainty" in result
        # Perfectly confident -> uncertainty near 0
        assert result["mean_uncertainty"] < 0.01


class TestAgreementMap:
    """Test agreement computation across models."""

    def test_full_agreement(self):
        from src.classification.ensemble import EnsembleClassifier

        pred = np.full((10, 10), 1, dtype=np.uint8)

        ec = EnsembleClassifier()
        result = ec.compute_agreement_map({"a": pred.copy(), "b": pred.copy()})

        assert result["full_agreement_pct"] == 100.0
        assert result["mean_agreement"] == 1.0

    def test_partial_agreement(self):
        from src.classification.ensemble import EnsembleClassifier

        pred_a = np.zeros((10, 10), dtype=np.uint8)
        pred_b = np.zeros((10, 10), dtype=np.uint8)
        pred_b[:5] = 1  # top half different

        ec = EnsembleClassifier()
        result = ec.compute_agreement_map({"a": pred_a, "b": pred_b})

        assert result["full_agreement_pct"] == 50.0

    def test_pairwise_agreement(self):
        from src.classification.ensemble import EnsembleClassifier

        pred_a = np.full((10, 10), 0, dtype=np.uint8)
        pred_b = np.full((10, 10), 0, dtype=np.uint8)
        pred_c = np.full((10, 10), 1, dtype=np.uint8)

        ec = EnsembleClassifier()
        result = ec.compute_agreement_map({"a": pred_a, "b": pred_b, "c": pred_c})

        assert result["pairwise_agreement"]["a_vs_b"] == 100.0
        assert result["pairwise_agreement"]["a_vs_c"] == 0.0

    def test_per_class_agreement(self):
        from src.classification.ensemble import EnsembleClassifier

        pred_a = np.zeros((10, 10), dtype=np.uint8)
        pred_a[:, 5:] = 1
        pred_b = pred_a.copy()

        ec = EnsembleClassifier()
        result = ec.compute_agreement_map({"a": pred_a, "b": pred_b})

        assert "water" in result["per_class"]
        assert "trees" in result["per_class"]
        assert result["per_class"]["water"]["mean_agreement"] == 1.0

    def test_needs_two_models(self):
        from src.classification.ensemble import EnsembleClassifier

        pred = np.full((10, 10), 1, dtype=np.uint8)
        ec = EnsembleClassifier()

        with pytest.raises(ValueError, match="at least 2"):
            ec.compute_agreement_map({"a": pred})


class TestHierarchicalFusion:
    """Test hierarchical multi-resolution fusion."""

    def test_high_res_priority(self):
        from src.classification.ensemble import HierarchicalFusion

        base = np.full((10, 10), 1, dtype=np.uint8)  # trees everywhere
        refinement = np.full((10, 10), 0, dtype=np.uint8)  # water everywhere

        fusion = HierarchicalFusion()
        result = fusion.fuse(base, refinement, strategy="high_res_priority")

        # Refinement should win where available
        assert np.all(result["classification"] == 0)
        assert result["refinement_pct"] == 100.0

    def test_base_fills_gaps(self):
        from src.classification.ensemble import HierarchicalFusion

        base = np.full((10, 10), 1, dtype=np.uint8)
        refinement = np.full((10, 10), 255, dtype=np.uint8)  # nodata
        refinement[:5] = 0  # only top half has refinement

        fusion = HierarchicalFusion()
        result = fusion.fuse(base, refinement, strategy="high_res_priority")

        # Top half should be refinement (0), bottom half should be base (1)
        assert np.all(result["classification"][:5] == 0)
        assert np.all(result["classification"][5:] == 1)
        assert result["base_pixels"] > 0
        assert result["refinement_pixels"] > 0

    def test_confidence_threshold(self):
        from src.classification.ensemble import HierarchicalFusion

        base = np.full((10, 10), 1, dtype=np.uint8)
        refinement = np.full((10, 10), 0, dtype=np.uint8)
        confidence = np.full((10, 10), 0.3, dtype=np.float32)  # low confidence

        fusion = HierarchicalFusion()
        result = fusion.fuse(
            base, refinement,
            refinement_confidence=confidence,
            confidence_threshold=0.5,
            strategy="high_res_priority",
        )

        # Low confidence -> revert to base
        assert np.all(result["classification"] == 1)
        assert result["base_pct"] == 100.0

    def test_confidence_weighted_strategy(self):
        from src.classification.ensemble import HierarchicalFusion

        base = np.full((10, 10), 1, dtype=np.uint8)
        refinement = np.full((10, 10), 0, dtype=np.uint8)
        confidence = np.full((10, 10), 0.8, dtype=np.float32)

        fusion = HierarchicalFusion()
        result = fusion.fuse(
            base, refinement,
            refinement_confidence=confidence,
            confidence_threshold=0.5,
            strategy="confidence_weighted",
        )

        # High confidence -> use refinement
        assert np.all(result["classification"] == 0)

    def test_shape_mismatch_raises(self):
        from src.classification.ensemble import HierarchicalFusion

        base = np.full((10, 10), 1, dtype=np.uint8)
        refinement = np.full((20, 20), 0, dtype=np.uint8)

        fusion = HierarchicalFusion()
        with pytest.raises(ValueError, match="Shape mismatch"):
            fusion.fuse(base, refinement)

    def test_source_map(self):
        from src.classification.ensemble import HierarchicalFusion

        base = np.full((10, 10), 1, dtype=np.uint8)
        refinement = np.full((10, 10), 255, dtype=np.uint8)
        refinement[:3] = 2  # small region

        fusion = HierarchicalFusion()
        result = fusion.fuse(base, refinement)

        assert "source_map" in result
        # Top 3 rows: source=2 (refinement)
        assert np.all(result["source_map"][:3] == 2)
        # Bottom 7 rows: source=1 (base)
        assert np.all(result["source_map"][3:] == 1)


@pytest.mark.skipif(not _RASTERIO_AVAILABLE, reason="rasterio required")
class TestSaveEnsembleOutputs:
    """Test saving ensemble products as GeoTIFFs."""

    def _make_profile(self):
        transform = from_bounds(0, 0, 100, 100, 10, 10)
        return {
            "driver": "GTiff", "dtype": "uint8", "count": 1,
            "height": 10, "width": 10, "crs": "EPSG:32610",
            "transform": transform, "nodata": 255,
        }

    def test_save_classification(self, tmp_path):
        from src.classification.ensemble import save_ensemble_outputs

        result = {
            "classification": np.full((10, 10), 1, dtype=np.uint8),
            "n_models": 2,
            "models": ["a", "b"],
        }

        outputs = save_ensemble_outputs(
            result, tmp_path / "out", self._make_profile(),
        )

        assert "classification" in outputs
        assert Path(outputs["classification"]).exists()
        assert "metadata" in outputs

    def test_save_full_products(self, tmp_path):
        from src.classification.ensemble import save_ensemble_outputs

        result = {
            "classification": np.full((10, 10), 1, dtype=np.uint8),
            "confidence": np.full((10, 10), 0.9, dtype=np.float32),
            "uncertainty": np.full((10, 10), 0.1, dtype=np.float32),
            "agreement_map": np.full((10, 10), 0.8, dtype=np.float32),
            "n_models": 3,
            "models": ["a", "b", "c"],
        }

        outputs = save_ensemble_outputs(
            result, tmp_path / "full", self._make_profile(),
        )

        assert "classification" in outputs
        assert "confidence" in outputs
        assert "uncertainty" in outputs
        assert "agreement" in outputs
        assert "metadata" in outputs

    def test_save_probabilities(self, tmp_path):
        from src.classification.ensemble import save_ensemble_outputs

        probs = np.random.rand(7, 10, 10).astype(np.float32)
        result = {
            "probabilities": probs,
            "classification": np.argmax(probs, axis=0).astype(np.uint8),
            "n_models": 1,
        }

        outputs = save_ensemble_outputs(
            result, tmp_path / "prob", self._make_profile(),
        )

        assert "probabilities" in outputs
        with rasterio.open(outputs["probabilities"]) as src:
            assert src.count == 7

    def test_save_source_map(self, tmp_path):
        from src.classification.ensemble import save_ensemble_outputs

        result = {
            "classification": np.full((10, 10), 1, dtype=np.uint8),
            "source_map": np.full((10, 10), 2, dtype=np.uint8),
            "strategy": "high_res_priority",
        }

        outputs = save_ensemble_outputs(
            result, tmp_path / "fused", self._make_profile(),
            name_prefix="fusion",
        )

        assert "source_map" in outputs
        assert "fusion_source.tif" in outputs["source_map"]
