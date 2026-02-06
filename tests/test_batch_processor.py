"""Tests for batch processor with checkpoint/resume."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.processing.batch_processor import BatchProcessor, EmbeddingCheckpoint


class TestEmbeddingCheckpoint:
    """Test checkpoint persistence."""

    def test_init_no_file(self, tmp_path):
        cp = EmbeddingCheckpoint(tmp_path / "checkpoint.json")
        assert cp.n_completed == 0
        assert cp.n_failed == 0

    def test_init_and_save(self, tmp_path):
        cp = EmbeddingCheckpoint(tmp_path / "checkpoint.json")
        cp.init("prithvi", 100)
        cp.save()
        assert (tmp_path / "checkpoint.json").exists()

    def test_resume(self, tmp_path):
        # Create a checkpoint
        cp1 = EmbeddingCheckpoint(tmp_path / "checkpoint.json")
        cp1.init("prithvi", 50)
        cp1.mark_completed(["tile_001", "tile_002"], 0)
        cp1.mark_failed("tile_003", "bad data")

        # Resume from it
        cp2 = EmbeddingCheckpoint(tmp_path / "checkpoint.json")
        assert cp2.n_completed == 2
        assert cp2.n_failed == 1
        assert "tile_001" in cp2.completed_set
        assert "tile_002" in cp2.completed_set

    def test_get_summary(self, tmp_path):
        cp = EmbeddingCheckpoint(tmp_path / "checkpoint.json")
        cp.init("satlas", 200)
        cp.mark_completed(["a", "b", "c"], 0)
        summary = cp.get_summary()
        assert summary["model"] == "satlas"
        assert summary["completed"] == 3
        assert summary["total"] == 200

    def test_mark_completed_incremental(self, tmp_path):
        cp = EmbeddingCheckpoint(tmp_path / "checkpoint.json")
        cp.init("ssl4eo", 10)
        cp.mark_completed(["t1", "t2"], 0)
        cp.mark_completed(["t3"], 1)
        assert cp.n_completed == 3
        assert cp.completed_set == {"t1", "t2", "t3"}


class TestBatchProcessor:
    """Test batch processing pipeline."""

    def _make_mock_model(self, embedding_dim=64):
        model = MagicMock()
        model.model_name = "test_model"
        model.embedding_dim = embedding_dim

        def extract(data, batch_size=32):
            return np.random.randn(data.shape[0], embedding_dim).astype(np.float32)

        model.extract_embeddings = MagicMock(side_effect=extract)
        return model

    def _make_mock_preprocessor(self, n_bands=6, size=64):
        preprocessor = MagicMock()

        def preprocess_file(path, return_tensor=False):
            return {
                "data": np.random.randn(n_bands, size, size).astype(np.float32),
                "valid_mask": np.ones((size, size), dtype=bool),
            }

        preprocessor.preprocess_file = MagicMock(side_effect=preprocess_file)
        return preprocessor

    def test_init(self, tmp_path):
        model = self._make_mock_model()
        preprocessor = self._make_mock_preprocessor()
        processor = BatchProcessor(
            model=model,
            preprocessor=preprocessor,
            output_dir=tmp_path / "embeddings",
            batch_size=4,
        )
        assert processor.batch_size == 4
        assert processor.output_dir.exists()

    def test_process_tiles(self, tmp_path):
        model = self._make_mock_model(embedding_dim=32)
        preprocessor = self._make_mock_preprocessor()

        # Create fake tile files
        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        tile_paths = {}
        for i in range(5):
            p = tile_dir / f"tile_{i:03d}.tif"
            p.write_bytes(b"fake")
            tile_paths[f"tile_{i:03d}"] = p

        processor = BatchProcessor(
            model=model,
            preprocessor=preprocessor,
            output_dir=tmp_path / "embeddings",
            batch_size=3,
        )

        result = processor.process_tiles(tile_paths, resume=False)

        assert "summary" in result
        assert result["summary"]["total_tiles"] == 5
        assert result["summary"]["processed"] == 5
        assert result["summary"]["failed"] == 0
        assert len(result["embeddings"]) == 5

    def test_cache_embeddings(self, tmp_path):
        model = self._make_mock_model(embedding_dim=16)
        preprocessor = self._make_mock_preprocessor()

        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        tile_paths = {}
        for i in range(3):
            p = tile_dir / f"t{i}.tif"
            p.write_bytes(b"fake")
            tile_paths[f"t{i}"] = p

        out_dir = tmp_path / "embeddings"
        processor = BatchProcessor(
            model=model,
            preprocessor=preprocessor,
            output_dir=out_dir,
            batch_size=10,
            cache_embeddings=True,
        )

        processor.process_tiles(tile_paths, resume=False)

        # Check cached files exist
        for tid in tile_paths:
            cache_path = out_dir / f"{tid}.npz"
            assert cache_path.exists()
            data = np.load(cache_path)
            assert "embedding" in data
            assert data["embedding"].shape == (16,)

    def test_resume_from_checkpoint(self, tmp_path):
        model = self._make_mock_model(embedding_dim=16)
        preprocessor = self._make_mock_preprocessor()

        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        tile_paths = {}
        for i in range(6):
            p = tile_dir / f"tile_{i}.tif"
            p.write_bytes(b"fake")
            tile_paths[f"tile_{i}"] = p

        out_dir = tmp_path / "embeddings"
        processor = BatchProcessor(
            model=model,
            preprocessor=preprocessor,
            output_dir=out_dir,
            batch_size=2,
            cache_embeddings=True,
        )

        # First run: process all
        result1 = processor.process_tiles(tile_paths, resume=False)
        assert result1["summary"]["processed"] == 6

        # Reset the model call count
        model.extract_embeddings.reset_mock()

        # Second run with resume: should use cached
        result2 = processor.process_tiles(tile_paths, resume=True)
        # All tiles already completed in checkpoint, so nothing new to process
        assert result2["summary"]["processed"] == 0

    def test_load_cached_embeddings(self, tmp_path):
        model = self._make_mock_model(embedding_dim=8)
        preprocessor = self._make_mock_preprocessor()

        out_dir = tmp_path / "embeddings"
        out_dir.mkdir(parents=True)

        # Manually save some cached embeddings
        for i in range(3):
            np.savez_compressed(
                out_dir / f"cached_{i}.npz",
                embedding=np.ones(8, dtype=np.float32) * i,
                tile_id=f"cached_{i}",
            )

        processor = BatchProcessor(
            model=model,
            preprocessor=preprocessor,
            output_dir=out_dir,
        )

        loaded = processor.load_cached_embeddings()
        assert len(loaded) == 3
        assert loaded["cached_0"].shape == (8,)
        np.testing.assert_array_almost_equal(loaded["cached_0"], np.zeros(8))
        np.testing.assert_array_almost_equal(loaded["cached_2"], np.ones(8) * 2)

    def test_load_cached_specific_tiles(self, tmp_path):
        model = self._make_mock_model(embedding_dim=4)
        preprocessor = self._make_mock_preprocessor()

        out_dir = tmp_path / "embeddings"
        out_dir.mkdir(parents=True)
        np.savez_compressed(out_dir / "a.npz", embedding=np.ones(4))
        np.savez_compressed(out_dir / "b.npz", embedding=np.zeros(4))

        processor = BatchProcessor(
            model=model, preprocessor=preprocessor, output_dir=out_dir,
        )

        loaded = processor.load_cached_embeddings(tile_ids=["a"])
        assert "a" in loaded
        assert "b" not in loaded

    def test_get_progress_empty(self, tmp_path):
        model = self._make_mock_model()
        preprocessor = self._make_mock_preprocessor()
        processor = BatchProcessor(
            model=model, preprocessor=preprocessor,
            output_dir=tmp_path / "emb",
        )
        progress = processor.get_progress()
        assert progress["completed"] == 0

    def test_clear_cache(self, tmp_path):
        model = self._make_mock_model(embedding_dim=4)
        preprocessor = self._make_mock_preprocessor()

        out_dir = tmp_path / "embeddings"
        out_dir.mkdir(parents=True)
        np.savez_compressed(out_dir / "t1.npz", embedding=np.ones(4))
        np.savez_compressed(out_dir / "t2.npz", embedding=np.ones(4))

        processor = BatchProcessor(
            model=model, preprocessor=preprocessor, output_dir=out_dir,
        )

        removed = processor.clear_cache()
        assert removed == 2
        assert len(list(out_dir.glob("*.npz"))) == 0

    def test_process_with_preprocess_failure(self, tmp_path):
        model = self._make_mock_model(embedding_dim=16)

        # Preprocessor that fails on specific tiles
        preprocessor = MagicMock()
        call_count = [0]

        def flaky_preprocess(path, return_tensor=False):
            call_count[0] += 1
            if "bad" in str(path):
                raise ValueError("bad tile data")
            return {
                "data": np.random.randn(6, 64, 64).astype(np.float32),
                "valid_mask": np.ones((64, 64), dtype=bool),
            }

        preprocessor.preprocess_file = MagicMock(side_effect=flaky_preprocess)

        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        tile_paths = {
            "good_1": tile_dir / "good_1.tif",
            "bad_tile": tile_dir / "bad_tile.tif",
            "good_2": tile_dir / "good_2.tif",
        }
        for p in tile_paths.values():
            p.write_bytes(b"fake")

        processor = BatchProcessor(
            model=model, preprocessor=preprocessor,
            output_dir=tmp_path / "emb",
            batch_size=10,
        )

        result = processor.process_tiles(tile_paths, resume=False)
        assert result["summary"]["processed"] == 2
        assert result["summary"]["failed"] == 1
