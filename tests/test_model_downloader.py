"""Tests for model downloader."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.model_downloader import MODEL_REGISTRY, ModelDownloader


class TestModelRegistry:
    """Test MODEL_REGISTRY configuration."""

    def test_all_models_present(self):
        assert "prithvi" in MODEL_REGISTRY
        assert "satlas" in MODEL_REGISTRY
        assert "ssl4eo" in MODEL_REGISTRY

    def test_prithvi_config(self):
        cfg = MODEL_REGISTRY["prithvi"]
        assert cfg["source"] == "huggingface"
        assert "repo_id" in cfg
        assert "filename" in cfg
        assert "weights_dir" in cfg

    def test_satlas_config(self):
        cfg = MODEL_REGISTRY["satlas"]
        assert cfg["source"] == "huggingface"
        assert "repo_id" in cfg

    def test_ssl4eo_config(self):
        cfg = MODEL_REGISTRY["ssl4eo"]
        assert "weights_dir" in cfg


class TestModelDownloader:
    """Test ModelDownloader functionality."""

    def test_init(self, tmp_path):
        downloader = ModelDownloader(models_dir=str(tmp_path))
        assert downloader.models_dir == tmp_path

    def test_get_weights_path_not_downloaded(self, tmp_path):
        downloader = ModelDownloader(models_dir=str(tmp_path))
        path = downloader.get_weights_path("prithvi")
        assert path is None

    def test_get_weights_path_exists(self, tmp_path):
        # Create fake weights
        weights_dir = tmp_path / "prithvi"
        weights_dir.mkdir()
        weights_file = weights_dir / MODEL_REGISTRY["prithvi"]["filename"]
        weights_file.write_bytes(b"fake_weights")

        downloader = ModelDownloader(models_dir=str(tmp_path))
        path = downloader.get_weights_path("prithvi")
        assert path is not None
        assert path.exists()

    def test_get_status_empty(self, tmp_path):
        downloader = ModelDownloader(models_dir=str(tmp_path))
        status = downloader.get_status()
        assert "prithvi" in status
        assert "satlas" in status
        assert "ssl4eo" in status
        assert not status["prithvi"]["downloaded"]

    def test_get_status_with_downloaded(self, tmp_path):
        # Create fake prithvi weights
        weights_dir = tmp_path / "prithvi"
        weights_dir.mkdir()
        weights_file = weights_dir / MODEL_REGISTRY["prithvi"]["filename"]
        weights_file.write_bytes(b"x" * 1024)

        downloader = ModelDownloader(models_dir=str(tmp_path))
        status = downloader.get_status()
        assert status["prithvi"]["downloaded"]
        assert status["prithvi"]["size_mb"] is not None
        assert not status["satlas"]["downloaded"]

    def test_download_unknown_model(self, tmp_path):
        downloader = ModelDownloader(models_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown model"):
            downloader.download("nonexistent")

    @patch("src.models.model_downloader._HF_AVAILABLE", False)
    def test_download_without_hf(self, tmp_path):
        downloader = ModelDownloader(models_dir=str(tmp_path))
        with pytest.raises(ImportError):
            downloader.download("prithvi")

    def test_download_returns_cached(self, tmp_path):
        # Pre-create weights
        weights_dir = tmp_path / "prithvi"
        weights_dir.mkdir()
        weights_file = weights_dir / MODEL_REGISTRY["prithvi"]["filename"]
        weights_file.write_bytes(b"cached_weights")

        downloader = ModelDownloader(models_dir=str(tmp_path))
        path = downloader.download("prithvi", force=False)
        assert path == weights_file
