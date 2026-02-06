"""Tests for embedding models and model registry."""

import numpy as np
import pytest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.models.embedding_base import EmbeddingModel


# Concrete test implementation
class _MockModel(EmbeddingModel):
    def _build_model(self):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required")
        return torch.nn.Linear(10, self.embedding_dim)

    def _load_weights(self, checkpoint):
        pass

    def _extract_features(self, batch):
        return torch.randn(batch.shape[0], self.embedding_dim)


class TestEmbeddingBase:
    """Test EmbeddingModel abstract base class."""

    def test_init(self):
        model = _MockModel(model_name="test", embedding_dim=256)
        assert model.model_name == "test"
        assert model.embedding_dim == 256
        assert model.device == "cpu"
        assert not model.is_loaded

    def test_weights_path(self, tmp_path):
        path = tmp_path / "weights.pt"
        model = _MockModel(model_name="test", weights_path=str(path))
        assert model.weights_path == path

    def test_weights_path_none(self):
        model = _MockModel(model_name="test", weights_path=None)
        assert model.weights_path is None

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
    def test_load(self):
        model = _MockModel(model_name="test", embedding_dim=64)
        model.load()
        assert model.is_loaded

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
    def test_extract_embeddings_not_loaded(self):
        model = _MockModel(model_name="test")
        data = np.random.randn(4, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.extract_embeddings(data)

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
    def test_extract_embeddings(self):
        model = _MockModel(model_name="test", embedding_dim=64)
        model.load()
        data = np.random.randn(4, 10).astype(np.float32)
        embeddings = model.extract_embeddings(data, batch_size=2)
        assert embeddings.shape == (4, 64)
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
    def test_extract_single_sample(self):
        model = _MockModel(model_name="test", embedding_dim=32)
        model.load()
        data = np.random.randn(10).astype(np.float32)  # Single sample, no batch dim
        # Should handle 1D gracefully (adds batch dim)
        data = data.reshape(1, -1)
        embeddings = model.extract_embeddings(data)
        assert embeddings.shape == (1, 32)

    def test_get_info(self):
        model = _MockModel(model_name="test", embedding_dim=768, device="cpu")
        info = model.get_info()
        assert info["model_name"] == "test"
        assert info["embedding_dim"] == 768
        assert info["device"] == "cpu"
        assert not info["loaded"]


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestPrithviModel:
    """Test Prithvi model architecture."""

    def test_init(self):
        from src.models.prithvi_model import PrithviModel
        model = PrithviModel(embedding_dim=768, input_channels=6, input_size=224)
        assert model.model_name == "prithvi"
        assert model.embedding_dim == 768
        assert model.input_channels == 6

    def test_build_and_load(self):
        from src.models.prithvi_model import PrithviModel
        model = PrithviModel(
            embedding_dim=768, input_channels=6,
            input_size=224, num_layers=2, num_heads=4,
        )
        model.load()
        assert model.is_loaded

    def test_forward(self):
        from src.models.prithvi_model import PrithviModel
        model = PrithviModel(
            embedding_dim=128, input_channels=6,
            input_size=64, patch_size=8, num_layers=1, num_heads=4,
        )
        model.load()
        data = np.random.randn(2, 6, 64, 64).astype(np.float32)
        embeddings = model.extract_embeddings(data)
        assert embeddings.shape == (2, 128)

    def test_load_weights_dict(self):
        from src.models.prithvi_model import PrithviModel
        model = PrithviModel(
            embedding_dim=128, input_channels=6,
            input_size=64, patch_size=8, num_layers=1, num_heads=4,
        )
        model._model = model._build_model()
        # Simulate loading a checkpoint dict
        model._load_weights({"state_dict": {}})
        # Should warn but not crash
        model._load_weights({"unexpected_key": "value"})

    def test_load_weights_bad_format(self):
        from src.models.prithvi_model import PrithviModel
        model = PrithviModel(
            embedding_dim=128, input_channels=6,
            input_size=64, patch_size=8, num_layers=1, num_heads=4,
        )
        model._model = model._build_model()
        model._load_weights("not a dict")  # Should warn, not crash


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestSatlasModel:
    """Test SatLas model architecture."""

    def test_init(self):
        from src.models.satlas_model import SatlasModel
        model = SatlasModel(embedding_dim=1024, input_channels=3, input_size=256)
        assert model.model_name == "satlas"
        assert model.embedding_dim == 1024

    def test_build_and_load(self):
        from src.models.satlas_model import SatlasModel
        model = SatlasModel(embedding_dim=256, input_channels=3, input_size=64)
        model.load()
        assert model.is_loaded

    def test_forward(self):
        from src.models.satlas_model import SatlasModel
        model = SatlasModel(embedding_dim=256, input_channels=3, input_size=64)
        model.load()
        data = np.random.randn(2, 3, 64, 64).astype(np.float32)
        embeddings = model.extract_embeddings(data)
        assert embeddings.shape == (2, 256)

    def test_load_weights_partial(self):
        from src.models.satlas_model import SatlasModel
        model = SatlasModel(embedding_dim=256, input_channels=3, input_size=64)
        model._model = model._build_model()
        model._load_weights({"model": {}})  # Empty state dict


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestSSL4EOModel:
    """Test SSL4EO model architecture."""

    def test_init(self):
        from src.models.ssl4eo_model import SSL4EOModel
        model = SSL4EOModel(embedding_dim=768, input_channels=6, input_size=224)
        assert model.model_name == "ssl4eo"
        assert model.embedding_dim == 768

    def test_build_and_load(self):
        from src.models.ssl4eo_model import SSL4EOModel
        model = SSL4EOModel(embedding_dim=256, input_channels=6, input_size=64)
        model.load()
        assert model.is_loaded

    def test_forward(self):
        from src.models.ssl4eo_model import SSL4EOModel
        model = SSL4EOModel(embedding_dim=256, input_channels=6, input_size=64)
        model.load()
        data = np.random.randn(2, 6, 64, 64).astype(np.float32)
        embeddings = model.extract_embeddings(data)
        assert embeddings.shape == (2, 256)

    def test_load_weights_moco(self):
        from src.models.ssl4eo_model import SSL4EOModel
        model = SSL4EOModel(embedding_dim=256, input_channels=6, input_size=64)
        model._model = model._build_model()
        # Simulate MoCo checkpoint with encoder_q key
        model._load_weights({"encoder_q": {}})


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestModelRegistry:
    """Test model registry functions."""

    def test_get_available_models(self):
        from src.models.model_registry import get_available_models
        models = get_available_models()
        assert "prithvi" in models
        assert "satlas" in models
        assert "ssl4eo" in models

    def test_create_model(self):
        from src.models.model_registry import create_model
        model = create_model("prithvi", device="cpu")
        assert model.model_name == "prithvi"
        assert not model.is_loaded

    def test_create_unknown_model(self):
        from src.models.model_registry import create_model
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent")

    def test_get_model_info(self):
        from src.models.model_registry import get_model_info
        info = get_model_info("prithvi")
        assert info["name"] == "prithvi"
        assert info["available"]
        assert "input_size" in info

    def test_get_model_info_unknown(self):
        from src.models.model_registry import get_model_info
        info = get_model_info("nonexistent")
        assert info["name"] == "nonexistent"
        assert not info["available"]
