"""Model registry for loading and managing embedding models.

Provides a unified interface for creating, loading, and accessing
all supported embedding models.
"""

import logging
from pathlib import Path
from typing import Any

from src.models.embedding_base import EmbeddingModel

logger = logging.getLogger("lccomparison.model_registry")

# Lazy imports to avoid torch dependency at module level
_MODEL_CLASSES: dict[str, type] = {}


def _ensure_registered():
    """Register model classes on first use."""
    if _MODEL_CLASSES:
        return
    from src.models.prithvi_model import PrithviModel
    from src.models.satlas_model import SatlasModel
    from src.models.ssl4eo_model import SSL4EOModel

    _MODEL_CLASSES["prithvi"] = PrithviModel
    _MODEL_CLASSES["satlas"] = SatlasModel
    _MODEL_CLASSES["ssl4eo"] = SSL4EOModel


def get_available_models() -> list[str]:
    """Get list of available model names."""
    _ensure_registered()
    return list(_MODEL_CLASSES.keys())


def create_model(
    model_name: str,
    weights_path: str | Path | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> EmbeddingModel:
    """Create an embedding model instance.

    Args:
        model_name: Model name (prithvi, satlas, ssl4eo).
        weights_path: Path to pre-trained weights.
        device: Device (cpu, cuda).
        **kwargs: Model-specific parameters.

    Returns:
        EmbeddingModel instance (not yet loaded).
    """
    _ensure_registered()

    if model_name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(_MODEL_CLASSES.keys())}"
        )

    model_cls = _MODEL_CLASSES[model_name]
    return model_cls(weights_path=weights_path, device=device, **kwargs)


def load_model(
    model_name: str,
    weights_path: str | Path | None = None,
    device: str = "cpu",
    auto_download: bool = False,
    models_dir: str | Path = "models",
    **kwargs: Any,
) -> EmbeddingModel:
    """Create and load an embedding model.

    Convenience function that creates, optionally downloads weights,
    and loads the model in one step.

    Args:
        model_name: Model name.
        weights_path: Explicit weights path. If None, looks in models_dir.
        device: Device.
        auto_download: Auto-download from HuggingFace if missing.
        models_dir: Base directory for model weights.
        **kwargs: Model-specific parameters.

    Returns:
        Loaded EmbeddingModel ready for inference.
    """
    # Resolve weights path
    if weights_path is None:
        from src.models.model_downloader import ModelDownloader
        downloader = ModelDownloader(models_dir=models_dir)

        if auto_download:
            weights_path = downloader.download(model_name)
        else:
            weights_path = downloader.get_weights_path(model_name)

    model = create_model(model_name, weights_path=weights_path, device=device, **kwargs)
    model.load()
    return model


def get_model_info(model_name: str) -> dict[str, Any]:
    """Get information about a model without loading it."""
    from src.models.model_downloader import MODEL_REGISTRY
    from src.data.preprocessor import MODEL_NORMALIZATION

    info = {
        "name": model_name,
        "available": model_name in get_available_models(),
    }

    if model_name in MODEL_REGISTRY:
        info.update(MODEL_REGISTRY[model_name])

    if model_name in MODEL_NORMALIZATION:
        norm = MODEL_NORMALIZATION[model_name]
        info["input_size"] = norm["input_size"]
        info["bands"] = norm["bands"]

    return info
