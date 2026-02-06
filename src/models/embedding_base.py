"""Abstract base class for embedding models.

All foundation model wrappers (Prithvi, SatLas, SSL4EO) inherit from this
base to provide a consistent interface for embedding generation.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.embedding_base")

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False


class EmbeddingModel(ABC):
    """Abstract base class for satellite image embedding models.

    Provides a consistent interface for:
    - Loading pre-trained weights
    - Extracting embeddings from preprocessed tiles
    - Batch processing with GPU support
    """

    def __init__(
        self,
        model_name: str,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        embedding_dim: int = 768,
    ):
        self.model_name = model_name
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device
        self.embedding_dim = embedding_dim
        self._model: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def _build_model(self) -> Any:
        """Build the model architecture. Returns nn.Module or similar."""
        ...

    @abstractmethod
    def _load_weights(self, checkpoint: Any) -> None:
        """Load pre-trained weights into the model."""
        ...

    @abstractmethod
    def _extract_features(self, batch: Any) -> np.ndarray:
        """Extract embeddings from a preprocessed batch.

        Args:
            batch: Tensor of shape (B, C, H, W).

        Returns:
            Embeddings array of shape (B, embedding_dim).
        """
        ...

    def load(self, weights_path: str | Path | None = None) -> None:
        """Load model and weights.

        Args:
            weights_path: Path to weights. Overrides constructor path.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for embedding models")

        if weights_path:
            self.weights_path = Path(weights_path)

        self._model = self._build_model()

        if self.weights_path and self.weights_path.exists():
            checkpoint = torch.load(
                self.weights_path, map_location=self.device, weights_only=False,
            )
            self._load_weights(checkpoint)
            logger.info(f"Loaded weights from {self.weights_path}")

        if isinstance(self._model, nn.Module):
            self._model = self._model.to(self.device)
            self._model.eval()

        self._loaded = True
        logger.info(f"Model {self.model_name} loaded on {self.device}")

    def extract_embeddings(
        self,
        data: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract embeddings from preprocessed data.

        Args:
            data: Array of shape (N, C, H, W) or (C, H, W) for single.
            batch_size: Processing batch size.

        Returns:
            Embeddings array of shape (N, embedding_dim).
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded. Call load() first.")

        if data.ndim == 3:
            data = data[np.newaxis]

        n_samples = data.shape[0]
        all_embeddings = []

        for i in range(0, n_samples, batch_size):
            batch_data = data[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch_data).float().to(self.device)

            with torch.no_grad():
                embeddings = self._extract_features(batch_tensor)

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def get_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "loaded": self._loaded,
            "weights_path": str(self.weights_path) if self.weights_path else None,
        }
