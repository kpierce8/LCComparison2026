"""Prithvi-100M embedding model wrapper.

IBM/NASA Geospatial Foundation Model based on Vision Transformer (ViT).
Designed for multi-spectral Sentinel-2 imagery at 10m resolution.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.prithvi_model")

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

from src.models.embedding_base import EmbeddingModel


class PrithviModel(EmbeddingModel):
    """Prithvi-100M embedding model.

    Architecture: ViT-based with 6 Sentinel-2 input bands.
    Input: (B, 6, 224, 224) preprocessed tiles.
    Output: (B, 768) embedding vectors.

    Uses a generic ViT encoder that works with or without the
    actual Prithvi weights. When weights are available, loads
    the pre-trained encoder.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        embedding_dim: int = 768,
        input_channels: int = 6,
        input_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
    ):
        super().__init__(
            model_name="prithvi",
            weights_path=weights_path,
            device=device,
            embedding_dim=embedding_dim,
        )
        self.input_channels = input_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads

    def _build_model(self) -> Any:
        """Build ViT encoder architecture."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Prithvi model")

        num_patches = (self.input_size // self.patch_size) ** 2

        model = nn.Sequential(
            # Patch embedding: project patches to embedding dim
            _PatchEmbed(
                self.input_channels, self.embedding_dim,
                self.patch_size, self.input_size,
            ),
            # Transformer encoder blocks
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.embedding_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=self.num_layers,
            ),
        )
        return model

    def _load_weights(self, checkpoint: Any) -> None:
        """Load Prithvi pre-trained weights.

        Handles various checkpoint formats from HuggingFace.
        """
        if isinstance(checkpoint, dict):
            # Try common checkpoint key patterns
            state_dict = None
            for key in ["model", "state_dict", "model_state_dict", "encoder"]:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint

            # Attempt partial load (skip mismatched keys)
            model_dict = self._model.state_dict()
            compatible = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            if compatible:
                model_dict.update(compatible)
                self._model.load_state_dict(model_dict)
                logger.info(
                    f"Loaded {len(compatible)}/{len(model_dict)} weight tensors"
                )
            else:
                logger.warning(
                    "No compatible weights found in checkpoint. "
                    "Using random initialization."
                )
        else:
            logger.warning("Unexpected checkpoint format. Using random initialization.")

    def _extract_features(self, batch: Any) -> np.ndarray:
        """Extract embeddings from preprocessed batch.

        Args:
            batch: Tensor (B, 6, 224, 224).

        Returns:
            Embeddings (B, 768).
        """
        features = self._model(batch)  # (B, num_patches, embedding_dim)
        # Global average pooling over patches
        if features.ndim == 3:
            embeddings = features.mean(dim=1)
        else:
            embeddings = features
        return embeddings


class _PatchEmbed(nn.Module):
    """Patch embedding layer for ViT."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, img_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

    def forward(self, x: Any) -> Any:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x
