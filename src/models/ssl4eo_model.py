"""SSL4EO-S12 embedding model wrapper.

Self-Supervised Learning for Earth Observation using Sentinel-1/2.
Based on ResNet-50 with MoCo pre-training.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.ssl4eo_model")

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

from src.models.embedding_base import EmbeddingModel


class SSL4EOModel(EmbeddingModel):
    """SSL4EO-S12 embedding model.

    Architecture: ResNet-50 with MoCo pre-training.
    Input: (B, 6, 224, 224) multi-spectral tiles.
    Output: (B, 768) embedding vectors.

    Uses a ResNet-like architecture. When weights are available,
    loads the MoCo encoder.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        embedding_dim: int = 768,
        input_channels: int = 6,
        input_size: int = 224,
    ):
        super().__init__(
            model_name="ssl4eo",
            weights_path=weights_path,
            device=device,
            embedding_dim=embedding_dim,
        )
        self.input_channels = input_channels
        self.input_size = input_size

    def _build_model(self) -> Any:
        """Build ResNet-50-like feature extractor."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SSL4EO model")

        model = nn.Sequential(
            # Stem
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Stage 1
            _ResBlock(64, 64),
            _ResBlock(64, 64),
            _ResBlock(64, 64),
            # Stage 2
            _ResBlock(64, 128, stride=2),
            _ResBlock(128, 128),
            _ResBlock(128, 128),
            _ResBlock(128, 128),
            # Stage 3
            _ResBlock(128, 256, stride=2),
            _ResBlock(256, 256),
            _ResBlock(256, 256),
            _ResBlock(256, 256),
            _ResBlock(256, 256),
            _ResBlock(256, 256),
            # Stage 4
            _ResBlock(256, 512, stride=2),
            _ResBlock(512, 512),
            _ResBlock(512, 512),
            # Pool + project
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.embedding_dim),
        )
        return model

    def _load_weights(self, checkpoint: Any) -> None:
        """Load SSL4EO MoCo pre-trained weights."""
        if isinstance(checkpoint, dict):
            state_dict = None
            # MoCo stores encoder under specific keys
            for key in ["state_dict", "model", "encoder_q", "backbone"]:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint

            # Strip 'module.' prefix if present (DataParallel)
            cleaned = {}
            for k, v in state_dict.items():
                clean_key = k.replace("module.", "").replace("encoder_q.", "")
                cleaned[clean_key] = v

            model_dict = self._model.state_dict()
            compatible = {
                k: v for k, v in cleaned.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            if compatible:
                model_dict.update(compatible)
                self._model.load_state_dict(model_dict)
                logger.info(f"Loaded {len(compatible)}/{len(model_dict)} weight tensors")
            else:
                logger.warning("No compatible weights found. Using random initialization.")
        else:
            logger.warning("Unexpected checkpoint format. Using random initialization.")

    def _extract_features(self, batch: Any) -> np.ndarray:
        """Extract embeddings from preprocessed batch.

        Args:
            batch: Tensor (B, 6, 224, 224).

        Returns:
            Embeddings (B, 768).
        """
        return self._model(batch)


class _ResBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Any) -> Any:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
