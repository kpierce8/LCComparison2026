"""HuggingFace model auto-download with caching and progress tracking.

Downloads foundation model weights from HuggingFace Hub or loads
from local checkpoints. Handles caching and provides download status.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("lccomparison.model_downloader")

try:
    from huggingface_hub import hf_hub_download, HfApi
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


# Model registry: name -> download metadata
MODEL_REGISTRY = {
    "prithvi": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-100M",
        "filename": "Prithvi_100M.pt",
        "source": "huggingface",
        "weights_dir": "models/prithvi",
    },
    "satlas": {
        "repo_id": "allenai/satlas-pretrain",
        "filename": "aerial_swinb_si.pth",
        "source": "huggingface",
        "weights_dir": "models/satlas",
    },
    "ssl4eo": {
        "repo_id": "ssl4eo-s12",
        "filename": "ssl4eo_moco.pth",
        "source": "manual",
        "weights_dir": "models/ssl4eo",
    },
}


class ModelDownloader:
    """Download and manage foundation model weights.

    Supports HuggingFace Hub auto-download with local caching,
    and manual checkpoint paths for models not on HF.
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        model_name: str,
        force: bool = False,
        repo_id: str | None = None,
        filename: str | None = None,
    ) -> Path:
        """Download model weights.

        Args:
            model_name: Model name (prithvi, satlas, ssl4eo).
            force: Force re-download even if cached.
            repo_id: Override HuggingFace repo ID.
            filename: Override filename to download.

        Returns:
            Path to the downloaded weights file.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        info = MODEL_REGISTRY[model_name]
        repo = repo_id or info.get("repo_id", "")
        fname = filename or info.get("filename", "")
        source = info.get("source", "huggingface")
        weights_dir = self.models_dir / model_name

        # Check for existing weights
        local_path = weights_dir / fname
        if local_path.exists() and not force:
            logger.info(f"Using cached weights: {local_path}")
            return local_path

        if source == "manual":
            logger.warning(
                f"Model '{model_name}' requires manual download. "
                f"Place weights at: {local_path}"
            )
            return local_path

        if not _HF_AVAILABLE:
            raise ImportError(
                "huggingface-hub is required for auto-download. "
                "Install with: pip install huggingface-hub"
            )

        weights_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {model_name} from {repo}/{fname}...")

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=str(weights_dir),
                force_download=force,
            )
            logger.info(f"Downloaded to: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as e:
            logger.error(f"Download failed for {model_name}: {e}")
            raise

    def get_weights_path(self, model_name: str) -> Path | None:
        """Get path to cached weights, or None if not downloaded.

        Args:
            model_name: Model name.

        Returns:
            Path to weights file, or None.
        """
        info = MODEL_REGISTRY.get(model_name, {})
        fname = info.get("filename", "")
        path = self.models_dir / model_name / fname
        return path if path.exists() else None

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get download status for all registered models.

        Returns:
            Dict mapping model name to status info.
        """
        status = {}
        for name, info in MODEL_REGISTRY.items():
            path = self.models_dir / name / info["filename"]
            status[name] = {
                "downloaded": path.exists(),
                "path": str(path) if path.exists() else None,
                "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else None,
                "source": info["source"],
                "repo_id": info.get("repo_id", "N/A"),
            }
        return status

    def load_weights(
        self,
        model_name: str,
        device: str = "cpu",
        download_if_missing: bool = True,
    ) -> Any:
        """Load model weights as a state dict.

        Args:
            model_name: Model name.
            device: Device to load to (cpu, cuda).
            download_if_missing: Auto-download if not cached.

        Returns:
            State dict or checkpoint object.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for loading model weights")

        path = self.get_weights_path(model_name)
        if path is None and download_if_missing:
            path = self.download(model_name)

        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Weights not found for {model_name}. "
                f"Download with: lccompare download-models --model {model_name}"
            )

        logger.info(f"Loading weights from {path} to {device}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        return checkpoint
