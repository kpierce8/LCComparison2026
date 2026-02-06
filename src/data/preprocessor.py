"""Image preprocessing pipeline for embedding models.

Handles normalization, band selection, resizing, nodata handling,
and edge effect minimization for Prithvi, SatLas, and SSL4EO models.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.preprocessor")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    _RASTERIO_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


# Default normalization stats per model
MODEL_NORMALIZATION = {
    "prithvi": {
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "input_size": 224,
        "mean": [494.905, 815.239, 924.944, 2968.881, 2634.621, 1739.579],
        "std": [284.925, 357.299, 575.512, 896.601, 951.455, 808.649],
    },
    "satlas": {
        "bands": ["B4", "B3", "B2"],  # RGB order
        "input_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "ssl4eo": {
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "input_size": 224,
        "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
}

# Sentinel-2 band name to index mapping (in typical export order)
S2_BAND_INDEX = {
    "B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5,
}


class Preprocessor:
    """Image preprocessing for embedding model input.

    Handles the full pipeline from raw GeoTIFF tiles to model-ready tensors:
    band selection, normalization, resizing, and nodata handling.
    """

    def __init__(self, model_name: str, config: dict[str, Any] | None = None):
        """
        Args:
            model_name: One of 'prithvi', 'satlas', 'ssl4eo'.
            config: Optional config overrides for normalization stats.
        """
        self.model_name = model_name
        self.config = config or {}

        # Load model-specific settings
        defaults = MODEL_NORMALIZATION.get(model_name, MODEL_NORMALIZATION["prithvi"])
        self.bands = self.config.get("bands", defaults["bands"])
        self.input_size = self.config.get("input_size", defaults["input_size"])
        self.mean = np.array(self.config.get("mean", defaults["mean"]), dtype=np.float32)
        self.std = np.array(self.config.get("std", defaults["std"]), dtype=np.float32)

    def preprocess_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        return_tensor: bool = False,
    ) -> dict[str, Any]:
        """Preprocess a single raster tile for model input.

        Steps:
        1. Read raster and select bands
        2. Handle nodata values
        3. Resize to model input size
        4. Normalize
        5. Optionally save or return as tensor

        Args:
            input_path: Path to input GeoTIFF.
            output_path: Path to save preprocessed file. None = don't save.
            return_tensor: Whether to return a PyTorch tensor.

        Returns:
            Dict with 'data' (ndarray or tensor), 'valid_mask', 'metadata'.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio required for preprocessing")

        input_path = Path(input_path)

        with rasterio.open(input_path) as src:
            # Select bands
            band_indices = self._get_band_indices(src)
            data = src.read(band_indices).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()

        # Handle nodata
        data, valid_mask = self._handle_nodata(data, nodata)

        # Resize
        data = self._resize(data, self.input_size)
        valid_mask = self._resize_mask(valid_mask, self.input_size)

        # Normalize
        data = self._normalize(data)

        # Handle edge effects
        data = self._minimize_edge_effects(data, valid_mask)

        result = {
            "data": data,
            "valid_mask": valid_mask,
            "metadata": {
                "source": str(input_path),
                "model": self.model_name,
                "bands": self.bands,
                "input_size": self.input_size,
                "shape": data.shape,
            },
        }

        # Save preprocessed
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                data=data,
                valid_mask=valid_mask,
            )
            result["metadata"]["output_path"] = str(output_path)

        # Convert to tensor
        if return_tensor and _TORCH_AVAILABLE:
            result["tensor"] = torch.from_numpy(data)

        return result

    def preprocess_array(
        self,
        data: np.ndarray,
        nodata: float | None = None,
    ) -> dict[str, Any]:
        """Preprocess an in-memory array.

        Args:
            data: Array of shape (C, H, W).
            nodata: Nodata value. None = skip nodata handling.

        Returns:
            Dict with 'data', 'valid_mask'.
        """
        data = data.astype(np.float32)

        # Handle nodata
        data, valid_mask = self._handle_nodata(data, nodata)

        # Resize
        data = self._resize(data, self.input_size)
        valid_mask = self._resize_mask(valid_mask, self.input_size)

        # Normalize
        data = self._normalize(data)

        # Edge effects
        data = self._minimize_edge_effects(data, valid_mask)

        result = {
            "data": data,
            "valid_mask": valid_mask,
        }

        if _TORCH_AVAILABLE:
            result["tensor"] = torch.from_numpy(data)

        return result

    def preprocess_batch(
        self,
        paths: list[str | Path],
        return_tensors: bool = True,
    ) -> dict[str, Any]:
        """Preprocess a batch of tiles.

        Args:
            paths: List of input raster paths.
            return_tensors: Whether to return stacked PyTorch tensors.

        Returns:
            Dict with 'batch' (stacked array/tensor), 'valid_masks', 'metadata'.
        """
        results = []
        for path in paths:
            try:
                result = self.preprocess_file(path, return_tensor=False)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to preprocess {path}: {e}")
                results.append(None)

        # Stack valid results
        valid = [r for r in results if r is not None]
        if not valid:
            return {"batch": None, "valid_masks": None, "metadata": {"count": 0}}

        batch = np.stack([r["data"] for r in valid])
        masks = np.stack([r["valid_mask"] for r in valid])

        output = {
            "batch": batch,
            "valid_masks": masks,
            "metadata": {
                "count": len(valid),
                "failed": len(results) - len(valid),
                "model": self.model_name,
                "shape": batch.shape,
            },
        }

        if return_tensors and _TORCH_AVAILABLE:
            output["tensors"] = torch.from_numpy(batch)

        return output

    def _get_band_indices(self, src: Any) -> list[int]:
        """Map band names to 1-based rasterio indices.

        Tries to use band descriptions, falls back to positional order.
        """
        # Try matching by band description
        descriptions = [src.descriptions[i] if src.descriptions[i] else ""
                        for i in range(src.count)]

        indices = []
        for band_name in self.bands:
            found = False
            for i, desc in enumerate(descriptions):
                if band_name.lower() in desc.lower():
                    indices.append(i + 1)  # 1-based
                    found = True
                    break

            if not found:
                # Fall back to positional index from S2_BAND_INDEX
                if band_name in S2_BAND_INDEX:
                    idx = S2_BAND_INDEX[band_name] + 1  # 1-based
                    if idx <= src.count:
                        indices.append(idx)
                    else:
                        logger.warning(f"Band {band_name} (index {idx}) exceeds band count {src.count}")
                        indices.append(min(idx, src.count))
                else:
                    # Last resort: sequential
                    indices.append(len(indices) + 1)

        return indices

    def _handle_nodata(
        self, data: np.ndarray, nodata: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Replace nodata values and create valid mask.

        Returns:
            Tuple of (cleaned_data, valid_mask).
        """
        # valid_mask: True where data is valid
        valid_mask = np.ones(data.shape[1:], dtype=bool)

        if nodata is not None:
            nodata_mask = np.any(data == nodata, axis=0)
            valid_mask = ~nodata_mask
            # Replace nodata with 0
            data[:, nodata_mask] = 0.0

        # Also mask NaN/Inf
        nan_mask = np.any(~np.isfinite(data), axis=0)
        if nan_mask.any():
            valid_mask &= ~nan_mask
            data[:, nan_mask] = 0.0

        return data, valid_mask

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply per-band normalization."""
        n_bands = min(len(self.mean), data.shape[0])
        for i in range(n_bands):
            if self.std[i] > 0:
                data[i] = (data[i] - self.mean[i]) / self.std[i]
        return data

    def _resize(self, data: np.ndarray, target_size: int) -> np.ndarray:
        """Resize data to target_size x target_size using bilinear interpolation."""
        if data.shape[1] == target_size and data.shape[2] == target_size:
            return data

        c, h, w = data.shape
        resized = np.empty((c, target_size, target_size), dtype=data.dtype)

        # Use simple bilinear via numpy
        y_scale = h / target_size
        x_scale = w / target_size
        y_coords = np.clip(np.arange(target_size) * y_scale, 0, h - 1).astype(int)
        x_coords = np.clip(np.arange(target_size) * x_scale, 0, w - 1).astype(int)
        resized = data[:, y_coords][:, :, x_coords]

        return resized

    def _resize_mask(self, mask: np.ndarray, target_size: int) -> np.ndarray:
        """Resize a boolean mask using nearest neighbor."""
        if mask.shape[0] == target_size and mask.shape[1] == target_size:
            return mask

        h, w = mask.shape
        y_coords = np.clip(np.arange(target_size) * (h / target_size), 0, h - 1).astype(int)
        x_coords = np.clip(np.arange(target_size) * (w / target_size), 0, w - 1).astype(int)
        return mask[y_coords][:, x_coords]

    def _minimize_edge_effects(
        self, data: np.ndarray, valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill invalid edge pixels with nearest valid values.

        Uses a simple expanding ring approach: fill invalid pixels
        with the mean of their valid neighbors.
        """
        if valid_mask.all():
            return data

        invalid = ~valid_mask
        n_invalid = invalid.sum()
        if n_invalid == 0 or n_invalid == valid_mask.size:
            return data

        # Fill with per-band mean of valid pixels
        for c in range(data.shape[0]):
            band_mean = data[c][valid_mask].mean() if valid_mask.any() else 0.0
            data[c][invalid] = band_mean

        return data
