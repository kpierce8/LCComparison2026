"""Batch embedding processor with checkpoint/resume support.

Processes tiles through the embedding pipeline in batches with:
- Memory-efficient GPU processing
- Checkpoint/resume after interruption
- Embedding caching to disk
- Integration with TileManager for status tracking
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.batch_processor")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


class EmbeddingCheckpoint:
    """Tracks embedding generation progress for checkpoint/resume."""

    def __init__(self, checkpoint_path: str | Path):
        self.path = Path(checkpoint_path)
        self.state: dict[str, Any] = {
            "model_name": None,
            "completed_tiles": [],
            "failed_tiles": [],
            "total_tiles": 0,
            "last_batch_index": 0,
            "started_at": None,
            "updated_at": None,
        }
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            self.state = json.load(f)
        logger.info(
            f"Resumed checkpoint: {len(self.state['completed_tiles'])}/"
            f"{self.state['total_tiles']} tiles completed"
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

    def init(self, model_name: str, total_tiles: int) -> None:
        self.state["model_name"] = model_name
        self.state["total_tiles"] = total_tiles
        self.state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save()

    def mark_completed(self, tile_ids: list[str], batch_index: int) -> None:
        self.state["completed_tiles"].extend(tile_ids)
        self.state["last_batch_index"] = batch_index
        self.save()

    def mark_failed(self, tile_id: str, error: str) -> None:
        self.state["failed_tiles"].append({"tile_id": tile_id, "error": error})
        self.save()

    @property
    def completed_set(self) -> set[str]:
        return set(self.state["completed_tiles"])

    @property
    def n_completed(self) -> int:
        return len(self.state["completed_tiles"])

    @property
    def n_failed(self) -> int:
        return len(self.state["failed_tiles"])

    def get_summary(self) -> dict[str, Any]:
        return {
            "model": self.state["model_name"],
            "completed": self.n_completed,
            "failed": self.n_failed,
            "total": self.state["total_tiles"],
            "started_at": self.state["started_at"],
            "updated_at": self.state["updated_at"],
        }


class BatchProcessor:
    """Process tiles through embedding model in batches.

    Integrates with:
    - Preprocessor for tile preparation
    - EmbeddingModel for feature extraction
    - TileManager for status tracking
    - EmbeddingCheckpoint for resume support
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        output_dir: str | Path,
        batch_size: int = 8,
        checkpoint_frequency: int = 100,
        cache_embeddings: bool = True,
        max_memory_gb: float = 16.0,
    ):
        """
        Args:
            model: Loaded EmbeddingModel instance.
            preprocessor: Preprocessor instance for the same model.
            output_dir: Directory for cached embeddings.
            batch_size: Tiles per batch.
            checkpoint_frequency: Save checkpoint every N tiles.
            cache_embeddings: Whether to cache embeddings to disk.
            max_memory_gb: Max GPU memory target.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.checkpoint_frequency = checkpoint_frequency
        self.cache_embeddings = cache_embeddings
        self.max_memory_gb = max_memory_gb

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_tiles(
        self,
        tile_paths: dict[str, Path],
        tile_manager: Any | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Process a set of tiles through the embedding pipeline.

        Args:
            tile_paths: Dict of {tile_id: path_to_tile}.
            tile_manager: Optional TileManager for status updates.
            resume: Whether to resume from checkpoint.

        Returns:
            Dict with processing results and summary.
        """
        model_name = self.model.model_name
        checkpoint_path = self.output_dir / f"{model_name}_checkpoint.json"
        checkpoint = EmbeddingCheckpoint(checkpoint_path)

        # Determine which tiles need processing
        all_tile_ids = list(tile_paths.keys())

        if resume and checkpoint.completed_set:
            pending_ids = [
                tid for tid in all_tile_ids
                if tid not in checkpoint.completed_set
            ]
            logger.info(
                f"Resuming: {len(pending_ids)} remaining "
                f"({checkpoint.n_completed} already done)"
            )
        else:
            pending_ids = all_tile_ids
            checkpoint.init(model_name, len(all_tile_ids))

        # Process in batches
        total_processed = 0
        total_failed = 0
        all_embeddings: dict[str, np.ndarray] = {}
        start_time = time.time()

        for batch_start in range(0, len(pending_ids), self.batch_size):
            batch_ids = pending_ids[batch_start:batch_start + self.batch_size]
            batch_idx = batch_start // self.batch_size

            try:
                batch_result = self._process_batch(
                    batch_ids, tile_paths, tile_manager,
                )
                all_embeddings.update(batch_result["embeddings"])

                # Update checkpoint
                completed_ids = list(batch_result["embeddings"].keys())
                failed_ids = batch_result.get("failed", [])
                total_processed += len(completed_ids)
                total_failed += len(failed_ids)

                checkpoint.mark_completed(completed_ids, batch_idx)
                for fid in failed_ids:
                    checkpoint.mark_failed(fid, "preprocessing or inference error")

                # Progress logging
                done = checkpoint.n_completed
                total = len(all_tile_ids)
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Batch {batch_idx}: {done}/{total} tiles "
                    f"({rate:.1f} tiles/sec)"
                )

            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                for tid in batch_ids:
                    checkpoint.mark_failed(tid, str(e))
                    total_failed += 1

        elapsed = time.time() - start_time
        summary = {
            "model": model_name,
            "total_tiles": len(all_tile_ids),
            "processed": total_processed,
            "failed": total_failed,
            "cached": checkpoint.n_completed,
            "elapsed_seconds": round(elapsed, 1),
            "tiles_per_second": round(total_processed / elapsed, 2) if elapsed > 0 else 0,
        }

        logger.info(
            f"Embedding generation complete: {summary['processed']} processed, "
            f"{summary['failed']} failed in {summary['elapsed_seconds']}s"
        )

        return {
            "embeddings": all_embeddings,
            "summary": summary,
            "checkpoint": checkpoint.get_summary(),
        }

    def _process_batch(
        self,
        tile_ids: list[str],
        tile_paths: dict[str, Path],
        tile_manager: Any | None,
    ) -> dict[str, Any]:
        """Process a single batch of tiles.

        Returns:
            Dict with 'embeddings' {tile_id: embedding} and 'failed' [tile_ids].
        """
        from src.data.tile_manager import TileStatus

        # Preprocess tiles
        batch_data = []
        valid_ids = []
        failed_ids = []

        for tile_id in tile_ids:
            tile_path = tile_paths[tile_id]

            # Check for cached embedding
            if self.cache_embeddings:
                cache_path = self._get_cache_path(tile_id)
                if cache_path.exists():
                    cached = np.load(cache_path)
                    batch_data.append(None)  # Placeholder
                    valid_ids.append((tile_id, cached["embedding"]))
                    continue

            # Update tile status
            if tile_manager is not None:
                try:
                    tile_manager.update_status(tile_id, TileStatus.EMBEDDING)
                except KeyError:
                    pass

            try:
                result = self.preprocessor.preprocess_file(
                    tile_path, return_tensor=False,
                )
                batch_data.append(result["data"])
                valid_ids.append((tile_id, None))  # Needs inference
            except Exception as e:
                logger.warning(f"Preprocess failed for {tile_id}: {e}")
                failed_ids.append(tile_id)
                if tile_manager is not None:
                    try:
                        tile_manager.update_status(
                            tile_id, TileStatus.FAILED, error=str(e),
                        )
                    except KeyError:
                        pass

        # Run inference on non-cached tiles
        needs_inference = [
            (i, tid) for i, (tid, emb) in enumerate(valid_ids)
            if emb is None
        ]

        embeddings_map: dict[str, np.ndarray] = {}

        # Collect already-cached
        for tid, emb in valid_ids:
            if emb is not None:
                embeddings_map[tid] = emb

        if needs_inference:
            # Build batch array
            inference_data = []
            inference_ids = []
            for idx, tid in needs_inference:
                if batch_data[idx] is not None:
                    inference_data.append(batch_data[idx])
                    inference_ids.append(tid)

            if inference_data:
                stacked = np.stack(inference_data, axis=0)
                try:
                    batch_embeddings = self.model.extract_embeddings(
                        stacked, batch_size=len(inference_data),
                    )

                    for i, tid in enumerate(inference_ids):
                        embedding = batch_embeddings[i]
                        embeddings_map[tid] = embedding

                        # Cache to disk
                        if self.cache_embeddings:
                            self._save_cached(tid, embedding)

                        # Update tile status
                        if tile_manager is not None:
                            try:
                                tile_manager.update_status(
                                    tid, TileStatus.EMBEDDED,
                                    file_paths={
                                        "embedding": str(self._get_cache_path(tid)),
                                    },
                                )
                            except KeyError:
                                pass

                except Exception as e:
                    logger.error(f"Inference failed: {e}")
                    failed_ids.extend(inference_ids)

        return {"embeddings": embeddings_map, "failed": failed_ids}

    def _get_cache_path(self, tile_id: str) -> Path:
        """Get cache file path for a tile's embedding."""
        return self.output_dir / f"{tile_id}.npz"

    def _save_cached(self, tile_id: str, embedding: np.ndarray) -> None:
        """Save embedding to disk cache."""
        cache_path = self._get_cache_path(tile_id)
        np.savez_compressed(cache_path, embedding=embedding, tile_id=tile_id)

    def load_cached_embeddings(
        self,
        tile_ids: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Load cached embeddings from disk.

        Args:
            tile_ids: Specific tiles to load. None = load all cached.

        Returns:
            Dict of {tile_id: embedding_array}.
        """
        embeddings = {}

        if tile_ids is not None:
            paths = [(tid, self._get_cache_path(tid)) for tid in tile_ids]
        else:
            paths = [
                (p.stem, p) for p in self.output_dir.glob("*.npz")
                if p.name != f"{self.model.model_name}_checkpoint.json"
            ]

        for tile_id, path in paths:
            if path.exists():
                try:
                    data = np.load(path)
                    embeddings[tile_id] = data["embedding"]
                except Exception as e:
                    logger.warning(f"Failed to load cached {tile_id}: {e}")

        return embeddings

    def get_progress(self) -> dict[str, Any]:
        """Get current processing progress."""
        checkpoint_path = self.output_dir / f"{self.model.model_name}_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint = EmbeddingCheckpoint(checkpoint_path)
            return checkpoint.get_summary()
        return {
            "model": self.model.model_name,
            "completed": 0,
            "failed": 0,
            "total": 0,
        }

    def clear_cache(self) -> int:
        """Clear cached embeddings. Returns number of files removed."""
        count = 0
        for f in self.output_dir.glob("*.npz"):
            f.unlink()
            count += 1
        # Also remove checkpoint
        cp = self.output_dir / f"{self.model.model_name}_checkpoint.json"
        if cp.exists():
            cp.unlink()
            count += 1
        return count
