"""Training use case: incrementally retrain PatchCore by extending the memory bank.

Pure application logic wired entirely against :mod:`application.ports`: the model
registry, the feature embedder and the memory-bank repository. It holds no
reference to MinIO, ONNX, numpy SDKs or Postgres — the composition root injects
concrete adapters.

For a ``train`` event it: resolves the model to retrain, embeds the event's image,
appends the new vectors to the current memory bank, recalibrates the anomaly-score
buffer zone against a rolling validation set of known-good images, writes the new
bank and bounds to a ``patch + 1`` version directory in object storage, and
registers that version as a candidate (``is_valid = false``) for a later promotion.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np

from application.ports.embedder import FeatureEmbedder
from application.ports.memory_bank import MemoryBankRepository
from application.ports.model_registry import ModelRegistry
from application.ports.image_repository import ImageRepository
from domain.models import TrainEvent

logger = logging.getLogger(__name__)

# Coreset ratio for the memory bank: keep ~1/98 of an image's patch vectors,
# matching the random subsampling the original memory bank was built with.
_CORESET_DIVISOR = 98
# Percentiles of the validation good-image score distribution used as the buffer
# zone's lower (good-product defence line) and upper bounds.
_LOWER_PERCENTILE = 75
_UPPER_PERCENTILE = 99
# Validation images are fetched from object storage and scored one batch at a
# time: each batch is pulled concurrently (one connection per image) before
# PatchCore scores it, bounding open connections and in-flight image bytes.
_BATCH_SIZE = 16


def _subsample(features: np.ndarray) -> np.ndarray:
    """Randomly keep ``len(features) // 98`` of the patch vectors (coreset),
    mirroring how the original memory bank was built."""
    n = len(features)
    size = max(1, n // _CORESET_DIVISOR)
    idx = np.random.choice(n, size=size, replace=False)
    return features[idx]


def _build_index(bank: np.ndarray) -> "faiss.Index":
    """Build a FAISS L2 index over the memory bank (as the inference service does)."""
    bank = np.ascontiguousarray(bank, dtype="float32")
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank)
    return index


def _image_score(index: "faiss.Index", features: np.ndarray) -> float:
    """Image-level anomaly score for one image's patch ``features``.

    The max over per-patch nearest-neighbour distances (``sqrt`` of L2), exactly
    mirroring the inference detector (PatchCore.inference) so the recomputed
    bounds are on the same scale the inference service classifies against.
    """
    features = np.ascontiguousarray(features, dtype="float32")
    distances, _ = index.search(features, 1)
    return float(np.max(np.sqrt(distances)))


class RunTraining:
    def __init__(
        self,
        registry: ModelRegistry,
        embedder: FeatureEmbedder,
        memory_bank: MemoryBankRepository,
        validation_set: ImageRepository,
        *,
        storage_get,
        validation_set_size: int = 200,
    ) -> None:
        self._registry = registry
        self._embedder = embedder
        self._memory_bank = memory_bank
        self._validation_set = validation_set
        self._validation_set_size = validation_set_size
        # Reads the event's image bytes from object storage; injected as a plain
        # callable so the use case stays free of the storage adapter's type.
        self._storage_get = storage_get

    def execute(self, event: TrainEvent) -> None:
        # 1. Analyse the event: which model, and where the image lives.
        model_type = event.model_type
        bucket, key = event.image_location()
        logger.info("Train '%s' on image %s/%s", model_type, bucket, key)

        # 2. Resolve the model to retrain (newest valid version).
        base = self._registry.latest(model_type)

        # 3. Embed the event's image into patch feature vectors.
        image = self._storage_get(bucket, key)
        features = self._embedder.embed(image)

        # 4. Subsample the new vectors into a coreset and append them to the bank.
        current_bank = self._memory_bank.load(base)
        new_bank = np.vstack([current_bank, _subsample(features)])
        logger.info(
            "Memory bank %s + %s -> %s", current_bank.shape, features.shape, new_bank.shape
        )

        # 5. Recalibrate the buffer zone: score a rolling validation set of recent
        #    known-good images against the new bank (image-level scores) and take
        #    P75/P99 as the lower/upper bounds.
        index = _build_index(new_bank)
        validation_images = self._validation_set.recent_good(model_type, self._validation_set_size)
        scores = self._score_validation_images(index, validation_images)

        # 6. Persist as a new patch+1 candidate: write the bank, then the
        #    recomputed buffer zone (or copy the previous one forward when there
        #    are no validation images yet), copy the unchanged backbone assets so
        #    the version directory is self-contained, and register it not-yet-valid.
        new_version = self._registry.next_version(base)
        self._memory_bank.save(new_version, new_bank)
        if scores:
            lower, upper = np.percentile(scores, [_LOWER_PERCENTILE, _UPPER_PERCENTILE])
            logger.info(
                "Recalibrated buffer zone from %d good image(s): lower=%s upper=%s",
                len(scores),
                lower,
                upper,
            )
            self._memory_bank.save_buffer_zone(new_version, float(lower), float(upper))
        else:
            logger.warning(
                "No validation good images for '%s'; copying previous buffer zone",
                model_type,
            )
            self._memory_bank.copy_buffer_zone(base, new_version)
        self._memory_bank.copy_unchanged_assets(base, new_version)
        self._registry.register(new_version, is_valid=False)
        logger.info(
            "Retrained '%s' v%s -> v%s (model_id=%s, is_valid=false)",
            model_type,
            base.version,
            new_version.version,
            new_version.model_id,
        )

    def _score_validation_images(
        self, index: "faiss.Index", locations: list[tuple[str, str]]
    ) -> list[float]:
        """Image-level anomaly scores for the ``(bucket, key)`` validation images.

        Processed in batches of ``_BATCH_SIZE``: each batch's image bytes are
        fetched from object storage concurrently (one connection per image), and
        each image is embedded and scored against ``index`` the moment its fetch
        completes before the next batch starts. This keeps at most one batch of
        connections open at a time. A fetch that fails is logged at WARNING and
        skipped rather than aborting the batch.
        """
        scores: list[float] = []
        for start in range(0, len(locations), _BATCH_SIZE):
            batch = locations[start : start + _BATCH_SIZE]
            with ThreadPoolExecutor(max_workers=_BATCH_SIZE) as pool:

                futures = {}
                for loc in batch:
                    futures[pool.submit(self._storage_get, *loc)] = loc

                for future in as_completed(futures):
                    bucket, key = futures[future]
                    try:
                        image = future.result()
                    except Exception:
                        logger.warning(
                            "Failed to fetch validation image %s/%s; skipping",
                            bucket,
                            key,
                        )
                        continue
                    scores.append(_image_score(index, self._embedder.embed(image)))
                    
        return scores
