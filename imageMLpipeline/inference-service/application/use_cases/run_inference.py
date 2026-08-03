"""Stage-one inference: score a batch of images, render heatmaps, store results.

Pure application logic wired entirely against :mod:`application.ports`: object
storage, the anomaly model, the heatmap renderer and the result repository. It
holds no reference to MinIO, ONNX, FAISS, matplotlib, Kafka or Postgres — the
composition root injects concrete adapters.

Two entry points, deliberately split so the caller controls failure granularity:

* :meth:`RunInferenceUseCase.score_batch` — the model call for a whole
  mini-batch, which is the expensive part and the part that benefits from being
  batched.
* :meth:`RunInferenceUseCase.finalise` — everything after it, per image: the
  buffer-zone verdict, the heatmap, and the stored result.

Stage-one is anomaly detection only. Handing defects on to stage-two is *not*
done here: :class:`~application.use_cases.run_stage_one_batch.RunStageOneBatch`
owns that, so triggers are emitted once per batch from one place — a use case
that runs on a Spark executor has no business talking to Kafka.
"""
from __future__ import annotations

import io
import logging
from pathlib import PurePosixPath
from typing import Sequence

from application.ports.heatmap import HeatmapRenderer
from application.ports.model import AnomalyModel
from application.ports.repository import ResultRepository
from application.ports.storage import ObjectStorage
from domain.models import BufferZone, InferenceEvent, InferenceResult

logger = logging.getLogger(__name__)


class RunInferenceUseCase:
    def __init__(
        self,
        storage: ObjectStorage,
        model: AnomalyModel,
        renderer: HeatmapRenderer,
        repository: ResultRepository,
        *,
        model_id: str,
        buffer_zone: BufferZone,
        heatmap_bucket: str = "images",
    ) -> None:
        self._storage = storage
        self._model = model
        self._renderer = renderer
        self._repository = repository
        self._model_id = model_id
        self._buffer_zone = buffer_zone
        self._heatmap_bucket = heatmap_bucket

    def score_batch(self, images: Sequence[bytes]) -> list[tuple]:
        """Run the model over a batch of image bytes, aligned to ``images``.

        The images arrive already fetched — the scorer reads them where the
        scoring happens — and the model sees the whole mini-batch in one call
        instead of one image at a time.

        Returns one ``(original_img_np, dist_score, anomaly_score)`` per image;
        :meth:`finalise` turns each into a stored result.
        """
        return self._model.inference_batch([io.BytesIO(image) for image in images])

    def finalise(
        self,
        event: InferenceEvent,
        original_img_np,
        dist_score,
        anomaly_score: float,
    ) -> InferenceResult:
        """Classify one scored image and store its heatmap and result."""
        status = self._buffer_zone.classify(anomaly_score)

        # Render the heatmap (technical concern, delegated to the renderer port)
        # and store it via this use case's own object storage.
        heatmap_key = f"heatmap/{PurePosixPath(event.object_key).stem}.png"
        heatmap_png = self._renderer.render(
            original_img_np,
            dist_score,
            anomaly_score,
            self._buffer_zone,
            status,
            title=event.object_key,
        )
        self._storage.put_object(
            self._heatmap_bucket, heatmap_key, heatmap_png, "image/png"
        )

        # Persist the scalar result (score + verdict + heatmap location).
        result = InferenceResult(
            bucket=event.bucket,
            object_key=event.object_key,
            anomaly_score=anomaly_score,
            status=status,
            model_id=self._model_id,
            heatmap_key=heatmap_key,
        )
        self._repository.save(result)
        logger.info(
            "Saved result for %s/%s: score=%.4f (%s), heatmap=%s",
            result.bucket,
            result.object_key,
            result.anomaly_score,
            result.status.upper(),
            result.heatmap_key,
        )
        return result
