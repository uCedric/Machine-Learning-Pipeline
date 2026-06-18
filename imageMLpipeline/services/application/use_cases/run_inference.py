"""Inference use case: fetch an image, score it, render a heatmap, store the result.

Pure application logic wired entirely against :mod:`application.ports`: object
storage, the anomaly model, the heatmap renderer and the result repository. It
holds no reference to MinIO, ONNX, FAISS, matplotlib or Postgres — the
composition root injects concrete adapters.
"""
from __future__ import annotations

import io
import logging
from pathlib import PurePosixPath

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

    def execute(self, event: InferenceEvent) -> InferenceResult:
        # Fetch the image bytes from object storage. The model reads the image
        # through PIL, which accepts a file-like object, so a BytesIO stream
        # stands in for the file path its ``inference`` signature expects.
        image = self._storage.get_object(event.bucket, event.object_key)

        # Run the model. Products: the preprocessed image, the per-patch L2
        # distance map and the scalar anomaly score.
        original_img_np, dist_score, anomaly_score = self._model.inference(io.BytesIO(image))
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
