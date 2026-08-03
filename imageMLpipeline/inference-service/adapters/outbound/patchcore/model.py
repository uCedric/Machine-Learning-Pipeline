"""PatchCore detector — a concrete :class:`~application.ports.model.AnomalyModel`.

ResNet (ONNX) backbone + FAISS memory-bank nearest-neighbour search. Built by
:class:`~adapters.outbound.factory.ModelFactory`.

Two entry points, same maths:

* :meth:`PatchCore.inference` scores one image — the in-process path.
* :meth:`PatchCore.inference_batch` scores a mini-batch in one go — what the
  Spark ``predict_batch_udf`` scorer calls. It stacks the preprocessed tensors
  into a single ``(B, 3, H, W)`` input and runs **one** FAISS search over every
  image's patches, which is where most of the batching win actually comes from.
"""
from __future__ import annotations

import logging
from abc import ABCMeta
from typing import Any, Sequence, Tuple

import numpy as np
from PIL import Image

from application.ports.model import AnomalyModel

logger = logging.getLogger(__name__)


class SingletonMeta(ABCMeta):
    """Metaclass that makes a class a singleton.

    Subclasses ``ABCMeta`` so it composes with abstract base classes. The first
    construction is cached per class; later calls return the same instance and do
    not re-run ``__init__``.
    """

    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class PatchCore(AnomalyModel, metaclass=SingletonMeta):
    """ResNet (ONNX) backbone + FAISS memory-bank anomaly detector (singleton)."""

    def __init__(self, backbone, index, transform=None, buffer_zone=None, model_id=None):
        self.backbone = backbone      # onnxruntime.InferenceSession
        self.index = index            # FAISS index over the memory bank
        self.transform = transform    # preprocessing transform (callable)
        self.buffer_zone = buffer_zone  # domain.models.BufferZone (decision band)
        self.model_id = model_id      # inference_model.model_id of the loaded version
        # Whether the exported graph really accepts a batch axis. Unknown until
        # the first multi-image call probes it; see :meth:`_batch_capable`.
        self._batchable: bool | None = None

    def inference(self, img_path, transform=None):
        transform = transform if transform is not None else self.transform

        # Preprocess the image and run the ONNX backbone. ``img_path`` may be a
        # path or any file-like object PIL can open.
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).numpy()
        ort_inputs = {self.backbone.get_inputs()[0].name: img_tensor}
        features = self.backbone.run(None, ort_inputs)[0].astype("float32")

        # Nearest-neighbour search against the memory bank.
        distances, _ = self.index.search(features, 1)
        dist_score = np.sqrt(distances)
        anomaly_score = float(np.max(dist_score))
        return img_tensor, dist_score, anomaly_score

    def inference_batch(
        self, images: Sequence[Any], transform: Any = None
    ) -> list[Tuple[Any, Any, float]]:
        """Score a mini-batch, returning ``(tensor, dist_score, score)`` per image.

        Each returned tuple is shaped exactly like :meth:`inference`'s — a
        ``(1, 3, H, W)`` tensor and a ``(patches, 1)`` distance map — so the
        heatmap renderer and the use case cannot tell which path produced them.
        """
        transform = transform if transform is not None else self.transform
        if not images:
            return []

        # Preprocess into one contiguous (B, 3, H, W) block.
        batch = np.stack(
            [transform(Image.open(image).convert("RGB")).numpy() for image in images]
        ).astype("float32")

        features = self._extract(batch)  # (B, patches, dim)
        count, patches, dim = features.shape

        # One FAISS search for the whole batch: the index is queried with
        # B*patches vectors at once instead of B separate calls, which is a
        # straight win even when the backbone itself has to run image by image.
        distances, _ = self.index.search(features.reshape(count * patches, dim), 1)
        dist_scores = np.sqrt(distances).reshape(count, patches, 1)

        return [
            (batch[i : i + 1], dist_scores[i], float(np.max(dist_scores[i])))
            for i in range(count)
        ]

    # ── backbone plumbing ────────────────────────────────────────────────────

    def _extract(self, batch: np.ndarray) -> np.ndarray:
        """Run the backbone over ``(B, 3, H, W)``, returning ``(B, patches, dim)``."""
        input_name = self.backbone.get_inputs()[0].name
        count = batch.shape[0]

        if count > 1 and self._batch_capable(batch, input_name):
            features = self.backbone.run(None, {input_name: batch})[0].astype("float32")
            # The graph flattens the batch into the patch axis, so split it back.
            return features.reshape(count, -1, features.shape[-1])

        # Fixed batch axis (or a graph that does not really batch): run the
        # images one at a time and stack the per-image patch features.
        per_image = [
            self.backbone.run(None, {input_name: batch[i : i + 1]})[0].astype("float32")
            for i in range(count)
        ]
        return np.stack(per_image)

    def _batch_capable(self, batch: np.ndarray, input_name: str) -> bool:
        """Decide once whether the exported graph accepts a real batch axis.

        The PatchCore export flattens its spatial grid into the leading axis, so
        a fixed ``batch=1`` input dim is entirely plausible and feeding it two
        images would either raise or — worse — silently return the wrong number
        of patch rows. Rather than trust the export, this probes it once per
        process: run two images, confirm the output grew by exactly the expected
        factor, and cache the verdict on the instance.
        """
        if self._batchable is not None:
            return self._batchable

        declared = self.backbone.get_inputs()[0].shape[0]
        if isinstance(declared, int) and declared > 0:
            logger.info(
                "ONNX backbone declares a fixed batch axis (%s); scoring image-by-image",
                declared,
            )
            self._batchable = False
            return False

        try:
            single = self.backbone.run(None, {input_name: batch[0:1]})[0]
            pair = self.backbone.run(None, {input_name: batch[0:2]})[0]
            self._batchable = (
                pair.shape[0] == 2 * single.shape[0]
                and pair.shape[-1] == single.shape[-1]
            )
        except Exception:  # noqa: BLE001 - a probe failure just means "no batching"
            logger.warning(
                "ONNX backbone rejected a 2-image probe; scoring image-by-image",
                exc_info=True,
            )
            self._batchable = False

        logger.info(
            "ONNX backbone batch support: %s",
            "enabled" if self._batchable else "disabled (per-image forward pass)",
        )
        return self._batchable
