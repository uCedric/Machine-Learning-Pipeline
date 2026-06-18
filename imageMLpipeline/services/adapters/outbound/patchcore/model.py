"""PatchCore detector — a concrete :class:`~application.ports.model.AnomalyModel`.

ResNet (ONNX) backbone + FAISS memory-bank nearest-neighbour search. Built by
:class:`~adapters.outbound.patchcore.factory.ModelFactory`.
"""
from __future__ import annotations

from abc import ABCMeta

import numpy as np
from PIL import Image

from application.ports.model import AnomalyModel


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

    def __init__(self, backbone, index, transform=None, buffer_zone=None):
        self.backbone = backbone      # onnxruntime.InferenceSession
        self.index = index            # FAISS index over the memory bank
        self.transform = transform    # preprocessing transform (callable)
        self.buffer_zone = buffer_zone  # domain.models.BufferZone (decision band)

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
