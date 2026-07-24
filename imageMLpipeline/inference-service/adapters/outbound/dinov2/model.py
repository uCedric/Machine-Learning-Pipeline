"""DINOv2 defect clusterer — a concrete :class:`~application.ports.clustering.ClusterModel`.

DINOv2 ViT-S/14 (ONNX) CLS embedding -> L2 normalise -> pretrained PCA
(384 -> 50) -> pretrained UMAP (50 -> 2) -> HDBSCAN ``approximate_predict``.
Built by :class:`~adapters.outbound.factory.ModelFactory`.
"""
from __future__ import annotations

import hdbscan
import numpy as np
from PIL import Image

from adapters.outbound.patchcore.model import SingletonMeta
from application.ports.clustering import ClusterModel
from domain.models import ClusterAssignment


class Dinov2ClusterModel(ClusterModel, metaclass=SingletonMeta):
    """DINOv2 (ONNX) embedding + pretrained PCA/UMAP/HDBSCAN clusterer (singleton)."""

    def __init__(self, backbone, pca, umap_reducer, clusterer, transform=None, model_id=None):
        self.backbone = backbone  # onnxruntime.InferenceSession
        self.pca = pca  # sklearn PCA (384 -> 50)
        self.umap = umap_reducer  # umap.UMAP reducer (50 -> 2)
        self.clusterer = clusterer  # hdbscan.HDBSCAN with prediction data
        self.transform = transform  # preprocessing transform (callable)
        self.model_id = model_id  # inference_model.model_id of the loaded version

    def assign(self, image) -> ClusterAssignment:
        # Preprocess the image and extract the CLS embedding. ``image`` may be
        # a path or any file-like object PIL can open.
        img = Image.open(image).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).numpy()
        ort_inputs = {self.backbone.get_inputs()[0].name: img_tensor}
        feature = self.backbone.run(None, ort_inputs)[0].astype("float32")  # (1, 384)

        # Cosine-oriented pipeline: L2-normalise, then apply the pretrained
        # projections and query the cluster hierarchy.
        feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        reduced = self.pca.transform(feature)  # (1, 50)
        embedding = self.umap.transform(reduced)  # (1, 2)
        labels, strengths = hdbscan.approximate_predict(self.clusterer, embedding)

        # Plain-python scalars: the domain is numpy-free and psycopg2 cannot
        # adapt numpy integer/float types.
        return ClusterAssignment(
            cluster_id=int(labels[0]),
            probability=float(strengths[0]),
            model_id=self.model_id,
        )
