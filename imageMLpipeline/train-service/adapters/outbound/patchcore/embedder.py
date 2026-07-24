"""PatchCore feature embedder — a concrete :class:`~application.ports.embedder.FeatureEmbedder`.

Runs the ResNet (ONNX) backbone over a preprocessed image to produce the patch
feature vectors that make up the memory bank. This is the embedding half of the
inference detector (:class:`adapters.outbound.patchcore.model.PatchCore` in the
inference service) without the FAISS nearest-neighbour search.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from application.ports.embedder import FeatureEmbedder


class PatchCoreEmbedder(FeatureEmbedder):
    def __init__(self, backbone, transform) -> None:
        self._backbone = backbone      # onnxruntime.InferenceSession
        self._transform = transform    # preprocessing transform (callable)

    def embed(self, image) -> np.ndarray:
        # Preprocess and run the ONNX backbone. ``image`` may be a path or any
        # file-like object PIL can open. Returns the patch features as a
        # ``(num_patches, dim)`` float32 array — the rows appended to the bank.
        img = Image.open(image).convert("RGB")
        img_tensor = self._transform(img).unsqueeze(0).numpy()
        ort_inputs = {self._backbone.get_inputs()[0].name: img_tensor}
        features = self._backbone.run(None, ort_inputs)[0].astype("float32")
        return features
