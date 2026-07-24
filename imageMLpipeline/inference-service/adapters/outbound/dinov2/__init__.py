"""DINOv2 defect-clustering adapter (ViT-S/14 ONNX backbone + PCA/UMAP/HDBSCAN).

Split by concern:

* :mod:`adapters.outbound.dinov2.model`     — the :class:`Dinov2ClusterModel`
  clusterer (implements the
  :class:`~application.ports.clustering.ClusterModel` port).
* :mod:`adapters.outbound.dinov2.resources` — fetch the assets and load the
  ONNX backbone, the pretrained PCA/UMAP/HDBSCAN artifacts and the transform.
* :mod:`adapters.outbound.dinov2.builder`   — assemble a ready-to-run clusterer
  from a registered model version.

The model factory that builds the clusterer lives one level up, at
:mod:`adapters.outbound.factory`, so it can manage other model types too.
"""
from adapters.outbound.dinov2.model import Dinov2ClusterModel

__all__ = [
    "Dinov2ClusterModel",
]
