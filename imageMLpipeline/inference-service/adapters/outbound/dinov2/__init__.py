"""DINOv2 stage-two defect-clustering adapter (batch, cosine memory bank + UMAP/HDBSCAN).

Split by concern:

* :mod:`adapters.outbound.dinov2.model`     — the batch :class:`Dinov2ClusterModel`
  clusterer (implements the
  :class:`~application.ports.model.ClusterModel` port).
* :mod:`adapters.outbound.dinov2.resources` — load the DINOv2 backbone (torch hub)
  and the intermediate-layer configuration + transform.
* :mod:`adapters.outbound.dinov2.builder`   — assemble a ready-to-run clusterer
  from a registered model version.

Registered but **not built** while its ``inference_model`` row is ``is_valid=false``
(a candidate awaiting the evaluate-service). The model factory that builds the
clusterer lives one level up, at :mod:`adapters.outbound.factory`.
"""
from adapters.outbound.dinov2.model import Dinov2ClusterModel

__all__ = [
    "Dinov2ClusterModel",
]
