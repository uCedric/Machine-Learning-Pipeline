"""ResNet50 stage-two defect-clustering adapter (torchvision Layer2/3 + UMAP/HDBSCAN).

Split by concern:

* :mod:`adapters.outbound.resnet50.model`       — the batch
  :class:`ResNet50ClusterModel` clusterer (implements the
  :class:`~application.ports.model.ClusterModel` port).
* :mod:`adapters.outbound.resnet50.descriptors` — pure numpy/skimage colour &
  shape descriptors read off the anomaly mask.
* :mod:`adapters.outbound.resnet50.resources`   — load the ResNet50 Layer2/3
  feature extractor and the eval transform.
* :mod:`adapters.outbound.resnet50.builder`     — assemble a ready-to-run
  clusterer from a registered model version.

The model factory that builds the clusterer lives one level up, at
:mod:`adapters.outbound.factory`, so it can manage other model types too.
"""
from adapters.outbound.resnet50.model import ResNet50ClusterModel

__all__ = [
    "ResNet50ClusterModel",
]
