"""PatchCore anomaly-detection adapter (ResNet ONNX backbone + FAISS memory bank).

Split by concern:

* :mod:`adapters.outbound.patchcore.model`     — the :class:`PatchCore` detector
  (implements the :class:`~application.ports.model.AnomalyModel` port).
* :mod:`adapters.outbound.patchcore.resources` — load the memory bank, FAISS
  index, ONNX backbone and the preprocessing transform.
* :mod:`adapters.outbound.patchcore.factory`   — build a ready-to-run model.
* :mod:`adapters.outbound.patchcore.heatmap`   — render + store the heatmap
  (implements the :class:`~application.ports.heatmap.HeatmapRenderer` port).
"""
from adapters.outbound.patchcore.factory import ModelFactory, build_patchcore, get_model
from adapters.outbound.patchcore.heatmap import MatplotlibHeatmapRenderer
from adapters.outbound.patchcore.model import PatchCore

__all__ = [
    "ModelFactory",
    "build_patchcore",
    "get_model",
    "MatplotlibHeatmapRenderer",
    "PatchCore",
]
