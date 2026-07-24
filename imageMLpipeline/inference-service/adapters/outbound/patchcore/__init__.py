"""PatchCore anomaly-detection adapter (ResNet ONNX backbone + FAISS memory bank).

Split by concern:

* :mod:`adapters.outbound.patchcore.model`     — the :class:`PatchCore` detector
  (implements the :class:`~application.ports.model.AnomalyModel` port).
* :mod:`adapters.outbound.patchcore.resources` — load the memory bank, FAISS
  index, ONNX backbone and the preprocessing transform.
* :mod:`adapters.outbound.patchcore.heatmap`   — render + store the heatmap
  (implements the :class:`~application.ports.heatmap.HeatmapRenderer` port).

The model factory that builds the detector lives one level up, at
:mod:`adapters.outbound.factory`, so it can manage other model types too.
"""
from adapters.outbound.patchcore.heatmap import MatplotlibHeatmapRenderer
from adapters.outbound.patchcore.model import PatchCore

__all__ = [
    "MatplotlibHeatmapRenderer",
    "PatchCore",
]
