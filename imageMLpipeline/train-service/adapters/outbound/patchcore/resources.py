"""PatchCore resource loading for retraining.

Pulls the backbone + memory bank from object storage so a retrain can extract
new feature vectors with the *same* ONNX backbone and preprocessing the inference
service uses, then append to the existing bank. No FAISS index is built here —
training only needs the raw embeddings, not nearest-neighbour search.

Asset object names mirror the inference service: they live in the models bucket
under ``{model_type}/{version}/``.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torchvision import transforms

from application.ports.storage import ObjectStorage

logger = logging.getLogger(__name__)

# The ONNX backbone keeps its weights in a sidecar ``.onnx.data`` file that
# onnxruntime loads by name from the model's directory, so it is downloaded
# alongside the model but never referenced directly.
ASSET_MEMORY_BANK = "memory_bank.npy"
ASSET_ONNX = "resnet_backbone.onnx"
ASSET_ONNX_DATA = "resnet_backbone.onnx.data"
ASSET_BUFFER_ZONE = "buffer_zone.txt"
# Assets that a retrain does not change and copies verbatim into the new version.
# The buffer zone is *not* here: a retrain recomputes its bounds from the updated
# memory bank and writes a fresh ``buffer_zone.txt`` (see serialize_buffer_zone).
UNCHANGED_ASSETS = (ASSET_ONNX, ASSET_ONNX_DATA)


def prefix_for(model_type: str, version: str) -> str:
    """Object-key prefix for a model version: ``{model_type}/{version}``."""
    return f"{model_type}/{version}"


def load_onnx_session(
    storage: ObjectStorage,
    bucket: str,
    model_type: str,
    version: str,
    cache_dir: str,
    providers: list[str] | None = None,
) -> ort.InferenceSession:
    """Download the ONNX backbone (+ sidecar) to disk and open a session.

    The ``.onnx.data`` sidecar is loaded implicitly by onnxruntime from the
    model's directory, so it is fetched next to the model but never passed in.
    """
    prefix = prefix_for(model_type, version)
    dest = Path(cache_dir) / model_type / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in (ASSET_ONNX, ASSET_ONNX_DATA):
        data = storage.get_object(bucket, f"{prefix}/{name}")
        (dest / name).write_bytes(data)
        logger.info("Fetched asset %s/%s/%s (%d bytes)", bucket, prefix, name, len(data))
    return ort.InferenceSession(
        str(dest / ASSET_ONNX), providers=providers or ["CPUExecutionProvider"]
    )


def load_memory_bank(
    storage: ObjectStorage, bucket: str, model_type: str, version: str
) -> np.ndarray:
    """Load the current memory bank ``.npy`` array from object storage."""
    prefix = prefix_for(model_type, version)
    data = storage.get_object(bucket, f"{prefix}/{ASSET_MEMORY_BANK}")
    bank = np.load(io.BytesIO(data)).astype("float32")
    logger.info("Loaded memory bank %s from %s/%s", bank.shape, bucket, prefix)
    return bank


def build_transform(image_size: int = 224):
    """Preprocessing pipeline: resize to a square and convert to a tensor.

    Must match the inference service exactly, or the appended vectors would be
    inconsistent with the existing bank.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def serialize_memory_bank(bank: np.ndarray) -> bytes:
    """Serialise a memory bank array to ``.npy`` bytes for object storage."""
    buf = io.BytesIO()
    np.save(buf, bank.astype("float32"))
    return buf.getvalue()


def serialize_buffer_zone(lower: float, upper: float) -> bytes:
    """Serialise the buffer-zone bounds to the ``key=value`` text format the
    inference service's ``load_buffer_zone`` parses."""
    return f"lower={lower}\nupper={upper}\n".encode("utf-8")
