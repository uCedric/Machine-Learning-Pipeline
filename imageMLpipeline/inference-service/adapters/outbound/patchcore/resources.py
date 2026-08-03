"""PatchCore resource loading (stage 1).

Fetch the assets from the ``models`` bucket, load the memory bank, build the
FAISS index, load the ONNX backbone and the preprocessing transform. Kept
separate from the detector so the model class stays free of file-system / SDK
setup concerns.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from torchvision import transforms

from application.ports.storage import ObjectStorage
from domain.models import BufferZone

logger = logging.getLogger(__name__)

# PatchCore asset object names, stored in the models bucket under
# ``{model_key}/{version}/``. The ONNX backbone keeps its weights in a sidecar
# ``.onnx.data`` file that onnxruntime loads by name from the model's directory,
# so it is downloaded alongside the model but never referenced directly.
ASSET_MEMORY_BANK = "memory_bank.npy"
ASSET_ONNX = "resnet_backbone.onnx"
ASSET_ONNX_DATA = "resnet_backbone.onnx.data"
ASSET_BUFFER_ZONE = "buffer_zone.txt"
ASSET_FILENAMES = (ASSET_MEMORY_BANK, ASSET_ONNX, ASSET_ONNX_DATA, ASSET_BUFFER_ZONE)


@dataclass
class AssetPaths:
    """Local filesystem locations of the downloaded PatchCore assets."""

    memory_bank: str
    onnx: str
    buffer_zone: str


def fetch_assets(
    storage: ObjectStorage,
    bucket: str,
    model_key: str,
    version: str,
    cache_dir: str,
) -> AssetPaths:
    """Download the PatchCore assets from ``bucket/{model_key}/{version}/`` to disk.

    Returns the local paths the loaders read from. The ONNX sidecar
    ``.onnx.data`` is downloaded next to the model but loaded implicitly by
    onnxruntime, so it is not part of the returned paths.

    The cache is per version, so an already-downloaded asset is reused rather
    than re-fetched: on a Spark worker several Python processes build the same
    model, and only the first pays for the 45 MB of weights. Each file is
    written to a temporary name and renamed into place, so a process that finds
    an asset present always finds it complete — a plain write would let one
    worker read a file another is still writing.
    """
    prefix = f"{model_key}/{version}"
    dest = Path(cache_dir) / model_key / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in ASSET_FILENAMES:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            logger.info("Reusing cached asset %s", target)
            continue
        data = storage.get_object(bucket, f"{prefix}/{name}")
        staged = target.with_name(f"{name}.{os.getpid()}.part")
        staged.write_bytes(data)
        os.replace(staged, target)
        logger.info("Fetched asset %s/%s/%s (%d bytes)", bucket, prefix, name, len(data))
    return AssetPaths(
        memory_bank=str(dest / ASSET_MEMORY_BANK),
        onnx=str(dest / ASSET_ONNX),
        buffer_zone=str(dest / ASSET_BUFFER_ZONE),
    )


def load_buffer_zone(path: str) -> BufferZone:
    """Parse the calibrated decision band from a ``lower=``/``upper=`` text file."""
    bounds: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            bounds[key.strip()] = float(value.strip())
    zone = BufferZone(lower=bounds["lower"], upper=bounds["upper"])
    logger.info("Loaded buffer zone lower=%.4f upper=%.4f", zone.lower, zone.upper)
    return zone


def onnx_providers(device: str) -> list:
    """Translate a ``ModelConfig.device`` string into onnxruntime providers.

    ``cpu`` → CPU only. ``cuda`` / ``cuda:<id>`` → the CUDA provider pinned to
    that board, with the CPU provider left behind it as a fallback: onnxruntime
    walks the list in order, so a node CUDA cannot execute still runs rather
    than failing the session.

    Note the fallback does **not** cover a missing CUDA provider — an image
    built with the CPU ``onnxruntime`` wheel has no CUDA execution provider at
    all and will simply run everything on the CPU while looking configured for
    GPU. Check the log line below when a GPU deployment seems slow.
    """
    if not device.startswith("cuda"):
        return ["CPUExecutionProvider"]

    _, _, board = device.partition(":")
    device_id = int(board) if board.isdigit() else 0
    return [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]


def build_transform(image_size: int = 224):
    """Preprocessing pipeline: resize to a square and convert to a tensor."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


@dataclass
class AnomalyResources:
    """Everything the inference stage needs, loaded once at start-up."""

    index: faiss.Index
    session: ort.InferenceSession
    transform: object
    buffer_zone: BufferZone


def load_resources(
    memory_bank_path: str,
    onnx_path: str,
    buffer_zone_path: str,
    image_size: int = 224,
    providers: list[str] | None = None,
) -> AnomalyResources:
    """Load the memory bank, build the FAISS index, load the model."""
    # 1. Load the memory bank from .npy.
    memory_bank = np.load(memory_bank_path).astype("float32")
    logger.info("Loaded memory bank %s", memory_bank.shape)

    # 2. (Re-)initialise the FAISS index with the loaded vectors.
    dim = memory_bank.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(memory_bank)

    # 3. Load the ONNX backbone.
    session = ort.InferenceSession(
        onnx_path, providers=providers or ["CPUExecutionProvider"]
    )
    # What onnxruntime actually bound to, which is not necessarily what was
    # requested: an unavailable provider is skipped silently.
    logger.info("ONNX backbone providers: %s", session.get_providers())

    # 4. Load the calibrated decision band.
    buffer_zone = load_buffer_zone(buffer_zone_path)

    return AnomalyResources(
        index=index,
        session=session,
        transform=build_transform(image_size),
        buffer_zone=buffer_zone,
    )
