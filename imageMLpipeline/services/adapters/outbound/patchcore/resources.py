"""PatchCore resource loading (stage 1).

Load the memory bank, build the FAISS index, load the ONNX backbone and the
preprocessing transform. Kept separate from the detector so the model class
stays free of file-system / SDK setup concerns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import faiss
import numpy as np
import onnxruntime as ort
from torchvision import transforms

logger = logging.getLogger(__name__)

# Default PatchCore asset locations inside the container image (see the Docker
# build, which copies ``adapters/`` to ``/app/adapters``).
DEFAULT_MEMORY_BANK_PATH = "/app/adapters/outbound/patchcore/asset/memory_bank.npy"
DEFAULT_ONNX_PATH = "/app/adapters/outbound/patchcore/asset/resnet_backbone.onnx"


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


def load_resources(
    memory_bank_path: str = DEFAULT_MEMORY_BANK_PATH,
    onnx_path: str = DEFAULT_ONNX_PATH,
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

    return AnomalyResources(
        index=index, session=session, transform=build_transform(image_size)
    )
