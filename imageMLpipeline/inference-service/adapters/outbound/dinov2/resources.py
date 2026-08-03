"""DINOv2 clustering resource loading (batch, cosine memory bank).

The stage-two DINOv2 backbone is a fixed pretrained ViT loaded from torch hub
(no per-version artifacts to fetch — the normal memory bank is built at batch
time from known-good images). This backbone is a *candidate*: it is registered
``is_valid=false`` and therefore not built at runtime until an evaluate-service
promotes it, so its weights are fetched lazily on first activation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ViT-B/14 with register tokens; intermediate blocks [3, 5] (the finalised choice
# in the experiment). PATCH 14 -> INPUT_SIZE must be a multiple of 14.
DINO_MODEL = "dinov2_vitb14_reg"
DINO_LAYERS = [3, 5]
PATCH = 14

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


@dataclass
class ClusterResources:
    """Everything the DINOv2 clustering stage needs, loaded once at start-up."""

    model: torch.nn.Module
    layers: list[int]        # intermediate block indices to pool over
    layer_names: list[str]   # stable names aligned with ``layers``
    transform: object
    device: str
    input_size: int


def load_resources(image_size: int = 448) -> ClusterResources:
    """Load the DINOv2 backbone and derive the intermediate-layer configuration."""
    input_size = (image_size // PATCH) * PATCH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL).to(device).eval()

    n_blocks = len(model.blocks)
    layers = sorted({(i if i >= 0 else n_blocks + i) for i in DINO_LAYERS})
    layers = [i for i in layers if 0 <= i < n_blocks]
    if not layers:
        raise ValueError("DINO_LAYERS out of the model's block range")
    layer_names = [f"blk{i}" for i in layers]
    logger.info(
        "Loaded DINOv2 %s on %s · %d blocks · using intermediate layers %s · input %d",
        DINO_MODEL, device, n_blocks, layers, input_size,
    )
    return ClusterResources(
        model=model,
        layers=layers,
        layer_names=layer_names,
        transform=build_transform(input_size),
        device=device,
        input_size=input_size,
    )
