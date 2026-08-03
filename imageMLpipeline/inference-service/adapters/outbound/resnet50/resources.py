"""ResNet50 clustering resource loading.

The **backbone** is a fixed pretrained network — the standard torchvision ResNet50
ImageNet checkpoint, pre-downloaded into the image at build time so runtime stays
offline — so there is nothing version-specific about the weights.

The **fit**, on the other hand, is now a per-version artifact. Everything the
clustering pipeline learns from a batch (the normal-texture prototype, the three
block scalers, PCA, UMAP, the HDBSCAN clusterer and the second-layer split
decision) is serialised into a single ``cluster_state.joblib`` under
``{bucket}/{model_type}/{version}/``, exactly like PatchCore's memory bank. That
is what lets a later batch be *assigned* to the same clusters instead of
re-fitting, so a ``cluster_id`` keeps its meaning.

The prototype crosses that boundary as numpy rather than torch tensors: the state
is a pickle, and pinning it to a torch version as well as a scikit-learn/UMAP one
would make it needlessly fragile.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import joblib
import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

logger = logging.getLogger(__name__)

# The frozen fit, stored alongside the model version it belongs to.
STATE_ASSET = "cluster_state.joblib"

# ImageNet statistics — the backbone was pretrained with this normalisation.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# The two intermediate feature levels the experiment pools over: Layer2 (512-ch,
# finer/colour-texture) + Layer3 (1024-ch, coarser/semantic).
LAYERS = ["layer2", "layer3"]


def build_transform(image_size: int = 448):
    """Square-resize + ImageNet-normalise eval transform (matches the experiment)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


@dataclass
class ClusterResources:
    """Everything the ResNet50 clustering stage needs, loaded once at start-up."""

    feature_extractor: torch.nn.Module
    transform: object
    device: str


def state_key(model_type: str, version: str) -> str:
    """Object key of a version's frozen fit."""
    return f"{model_type}/{version}/{STATE_ASSET}"


def dump_state(state: dict) -> bytes:
    """Serialise a fitted-state bundle, converting the prototype to numpy.

    ``prototype`` arrives as ``{layer: (mu, std)}`` torch tensors; they are stored
    as numpy so the artifact does not additionally depend on a torch version.
    """
    payload = dict(state)
    payload["prototype"] = {
        layer: (mu.cpu().numpy(), std.cpu().numpy())
        for layer, (mu, std) in state["prototype"].items()
    }
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    return buffer.getvalue()


def load_state(data: bytes, device: str = "cpu") -> dict:
    """Inverse of :func:`dump_state`, restoring the prototype as torch tensors."""
    state = joblib.load(io.BytesIO(data))
    state["prototype"] = {
        layer: (
            torch.as_tensor(mu, device=device),
            torch.as_tensor(std, device=device),
        )
        for layer, (mu, std) in state["prototype"].items()
    }
    return state


def save_state(storage, version, data: bytes) -> None:
    """Upload a frozen fit to its version's prefix in object storage.

    Lives here rather than in the composition root so the object-key layout stays
    with the model package that defines it — the root only needs a callable.
    """
    key = state_key(version.model_type, version.version)
    storage.put_object(version.bucket, key, data, "application/octet-stream")
    logger.info("Saved frozen fit to %s/%s (%d bytes)", version.bucket, key, len(data))


def load_resources(image_size: int = 448) -> ClusterResources:
    """Load the ResNet50 Layer2/Layer3 feature extractor and the eval transform."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    extractor = create_feature_extractor(
        base, {"layer2": "layer2", "layer3": "layer3"}
    ).eval()
    logger.info("Loaded ResNet50 (Layer2+Layer3) feature extractor on %s", device)
    return ClusterResources(
        feature_extractor=extractor,
        transform=build_transform(image_size),
        device=device,
    )
