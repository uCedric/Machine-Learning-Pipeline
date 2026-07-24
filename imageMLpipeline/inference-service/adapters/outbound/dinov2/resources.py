"""DINOv2 clustering resource loading.

Fetch the assets from the ``models`` bucket, load the ONNX backbone, the
pretrained PCA / UMAP / HDBSCAN artifacts and the preprocessing transform. Kept
separate from the clusterer so the model class stays free of file-system / SDK
setup concerns.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import hdbscan as hdbscan_lib
import joblib
import numpy as np
import onnxruntime as ort
from torchvision import transforms

from application.ports.storage import ObjectStorage

logger = logging.getLogger(__name__)

# DINOv2 asset object names, stored in the models bucket under
# ``{model_key}/{version}/``. The three .joblib artifacts are version-coupled
# pickles: they must be fitted and dumped with the exact scikit-learn /
# umap-learn / hdbscan versions pinned in requirements.txt.
ASSET_ONNX = "dinov2_vits14.onnx"
ASSET_PCA = "pca.joblib"
ASSET_UMAP = "umap.joblib"
ASSET_HDBSCAN = "hdbscan.joblib"
ASSET_FILENAMES = (ASSET_ONNX, ASSET_PCA, ASSET_UMAP, ASSET_HDBSCAN)

# ImageNet statistics: the artifacts must be fitted on features extracted with
# this exact normalisation, or the cluster assignments are meaningless.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class AssetPaths:
    """Local filesystem locations of the downloaded DINOv2 assets."""

    onnx: str
    pca: str
    umap: str
    hdbscan: str


def fetch_assets(
    storage: ObjectStorage,
    bucket: str,
    model_key: str,
    version: str,
    cache_dir: str,
) -> AssetPaths:
    """Download the DINOv2 assets from ``bucket/{model_key}/{version}/`` to disk.

    Returns the local paths the loaders read from.
    """
    prefix = f"{model_key}/{version}"
    dest = Path(cache_dir) / model_key / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in ASSET_FILENAMES:
        data = storage.get_object(bucket, f"{prefix}/{name}")
        (dest / name).write_bytes(data)
        logger.info("Fetched asset %s/%s/%s (%d bytes)", bucket, prefix, name, len(data))
    return AssetPaths(
        onnx=str(dest / ASSET_ONNX),
        pca=str(dest / ASSET_PCA),
        umap=str(dest / ASSET_UMAP),
        hdbscan=str(dest / ASSET_HDBSCAN),
    )


def build_transform(image_size: int = 224):
    """DINOv2 eval preprocessing: shorter-side resize, centre crop, normalise.

    Deliberately different from PatchCore's square resize — it must match the
    transform the PCA/UMAP/HDBSCAN artifacts were fitted with.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


@dataclass
class ClusterResources:
    """Everything the clustering stage needs, loaded once at start-up."""

    session: ort.InferenceSession
    pca: object  # sklearn.decomposition.PCA (384 -> 50)
    umap: object  # umap.UMAP reducer (50 -> 2)
    clusterer: object  # hdbscan.HDBSCAN fitted with prediction_data=True
    transform: object


def load_resources(
    paths: AssetPaths,
    image_size: int = 224,
    providers: list[str] | None = None,
) -> ClusterResources:
    """Load the ONNX backbone and the pretrained clustering artifacts."""
    # 1. Load the ONNX backbone.
    session = ort.InferenceSession(
        paths.onnx, providers=providers or ["CPUExecutionProvider"]
    )

    # 2. Load the pretrained projection / clustering artifacts.
    pca = joblib.load(paths.pca)
    umap_reducer = joblib.load(paths.umap)
    clusterer = joblib.load(paths.hdbscan)

    # ``approximate_predict`` needs prediction data generated at fit time; fail
    # loudly at start-up rather than on every event.
    if getattr(clusterer, "_prediction_data", None) is None:
        raise ValueError(
            "HDBSCAN artifact carries no prediction data; refit it with "
            "prediction_data=True (or call generate_prediction_data() before "
            "dumping) and re-upload."
        )

    # 3. Warm up the numba-JIT'd UMAP/HDBSCAN path: the first call compiles for
    # tens of seconds and must not land on the first Kafka event.
    start = time.perf_counter()
    warm = pca.transform(np.zeros((1, pca.n_features_in_), dtype=np.float32))
    hdbscan_lib.approximate_predict(clusterer, umap_reducer.transform(warm))
    logger.info("UMAP/HDBSCAN warmup finished in %.1fs", time.perf_counter() - start)

    return ClusterResources(
        session=session,
        pca=pca,
        umap=umap_reducer,
        clusterer=clusterer,
        transform=build_transform(image_size),
    )
