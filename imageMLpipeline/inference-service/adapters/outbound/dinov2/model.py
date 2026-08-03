"""DINOv2 stage-two defect clusterer — a concrete batch ``ClusterModel``.

A faithful port of ``stage-two_experiment/DINOv2_cluster.py`` (ablation step 5:
cosine nearest-neighbour anomaly scoring against a normal memory bank + deep+colour
first layer + colour-gated ecc second layer). Shares the compact colour/shape
descriptors with the ResNet50 clusterer.

This backbone is a **candidate**: registered ``is_valid=false`` and awaiting an
evaluate-service, so it is not built at runtime today. It is implemented here so
promoting it is a one-row change, not a code change. Built (when valid) by
:class:`~adapters.outbound.factory.ModelFactory`; a singleton so the backbone
loads once.
"""
from __future__ import annotations

import io
import logging
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

import hdbscan
import umap

from adapters.outbound.patchcore.model import SingletonMeta
from adapters.outbound.resnet50 import descriptors
from application.ports.model import ClusterModel
from domain.models import ClusterAssignment, ClusterFit

logger = logging.getLogger(__name__)

# --- deep: cosine nearest-neighbour scoring + weighted pooling ---
WEIGHT_POWER = 3.0
BANK_SIZE = 20000        # max normal patches kept per layer (random downsample)
KNN_K = 1                # anomaly score = 1 - mean top-K cosine similarity

# --- first layer: block-balanced deep + colour -> KernelPCA(cosine) -> UMAP -> HDBSCAN ---
W_DEEP = 1.0
W_COLOR = 0.3
MIN_CLUSTER_SIZE = 3
KPCA_COMPONENTS = 20
UMAP_N_NEIGHBORS = 15

# --- second layer: colour intensity excludes colour/metal, then ecc splits cut/hole ---
SPLIT_ECC_GAP = 0.25
BALANCE_FRAC = 0.30
MIN_SUB = 4

SEED = 42


def _block_prep(X: np.ndarray, weight: float) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    return Xs * (weight / np.sqrt(X.shape[1]))


class Dinov2ClusterModel(ClusterModel, metaclass=SingletonMeta):
    """DINOv2 cosine memory-bank + hierarchical UMAP/HDBSCAN batch clusterer (singleton)."""

    def __init__(self, model, layers, layer_names, transform, device, input_size, model_id):
        self.model = model
        self.layers = layers
        self.layer_names = layer_names
        self.transform = transform
        self.device = device
        self.input_size = input_size
        self.model_id = model_id

    # ---- feature extraction -------------------------------------------------
    def _extract_maps(self, img: Image.Image) -> dict:
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(
                img_t, n=self.layers, reshape=True, norm=True
            )
        return {self.layer_names[i]: feats[i][0] for i in range(len(self.layer_names))}

    def _build_memory_bank(self, images: Sequence[Image.Image]) -> dict:
        feats = {L: [] for L in self.layer_names}
        for img in images:
            maps = self._extract_maps(img)
            for L in self.layer_names:
                v = maps[L].reshape(maps[L].shape[0], -1).T
                v = F.normalize(v, dim=1)
                feats[L].append(v.cpu())
        bank = {}
        for L in self.layer_names:
            allv = torch.cat(feats[L], dim=0)
            if allv.shape[0] > BANK_SIZE:
                idx = torch.randperm(allv.shape[0])[:BANK_SIZE]
                allv = allv[idx]
            bank[L] = allv.to(self.device)
        return bank

    def _extract_features(self, img: Image.Image, bank: dict):
        rgb = np.asarray(
            img.resize((self.input_size, self.input_size), Image.BILINEAR).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        maps = self._extract_maps(img)
        deep_vecs, score_maps = [], []
        for L in self.layer_names:
            m = maps[L]
            C, H, W = m.shape
            f = m.reshape(C, -1)
            fn = F.normalize(f, dim=0)
            sim = fn.T @ bank[L].T
            topk = sim.topk(min(KNN_K, sim.shape[1]), dim=1).values.mean(dim=1)
            score = 1.0 - topk
            w = score.clamp(min=0).pow(WEIGHT_POWER)
            w = w / (w.sum() + 1e-8)
            weighted = (f * w[None, :]).sum(1)
            vec = F.normalize(weighted, dim=0)
            deep_vecs.append(vec)
            sm = score.reshape(H, W).float()
            sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
            sm = F.interpolate(
                sm[None, None], size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )[0, 0]
            score_maps.append(sm)
        deep = torch.cat(deep_vecs).cpu().numpy()
        amap = torch.stack(score_maps).mean(0).cpu().numpy()
        mask, amap_s = descriptors.defect_mask(amap)
        return (
            deep,
            descriptors.color_descriptor(rgb, amap_s, mask),
            descriptors.shape_descriptor(mask),
        )

    # ---- second layer -------------------------------------------------------
    @staticmethod
    def _refine_by_shape(labels, shape_all, color_all):
        ecc = shape_all[:, 1]
        col_mag = np.linalg.norm(color_all, axis=1)
        new = labels.copy()
        cl_ids = [k for k in sorted(set(labels)) if k != -1]
        cl_colmag = {k: float(col_mag[labels == k].mean()) for k in cl_ids}
        if not cl_colmag:
            return new
        med = float(np.median(list(cl_colmag.values())))
        candidates = []
        for k in cl_ids:
            idx = np.where(labels == k)[0]
            if len(idx) < 2 * MIN_SUB:
                continue
            if cl_colmag[k] > med:            # exclude clear colour-difference groups
                continue
            km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(ecc[idx].reshape(-1, 1))
            a, b = idx[km.labels_ == 0], idx[km.labels_ == 1]
            nmin = min(len(a), len(b))
            gap = abs(ecc[a].mean() - ecc[b].mean())
            if gap >= SPLIT_ECC_GAP and nmin >= max(MIN_SUB, BALANCE_FRAC * len(idx)):
                candidates.append((nmin, gap, k, idx, a, b))
        if not candidates:
            logger.info("Second layer: no low-colour blob is ecc-bimodal enough; no split")
            return new
        nmin, gap, k, idx, a, b = max(candidates, key=lambda t: t[0])
        minor = a if len(a) < len(b) else b
        next_id = int(labels.max()) + 1
        new[minor] = next_id
        logger.info(
            "Second layer: split blob %d (n=%d) -> cluster %d (n=%d), ecc gap=%.3f",
            k, len(idx), next_id, len(minor), gap,
        )
        return new

    # ---- port ---------------------------------------------------------------
    def fit(
        self, defect_images: Sequence[bytes], normal_images: Sequence[bytes]
    ) -> ClusterFit:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        defects = [Image.open(io.BytesIO(b)).convert("RGB") for b in defect_images]
        normals = [Image.open(io.BytesIO(b)).convert("RGB") for b in normal_images]

        if normals:
            logger.info("Building normal memory bank from %d known-good image(s)", len(normals))
            bank = self._build_memory_bank(normals)
        else:
            logger.warning("No known-good images; building the memory bank from the defect set")
            bank = self._build_memory_bank(defects)

        deep_list, color_list, shape_list = [], [], []
        for img in defects:
            d, c, s = self._extract_features(img, bank)
            deep_list.append(d)
            color_list.append(c)
            shape_list.append(s)
        deep_all = np.asarray(deep_list)
        color_all = np.asarray(color_list)
        shape_all = np.asarray(shape_list)

        feat = np.concatenate(
            [_block_prep(deep_all, W_DEEP), _block_prep(color_all, W_COLOR)], axis=1
        )
        n_components = min(KPCA_COMPONENTS, len(feat) - 1)
        x_kpca = KernelPCA(
            n_components=n_components, kernel="cosine", random_state=SEED
        ).fit_transform(feat)
        n_neighbors = min(UMAP_N_NEIGHBORS, len(feat) - 1)
        x_umap = umap.UMAP(
            n_neighbors=n_neighbors, n_components=2, min_dist=0.0,
            random_state=SEED, metric="cosine",
        ).fit_transform(x_kpca)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
        labels_l1 = clusterer.fit_predict(x_umap)
        probabilities = clusterer.probabilities_

        labels_final = self._refine_by_shape(labels_l1, shape_all, color_all)

        n_clusters = len({int(k) for k in labels_final if k != -1})
        n_noise = int(np.sum(labels_final == -1))
        logger.info(
            "Clustered %d defect image(s) into %d cluster(s) (%d noise)",
            len(defects), n_clusters, n_noise,
        )

        # No ``state``: this candidate's KernelPCA/UMAP pipeline has not been
        # split into fit/assign, so every run against it is a full re-fit and its
        # cluster ids are not comparable between runs. Promoting it for production
        # use means giving it the same treatment ResNet50 got.
        return ClusterFit(
            assignments=[
                ClusterAssignment(
                    cluster_id=int(labels_final[i]),
                    probability=float(probabilities[i]),
                    model_id=self.model_id,
                )
                for i in range(len(defects))
            ],
            state=None,
        )
