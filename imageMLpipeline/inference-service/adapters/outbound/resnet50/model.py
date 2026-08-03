"""ResNet50 stage-two defect clusterer — a concrete batch ``ClusterModel``.

A faithful port of ``stage-two_experiment/ResNet50_cluster.py`` (the validated
hierarchical-v2 pipeline), adapted to the pipeline's hexagon. The ground-truth
benchmark, matplotlib plots and per-cluster folder copying are dropped (there is
no ground truth in production); everything that produces the cluster labels is
kept verbatim:

  1. ResNet50 Layer2+Layer3 maps -> a normal-texture prototype (mean/std per
     channel) built from recent known-good images.
  2. per defect image: anomaly-weighted deep pooling (deep) + a compact
     single-peak anomaly mask -> residual colour (5-d) and shape (5-d).
  3. first layer: block-balanced deep+colour+shape -> PCA(0.95) -> UMAP ->
     HDBSCAN  (clean colour / metal / thread + a merged cut+hole blob).
  4. second layer: split only the most-balanced ecc-bimodal blob (cut vs hole)
     with a 1-D KMeans, guarded by an ecc gap and a balance fraction.

Two entry points. ``fit`` runs the four steps above over a whole defect set and
returns the fitted state alongside the labels; ``assign`` replays that frozen
state over a new batch, so later images join the *same* clusters instead of
provoking a re-fit that would renumber everything.

Built by :class:`~adapters.outbound.factory.ModelFactory`; a singleton so the
heavy backbone loads once.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

import hdbscan
import umap

from adapters.outbound.patchcore.model import SingletonMeta
from adapters.outbound.resnet50 import descriptors, resources
from adapters.outbound.resnet50.resources import LAYERS
from application.ports.model import ClusterModel
from domain.models import ClusterAssignment, ClusterFit

logger = logging.getLogger(__name__)

# --- deep anomaly-weighted pooling ---
FEATURE_MODE = "both"       # weighted - mu (deep residual + weighted pooling)
WEIGHT_POWER = 3.0          # concentrates pooling on the most anomalous patches

# --- first layer: block weights (= v2 config: deep + colour + shape) ---
W_DEEP = 1.0
W_COLOR = 0.3
W_SHAPE = 0.3
MIN_CLUSTER_SIZE = 3
UMAP_N_NEIGHBORS = 15

# --- second layer: split only the balanced ecc-bimodal (cut+hole) blob ---
SPLIT_ECC_GAP = 0.25        # the two subgroups' mean ecc gap must be >= this
BALANCE_FRAC = 0.30         # the minority subgroup must be >= this fraction
MIN_SUB = 4

SEED = 42


def _fit_block(X: np.ndarray, weight: float) -> tuple[np.ndarray, StandardScaler]:
    """StandardScale a feature block, then scale so its L2 contribution ~= weight.

    Returns the transformed block **and the fitted scaler**. The scaler used to be
    discarded, which is precisely what made the pipeline impossible to reuse: a
    later batch standardised against its own mean/std would land somewhere else in
    the same nominal space.
    """
    scaler = StandardScaler().fit(X)
    return _apply_block(X, weight, scaler), scaler


def _apply_block(X: np.ndarray, weight: float, scaler: StandardScaler) -> np.ndarray:
    """Re-apply a fitted block scaling to new rows."""
    return scaler.transform(X) * (weight / np.sqrt(X.shape[1]))


class ResNet50ClusterModel(ClusterModel, metaclass=SingletonMeta):
    """ResNet50 Layer2/3 + hierarchical UMAP/HDBSCAN batch clusterer (singleton)."""

    def __init__(
        self, feature_extractor, transform, device, image_size, model_id, state=None
    ):
        self.feature_extractor = feature_extractor  # torchvision feature extractor
        self.transform = transform                  # eval preprocessing (callable)
        self.device = device
        self.image_size = image_size
        self.model_id = model_id                    # inference_model.model_id
        # The frozen fit for this version, if it has one. None means this version
        # has never been fitted, so only ``fit`` is available.
        self.state = state

    # ---- feature extraction -------------------------------------------------
    def _extract_maps(self, img: Image.Image) -> dict:
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.feature_extractor(img_t)
        return {L: out[L][0] for L in LAYERS}

    def _build_prototype(self, images: Sequence[Image.Image]) -> dict:
        """Per-channel (mu, std) of Layer2/Layer3 features over normal images."""
        stats = {L: {"sum": 0.0, "sqsum": 0.0, "n": 0} for L in LAYERS}
        for img in images:
            maps = self._extract_maps(img)
            for L in LAYERS:
                v = maps[L].reshape(maps[L].shape[0], -1).T
                stats[L]["sum"] += v.sum(0)
                stats[L]["sqsum"] += (v * v).sum(0)
                stats[L]["n"] += v.shape[0]
        proto = {}
        for L in LAYERS:
            cnt = stats[L]["n"]
            mu = stats[L]["sum"] / cnt
            var = stats[L]["sqsum"] / cnt - mu * mu
            proto[L] = (mu, torch.sqrt(torch.clamp(var, min=1e-6)))
        return proto

    def _extract_features(self, img: Image.Image, prototype: dict):
        """(deep vector, colour 5-d, shape 5-d) for one defect image."""
        rgb = np.asarray(
            img.resize((self.image_size, self.image_size), Image.BILINEAR).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        maps = self._extract_maps(img)
        deep_vecs, score_maps = [], []
        for L in LAYERS:
            m = maps[L]
            C, H, W = m.shape
            f = m.reshape(C, -1)
            mu, std = prototype[L]
            fz = (f - mu[:, None]) / std[:, None]
            score = fz.norm(dim=0)
            w = score.pow(WEIGHT_POWER)
            w = w / (w.sum() + 1e-8)
            weighted = (f * w[None, :]).sum(1)
            if FEATURE_MODE == "weighted":
                vec = weighted
            elif FEATURE_MODE == "residual":
                vec = f.mean(1) - mu
            else:
                vec = weighted - mu
            deep_vecs.append(vec)
            sm = score.reshape(H, W).float()
            sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
            sm = F.interpolate(
                sm[None, None], size=(self.image_size, self.image_size),
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
    def _refine_by_shape(
        labels: np.ndarray, shape_all: np.ndarray
    ) -> tuple[np.ndarray, dict | None]:
        """Split only the most-balanced ecc-bimodal group (cut+hole); leave the rest.

        Also returns the *decision* as a reusable dict, because this step is a
        search rather than a model: it scans every cluster for the best ecc-bimodal
        candidate and splits it with a throwaway KMeans. Freezing which cluster was
        split, the KMeans that split it, and which side became the new id is what
        lets :meth:`_apply_refine` reproduce the same split for later batches.
        """
        ecc = shape_all[:, 1]
        new = labels.copy()
        candidates = []
        for k in sorted(set(labels)):
            if k == -1:
                continue
            idx = np.where(labels == k)[0]
            if len(idx) < 2 * MIN_SUB:
                continue
            km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(ecc[idx].reshape(-1, 1))
            a, b = idx[km.labels_ == 0], idx[km.labels_ == 1]
            nmin = min(len(a), len(b))
            gap = abs(ecc[a].mean() - ecc[b].mean())
            if gap >= SPLIT_ECC_GAP and nmin >= max(MIN_SUB, BALANCE_FRAC * len(idx)):
                candidates.append((nmin, gap, k, idx, a, b, km))
        if not candidates:
            logger.info(
                "Second layer: no group satisfies ecc gap >= %.2f and balance >= %.2f; no split",
                SPLIT_ECC_GAP, BALANCE_FRAC,
            )
            return new, None
        nmin, gap, k, idx, a, b, km = max(candidates, key=lambda t: t[0])
        minor = a if len(a) < len(b) else b
        # Which KMeans side became the new cluster, so a later batch can be sent
        # the same way rather than re-deciding "which side is the minority".
        minor_side = int(km.labels_[np.searchsorted(idx, minor[0])])
        next_id = int(labels.max()) + 1
        new[minor] = next_id
        logger.info(
            "Second layer: split cluster %d (n=%d) -> cluster %d (n=%d), ecc gap=%.3f, minority=%.0f%%",
            k, len(idx), next_id, len(minor), gap, 100 * nmin / len(idx),
        )
        return new, {"source_cluster": int(k), "new_cluster": next_id,
                     "kmeans": km, "minor_side": minor_side}

    @staticmethod
    def _apply_refine(
        labels: np.ndarray, shape_all: np.ndarray, refine: dict | None
    ) -> np.ndarray:
        """Re-apply a frozen second-layer split to newly assigned labels."""
        if refine is None:
            return labels
        new = labels.copy()
        idx = np.where(labels == refine["source_cluster"])[0]
        if len(idx) == 0:
            return new
        sides = refine["kmeans"].predict(shape_all[idx, 1].reshape(-1, 1))
        new[idx[sides == refine["minor_side"]]] = refine["new_cluster"]
        return new

    # ---- shared descriptor extraction ---------------------------------------
    def _descriptors(self, defects: Sequence[Image.Image], prototype: dict):
        """(deep, colour, shape) matrices for a batch, against a given prototype."""
        deep_list, color_list, shape_list = [], [], []
        for img in defects:
            d, c, s = self._extract_features(img, prototype)
            deep_list.append(d)
            color_list.append(c)
            shape_list.append(s)
        # shape_all[:, 1] is eccentricity, which the second layer splits on.
        return (
            np.asarray(deep_list),
            np.asarray(color_list),
            np.asarray(shape_list),
        )

    @staticmethod
    def _open(images: Sequence[bytes]) -> list[Image.Image]:
        return [Image.open(io.BytesIO(b)).convert("RGB") for b in images]

    # ---- port ---------------------------------------------------------------
    def fit(
        self, defect_images: Sequence[bytes], normal_images: Sequence[bytes]
    ) -> ClusterFit:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        defects = self._open(defect_images)
        normals = self._open(normal_images)

        # Normal-texture prototype. Fall back to the defect images' own stats if
        # no known-good images are available yet (mirrors the experiment).
        if normals:
            logger.info("Building normal prototype from %d known-good image(s)", len(normals))
            prototype = self._build_prototype(normals)
        else:
            logger.warning("No known-good images; using the defect set as the prototype")
            prototype = self._build_prototype(defects)

        deep_all, color_all, shape_all = self._descriptors(defects, prototype)

        # First layer: block-balanced deep+colour+shape -> PCA(0.95) -> UMAP ->
        # HDBSCAN. Every estimator is now *kept*, not discarded — that bundle is
        # what `assign` replays.
        deep_x, deep_scaler = _fit_block(deep_all, W_DEEP)
        color_x, color_scaler = _fit_block(color_all, W_COLOR)
        shape_x, shape_scaler = _fit_block(shape_all, W_SHAPE)
        feat = normalize(np.concatenate([deep_x, color_x, shape_x], axis=1), norm="l2")

        pca = PCA(n_components=0.95, random_state=SEED).fit(feat)
        n_neighbors = min(UMAP_N_NEIGHBORS, len(feat) - 1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, n_components=2, min_dist=0.0,
            random_state=SEED, metric="cosine",
        ).fit(pca.transform(feat))
        # prediction_data=True is what makes hdbscan.approximate_predict possible
        # later; without it the frozen clusterer cannot label a new point at all.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE, prediction_data=True
        )
        labels_l1 = clusterer.fit_predict(reducer.embedding_)
        probabilities = clusterer.probabilities_

        # Second layer: split the cut+hole blob, keeping the decision.
        labels_final, refine = self._refine_by_shape(labels_l1, shape_all)

        n_clusters = len({int(k) for k in labels_final if k != -1})
        n_noise = int(np.sum(labels_final == -1))
        logger.info(
            "Fitted %d defect image(s) into %d cluster(s) (%d noise)",
            len(defects), n_clusters, n_noise,
        )

        live_state = {
                "prototype": prototype,
                "scalers": {
                    "deep": deep_scaler,
                    "color": color_scaler,
                    "shape": shape_scaler,
                },
                "pca": pca,
                "umap": reducer,
                "hdbscan": clusterer,
                "refine": refine,
                # Recorded so a state can be recognised as incompatible rather
                # than silently misused if these constants ever change.
                "meta": {
                    "seed": SEED,
                    "image_size": self.image_size,
                    "weights": [W_DEEP, W_COLOR, W_SHAPE],
                    "feature_mode": FEATURE_MODE,
                    "min_cluster_size": MIN_CLUSTER_SIZE,
                },
        }
        # Adopt the fit immediately. This class is a singleton built once at
        # start-up, so without this the running process would keep `state=None`,
        # never be able to assign, and re-fit forever. A restart would pick the
        # state up from object storage, but nothing should need a restart.
        self.state = live_state
        return ClusterFit(
            assignments=self._assignments(labels_final, probabilities),
            state=resources.dump_state(live_state),
        )

    def assign(self, defect_images: Sequence[bytes]) -> list[ClusterAssignment]:
        if self.state is None:
            raise NotImplementedError(
                "No frozen fit loaded for this model version; re-fit instead"
            )
        state = self.state
        defects = self._open(defect_images)

        # The frozen prototype, not a fresh one: rebuilding it from today's
        # known-good images would shift the whole deep space and make the
        # existing cluster ids meaningless.
        deep_all, color_all, shape_all = self._descriptors(defects, state["prototype"])

        scalers = state["scalers"]
        feat = normalize(
            np.concatenate(
                [
                    _apply_block(deep_all, W_DEEP, scalers["deep"]),
                    _apply_block(color_all, W_COLOR, scalers["color"]),
                    _apply_block(shape_all, W_SHAPE, scalers["shape"]),
                ],
                axis=1,
            ),
            norm="l2",
        )
        embedding = state["umap"].transform(state["pca"].transform(feat))
        labels, strengths = hdbscan.approximate_predict(state["hdbscan"], embedding)
        labels = self._apply_refine(np.asarray(labels), shape_all, state["refine"])

        n_noise = int(np.sum(labels == -1))
        logger.info(
            "Assigned %d defect image(s) to the frozen fit (%d matched no cluster)",
            len(defects), n_noise,
        )
        return self._assignments(labels, strengths)

    def _assignments(self, labels, probabilities) -> list[ClusterAssignment]:
        return [
            ClusterAssignment(
                cluster_id=int(labels[i]),
                probability=float(probabilities[i]),
                model_id=self.model_id,
            )
            for i in range(len(labels))
        ]
