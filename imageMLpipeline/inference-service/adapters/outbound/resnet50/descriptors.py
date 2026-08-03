"""Compact per-defect descriptors (colour + shape), ported from the experiment.

These are pure numpy / scipy / scikit-image helpers lifted verbatim from
``stage-two_experiment/ResNet50_cluster.py`` (the ``defect_mask``,
``color_descriptor`` and ``shape_descriptor`` functions). They turn a per-image
anomaly heatmap into a compact single-peak defect mask and read residual colour
and geometric shape features off it — the "v2 descriptors" that let the first
clustering layer separate colour / metal / thread and the second layer split
cut vs hole.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

# --- descriptor localisation (compact single-peak mask) ---
SMOOTH_SIGMA = 4.0
MASK_K = 3.0
MASK_PCTL = 97.5
BG_PCTL = 50


def defect_mask(amap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compact single-peak defect mask: keep the connected component holding the
    highest-scoring peak of the (smoothed) anomaly map."""
    a = ndi.gaussian_filter(amap, sigma=SMOOTH_SIGMA)
    thr = max(a.mean() + MASK_K * a.std(), np.percentile(a, MASK_PCTL))
    m = a >= thr
    if m.sum() < 4:
        m = a >= np.percentile(a, 99)
    m = ndi.binary_closing(m, iterations=1)
    lbl = measure.label(m)
    if lbl.max() == 0:
        return m, a
    peak = np.unravel_index(np.argmax(a), a.shape)
    peak_label = lbl[peak]
    if peak_label == 0:
        counts = np.bincount(lbl.ravel())
        peak_label = int(np.argmax(counts[1:])) + 1
    return (lbl == peak_label), a


def color_descriptor(rgb: np.ndarray, amap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Residual colour, 5 dims: residual RGB(3) + brightness residual(1) + defect saturation(1)."""
    bg = amap <= np.percentile(amap, BG_PCTL)
    if mask.sum() < 3:
        mask = amap >= np.percentile(amap, 99)
    if bg.sum() < 10:
        bg = ~mask
    d, b = rgb[mask], rgb[bg]
    res_rgb = d.mean(0) - b.mean(0)
    res_bright = np.array([d.mean(1).mean() - b.mean(1).mean()])
    sat = (d.max(1) - d.min(1)).mean()
    return np.concatenate([res_rgb, res_bright, [sat]]).astype(np.float32)


def shape_descriptor(mask: np.ndarray) -> np.ndarray:
    """v2 shape, 5 dims: area / ecc / solidity / extent / aspect (ecc at index 1)."""
    default = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    try:
        lbl = measure.label(mask)
        props = measure.regionprops(lbl)
        if not props:
            return default
        r = max(props, key=lambda p: p.area)
        if r.area < 4:
            return default
        minr, minc, maxr, maxc = r.bbox
        h, w = (maxr - minr), (maxc - minc)
        aspect = max(h, w) / (min(h, w) + 1e-6)
        return np.array(
            [r.area / mask.size, r.eccentricity, r.solidity, r.extent, aspect],
            dtype=np.float32,
        )
    except Exception:
        return default
