"""
Stage-2 Clustering — DINOv2 cosine nearest-neighbor anomaly scoring (drop Euclidean residual) + deep+color + second layer (ABLATION step 5)
================================================================================
The first four steps used the 'Euclidean z-score residual' as the anomaly score; Hungarian stalled at 0.494 (and 448/784 were identical -> resolution is not the bottleneck).
This step switches to 'cosine similarity' to leverage the strength of DINOv2 features:

  anomaly score:  ‖(f−μ)/σ‖ (Euclidean z-score)   ->   1 − nearest-neighbor cosine similarity to the normal memory bank
  deep feature:   weighted − μ (Euclidean residual) ->   L2-normalize(cosine anomaly-weighted pooling)

Why cosine fits better: DINOv2 is a LayerNorm feature, semantics are encoded in 'direction' rather than 'per-channel magnitude'; a cosine / nearest-neighbor memory bank
(PatchCore / AnomalyDINO style) matches its geometry better than a 'per-channel z-score against a single mean', and the memory bank preserves the diversity (multi-modality) of the normal texture
instead of collapsing diverse normals into one mean.

The rest follows step 4: first layer deep+color (block_prep balanced 1.0/0.3) -> HDBSCAN; second layer splits cut/hole
(★use color intensity to exclude color/metal, then use ecc to split cut/hole; the elongation-ratio variant was tried but didn't split at the second layer so we reverted to ecc).
color/shape still use the anomaly map for localization, only the map is now produced from cosine scores (localization may be more accurate).

★Folder assumption (MVTec standard structure):
   IMAGE_DIR  = .../carpet/test/          test/<defecttype>/*.png  (label = parent folder name)
   NORMAL_DIR = .../carpet/train/good/    normal images, used to build the memory bank
"""

import os
import glob
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans   # used by the second-layer 1D-ecc splitter
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.optimize import linear_sum_assignment
from scipy import ndimage as ndi
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt

import umap
import hdbscan

# ==========================================
# 0. Configuration
# ==========================================
IMAGE_DIR    = "/content/carpet/test/"
NORMAL_DIR   = "/content/carpet/train/good/"   # build the normal texture prototype; if not found, fall back to the mean of all defect images
INCLUDE_GOOD = False

DINO_MODEL   = "dinov2_vitb14_reg"   # vits14/vitb14/vitl14/vitg14 (+_reg); if it fails, drop _reg
DINO_LAYERS  = [3, 5]                # ★finalized. [3,5,8] retested (with the reworked second layer) = 0.708 < 0.753: its cut/hole
#   can't be separated by ecc, the second layer safely doesn't split, and hole is lost. [3,5]'s cut/hole is separable -> 0.753 is best.
PATCH        = 14
INPUT_SIZE   = 448                   # a multiple of 14 -> 32x32 patch grid. ★The best cosine result is exactly 448 (0.753);
#   tried 784 (56x56) but it was worse (0.627, finer map is noisier, cut/hole/thread more fragmented, second layer finds no bimodality and doesn't split).
#   So 448 cosine is the optimization baseline; we don't push resolution higher.

# --- deep: cosine nearest-neighbor anomaly scoring + weighted pooling (drop Euclidean z-score residual) ---
MAX_NORMAL   = 200                   # max number of normal images used to build the normal memory bank
WEIGHT_POWER = 3.0                   # weighted-pooling exponent: larger concentrates more on the most anomalous patches
BANK_SIZE    = 20000                 # max patches kept per layer in the normal memory bank (random downsample, controls compute/memory)
KNN_K        = 1                     # anomaly score = 1 - mean of top-K cosine similarity (K=1 is pure nearest-neighbor)

# --- residual color descriptor: compact single-peak localization (same as the hierarchical version) ---
SMOOTH_SIGMA = 4.0
MASK_K       = 3.0
MASK_PCTL    = 97.5
BG_PCTL      = 50

# --- first layer, block-balanced concatenation: deep + color (shape not in first layer; step 3 proved global shape is harmful) ---
W_DEEP  = 1.0
W_COLOR = 0.3    # diagnostic A confirmed color is a net contribution (removing it drops 0.11 and scatters thread more), restored to 0.3
MIN_CLUSTER_SIZE = 3
KPCA_COMPONENTS  = 20    # KernelPCA (cosine kernel) retained dimensions; sweep {10,15,20,30(=0.793),50}. Replaces linear PCA(0.95)

# --- second layer (reworked): color intensity excludes color/metal, then ecc splits cut/hole (elongation version didn't split so reverted to ecc) ---
SPLIT_ECC_GAP = 0.25   # the two subgroups' mean ecc difference must be >= this to count as a true cut/hole bimodality
BALANCE_FRAC  = 0.30   # the minority subgroup must be >= 30% of the whole group (blocks fake candidates that only fling off outliers)
MIN_SUB       = 4      # minimum number of images in a subgroup
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

INPUT_SIZE = (INPUT_SIZE // PATCH) * PATCH
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{device.upper()}] Loading DINOv2 ({DINO_MODEL}) ... INPUT_SIZE={INPUT_SIZE} -> patch grid {INPUT_SIZE//PATCH}x{INPUT_SIZE//PATCH}")

# ==========================================
# 1. Feature extractor (DINOv2 intermediate-layer patch tokens -> spatial feature maps)
# ==========================================
model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL).to(device).eval()
EMBED_DIM = getattr(model, "embed_dim", None)

N_BLOCKS = len(model.blocks)
DINO_LAYERS = sorted({(i if i >= 0 else N_BLOCKS + i) for i in DINO_LAYERS})
DINO_LAYERS = [i for i in DINO_LAYERS if 0 <= i < N_BLOCKS]
assert DINO_LAYERS, "DINO_LAYERS is empty: indices out of the model's layer range"
print(f"DINOv2 {DINO_MODEL} · embed_dim={EMBED_DIM} · {N_BLOCKS} blocks total · using intermediate layers {DINO_LAYERS}")

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LAYERS = [f"blk{i}" for i in DINO_LAYERS]   # each selected intermediate layer = one "feature level"


def extract_maps(img):
    """Return {level: (D, Hp, Wp)} patch-token spatial feature maps (from the specified intermediate layers)."""
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # reshape=True -> (B, D, Hp, Wp), CLS/register tokens already stripped; order increases by block index, aligned with LAYERS
        feats = model.get_intermediate_layers(img_t, n=DINO_LAYERS, reshape=True, norm=True)
    return {LAYERS[i]: feats[i][0] for i in range(len(LAYERS))}


# ==========================================
# 2. Read the full defect images
# ==========================================
image_paths, ground_truth_labels = [], []
for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "**", "*"), recursive=True)):
    if not p.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    label = os.path.basename(os.path.dirname(p))
    if label == "good" and not INCLUDE_GOOD:
        continue
    image_paths.append(p)
    ground_truth_labels.append(label)

unique_labels = sorted(set(ground_truth_labels))
print(f"Found {len(image_paths)} full images, ground-truth defect classes: {unique_labels}")

# ==========================================
# 3. Normal feature memory bank (one set of L2-normalized patch features per layer, for cosine nearest-neighbor scoring)
# ==========================================
# Replaces the old per-channel mean/std prototype: instead store the normal patch features themselves (L2 normalized). Then each test patch's
# anomaly score = 1 - nearest-neighbor cosine similarity to the memory bank. Cosine fits DINOv2 (LayerNorm) features better than Euclidean z-score;
# the memory bank also preserves the diversity (multi-modality) of the normal texture, unlike a single mean that collapses diverse normals to one point.

def build_memory_bank(sources, max_n):
    feats = {L: [] for L in LAYERS}
    for p in tqdm(sources[:max_n], desc="Building normal feature memory bank"):
        maps = extract_maps(Image.open(p).convert("RGB"))
        for L in LAYERS:
            v = maps[L].reshape(maps[L].shape[0], -1).T      # (N_patch, D)
            v = F.normalize(v, dim=1)                         # per-patch L2 normalize -> cosine space
            feats[L].append(v.cpu())
    bank = {}
    for L in LAYERS:
        allv = torch.cat(feats[L], dim=0)                     # (N_total, D)
        if allv.shape[0] > BANK_SIZE:                         # random downsample to control size
            idx = torch.randperm(allv.shape[0])[:BANK_SIZE]
            allv = allv[idx]
        bank[L] = allv.to(device)                             # (BANK_SIZE, D) on device
        print(f"  {L}: memory bank {bank[L].shape[0]} patches × {bank[L].shape[1]}D")
    return bank


normal_paths = []
if NORMAL_DIR and os.path.isdir(NORMAL_DIR):
    normal_paths = sorted(p for p in glob.glob(os.path.join(NORMAL_DIR, "**", "*"), recursive=True)
                          if p.lower().endswith((".png", ".jpg", ".jpeg")))
if normal_paths:
    print(f"Using {min(len(normal_paths), MAX_NORMAL)} / {len(normal_paths)} normal images to build the memory bank")
    memory_bank = build_memory_bank(normal_paths, MAX_NORMAL)
else:
    print("⚠ NORMAL_DIR not found -> falling back to building the memory bank from 'all defect images' (less ideal)")
    memory_bank = build_memory_bank(image_paths, len(image_paths))

# ==========================================
# 4. Feature extraction: deep (weighted pooling + residual) + anomaly map + residual color descriptor
# ==========================================

def defect_mask(amap):
    """Take a 'compact single-peak' defect mask from the anomaly map (keep only the connected component containing the highest-scoring peak)."""
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


def color_descriptor(rgb, amap, mask):
    """Residual color, 5 dims: residual RGB(3) + brightness residual(1) + defect saturation(1). All are defect relative to background."""
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


def shape_descriptor(mask):
    """Shape, 5 dims: area / ecc / solidity / extent / aspect (ecc at index 1)."""
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
        return np.array([r.area / mask.size, r.eccentricity,
                         r.solidity, r.extent, aspect], dtype=np.float32)
    except Exception:
        return default


def elongation_score(amap, mask):
    """Mathematical measure of circle vs elongated: second moment of defect-region coordinates weighted by anomaly score -> anisotropy √(λ1/λ2).
    Circular hole -> λ1≈λ2 -> ≈1; elongated cut -> λ1≫λ2 -> far greater than 1. Doesn't rely on a binary threshold, more stable than ecc.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return 1.0
    w = amap[ys, xs].astype(np.float64)
    w = w - w.min() + 1e-6                        # guarantee positive weights
    wsum = w.sum()
    cx = (w * xs).sum() / wsum; cy = (w * ys).sum() / wsum
    dx = xs - cx; dy = ys - cy
    cxx = (w * dx * dx).sum() / wsum
    cyy = (w * dy * dy).sum() / wsum
    cxy = (w * dx * dy).sum() / wsum
    ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))   # ascending -> ev[0]=λ2, ev[1]=λ1
    l2, l1 = max(float(ev[0]), 1e-9), max(float(ev[1]), 1e-9)
    return float(np.sqrt(l1 / l2))                # elongation ratio: circle ~1, elongated large


def extract_features(img):
    """Return (deep vector, color 5-dim, shape 5-dim, elong elongation-ratio scalar).
    deep  = per-layer 'cosine anomaly-weighted pooling' then L2-normalized concatenation (no Euclidean residual).
    color = anomaly map averaged from each layer's cosine anomaly-score maps -> compact single-peak mask -> residual color.
    shape = geometry of the compact single-peak mask (area/ecc/solidity/extent/aspect).
    elong = anisotropy √(λ1/λ2) of the anomaly-heatmap-weighted second moment: circular hole ~1 / elongated cut large (for the second-layer cut/hole split).
    """
    rgb = np.asarray(img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR).convert("RGB"),
                     dtype=np.float32) / 255.0
    maps = extract_maps(img)
    deep_vecs, score_maps = [], []
    for L in LAYERS:
        m = maps[L]
        C, H, W = m.shape
        f = m.reshape(C, -1)                             # (C, N_patch)
        fn = F.normalize(f, dim=0)                        # per-patch L2 normalize -> cosine space
        # cosine nearest-neighbor anomaly score: 1 - mean of top-K similarity to the normal memory bank (the less like normal, the higher)
        sim = fn.T @ memory_bank[L].T                     # (N_patch, BANK_SIZE) cosine similarity
        topk = sim.topk(min(KNN_K, sim.shape[1]), dim=1).values.mean(dim=1)
        score = 1.0 - topk                                # (N,) anomaly score
        w = score.clamp(min=0).pow(WEIGHT_POWER); w = w / (w.sum() + 1e-8)
        weighted = (f * w[None, :]).sum(1)                # cosine anomaly-weighted pooling (pooled over the raw features)
        vec = F.normalize(weighted, dim=0)                # L2 normalize -> stay in cosine space (no Euclidean residual)
        deep_vecs.append(vec)
        # anomaly-score map -> normalize -> upsample to image size (for color/shape localization)
        sm = score.reshape(H, W).float()
        sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
        sm = F.interpolate(sm[None, None], size=(INPUT_SIZE, INPUT_SIZE),
                           mode="bilinear", align_corners=False)[0, 0]
        score_maps.append(sm)
    deep = torch.cat(deep_vecs).cpu().numpy()
    amap = torch.stack(score_maps).mean(0).cpu().numpy()   # per-layer averaged anomaly map
    mask, amap_s = defect_mask(amap)
    return deep, color_descriptor(rgb, amap_s, mask), shape_descriptor(mask), elongation_score(amap_s, mask)


deep_list, color_list, shape_list, elong_list = [], [], [], []
for img_path in tqdm(image_paths, desc=f"Extracting deep(cosine)+color+shape+elong (power={WEIGHT_POWER}, K={KNN_K})"):
    d, c, s, e = extract_features(Image.open(img_path).convert("RGB"))
    deep_list.append(d); color_list.append(c); shape_list.append(s); elong_list.append(e)

deep_all  = np.asarray(deep_list)
color_all = np.asarray(color_list)
shape_all = np.asarray(shape_list)
elong_all = np.asarray(elong_list)   # circle-vs-elongated elongation-ratio scalar; for the second-layer cut/hole split
print(f"Feature dims  deep={deep_all.shape[1]}  color={color_all.shape[1]}  shape={shape_all.shape[1]}  elong range=[{elong_all.min():.2f},{elong_all.max():.2f}]")

# ==========================================
# 5. First layer: block-balanced concat deep+color -> KernelPCA(cosine) -> UMAP -> HDBSCAN (L2 norm dropped, cosine doesn't need it)
# ==========================================
# block_prep: per-channel StandardScaler on each block then ×(weight/√d). StandardScaler suppresses deep's high-variance
#   magnitude channels (step 1 proved this necessary, otherwise it collapses to 2 clusters); /√d makes each block's L2 contribution ≈ weight, not skewed by dimensionality.
# shape not in the first layer (step 3 proved global shape is harmful), kept only to supply ecc to the second layer.

def block_prep(X, weight):
    Xs = StandardScaler().fit_transform(X)
    return Xs * (weight / np.sqrt(X.shape[1]))


feat = np.concatenate([block_prep(deep_all,  W_DEEP),
                       block_prep(color_all, W_COLOR)], axis=1)
# No longer L2-norm the whole vector: the cosine kernel / cosine-UMAP is insensitive to per-sample length, so L2 here is redundant
# (results are identical with or without it). The real balancing is StandardScaler in block_prep.

# Manifold dimensionality reduction: KernelPCA (cosine kernel) -> project into the cosine kernel space, then feed UMAP.
kpca = KernelPCA(n_components=KPCA_COMPONENTS, kernel="cosine", random_state=SEED)
X_kpca = kpca.fit_transform(feat)
print(f"Dims after KernelPCA(cosine): {X_kpca.shape[1]} / original {feat.shape[1]}  (W_DEEP={W_DEEP}, W_COLOR={W_COLOR})")

X_umap = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0,
                   random_state=SEED, metric="cosine").fit_transform(X_kpca)
labels_L1 = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit_predict(X_umap)

# ==========================================
# 6. Second layer (reworked): color intensity excludes color/metal, then ecc splits cut/hole
# ==========================================
# Gate one: each group's mean residual color intensity ‖color‖ -> exclude color/metal (clear color difference), only cut into low-color-difference geometric blobs.
# Gate two: inside that blob, do a 1D binary split by ecc (eccentricity) —— cut (high ecc) vs hole (low ecc).
#   (The elongation version was tried but didn't split at the second layer, likely because the coarse mask makes most cut/hole elong revert to 1.0 -> no bimodality; so we reverted to ecc.
#    elongation_score / elong_all are kept only to print the elong range for diagnostics.)

def refine_by_shape(labels, shape_all, color_all):
    ecc = shape_all[:, 1]
    col_mag = np.linalg.norm(color_all, axis=1)          # per-image residual color intensity: high = color/metal, low = cut/hole
    new = labels.copy()
    cl_ids = [k for k in sorted(set(labels)) if k != -1]
    cl_colmag = {k: float(col_mag[labels == k].mean()) for k in cl_ids}   # per-group mean color intensity
    med = float(np.median(list(cl_colmag.values())))
    candidates = []
    for k in cl_ids:
        idx = np.where(labels == k)[0]
        if len(idx) < 2 * MIN_SUB:
            continue
        if cl_colmag[k] > med:                            # exclude groups with clear color difference (color/metal)
            continue
        km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(ecc[idx].reshape(-1, 1))
        a, b = idx[km.labels_ == 0], idx[km.labels_ == 1]
        nmin = min(len(a), len(b))
        gap = abs(ecc[a].mean() - ecc[b].mean())
        # ecc gap large enough (cut high / hole low) and balanced (minority subgroup >= 30%)
        if gap >= SPLIT_ECC_GAP and nmin >= max(MIN_SUB, BALANCE_FRAC * len(idx)):
            candidates.append((nmin, gap, k, idx, a, b))
    if not candidates:
        print(f"\nSecond layer: no low-color-difference group satisfies ecc gap >= {SPLIT_ECC_GAP} and balance >= {BALANCE_FRAC} -> no split")
        return new
    nmin, gap, k, idx, a, b = max(candidates, key=lambda t: t[0])
    minor = a if len(a) < len(b) else b
    next_id = int(labels.max()) + 1
    new[minor] = next_id
    tag = "high ecc (cut)" if ecc[minor].mean() > ecc[idx].mean() else "low ecc (hole)"
    print(f"\nSecond layer: split low-color-difference geometric blob cluster{k}(n={len(idx)}, color intensity={cl_colmag[k]:.3f}) -> "
          f"cluster{next_id}(n={len(minor)}, leans {tag}), ecc gap={gap:.3f}, minority fraction={nmin/len(idx):.0%}")
    return new


cluster_labels = refine_by_shape(labels_L1, shape_all, color_all)

# ==========================================
# 7. Clustering Benchmark (first layer vs final)
# ==========================================
gt = np.array([unique_labels.index(l) for l in ground_truth_labels])


def clustering_benchmark(cluster_labels, gt, unique_labels, space_pca, tag=""):
    non_noise = cluster_labels != -1
    labels_c = cluster_labels[non_noise]
    labels_t = gt[non_noise]
    n_true = len(unique_labels)
    cl_ids = sorted(set(labels_c))
    n_pred = len(cl_ids)

    print("\n" + "=" * 56)
    print(f" Clustering Benchmark {tag}")
    print("=" * 56)
    print(f" Samples N / true classes K : {len(cluster_labels)} / {n_true}")
    print(f" Number of clusters found   : {n_pred}")
    print(f" noise / Coverage           : {int((~non_noise).sum())} ({np.mean(~non_noise):.1%}) / {np.mean(non_noise):.1%}")
    if non_noise.sum() < 2 or n_pred < 1:
        print(" Too few valid clustered points, skipping.")
        return None

    ari = adjusted_rand_score(labels_t, labels_c)
    ami = adjusted_mutual_info_score(labels_t, labels_c)
    nmi = normalized_mutual_info_score(labels_t, labels_c)
    homo, comp, vmea = homogeneity_completeness_v_measure(labels_t, labels_c)
    fmi = fowlkes_mallows_score(labels_t, labels_c)

    cm = np.zeros((n_pred, n_true), dtype=int)
    for a, k in enumerate(cl_ids):
        for t in range(n_true):
            cm[a, t] = np.sum((labels_c == k) & (labels_t == t))
    purity = cm.max(axis=1).sum() / len(labels_c)
    row, col = linear_sum_assignment(-cm)
    hung_acc = cm[row, col].sum() / len(labels_c)

    print(f"   ARI={ari:.3f}  AMI={ami:.3f}  NMI={nmi:.3f}  V-measure={vmea:.3f}  FMI={fmi:.3f}")
    print(f"   Homogeneity={homo:.3f}  Completeness={comp:.3f}")
    print(f"   Purity(weighted)={purity:.3f}   Hungarian accuracy={hung_acc:.3f}")
    if n_pred >= 2:
        try:
            sil = silhouette_score(space_pca[non_noise], labels_c)
            db = davies_bouldin_score(space_pca[non_noise], labels_c)
            ch = calinski_harabasz_score(space_pca[non_noise], labels_c)
            print(f"   [Internal/PCA] Silhouette={sil:+.3f}  Davies-Bouldin={db:.3f}  Calinski-Harabasz={ch:.1f}")
        except ValueError as e:
            print(f"   [Internal] skipped ({e})")

    df = pd.DataFrame(cm, index=[f"cluster{k}" for k in cl_ids], columns=unique_labels)
    df["main class"] = df[unique_labels].idxmax(axis=1)
    df["purity"] = (cm.max(axis=1) / cm.sum(axis=1)).round(2)
    print("\n [Contingency  cluster(row) × true defect class(col)]")
    print(df.to_string())
    return df


clustering_benchmark(labels_L1, gt, unique_labels, X_kpca,
                     tag=f"(first layer, cosine scoring = DINOv2 {DINO_MODEL} deep+color · power={WEIGHT_POWER}, K={KNN_K})")
benchmark_df = clustering_benchmark(cluster_labels, gt, unique_labels, X_kpca,
                                    tag="(final = first layer + shape second-layer cut/hole split)")

# ==========================================
# 7. Visualization
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap="Spectral", s=40)
axes[0].set_title(f"Final clustering (DINOv2 blocks {DINO_LAYERS} · deep+color + shape second-layer split)")
for li, lab in enumerate(unique_labels):
    m = gt == li
    axes[1].scatter(X_umap[m, 0], X_umap[m, 1], s=40, label=lab)
axes[1].set_title("Ground-truth defect types")
axes[1].legend(markerscale=1.2, fontsize=8)
plt.tight_layout()
plt.savefig("/content/cluster_vs_gt_dinov2_cosine.png", dpi=120)
plt.show()

if benchmark_df is not None:
    cm = benchmark_df[unique_labels].values
    fig, ax = plt.subplots(figsize=(1.4 * len(unique_labels) + 2, 0.5 * cm.shape[0] + 2))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(unique_labels)), unique_labels, rotation=45, ha="right")
    ax.set_yticks(range(cm.shape[0]), list(benchmark_df.index))
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, cm[r, c], ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black")
    ax.set_title("Contingency: cluster × true defect class (DINOv2 final deep+color + second layer)")
    fig.colorbar(im, ax=ax, label="count")
    plt.tight_layout()
    plt.savefig("/content/benchmark_contingency_dinov2_cosine.png", dpi=120)
    plt.show()

# ==========================================
# 8. Sort images into folders by cluster
# ==========================================
print("\nSorting images into per-cluster folders...")
for i, label in enumerate(cluster_labels):
    folder = f"cluster{label}" if label != -1 else "noise"
    target_dir = os.path.join("/content", "dinov2_cosine_clusters", folder)
    os.makedirs(target_dir, exist_ok=True)
    dst_name = f"{ground_truth_labels[i]}_{os.path.basename(image_paths[i])}"
    shutil.copy(image_paths[i], os.path.join(target_dir, dst_name))

print("\n=== Cluster statistics ===")
for k in np.unique(cluster_labels):
    name = f"cluster{k}" if k != -1 else "noise"
    print(f"{name}: {int(np.sum(cluster_labels == k))} images")
