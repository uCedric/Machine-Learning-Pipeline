"""
Stage-2 Clustering (hierarchical v2: use v2(0.75) as the first layer, the second layer only adds a single cut to split cut/hole)
================================================================================
The mistake in the previous hierarchical version: the first layer used only deep+color, and color/cut/hole/thread all blurred into one blob
—— because the shape block in v2 not only separates cut/hole but also helps color stand out; removing it is like tearing down a load-bearing wall.

Fix:
  First layer = full v2 configuration (deep + color + shape, weights 1.0/0.3/0.3)
           -> reproduces v2's 4 clean groups: color / metal / thread + (cut+hole merged).
  Second layer = only add one cut to the 'ecc bimodal' group (= the cut+hole mixed group), splitting it along shape with KMeans(2).
           Two safeguards (ecc gap large enough + group's ecc dispersed enough) ensure metal/thread/color are not mistakenly split.

★Folder assumption (MVTec standard structure):
   IMAGE_DIR  = .../carpet/test/          test/<defecttype>/*.png  (label = parent folder name)
   NORMAL_DIR = .../carpet/train/good/    normal images, used to build the prototype
"""

import os
import glob
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
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
IMAGE_DIR   = "/content/carpet/test/"
NORMAL_DIR  = "/content/carpet/train/good/"
INCLUDE_GOOD = False
INPUT_SIZE   = 448
FEATURE_MODE = "both"
MAX_NORMAL   = 200
WEIGHT_POWER = 3.0

# --- descriptor: compact localization ---
SMOOTH_SIGMA = 4.0
MASK_K       = 3.0
MASK_PCTL    = 97.5
BG_PCTL      = 50

# --- first layer: deep + color + shape (= v2 configuration, produces clean 4 groups with cut+hole merged) ---
W_DEEP  = 1.0
W_COLOR = 0.3
W_SHAPE = 0.3
MIN_CLUSTER_SIZE = 3

# --- second layer: only split the 'most balanced two-way split' group (cut+hole); use balance to tell 'true bimodal vs has outliers' apart ---
SPLIT_ECC_GAP = 0.25         # the two subgroups' mean ecc gap must be >= this (cut vs hole ~0.3+)
BALANCE_FRAC  = 0.30         # the minority subgroup must be >= this fraction of the group (cut+hole ~0.47; color flinging outliers ~0.18 is blocked)
MIN_SUB       = 4

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{device.upper()}] Loading ResNet50 (Layer2+3) feature extractor...")

# ==========================================
# 1. Feature extractor
# ==========================================
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
feature_extractor = create_feature_extractor(base_model, {"layer2": "layer2", "layer3": "layer3"})
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
LAYERS = ["layer2", "layer3"]


def extract_maps(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = feature_extractor(img_t)
    return {L: out[L][0] for L in LAYERS}


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
# 3. Normal texture prototype
# ==========================================

def build_prototype(sources, max_n):
    stats = {L: {"sum": 0.0, "sqsum": 0.0, "n": 0} for L in LAYERS}
    for p in tqdm(sources[:max_n], desc="Estimating normal texture prototype"):
        maps = extract_maps(Image.open(p).convert("RGB"))

        for L in LAYERS:
            v = maps[L].reshape(maps[L].shape[0], -1).T
            stats[L]["sum"]   += v.sum(0)
            stats[L]["sqsum"] += (v * v).sum(0)
            stats[L]["n"]     += v.shape[0]

    proto = {}
    for L in LAYERS:
        cnt = stats[L]["n"]
        mu = stats[L]["sum"] / cnt
        var = stats[L]["sqsum"] / cnt - mu * mu
        proto[L] = (mu, torch.sqrt(torch.clamp(var, min=1e-6)))
    return proto


normal_paths = []
if NORMAL_DIR and os.path.isdir(NORMAL_DIR):
    normal_paths = sorted(p for p in glob.glob(os.path.join(NORMAL_DIR, "**", "*"), recursive=True)
                          if p.lower().endswith((".png", ".jpg", ".jpeg")))
if normal_paths:
    print(f"Using {min(len(normal_paths), MAX_NORMAL)} / {len(normal_paths)} normal images to build the prototype")
    prototype = build_prototype(normal_paths, MAX_NORMAL)
else:
    print("⚠ NORMAL_DIR not found -> falling back to using 'the mean of all defect images' as the prototype")
    prototype = build_prototype(image_paths, len(image_paths))

# ==========================================
# 4. Feature extraction: deep features + compact anomaly map + residual color/shape (v2 descriptors)
# ==========================================

def defect_mask(amap):
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
    """Residual color, 5 dims: residual RGB(3) + brightness residual(1) + defect saturation(1)."""
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
    """v2 shape, 5 dims: area / ecc / solidity / extent / aspect  (ecc at index 1)."""
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


def extract_features(img):
    rgb = np.asarray(img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR).convert("RGB"),
                     dtype=np.float32) / 255.0
    maps = extract_maps(img)
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
        sm = F.interpolate(sm[None, None], size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)[0, 0]
        score_maps.append(sm)
    deep = torch.cat(deep_vecs).cpu().numpy()
    amap = torch.stack(score_maps).mean(0).cpu().numpy()
    mask, amap_s = defect_mask(amap)
    return deep, color_descriptor(rgb, amap_s, mask), shape_descriptor(mask)


deep_list, color_list, shape_list = [], [], []
for img_path in tqdm(image_paths, desc=f"Extracting features + descriptors (mode={FEATURE_MODE})"):
    d, c, s = extract_features(Image.open(img_path).convert("RGB"))
    deep_list.append(d); color_list.append(c); shape_list.append(s)

deep_all  = np.asarray(deep_list)
color_all = np.asarray(color_list)
shape_all = np.asarray(shape_list)     # 5 dims; ecc = [:,1]
print(f"Feature dims  deep={deep_all.shape[1]}  color={color_all.shape[1]}  shape={shape_all.shape[1]}")

# ==========================================
# 5. First layer: deep + color + shape (= v2), produces clean 4 groups
# ==========================================

def block_prep(X, weight):
    Xs = StandardScaler().fit_transform(X)
    return Xs * (weight / np.sqrt(X.shape[1]))

feat_L1 = np.concatenate([block_prep(deep_all,  W_DEEP),
                          block_prep(color_all, W_COLOR),
                          block_prep(shape_all, W_SHAPE)], axis=1)
feat_L1 = normalize(feat_L1, norm="l2")

pca = PCA(n_components=0.95, random_state=SEED)
pca_L1 = pca.fit_transform(feat_L1)
print(f"First-layer dims after PCA: {pca_L1.shape[1]}")

umap_L1 = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0,
                    random_state=SEED, metric="cosine").fit_transform(pca_L1)
labels_L1 = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit_predict(umap_L1)

# ==========================================
# 6. Second layer: only split the 'ecc bimodal' group (cut+hole)
# ==========================================

def refine_by_shape(labels, shape_all):
    """
    Only split the 'most balanced two-way split' group (= cut+hole mixed group), leave everything else untouched.
    A true bimodal (cut+hole) gets split by KMeans into two large halves; a group with only outliers (color) merely
    flings off a few points -> extremely unbalanced, blocked by BALANCE_FRAC. The split uses only ecc (1D), cut vs hole is cleanest.
    """
    ecc = shape_all[:, 1]                      # ecc of v2 shape is at index 1
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
        # requires both: the two subgroups' ecc gap large enough, and the split balanced enough (not just flinging a few outliers)
        if gap >= SPLIT_ECC_GAP and nmin >= max(MIN_SUB, BALANCE_FRAC * len(idx)):
            candidates.append((nmin, gap, k, idx, a, b))

    if not candidates:
        print(f"\nSecond layer: no group satisfies both ecc gap >= {SPLIT_ECC_GAP} and balance >= {BALANCE_FRAC} -> no split")
        return new

    # among the qualifying candidates, pick the 'largest balanced split' -> the truly two-way cut+hole
    nmin, gap, k, idx, a, b = max(candidates, key=lambda t: t[0])
    minor = a if len(a) < len(b) else b
    next_id = int(labels.max()) + 1
    new[minor] = next_id
    print(f"\nSecond layer: split the most balanced bimodal cluster{k}(n={len(idx)}) -> "
          f"cluster{next_id}(n={len(minor)}), ecc gap={gap:.3f}, minority fraction={nmin/len(idx):.0%}")
    return new


labels_final = refine_by_shape(labels_L1, shape_all)

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


clustering_benchmark(labels_L1, gt, unique_labels, pca_L1, tag="(first layer = v2 deep+color+shape)")
benchmark_df = clustering_benchmark(labels_final, gt, unique_labels, pca_L1, tag="(final + shape second-layer cut/hole split)")

# ==========================================
# 8. Visualization
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(umap_L1[:, 0], umap_L1[:, 1], c=labels_final, cmap="Spectral", s=40)
axes[0].set_title("Final clustering")
for li, lab in enumerate(unique_labels):
    m = gt == li
    axes[1].scatter(umap_L1[m, 0], umap_L1[m, 1], s=40, label=lab)
axes[1].set_title("ground truth of defection")
axes[1].legend(markerscale=1.2, fontsize=8)
plt.tight_layout()
plt.savefig("/content/cluster_vs_gt.png", dpi=120)
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
    ax.set_title("Contingency: cluster × gournd truth of defection")
    fig.colorbar(im, ax=ax, label="count")
    plt.tight_layout()
    plt.savefig("/content/benchmark_contingency.png", dpi=120)
    plt.show()

# ==========================================
# 9. Sort images into folders by cluster
# ==========================================
print("\nSorting images into per-cluster folders...")
for i, label in enumerate(labels_final):
    folder = f"cluster{label}" if label != -1 else "noise"
    target_dir = os.path.join("/content", folder)
    os.makedirs(target_dir, exist_ok=True)
    dst_name = f"{ground_truth_labels[i]}_{os.path.basename(image_paths[i])}"
    shutil.copy(image_paths[i], os.path.join(target_dir, dst_name))

print("\n=== Cluster statistics ===")
for k in np.unique(labels_final):
    name = f"cluster{k}" if k != -1 else "noise"
    print(f"{name}: {int(np.sum(labels_final == k))} images")
