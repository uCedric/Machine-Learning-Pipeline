# Stage-2 · Unsupervised Defect Clustering · Ablation Study Report

# Unsupervised clustering of carpet defects using DINOv2 self-supervised features

Switching the defect-clustering backbone from ResNet50 to DINOv2, and using item-by-item ablation to find the key designs that let DINOv2 features shine. Hungarian accuracy is pushed all the way from **0.494** to **0.805**.

> **Dataset**: MVTec **carpet** test images · 89 images · 5 defect classes (color / cut / hole / metal contamination / thread)

| Final Hungarian | vs. starting point | ResNet50 reference | Coverage |
| :--- | :--- | :--- | :--- |
| **0.805** | **+0.31** | **0.865** | **97.8%** |
| best one-cluster-to-one-class assignment accuracy | 0.494 → 0.805 | existing solution · target line | 5 clusters · 2 noise images |

---

## 01 — Background and objective

### Problem setup

The overall flow is two-stage: Stage-1 uses PatchCore for anomaly detection, Stage-2 does **unsupervised** defect-type clustering on the "full carpet test images". This report focuses on the Stage-2 backbone-swap experiment.

* The existing **ResNet50 solution** already reaches Hungarian **0.865**, five classes one-cluster-each. This experiment explores whether switching to **DINOv2** (self-supervised ViT, patch tokens as dense features) can reach the same level.
* Methodologically we use **item-by-item ablation**: starting from a plain global-average-pooling baseline, adding back or replacing only "one" design at a time, and quantifying its contribution to clustering quality.
* The main metric is **Hungarian matching accuracy** (aligned with the ResNet solution's evaluation), supplemented by ARI / AMI / NMI and weighted purity.

---

## 02 — Method overview

### Final pipeline

The complete data flow from patch tokens to defect-cluster labels. The core is "cosine nearest-neighbor anomaly scoring" and "KernelPCA manifold reduction" — the two keys that let DINOv2 features win.

1. **DINOv2 dense features**: patch tokens from intermediate layers `blocks[3,5]` of `dinov2_vitb14_reg`, input 448×448 → 32×32 patch grid. Using intermediate layers rather than the last one preserves low-level texture.
2. **Cosine nearest-neighbor anomaly scoring**: build a memory bank from the normal patch features (L2 normalized) of `train/good`; each patch's anomaly score = $1 - \text{nearest-neighbor cosine similarity to the memory bank}$. Replaces the original Euclidean z-score.
3. **deep + color descriptors**:
   * **deep**: weighted pooling by $\text{anomaly score}^3$ then L2 normalized (stays in cosine space, no residual).
   * **color**: residual color, 5 dims, taken from the compact single-peak anomaly mask.
4. **Block-balanced concatenation**: `block_prep`: per-channel StandardScaler on each block then $\times (\text{weight} / \sqrt{\text{dim}})$, so deep(1.0) and color(0.3) contribute L2 without dimensional imbalance.
5. **Manifold reduction → clustering**: `KernelPCA(kernel='cosine', n=20)` → `UMAP(2D, cosine)` → `HDBSCAN(min_cluster_size=3)`.
6. **Second layer: split cut / hole**: use "per-group mean residual color intensity" to exclude color / metal (clear color difference), and only within low-color-difference geometric groups, use `ecc` (eccentricity) for a 1D binary split to separate elongated cut from circular hole.

---

## 03 — Ablation history

### From 0.494 to 0.805

Two turning points decide the whole curve: ① "Euclidean scoring → cosine scoring", ② "linear PCA → KernelPCA". The rest is tuning the second-layer splitter and the reduction dimensionality into place.

#### Hungarian-accuracy evolution across milestones

* **Start (z-score)** `0.494` ── best under Euclidean z-score scoring: deep (weighted pooling + residual, power=3) + residual color, first layer
* **Cosine scoring (+0.20)** `0.697` ── switch to cosine nearest-neighbor memory-bank scoring, first-layer deep+color (key breakthrough)
* **+ ecc second layer (split cut/hole)** `0.753` ── add the second-layer ecc splitter (color-gated), recover the hole that was merged in
* **+ KernelPCA (cosine kernel)** `0.793` ── switch reduction from linear PCA to KernelPCA (cosine kernel)
* **+ dim=20 (finalized)** `0.805` ── tune KernelPCA dim from 30 to 20 (sweet spot) —— final decision
* **ResNet50 (reference)** `0.865` ── existing ResNet50 solution, same reduction and clustering, Euclidean scoring + a finer 56×56 anomaly map

| Stage | design added / changed | Hungarian | Status |
| :--- | :--- | :--- | :--- |
| Start | deep(weighted pooling + residual, power=3)+ color · Euclidean z-score | 0.494 | baseline |
| **Turning point ①** | **switch to cosine nearest-neighbor memory-bank scoring (deep+color)** | **0.697** | **+0.20** |
| — | + second-layer **ecc** splitter (color-intensity gated) | 0.753 | +0.056 |
| **Turning point ②** | **+ reduction switched to KernelPCA(cosine)** | **0.793** | **+0.040** |
| **Finalized** | **+ KPCA dim tuned to 20 (sweet spot)** | **0.805** | **best** |
| Reference | existing ResNet50 solution (same downstream) | 0.865 | target |

> 💡 **Why turning point ① is the key:** DINOv2 is a LayerNorm feature, semantics are encoded in "direction" rather than "per-channel magnitude". Euclidean z-score suits ResNet (BN channels), but buries DINOv2's feature strength; switching to cosine nearest-neighbor gains +0.20 on the same data in one shot.
>
> 📌 **About resolution:** Under Euclidean scoring both 448 and 784 stall at 0.494, which was once misjudged as "the feature limit" rather than a scoring-method problem —— this wrong conclusion was later overturned by cosine scoring.

#### Dead ends walked (recorded to avoid retrying)

| Attempt | Result | Why it failed |
| :--- | :--- | :--- |
| shape descriptor into the **first layer** | 0.494 → 0.404 | global shape crushes the structure; the coarse mask's shape is too noisy and mostly reverts to defaults |
| resolution 448 → **784** | flat / worse | resolution is not the bottleneck; the finer map is noisier and thread/cut more fragmented |
| add deeper layer **[3,5,8]** | 0.708 | thread stuck with color is intrinsic to the embedding, adding a semantic layer doesn't help |
| **elongation ratio** (heatmap second moment) to split cut/hole | = ecc | the coarse mask makes most elongation revert to 1.0, no bimodality to split |
| multi-dim shape **SpectralClustering** | = ecc | the extra shape features carry no new information on the coarse mask |

---

## 04 — Key findings

### Transferable lessons

These conclusions are not limited to this dataset; they are general design principles when "swapping the backbone for unsupervised clustering".

* **FINDING 01｜The scoring method must match the feature geometry**  
  **The single largest contribution.** The anomaly-scoring method must match the backbone feature's geometry: Euclidean ↔ ResNet(BN), cosine ↔ DINOv2(LayerNorm). Getting the scoring right: 0.494 → 0.697.
* **FINDING 02｜Feature standardization cannot be skipped**  
  Weighted residuals produce a few high-variance "anomaly-magnitude" channels; without per-channel StandardScaler, the PCA principal axes get dominated by the magnitude axis and **clustering collapses to 2 groups**.
* **FINDING 03｜Shape is the second layer's targeted knife**  
  The shape descriptor **should not go into the first layer** (adding it globally is harmful). It should serve as a targeted second-layer knife, gated by "residual color intensity", cutting only the geometric cut/hole.
* **FINDING 04｜KernelPCA + dimensionality sweet spot**  
  Switching reduction to KernelPCA(cosine) gives a small gain; the dimensionality has a clear sweet spot: **20 is best**, too many (50) merges thread into metal and the pure group disappears.
* **FINDING 05｜The bottleneck is the input, not the algorithm**  
  The residual cut/hole mixing gives **identical results** with ecc, elongation ratio, or SpectralClustering —— the limit comes from the coarse mask's shape input, not the clustering method.
* **FINDING 06｜Resolution is no panacea**  
  Each ViT block has the same spatial resolution; the only knob is the input size. But refining the mask didn't solve type separation and instead added noise —— the problem is feature/localization quality.

---

## 05 — Final configuration and results

### Finalized configuration

| Hungarian | Purity | ARI | AMI | NMI | # clusters |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.805** | **0.805** | **0.617** | **0.699** | **0.719** | **5** |

#### Hyperparameters and module settings

| Module | Setting |
| :--- | :--- |
| **Backbone** | `dinov2_vitb14_reg` · blocks `[3, 5]` · input `448` (32×32 patch) |
| **Anomaly scoring** | cosine nearest-neighbor memory bank · `K=1` · `BANK = 20000` · source `train/good` |
| **deep feature** | weighted pooling by $\text{anomaly score}^3$ → L2 normalized (no residual) · `WEIGHT_POWER=3` |
| **color feature** | residual RGB(3) + brightness residual(1) + saturation(1), from the compact single-peak mask |
| **Block balancing** | `block_prep` · W-deep `1.0` / W-color `0.3` |
| **Reduction** | `KernelPCA(kernel='cosine', n_components=20)` → `UMAP(15, 2D, cosine)` |
| **Clustering** | `HDBSCAN(min_cluster_size=3)` |
| **Second layer** | color-intensity gate + `ecc` 1D-KMeans(2) split of cut/hole |

#### Contingency: clusters found (rows) × true defect classes (cols)

| Cluster | color | cut | hole | metal | thread | main class | purity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **cluster 0** | 0 | **14** | 6 | 0 | 0 | cut | 0.70 |
| **cluster 1** | **19** | 0 | 1 | 0 | 6 | color | 0.73 |
| **cluster 2** | 0 | 0 | 0 | **17** | 0 | metal | **1.00** |
| **cluster 3** | 0 | 0 | 0 | 0 | **10** | thread | **1.00** |
| **cluster 4** | 0 | 3 | **10** | 0 | 1 | hole | 0.71 |

> Note: bold numbers are each group's main diagonal hits. metal and thread are fully pure groups; each of the 5 true classes has one corresponding main group, and another 2 images are judged as noise by HDBSCAN (not counted).

---

## 06 — Limitations and future work

### Residual losses and next directions

Still about 0.060 short of ResNet's 0.865; diagnosis points both residual losses to the **same root cause: DINOv2's localization/features for borderline cases are not fine enough**.

* ❌ **6 thread images stuck in the color group** ── the same across layers and resolutions, deeper layers don't help; it's an intrinsic embedding-level limitation.
* ❌ **9 cut / hole images mixed together** ── those high-eccentricity holes and short cuts are inherently ambiguous in shape under a 32×32 mask; all three split methods giving the same result already proves this is an input limitation.

> 🔍 **The only path that might truly break through:** re-extract the mask at higher resolution for the defect region purely for "computing shape", or do classical segmentation on RGB, to obtain sharp cut/hole outlines —— but that is new engineering with no guaranteed payoff.
>
> ⚠️ **Not yet tried (lower expected value):** RBF-kernel KernelPCA, KNN K=3, memory-bank coreset. **If the goal is to deliver the highest score,** the existing ResNet50 solution's 0.865 is still a ready and better choice.

---

*Stage-2 DINOv2 unsupervised defect clustering ablation study · MVTec carpet · 89 images / 5 classes · final Hungarian 0.805 · report date 2026-07-21*
