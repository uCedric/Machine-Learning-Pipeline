# Stage-two 一次 fit 的凍結產物

查閱用速查表。敘述版見 [stage-two 訓練交接講稿](stage-two-retrain-handover.md) 第 3 節。

產物分三個地方:一個物件儲存的 artifact、兩列 Postgres。

---

## ① MinIO:`models/resnet50/{version}/cluster_state.joblib`

由 `resnet50/model.py` 的 `fit()` 組出 `live_state`,`resources.dump_state()` 序列化。七個 key:

| key | 內容 | 為什麼非凍不可 |
|---|---|---|
| `prototype` | layer2 / layer3 每通道的 `(mu, std)` | 深層特徵 z-score 的基準。改用今天的良品重建 → 整個特徵空間平移,**影響最大** |
| `scalers` | `deep` / `color` / `shape` 三個 `StandardScaler` | 決定三塊描述子的相對權重。原本 `fit_transform` 後即丟棄 |
| `pca` | `PCA(n_components=0.95)` | 投影軸與**保留的維度數**都會變 |
| `umap` | fit 過的 `UMAP` | `transform()` 要靠它把新點投進同一個 2D 空間 |
| `hdbscan` | 帶 `prediction_data=True` 的 clusterer | 少了這個 flag,`approximate_predict()` 直接拋錯 |
| `refine` | `{source_cluster, new_cluster, kmeans, minor_side}` | 第二層是**搜尋而非模型**,所以存的是決策:切了哪一群、用哪顆 1-D KMeans、哪一側成為新 id |
| `meta` | `seed` / `image_size` / `weights` / `feature_mode` / `min_cluster_size` | 供辨識不相容的 state,不參與運算 |

`prototype` 序列化時從 torch tensor 轉成 numpy(`dump_state`),載入時轉回(`load_state`)。
理由:這個 pickle 已經綁在 scikit-learn / umap-learn / hdbscan 三個版本上,不必再多綁一個 torch 版本。

> **artifact 先寫,registry 後寫。** 反過來會留下指向不存在 prefix 的 registry 列,之後每次載入都失敗。

---

## ② Postgres `inference_model`:一列新版本

`next_minor()` 產生(`1.0.0 → 1.1.0`),`is_valid = true` **直接生效**,不像 stage-one 重訓要等人批准。

用 minor 而非 patch,是因為一次 fit 會**重新編號所有 cluster** —— 對讀 `cluster_results` 的人來說是行為改變,不是無聲修補。

---

## ③ Postgres `stage_two_run`:一列 fit 紀錄

`mode = 'fit'`、`covered_through`(watermark)、`labelled_count` / `noise_count` / `mean_probability`。

後三個是**這次 fit 的品質基準**:之後每次 assign 的 noise 比例都會拿來與它比較,是儀錶板 drift 判定的依據
(門檻 = `max(STAGE_TWO_DRIFT_NOISE_RATIO, 本次 fit 的 noise 比例 × 2)`)。

---

## 不在凍結產物裡的東西

| 沒有凍結 | 原因 |
|---|---|
| **ResNet50 backbone** | torchvision 的 ImageNet 預訓練權重,build 時就烤進 image,每個版本都一樣。這點與 stage-one 不同 —— PatchCore 的版本目錄會複製一份 `resnet_backbone.onnx` 讓目錄自足 |
| **`cluster_results` 的那些列** | 那是這次 fit 的**輸出**(每張圖一列),不是產物。凍結的是「怎麼分」,不是「分完的結果」 |
| **良品影像本身** | 只留下它們算出來的 `prototype`,影像不複製 |

---

## 兩件必須一起記住的事

**一、pickle 綁函式庫版本。** artifact 綁在 `scikit-learn==1.5.2` / `umap-learn==0.5.7` / `hdbscan==0.8.40`。
升級任何一個都會讓既有的 fit 失效,必須重 fit。

**二、fit 會立刻自我採用。** `fit()` 結尾有 `self.state = live_state`,所以不必重啟服務就能開始 assign。
少了這行,行程內的 singleton 會一直是 `state=None`,永遠走不到 assign。重啟時則由 builder
從 MinIO 載回(`_load_frozen_fit`),兩條路殊途同歸。

---

## 對照:程式碼位置

| 動作 | 位置 |
|---|---|
| 組出 `live_state` | `adapters/outbound/resnet50/model.py` 的 `fit()` |
| 序列化 / 反序列化 / 上傳 | `adapters/outbound/resnet50/resources.py` 的 `dump_state` / `load_state` / `save_state` |
| 啟動時載回 | `adapters/outbound/resnet50/builder.py` 的 `_load_frozen_fit` |
| 發布版本與紀錄 run | `application/use_cases/run_stage_two.py` 的 `_do_fit()` |
