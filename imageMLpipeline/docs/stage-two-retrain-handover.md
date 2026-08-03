# Stage-two 訓練交接講稿(fit:重新畫出分群)

> 用途:把 stage-two 的**訓練側**(全集重新分群並凍結成新版本)交接給接手的人。
> 建議時間:20–25 分鐘 + 問答。
> 系列:[stage-one 推論](stage-one-handover.md) · [stage-one 重訓](stage-one-retrain-handover.md) ·
> [stage-two 推論](stage-two-handover.md) · **本篇**

---

## 0. 開場(2 分鐘)

「這份講 stage-two 的 fit —— 也就是「訓練」。但要先破除一個預期:

**這裡沒有梯度下降,也沒有可學習的權重。** backbone 是 torchvision 的 ResNet50 ImageNet 預訓練權重,從頭到尾不動。所謂「訓練」是:

> 對一整批缺陷影像跑一次無監督分群管線,把管線中**每一個被 fit 出來的統計物件**存起來,之後的新影像就能投影進同一個空間。

所以 stage-two 的 fit 跟 stage-one 的重訓在性質上完全不同:

| | stage-one 重訓 | stage-two fit |
|---|---|---|
| 學到什麼 | 記憶庫多幾個向量 + 新的閾值 | 一整套 fit 過的變換(prototype / scaler / PCA / UMAP / HDBSCAN) |
| 觸發者 | **人**在 dashboard 挑圖 | **系統**累積到 1000 張自動觸發 |
| 產出的版本 | `is_valid=false` 候選,等人批准 | `is_valid=true`,**自我上線** |
| 為什麼可以自我上線 | — | 它不是新 backbone,只是同一個已驗證 backbone 在新資料上重新 fit |

最後一列是這份文件最需要你理解的設計決定,第 6 節會展開。」

---

## 1. 什麼時候會 fit(2 分鐘)

```
type=stage-two 事件 → RunStageTwo.execute()
      │
      ▼
自上次 FIT 以來累積 ≥ STAGE_TWO_REFIT_COUNT(1000)張 anomaly?
      ├─ 是 ──────────────────────────────► FIT
      └─ 否 ──► 沒有凍結狀態? ─ 是 ──────► 只 log,等累積到 1000
                     │否
                     └─► 走 assign(另一份文件)
```

「兩種進入 fit 的情況:

1. **bootstrap** —— 系統剛上線,沒有任何凍結狀態,assign 不可能執行。所以第一次動作一定是 fit。
2. **週期性重畫** —— 自上次 fit 以來又累積了 1000 張新缺陷。

**fit 優先於 assign**:兩道閘門同時開時走 fit,因為它反正會重新標記全部。

程式碼是 `run_stage_two.py` 的 `execute()`,約 20 行。」

---

## 2. 走一次 `fit()`(6 分鐘,**主體**)

`adapters/outbound/resnet50/model.py` 的 `fit()`。這是原本 `stage-two_experiment/ResNet50_cluster.py` 的忠實移植,四個階段。

### 階段 1:normal prototype

```python
prototype = self._build_prototype(normals)   # 最近 200 張 status='normal'
```

「對每張已知良品跑 ResNet50,取 Layer2(512 通道)+ Layer3(1024 通道)的特徵圖,算**每個通道的 mean / std**。

這是「正常紋理長什麼樣」的基準。後面每張缺陷影像的深層特徵都要對它做 z-score,異常的地方才會凸顯出來。

**沒有良品時**會退而用缺陷集自己算 prototype,並記一行 WARNING。這是實驗程式碼原本的行為,保留下來當 bootstrap 的退路 —— 但要知道那樣算出來的分群品質會比較差。」

### 階段 2:每張缺陷影像的三組描述子

```python
deep_all, color_all, shape_all = self._descriptors(defects, prototype)
```

「每張圖產出三個東西(`_extract_features`):

- **deep** —— 對 prototype 做 z-score 後得到異常分數圖,用它當權重做**加權池化**(`WEIGHT_POWER=3.0`,讓池化集中在最異常的 patch),再減掉 mu。這是主要的語意特徵。
- **colour(5 維)** —— 從異常區域的遮罩內取殘差顏色統計
- **shape(5 維)** —— 從遮罩的形狀取,其中 `shape_all[:, 1]` 是**離心率**,第二層會專門用它

【提示】遮罩是 `descriptors.defect_mask(amap)` 產生的單峰緊緻遮罩 —— 只圈出最主要的那一塊缺陷。」

### 階段 3:第一層分群

```python
deep_x, deep_scaler = _fit_block(deep_all, W_DEEP)      # 1.0
color_x, color_scaler = _fit_block(color_all, W_COLOR)  # 0.3
shape_x, shape_scaler = _fit_block(shape_all, W_SHAPE)  # 0.3
feat = normalize(concat([deep_x, color_x, shape_x]), norm="l2")
pca = PCA(n_components=0.95, random_state=42).fit(feat)
reducer = umap.UMAP(n_neighbors=min(15, n-1), n_components=2,
                    min_dist=0.0, random_state=42, metric="cosine").fit(pca.transform(feat))
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, prediction_data=True)
labels_l1 = clusterer.fit_predict(reducer.embedding_)
```

「**block balancing 要解釋一下。** 三組描述子維度差很多(deep 是 1536 維,colour/shape 各 5 維)。如果直接接起來,deep 會完全主導。所以 `_fit_block` 先 StandardScale,再乘上 `weight / sqrt(維度)` —— 讓每個 block 對 L2 距離的貢獻大約等於它的權重。權重是實驗調出來的 1.0 / 0.3 / 0.3。

**`prediction_data=True` 一定要有。** 少了它,`hdbscan.approximate_predict()` 會直接拋錯,assign 就完全做不了。這是我在拆 fit/assign 時加上的。」

### 階段 4:第二層(切開 cut+hole)

```python
labels_final, refine = self._refine_by_shape(labels_l1, shape_all)
```

「第一層通常會把「切痕」和「破洞」混成同一個 blob(它們的深層特徵很像)。第二層專門處理這件事:掃過每個 cluster,用 1-D KMeans 對**離心率**切成兩半,只有同時滿足兩個條件才真的切:

- 兩群的平均離心率差距 ≥ `SPLIT_ECC_GAP`(0.25)
- 少數群至少佔 `BALANCE_FRAC`(0.30)且不少於 `MIN_SUB`(4)張

滿足條件的候選裡挑「少數群最大」的那一個切開,少數群拿一個新的 cluster id。

**這一步是搜尋而不是模型**,所以凍結時存的是決策:被切的是哪個 cluster、那顆 KMeans、哪一側變成新 id。assign 時 `_apply_refine()` 用同一顆 KMeans 判斷新影像歸哪邊。這是整個凍結過程中最不直觀的部分。」

---

## 3. 凍結:七個必須存的東西(4 分鐘,**這節最重要**)

> 速查表版本(含存放位置與『沒有凍結什麼』):[stage-two-frozen-artifacts.md](stage-two-frozen-artifacts.md)

「fit 的產出**不是分群結果,而是 fit 出來的狀態**。這是我要交接的核心概念。

原本的程式碼每一步都是 `fit_transform` 然後把估計器丟掉。要讓後續的 assign 可能,每一個都得留下來:」

| # | 凍結物 | 不凍結會怎樣 |
|---|---|---|
| 1 | `prototype`(Layer2/3 每通道 mu/std) | z-score 基準改變 → 整個 deep 空間平移 → **最致命** |
| 2 | 三個 `StandardScaler` | 三個 block 的相對權重跑掉 |
| 3 | `PCA(0.95)` | 投影軸與**維度數**都變 |
| 4 | `UMAP` | 2-D 嵌入空間完全不同 |
| 5 | `HDBSCAN`(帶 `prediction_data`) | 無法對新點 `approximate_predict` |
| 6 | 第二層切分決策 | 新影像無法被送到同一側 |
| 7 | `meta`(seed / image_size / 權重 / feature_mode) | 無法辨識不相容的 state |

「全部 joblib 成單一 artifact:

```
models/resnet50/{version}/cluster_state.joblib
```

**`prototype` 特別處理**:它原本是 torch tensor,存的時候轉成 numpy、載入時轉回來(`dump_state` / `load_state`)。理由是這個 pickle 已經綁在 sklearn / umap / hdbscan 三個版本上了,再多綁一個 torch 版本沒必要。」

### 寫入順序很重要

```python
state_version = self._registry.next_minor(self._model_key)   # 1.0.0 → 1.1.0
self._state_put(state_version, fit.state)                    # ① 先上傳 artifact
self._registry.register(state_version)                       # ② 後寫 registry
```

「**artifact 先、registry 後。** 反過來的話,若中間掛掉,registry 會有一列指向不存在的 prefix,之後每次載入都會失敗。」

### 為什麼是 minor 而不是 patch

「stage-one 重訓用 patch+1(`1.0.0 → 1.0.1`),stage-two fit 用 **minor+1**(`1.0.0 → 1.1.0`)。理由:**一次 fit 會重新編號所有 cluster**,對任何讀 `cluster_results` 的人來說這是行為改變,不是無聲的修補。版本號應該反映這件事。

另外 `next_minor` 是對「所有同 model_type 的列」取最大版本再 +1,**不論 valid 與否** —— 避免跟已註冊的候選(例如 DINOv2 的 1.0.0)撞號。」

---

## 4. fit 會立刻自我採用(2 分鐘)

「這一段是我修掉的一個會讓整個功能失效的缺陷,值得單獨講,因為看程式碼會覺得那行是多餘的。

`ResNet50ClusterModel` 是 **singleton**,在服務啟動時建立一次。如果 fit 只把 state 寫到 MinIO,行程內那個實例的 `state` 仍然是 `None`:

```
_can_assign → False → 每次都只評估 fit 閘門 → 每 1000 張重 fit 一次,永遠不會 assign
```

整個 assign 功能等於沒做。所以 `fit()` 結尾有:

```python
self.state = live_state    # 立刻採用自己的成果,不繞道 MinIO
```

以及使用端把 model_id 切到新版本:

```python
self._cluster_model.model_id = state_version.model_id
self._model_id = state_version.model_id
```

「後者是必要的,因為 cluster 現在是用新版本的編號,之後的 assign 必須歸屬到新版本。

重啟時走另一條路:builder 從 MinIO 載入最新有效版本的 state(`_load_frozen_fit`)。兩條路最後到同一個狀態。」

---

## 5. 訓練資料從哪來(2 分鐘)

「兩個集合,都是從 `inference_results` 即時查出來的,沒有固定的資料集資產:

| 集合 | 查詢 | 上限 |
|---|---|---|
| 缺陷集(要分群的) | `status = 'anomaly'`,`DISTINCT ON (object_key)` 取最新判定,按 `inferred_at` 倒序 | `STAGE_TWO_DEFECT_SET_SIZE` = 5000 |
| 良品基準 | `status = 'normal'` 且 model_type 相符,倒序 | `STAGE_TWO_NORMAL_SET_SIZE` = 200 |

三個要注意的地方:

**① `pending` 完全不在裡面。** buffer zone 的中間帶走人工審閱 → stage-one 重訓,不進分群。理由:把「不確定是不是缺陷」的影像餵進一個要**命名缺陷型態**的無監督分群,會模糊掉型態邊界。

**② `defect_set_size` 必須 ≥ `refit_count`。** 5000 ≥ 1000。調小到低於門檻的話,一次 fit 會看不完觸發它的那批影像。

**③ 這是一個滑動視窗。** 取「最新的 5000 張」,所以久遠的缺陷會自然被排除。好處是跟得上產線變化,壞處是罕見缺陷型態可能因為樣本掉出視窗而失去自己的 cluster。」

---

## 6. 為什麼 fit 可以自我上線(2 分鐘)

「這是跟 stage-one 重訓最大的差異,一定會被問到。

stage-one 重訓產出 `is_valid=false`,要人批准。stage-two fit 產出 `is_valid=true`,直接生效。看起來不一致,理由是:

**stage-one 的候選版本改變了『判定』。** 記憶庫加了向量、buffer zone 換了界線 —— 一張圖從 normal 變 anomaly 是有後果的決定,值得人看一眼。

**stage-two 的 fit 沒有改變任何判定,只改變了『分組方式』。** backbone 沒換(而且它本來就是已驗證的 `resnet50`),沒有新的閾值,也不會讓任何影像從「缺陷」變成「非缺陷」。它只是把同一批缺陷重新分成幾群。

而且更實務的理由:**如果 fit 需要人批准,系統就會卡死。** bootstrap 時沒有凍結狀態,assign 不能跑;如果那個 fit 又要等人批准,stage-two 就永遠不會開始工作。

**代價要誠實講:沒有任何品質把關。** 一次很糟的 fit 會直接上線,而且會把後續 assign 全部帶到爛空間裡。目前唯一的防線是儀錶板的 drift 判定 —— 它會**事後**告訴你有問題,不會事前阻止。這是已知的取捨(第 8 節①)。」

---

## 7. 怎麼確認正常工作(2 分鐘)

一次成功的 fit:

```
Stage-two fit run starting: 1043 new defect image(s)
Defect set: 4820 anomaly image(s) (limit 5000)
Normal baseline: 200 known-good image(s) for 'patchcore' (limit 200)
Fitting 4820 defect image(s) against 200 known-good image(s)
Building normal prototype from 200 known-good image(s)
Second layer: split cluster 2 (n=310) -> cluster 5 (n=104), ecc gap=0.412, minority=34%
Fitted 4820 defect image(s) into 6 cluster(s) (587 noise)
Saved frozen fit to models/resnet50/1.1.0/cluster_state.joblib (18432000 bytes)
Registered 'resnet50' v1.1.0 as valid (model_id=...)
Froze the fit as 'resnet50' v1.1.0 (18432000 bytes); later batches assign to it
Saved 4820 cluster result(s) for model_id=...
Recorded stage-two fit run <uuid>: completed, 1043 defect image(s), watermark=...
```

「幾個判讀重點:

- **`Fitted N into K cluster(s) (M noise)`** —— K 是這次分出幾群,M/N 就是這個 fit 的**基準 noise 比例**。之後所有 assign 都會跟它比。K 如果是 0 或 1,分群基本上失敗了。
- **`Second layer: ... no split`** 也是正常的,代表這批沒有需要切開的 cut+hole blob。
- **`Registered ... as valid`** —— 跟 stage-one 相反,這裡是 valid 才對。

要小心的兩行:

```
No known-good images; using the defect set as the prototype     ← 基準不可靠
Model produced no reusable state; every stage-two run will re-fit  ← DINOv2 才會出現
```

SQL:

```sql
-- fit 的品質歷史
SELECT started_at, defect_count, labelled_count, cluster_count, noise_count,
       round(noise_count::numeric / NULLIF(labelled_count,0), 3) AS noise_ratio,
       mean_probability
FROM stage_two_run WHERE mode = 'fit' AND status = 'completed'
ORDER BY started_at DESC;
```

「這張表就是判斷「分群演算法是否在退化」的依據 —— 如果連續幾次 fit 的 `noise_ratio` 都在上升、`cluster_count` 在下降,那不是資料的問題,是描述子不再能分開型態了。」

MinIO:

```bash
docker compose exec minio mc ls -r local/models/resnet50/
# 每個 fit 過的版本目錄應該有一個 cluster_state.joblib
```

---

## 8. 已知風險與未完成的事(3 分鐘,**誠實交代**)

**① fit 沒有品質把關,而且會自我上線。** 一次糟糕的 fit 直接生效,後續 assign 全部投影到爛空間。目前只有事後的 drift 判定。要補的話,最小的版本是:fit 完成時檢查 `noise_ratio` 與 `cluster_count`,超出合理範圍就註冊成 `is_valid=false` 並在儀錶板告警 —— 這樣壞的 fit 不會上線,但也代表 stage-two 會暫停,需要人介入。這個取捨我沒有替你決定。

**② `umap.transform()` 沒有實測。** 這影響 assign 而不是 fit,但如果它不可用,整個 fit/assign 架構的價值就沒了(每次都只能重 fit)。**接手後第一個該驗證的東西。** 方法:把 `STAGE_TWO_REFIT_COUNT=20`、`STAGE_TWO_ASSIGN_COUNT=10`,餵幾十張圖走完一圈。

**③ artifact 綁函式庫版本。** pickle 綁在 `scikit-learn==1.5.2` / `umap-learn==0.5.7` / `hdbscan==0.8.40`。升版會讓既有 fit 失效必須重 fit。這也逆轉了原本「不要有 version-coupled joblib artifact」的設計決定 —— 是這次改動刻意付的代價。

**④ 舊版本與 artifact 沒有清理機制。** 每次 fit 多一個 `cluster_state.joblib`(依資料量可能數十 MB)。沒有任何地方會刪。

**⑤ fit 會阻塞 Kafka 消費。** 對 5000 張圖跑 ResNet50 + PCA + UMAP + HDBSCAN,那段時間 stage-one 事件完全不會被消費。UMAP/HDBSCAN 對點數是超線性的,5000 張跟 500 張不是差 10 倍。如果超過 `max.poll.interval.ms`(預設 5 分鐘),consumer 會被踢出 group 觸發 rebalance。**這是量一上來第一個會爆的地方。**

**⑥ `SEED=42` 固定,但不保證完全可重現。** UMAP 在多執行緒下即使固定 seed 也可能有微小差異。所以兩次對同一批資料 fit,cluster 編號**可能**不同。不要依賴 fit 的可重現性。

**⑦ DINOv2 沒有拆 fit/assign。** 它只實作 `fit` 並回傳 `state=None`。若 promote,`Model produced no reusable state` 那行會出現,而且每次都全集重 fit。

---

## 9. 接手後建議的前三件事(1 分鐘)

「1. **調小門檻走完一圈 fit → assign**(風險 ②)。這是唯一能確認整條路通的方式,而且會讓你看懂 log 的每一行。記得跑完把門檻調回去。

2. **決定要不要為 fit 加品質閘門**(風險 ①)。這是這個設計目前最大的缺口,而且取捨很實在:嚴格的閘門會讓壞 fit 不上線,但也會讓 stage-two 停擺等人。

3. **量測一次 fit 的實際耗時**(風險 ⑤),用你的真實資料量。這個數字決定 stage-two 什麼時候必須搬離 driver 行程。

程式碼入口:`adapters/outbound/resnet50/model.py` 的 `fit()`(分群管線本體)、`resources.py` 的 `dump_state`/`load_state`/`save_state`(凍結)、`application/use_cases/run_stage_two.py` 的 `_do_fit()`(版本發布與紀錄)、`descriptors.py`(colour/shape 描述子,調參最可能動到的地方)。」

---

## 附錄:一頁流程圖

```
累積 ≥ 1000 張新 anomaly(或還沒有任何凍結狀態)
      │
      ▼  FIT(在 inference-service 行程內,阻塞 Kafka 消費)
┌──────────────────────────────────────────────────────────────┐
│ 缺陷集 = 最新 5000 張 status='anomaly'                       │
│ 良品集 = 最新 200 張 status='normal'                         │
│                                                              │
│ ① prototype:良品的 Layer2/3 每通道 mu/std                   │
│ ② 每張缺陷 → deep(加權池化)+ colour(5d)+ shape(5d)     │
│ ③ block balance(1.0/0.3/0.3)→ L2 norm → PCA(0.95)        │
│      → UMAP(cosine, 2d)→ HDBSCAN(prediction_data=True)   │
│ ④ 第二層:用離心率把 cut+hole blob 切開(有條件)            │
└──────────────────────────────────────────────────────────────┘
      │
      ▼  凍結七個物件 → 單一 joblib
  ① 上傳 models/resnet50/{minor+1}/cluster_state.joblib
  ② 註冊 inference_model(is_valid = TRUE,自我上線)
  ③ self.state = live_state       ← 立刻採用,不必重啟
  ④ model_id 切到新版本
      │
      ▼
  cluster_results(全集重新編號)+ stage_two_run(mode=fit)
      │
      ▼
  之後每 500 張走 assign,直到再累積 1000 張才重畫
      │
      ▼
  儀錶板持續比對 assign 的 noise 比例 vs 這次 fit 的基準
```
