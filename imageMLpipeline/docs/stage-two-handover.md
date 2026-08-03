# Stage-two 推論交接講稿(assign:把新缺陷歸類)

> 用途:把 stage-two 的**推論側**(用既有分群結果替新缺陷歸類)交接給接手的人。
> 建議時間:18–22 分鐘 + 問答。
> 系列:[stage-one 推論](stage-one-handover.md) · [stage-one 重訓](stage-one-retrain-handover.md) ·
> **本篇** · [stage-two 訓練](stage-two-retrain-handover.md)

---

## 0. 開場(1 分鐘)

「stage-two 做的事情是:**把已確認的缺陷影像分成缺陷型態**(刮痕 / 破洞 / 髒污…),寫進 `cluster_results`。

它跟 stage-one 有三個根本差異,先講清楚,後面才不會混:

| | stage-one | stage-two |
|---|---|---|
| 粒度 | 單張影像可獨立判定 | **whole-set**,必須一起看才能分群 |
| 跑在哪 | Spark executor | **inference-service 行程內**(沒有用 Spark) |
| 監督性 | 有校準過的閾值 | **無監督**,沒有標準答案 |

而 stage-two 自己又分成兩種模式:**assign**(用既有分群結果歸類新缺陷,便宜)和 **fit**(全集重新分群,昂貴)。這份講 assign,fit 在另一份。」

---

## 1. 心智模型:兩道閘門(3 分鐘)

「stage-one 每判定一張 `anomaly` 就發一個 `type=stage-two` 事件。但**絕大多數事件什麼都不會做** —— 它們只是去看一下閘門。」

```
type=stage-two 事件進來
      │
      ▼  查兩個累積量(都從 stage_two_run 推導)
  自上次 FIT 以來 ≥ 1000 張?  ──是──►  FIT(全集重新分群,另一份講)
      │否
      ▼
  有凍結的分群狀態可用?      ──否──►  只 log,返回(等 FIT)
      │是
      ▼
  自上次任何 run 以來 ≥ 500 張? ──是──►  ASSIGN(只歸類新的那批)
      │否
      ▼
  只 log,返回  ←── 這是最常見的路徑
```

「三個必須說清楚的點:

1. **FIT 優先。** 兩道閘門同時開時走 FIT,因為它反正會重新標記全部。
2. **沒有凍結狀態就只有 FIT 這條路。** 系統剛上線時沒有任何 fit,所以第一次動作一定是 FIT。
3. **兩道閘門用兩個獨立的 watermark。** 這點很反直覺但很關鍵,第 3 節會單獨講。」

程式碼:`inference-service/application/use_cases/run_stage_two.py` 的 `execute()`,大約 20 行,是整個決策的全部。

---

## 2. 為什麼要有 assign 這個模式(2 分鐘)

「原本的設計是每次觸發都全集重新分群。那有兩個問題:

**一、浪費。** 為了一張新缺陷把 5000 張重跑一次 UMAP + HDBSCAN。

**二、比較嚴重的:`cluster_id` 會漂移。** UMAP 和 HDBSCAN 都是 transductive 的 —— 每次 fit 出來的分群編號跟上一次沒有對應關係。今天的「cluster 3」和昨天的「cluster 3」可能是完全不同的缺陷型態。這讓 `cluster_results` 幾乎無法做趨勢分析。

assign 解決的就是第二個問題:**凍結一次 fit,之後的新影像投影進同一個空間、拿同一套編號。** 所以:

> **在同一個 `model_id` 內,`cluster_id` 是穩定的。跨 fit 會重新編號。**

這是查 `cluster_results` 時最重要的一條規則 —— 比較 cluster_id 一定要限定同一個 `model_id`。」

---

## 3. 兩個 watermark(3 分鐘,**不要跳**)

「兩個累積量都不是存在某個計數器裡,而是**從 `stage_two_run` 這張表推導出來的**:」

```sql
-- assign 閘門:自「任何模式」的上次成功 run 以來
SELECT max(covered_through) FROM stage_two_run WHERE status = 'completed';

-- fit 閘門:自上次「fit」以來
SELECT max(covered_through) FROM stage_two_run
 WHERE status = 'completed' AND mode = 'fit';

-- 然後對任一個 watermark 數
SELECT count(*), max(inferred_at) FROM inference_results
 WHERE status = 'anomaly' AND inferred_at > <那個 watermark>;
```

### 為什麼一定要分成兩個

「這是我實作時撞到的問題,值得完整講一次:

assign 必須推進**某個** watermark,否則它會永遠重複歸類同一批 500 張。但如果它同時推進 fit 的 watermark,那 fit 的計數就會每 500 張歸零一次 —— **永遠到不了 1000,分群永遠不會重畫**。而分群不重畫,新的缺陷型態就永遠無法形成新 cluster。

所以:assign 只推進「任何 run」的線,fit 推進兩條。」

### 為什麼不用計數器

「因為 stage-one 的 Kafka 重播會產生**重複的 `inference_results` 列**(`event_id` 每次都是新 uuid)。如果每寫一列就把計數器 +1,計數會偏掉而且無法校正。從單一事實來源推導就沒有這個問題。

另外三個好處:

- **重啟後續算**,累積量在 Postgres 不在記憶體
- **失敗的 run 會重試** —— `covered_through` 只在成功持久化後才前進
- **不會死鎖** —— 如果改用 `max(cluster_results.clustered_at)` 推導,一次全部是噪聲的 run 不會產生新列,watermark 就永遠不動了

`covered_through` 是在 run **開始前**取樣的,所以分群期間新進來的缺陷算下一輪,不會被這一輪跳過。」

---

## 4. assign 實際做什麼(5 分鐘)

`run_stage_two.py` 的 `_do_assign()` → `resnet50/model.py` 的 `assign()`

### 4.1 取哪些影像

```python
defect_locs = self._image_source.anomalous_since(watermark, defect_set_size)
```

「**只取新的**(watermark 之後的),不是全集。舊影像早就有 cluster_id 了,重貼一次只是寫入相同的列。這是 assign 便宜的主因。」

### 4.2 投影進凍結的空間

```python
deep_all, color_all, shape_all = self._descriptors(defects, state["prototype"])
feat = normalize(concat([_apply_block(deep_all, W_DEEP, scalers["deep"]), ...]))
embedding = state["umap"].transform(state["pca"].transform(feat))
labels, strengths = hdbscan.approximate_predict(state["hdbscan"], embedding)
labels = self._apply_refine(labels, shape_all, state["refine"])
```

「每一步都用**凍結的**那一份,不是重新 fit 的:

| 用凍結的 | 如果重建會怎樣 |
|---|---|
| `state["prototype"]` | deep 特徵的 z-score 基準改變 → 整個空間平移 → **最致命** |
| 三個 `StandardScaler` | 三個 block 的相對權重跑掉 |
| `PCA` | 投影軸與維度數都變 |
| `UMAP.transform()` | 2-D 嵌入空間完全不同 |
| `HDBSCAN` + `approximate_predict` | 沒有 `prediction_data=True` 就直接拋錯 |
| 第二層的切分決策 | 見下 |

**第二層要特別講。** 原本的第二層(`_refine_by_shape`)是一個**搜尋**而不是模型 —— 它掃過所有 cluster,找出離心率雙峰最平衡的那個,用臨時 KMeans 切開。它的產物是一個決策,不是分類器。所以凍結的是「被切的是哪個 cluster、那顆 1-D KMeans、哪一側變成新 id」,assign 時 `_apply_refine()` 用同一顆 KMeans 判斷新影像該歸哪一邊。」

### 4.3 寫入

「一張影像一列 `cluster_results`,整批共用同一個 `clustered_at`。`cluster_id = -1` 代表**沒有對應到任何 cluster**。」

---

## 5. `-1` 是訊號,不是錯誤(3 分鐘)

「這節是 stage-two 運維的核心概念。

**assign 只能把影像放進已經存在的 cluster。** 所以如果產線出現了 fit 當時沒見過的缺陷型態,它不可能憑空生出一個新 cluster —— 只能回 `-1`。

因此:**unmatched 比例上升 = 特徵遷移的訊號。**

每個 run 都會記下自己的品質數字:

| 欄位 | 意義 |
|---|---|
| `defect_count` | 觸發這次 run 的**新增**張數。**不是分母** |
| `labelled_count` | 這次真正貼標的張數 —— **分母** |
| `noise_count` | 其中 `-1` 的張數 |
| `mean_probability` | HDBSCAN 歸屬強度平均。下降代表點落在 cluster **邊緣**,比噪聲更早出現的徵兆 |

【提示】`defect_count` 不是分母這件事要強調 —— fit 的閘門是 1000 但可能貼標 5000 張,拿它當分母會算出荒謬的比例。」

### 儀錶板怎麼判定

「`monitor-service` 拿每次 assign 跟**那個 fit 自己達到的 noise 比例**比較:

```
門檻 = max(STAGE_TWO_DRIFT_NOISE_RATIO(0.30), fit 自身的 noise 比例 × 2)
```

相對的那一半很重要:一個本來就會留下 25% 噪聲的 fit,如果只看絕對門檻,從建立的第一天就會被誤報成 drifting。

儀錶板上是 `unknown` / `healthy` / `drifting` 三種判定,加上三個累積讀數(`towards next assign 487`、`towards next re-fit 912`、`fit baseline unmatched 21.0%`)—— 後者是為了讓操作者看得出「為什麼現在沒有動作」,而不是乾等。」

### drifting 之後呢

「**刻意沒有做自動 re-fit 按鈕。** 理由是:re-fit 會重新編號,但如果問題是**描述子本身不再能分開缺陷型態**,重 fit 也救不了 —— 你只會得到一組同樣糟糕但編號不同的 cluster。那時需要的是人去重新檢視分群演算法(換描述子、調 block 權重、改 UMAP 參數)。

把它自動化掉會讓這個問題被藏起來。所以儀錶板只呈現證據。」

---

## 6. 你會踩的坑(2 分鐘)

**① `cluster_id` 只在同一個 `model_id` 內可比。** 已經講過,但這是最容易寫錯查詢的地方。跨 fit 比較 cluster 3 是沒有意義的。

**② `defect_set_size` 必須 ≥ `refit_count`。** 目前是 5000 ≥ 1000。如果調小到低於 refit 門檻,一次 fit 會**看不完觸發它的那批影像**(舊的被 `LIMIT` 截掉)。

**③ stage-two 會阻塞 Kafka 消費。** 它是在 consumer 迴圈裡同步跑的。一次 fit 要對數千張圖跑 ResNet50 + UMAP + HDBSCAN,那段時間 stage-one 事件完全不會被消費。量大之後這是第一個會爆的地方(見第 8 節)。

**④ `pending` 不在這條路上。** buffer zone 的中間帶完全不進 stage-two:不計數、不分群、也不發事件。它走 monitor 的人工審閱 → 重訓。不要看到 pending 沒有 cluster_results 就以為是 bug。

---

## 7. 怎麼確認正常工作(2 分鐘)

「最常見的 log 是「什麼都沒做」,這是正常的:」

```
Stage-two assign gate: 37/500 new defect image(s) since 2026-07-30T...; waiting
```

一次真正的 assign:

```
Stage-two assign run starting: 512 new defect image(s)
New defect set: 512 anomaly image(s) since 2026-07-30T... (limit 5000)
Assigning 512 new defect image(s) to the frozen fit
Assigned 512 defect image(s) to the frozen fit (63 matched no cluster)
Saved 512 cluster result(s) for model_id=...
Recorded stage-two assign run <uuid>: completed, 512 defect image(s), watermark=...
```

啟動時要確認凍結狀態載到了:

```
Loaded frozen fit models/resnet50/1.1.0/cluster_state.joblib (18432000 bytes); assign is available
```

「如果看到的是這行,代表這個版本還沒被 fit 過,只能走 fit:」

```
No frozen fit at models/resnet50/1.0.0/cluster_state.joblib (S3Error); this version can fit but not assign
```

儀錶板:`http://localhost:8000` 的「Stage-two clustering」區塊 —— verdict、三個累積讀數、run 歷史表。

SQL:

```sql
SELECT mode, status, defect_count, labelled_count, noise_count,
       round(noise_count::numeric / NULLIF(labelled_count,0), 3) AS noise_ratio,
       mean_probability, started_at
FROM stage_two_run ORDER BY started_at DESC LIMIT 20;
```

---

## 8. 已知風險與未完成的事(2 分鐘)

**① `umap.transform()` 在 `metric="cosine"` + 固定 `random_state` 下沒有實測過。** 這是整條 assign 路徑上唯一我無法靜態確認的環節。如果不支援,assign 會在這一步拋錯 —— 但會被記成 `failed` 且**不推進 watermark**,所以不會靜默出錯,而是會一直重試失敗。儀錶板的 run 表會直接顯示紅色 `failed` 與錯誤訊息。**這是你接手後第一個該驗證的東西。**

**② 凍結狀態與函式庫版本綁定。** artifact 是 pickle,綁在 `scikit-learn==1.5.2` / `umap-learn==0.5.7` / `hdbscan==0.8.40`。升版會讓既有 fit 失效、必須重 fit。

**③ stage-two 沒有搬上 Spark。** 它形態上其實比 stage-one 更適合(本來就是整批作業),但目前在 driver 行程內。量大之後要考慮搬走,或至少獨立成自己的 consumer group,否則會拖累 stage-one。

**④ DINOv2 候選版本沒有跟上。** 它只實作了 `fit` 並回傳 `state=None`,所以如果被 promote,每次都是全集重 fit、`cluster_id` 不可比。要用在生產得比照 ResNet50 拆成 fit/assign。

**⑤ 沒有 cluster 的語意標籤。** `cluster_id = 3` 沒有任何地方記錄它代表「刮痕」。這需要人看過樣本再命名,目前完全沒有這個機制。若要做趨勢報表,這是必要的一層。

---

## 9. 接手後建議的前三件事(1 分鐘)

「1. **先驗證 `umap.transform()`**(風險 ①)。最快的方法是把兩個門檻暫時調小(`STAGE_TWO_REFIT_COUNT=20`、`STAGE_TWO_ASSIGN_COUNT=10`),餵幾十張圖走完 fit → assign 一圈。這比等累積到 1000 快得多,而且是唯一能確認整條路通的方式。

2. **看懂儀錶板的三個累積讀數**,那是之後所有「為什麼沒動作」問題的答案。

3. **決定要不要做 cluster 命名**(風險 ⑤)。如果 stage-two 的產出要給人看,這是必要的;純技術指標的話可以先不做。

程式碼入口三個:`application/use_cases/run_stage_two.py`(兩道閘門 + 兩條路徑,約 300 行,主體)、`adapters/outbound/resnet50/model.py` 的 `assign()`、`monitor-service/db/stage_two_runs.py`(drift 判定)。」

---

## 附錄:一頁流程圖

```
stage-one 判定 anomaly → 發 Kafka type=stage-two
      │
      ▼  絕大多數事件到這裡就結束
┌─ RunStageTwo.execute ─────────────────────────────────────┐
│  since_fit = count(anomaly > fit watermark)               │
│  if since_fit >= 1000        → FIT(另一份文件)           │
│  if 沒有凍結狀態             → 只 log,等 FIT             │
│  since_run = count(anomaly > any-run watermark)           │
│  if since_run >= 500         → ASSIGN ↓                   │
│  else                        → 只 log                     │
└───────────────────────────────────────────────────────────┘
      │
      ▼  ASSIGN(在 inference-service 行程內,沒有 Spark)
  anomalous_since(watermark) → 只取新的缺陷影像
      → MinIO 併發下載
      → 凍結的 prototype / scalers / PCA / UMAP.transform
      → hdbscan.approximate_predict → 凍結的第二層切分
      → cluster_results(一張一列,-1 = 沒對應到)
      → stage_two_run(mode=assign + labelled/noise/mean_prob)
      │
      ▼
monitor 儀錶板:noise 比例 vs fit 基準 → healthy / drifting
      │
      ▼  drifting 時
  人工檢視分群演算法(不是按一個 re-fit 按鈕就好)
```
