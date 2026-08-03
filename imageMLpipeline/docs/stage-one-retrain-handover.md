# Stage-one 重訓流程交接講稿

> 用途:把 stage-one 的 train / retrain(`train-service`)交接給接手的人。
> 建議時間:18–22 分鐘 + 問答。標【略】的段落時間不夠可跳過。
> 搭配 [`stage-one-handover.md`](stage-one-handover.md) 一起看 —— 那份講推論,這份講模型怎麼更新。

---

## 0. 開場(1 分鐘)

「這份講 stage-one 的模型怎麼更新。先說結論式的三句話,後面都是展開:

1. **PatchCore 的『重訓』不是梯度下降,是往記憶庫裡加向量。** 沒有反向傳播、沒有 loss、沒有 epoch。backbone 從頭到尾都是同一個 ONNX 檔案,永遠不動。
2. **人在迴圈裡,而且是必要的。** 重訓只能由人在 dashboard 上挑選 `pending` 影像來觸發,產出的新版本一律是 `is_valid=false` 的候選,要再由人批准才會上線。
3. **每個 train 事件會產生一個獨立的新版本,而且它們不累積。** 這是最容易誤解的行為,我會單獨用一節講。」

---

## 1. 心智模型:三個角色(2 分鐘)

```
① 誰觸發          monitor-service(人工挑選)
                      │  POST /api/retrain
                      ▼  Kafka topic "train"(一張圖一個事件)
② 誰執行          train-service
                      │  加向量 → 重算 buffer zone → 寫新版本
                      ▼  MinIO models/patchcore/1.0.N/ + inference_model 一列
③ 誰批准          人工 SQL(is_valid=true)+ 重啟 Spark worker
```

「注意三件事:

- **train-service 完全沒有排程,也不會自己找資料。** 它只是一個 Kafka consumer,沒有事件就閒著。
- **`pending` 這個判定是這條路的入口。** buffer zone 的中間帶(不確定、需要人看)專門走這裡 —— 它不會進 stage-two 分群,只會被人審閱後送來重訓。
- **角色 ③ 目前完全沒有程式碼。** 全 repo 沒有任何一行會把 `is_valid` 設成 true,只有 README 裡的 SQL 範例。這是刻意的(等 evaluate-service),但你要知道現況。」

---

## 2. 觸發端:monitor-service(2 分鐘)

`monitor-service/app.py` 的 `POST /api/retrain`

「操作是:dashboard 上把 `pending` 的列勾起來 → 按「Retrain selected」。前端會產一個 `crypto.randomUUID()` 當 `Idempotency-Key`。

伺服器端做三件事:

1. **獨立重新驗證。** 不信前端送來的東西 —— 用 `pending_by_ids()` 回查 Postgres,只有真的是 `status='pending'` 的列才會被發布。
2. **依 key 做幂等。** 同一個 key 重送會直接回傳快取的結果、不重複發布;同一個 key 併發重送會拿到 `409`。快取是行程內的有界 `OrderedDict`(1024 筆)。
3. **一張圖一個事件**,payload 很簡單:

```json
{"event": "train", "model": "patchcore", "images": "/images/<uuid>.png"}
```

【提示】要強調:**幂等只在 monitor 這一端。** train-service 本身沒有幂等 —— 同一個 train 事件被重播就會再產生一個新版本。」

---

## 3. train-service 啟動時做什麼(2 分鐘)

`train-service/adapters/inbound/train_consumer.py` 的 `initialize()`

「啟動時只做一次的事:

```python
base = registry.latest(settings.model.model_key)     # 最新的『已驗證』版本
session = load_onnx_session(storage, base.bucket, ...)  # 下載並載入 ONNX backbone
embedder = PatchCoreEmbedder(session, build_transform(224))
```

**backbone 在啟動時載入一次,之後不再重載。** 理由很正當:重訓只會擴充記憶庫,backbone 永遠是同一個檔案,所以每個事件重載毫無意義。

但推論出一個坑:**如果有人批准了一個 backbone 不同的版本,train-service 必須重啟才會用到它。** 目前的重訓流程不會產生不同的 backbone,所以還不是問題 —— 但如果之後引入真正的 fine-tune,這裡要一起改。」

---

## 4. 走一次 `RunTraining.execute`(5 分鐘)

`train-service/application/use_cases/run_training.py` —— 整個重訓邏輯就這一個檔案,六步。

### 步驟 1–2:解析事件、決定基準版本

```python
model_type = event.model_type            # payload 的 "model"
bucket, key = event.image_location()     # 解析 "/{bucket}/{key}"
base = self._registry.latest(model_type) # 最新的『已驗證』版本
```

「`latest()` 的 SQL 有 `WHERE is_valid = true`。**記住這一點**,第 5 節整節都在講它的後果。」

### 步驟 3:把影像變成向量

```python
features = self._embedder.embed(image)   # ONNX ResNet50 → (num_patches, dim)
```

「用的是**跟推論完全相同的 backbone 與前處理**(`build_transform(224)`)。這是必要的 —— 記憶庫裡新舊向量必須在同一個特徵空間裡,否則距離沒有意義。」

### 步驟 4:coreset 抽樣後併入記憶庫

```python
_CORESET_DIVISOR = 98
size = max(1, n // 98)                   # 隨機保留 1/98
new_bank = np.vstack([current_bank, _subsample(features)])
```

「**只保留約 1/98 的向量**,這個比例是為了對齊原始記憶庫的建立方式。一張 224×224 的圖大約 784 個 patch,所以一張圖大概只貢獻 8 個向量。

現有的記憶庫是 1704 × 1536 —— 所以單張圖的影響**很小**。這是設計上的保守,但也意味著:想真的改變模型行為,需要不少張圖。」

### 步驟 5:重算 buffer zone(**這步最容易被忽略**)

```python
index = _build_index(new_bank)                                    # FAISS over 新 bank
validation_images = self._validation_set.recent_good(model_type, 200)
scores = self._score_validation_images(index, validation_images)
lower, upper = np.percentile(scores, [75, 99])
```

「三個要講的點:

**一、為什麼要重算。** 記憶庫變大 → 任何影像的最近鄰距離都可能變小 → 整個分數尺度往下平移。如果沿用舊的 buffer zone,判定會整體偏向 `normal`。所以 bank 一動,界線就必須跟著動。

**二、用什麼資料算。** 最近 200 張 `status='normal'` 的影像(`VALIDATION_SET_SIZE`),條件是 `r.status = 'normal' AND m.model_type = ?`,按 `inferred_at` 倒序。這是一個**滾動的驗證集** —— 沒有固定資產要維護,但也代表它會隨著產線漂移。

**三、分數怎麼算。** `_image_score()` 是刻意逐行對齊 `PatchCore.inference` 的:`index.search(features, 1)` → `sqrt` → `max`。**這兩處必須一致,否則算出來的界線跟推論用的尺度不同。** 如果你改了推論端的評分方式,這裡一定要跟著改。

抓驗證影像是分批的(每批 16 張,併發下載後立刻評分),抓失敗的那張只會 WARNING 跳過,不會中斷整批。」

### 步驟 6:寫成 patch+1 候選版本

```python
new_version = self._registry.next_version(base)   # 1.0.0 → 1.0.1
memory_bank.save(new_version, new_bank)           # memory_bank.npy
memory_bank.save_buffer_zone(new_version, lower, upper)   # buffer_zone.txt
memory_bank.copy_unchanged_assets(base, new_version)      # onnx + onnx.data
registry.register(new_version, is_valid=False)
```

「**版本目錄是自足的。** 即使 backbone 沒變,也會把 `resnet_backbone.onnx` 和 `.onnx.data` 複製過去。這樣 `models/patchcore/1.0.1/` 底下四個檔案齊全,推論端載入任何版本都不需要知道「backbone 要去哪個舊版本找」。代價是每個候選版本多佔 34MB。

**版本號怎麼決定:** `max(patch_version) + 1`,範圍是所有同 `(model_type, major, minor)` 的列 —— **包含 `is_valid=false` 的**。所以連續重訓不會撞號。

**沒有驗證影像時**(例如系統剛上線,還沒有 `normal` 結果),會把舊的 `buffer_zone.txt` 直接複製過去,並記一行 WARNING。這是安全的退路 —— 界線沒有變好,但至少沒有變成亂數。」

---

## 5. 最重要的坑:候選版本不累積(3 分鐘,**不要跳**)

「這是這份文件最需要你記住的一件事。」

「回到步驟 2:`base = registry.latest()`,而 `latest()` 只看 `is_valid = true`。而步驟 6 註冊的新版本是 `is_valid = false`。

**所以每個 train 事件的基準都是同一個已驗證版本。**

在 dashboard 上勾 10 張圖按下重訓,結果不是「一個含 10 張圖的候選版本」,而是:」

```
1.0.0(已驗證,基準)
 ├─ 1.0.1 = 原始 bank + 圖A 的 8 個向量
 ├─ 1.0.2 = 原始 bank + 圖B 的 8 個向量      ← 不含圖A
 ├─ 1.0.3 = 原始 bank + 圖C 的 8 個向量      ← 不含A、B
 └─ ...
1.0.10 = 原始 bank + 圖J 的 8 個向量        ← 只含圖J
```

「十個互不相干的候選版本,每個只多了一張圖。而你只能批准其中一個 —— 批准 `1.0.10` 就等於**丟掉另外九張圖的貢獻**。

要做到累積,目前只有兩條路:

- **一張一張來**:重訓 → 批准(`is_valid=true`)→ 再重訓下一張。這樣第二次的 `latest()` 才會拿到含第一張圖的版本。很慢,但符合現有程式碼。
- **改成一個事件帶多張圖**:`TrainEvent.image_location()` 現在解析單一路徑,payload 的 `images` 欄位改成陣列、`execute()` 改成迴圈累積後只寫一個版本。這是我認為正確的方向,但我沒做 —— 你接手後如果重訓會常用,這應該是第一個要處理的。

【提示】如果被問「那為什麼現在沒壞?」—— 因為目前是示範規模,一次只重訓一兩張。量一上來就會很明顯。」

---

## 6. 批准與上線(2 分鐘)

「候選版本產生後,**沒有任何程式會自動啟用它**。要上線是兩個動作:

```sql
-- 1. 批准(目前純手動;未來是 evaluate-service 的職責)
UPDATE inference_model SET is_valid = true WHERE model_id = '<新版本的 model_id>';
```

```bash
# 2. 重啟 Spark worker —— 這一步很容易漏
docker compose restart spark-worker-1 spark-worker-2 spark-worker-3
```

「**為什麼一定要重啟 worker:** 推論端的模型是快取在每個 Python worker 行程的 `_STATE` 裡的,而且 `PatchCore` 是 singleton。`_STATE` 沒有失效機制,`refresh` 事件現在也只剩一行 log。所以不重啟的話,executor 會繼續用舊模型,而且會繼續往 `inference_results` 寫**舊的 `model_id`** —— 從資料上看你會以為新模型沒生效,實際上是根本沒載入。

這是我在推論那份文件裡也列為坑的同一件事,兩邊都要記得。

驗證方式:重啟後看 worker log 的 `Executor ready: PatchCore model_id=...`,那個 model_id 應該是你剛批准的那個。」

---

## 7. 失敗語意(1 分鐘)

`train_consumer.py` 的 `run()`:

```python
for event in self._consumer.events():
    try:
        self._use_case.execute(event)
        self._consumer.commit()          # 成功才 commit
    except Exception:
        logger.exception("Failed to process train event")
```

| 情況 | 行為 |
|---|---|
| 某張驗證影像抓不到 | WARNING 跳過,其餘照算 |
| 完全沒有驗證影像 | 複製舊的 buffer zone,記 WARNING |
| 事件處理中任何例外 | 記 log、**不 commit**,繼續下一個事件 |
| 事件格式錯誤(`images` 路徑不合法) | `ValueError`,同上 |

「注意跟推論端一樣的陷阱:**不 commit 不等於會立刻重試**。kafka-python 的讀取位置已經前進,所以那個事件要等服務重啟或 rebalance 才會從上次 commit 的位置重播。實務上就是:失敗的重訓要人重新在 dashboard 上按一次。」

---

## 8. 怎麼確認正常工作(2 分鐘)

「照著 log 的順序看,一次成功的重訓會留下這串:」

```
Train 'patchcore' on image images/<uuid>.png
Memory bank (1704, 1536) + (784, 1536) -> (1712, 1536)     ← 加了 8 個向量
Recalibrated buffer zone from 200 good image(s): lower=... upper=...
Saved memory bank (1712, 1536) to models/patchcore/1.0.1/memory_bank.npy
Saved buffer zone lower=... upper=... to .../buffer_zone.txt
Copied asset resnet_backbone.onnx: ... -> ...
Registered 'patchcore' v1.0.1 (model_id=..., is_valid=False)
Retrained 'patchcore' v1.0.0 -> v1.0.1 (model_id=..., is_valid=false)
```

「三個檢查點:

- **`Memory bank A + B -> C`** —— 確認向量真的加進去了,且增量約是 `B ÷ 98`
- **`Recalibrated buffer zone from N good image(s)`** —— 如果 N 是 0,代表沒有 `normal` 結果,界線只是複製舊的
- **`is_valid=false`** —— 這是正常的,不是錯誤

資料庫與 MinIO:

```sql
SELECT model_id, major_version, minor_version, patch_version, is_valid, created_at
FROM inference_model WHERE model_type = 'patchcore' ORDER BY created_at DESC;
```

```bash
docker compose exec minio mc ls -r local/models/patchcore/
# 每個版本目錄應該有 4 個檔案:memory_bank.npy / buffer_zone.txt
#                            resnet_backbone.onnx / .onnx.data
```

---

## 9. 已知風險與未完成的事(2 分鐘,**誠實交代**)

**① 候選版本不累積**(第 5 節)—— 目前最該修的東西。

**② 沒有 evaluate-service,所以批准是純人工。** 沒有任何自動的品質把關:候選版本的 buffer zone 是算出來了,但沒有人比較「新版本在測試集上是否真的更好」。批准的人實際上是憑信任在按下去。

**③ 候選版本會無限累積,沒有清理機制。** 每次重訓多一個目錄、多 34MB(複製的 backbone)+ 記憶庫。沒有任何地方會刪掉未被批准的候選。跑久了 `models` bucket 會長很大,`inference_model` 也會塞滿 `is_valid=false` 的列。

**④ 滾動驗證集會隨產線漂移。** buffer zone 用「最近 200 張 normal」算,而「什麼算 normal」又是由當時生效的模型判定的。這是一個回饋迴路:模型判定 normal → 那些圖成為驗證集 → 驗證集決定新的界線。如果模型開始把真正的缺陷判成 normal,這個迴路會自我強化。目前規模看不出來,但這是設計上的隱憂,值得記著。

**⑤ 重訓沒有併發保護。** train-service 是單一 consumer 循序處理,所以目前安全。但如果之後開多個實例,兩個事件可能同時 `next_version(base)` 拿到同一個 patch 號 —— `register` 有 `ON CONFLICT (model_id) DO NOTHING`,而 model_id 是各自新產生的 uuid,所以**不會衝突報錯,而是會產生兩個版本號相同的列**。要擴展的話這裡要先加約束。

---

## 10. 接手後建議的前三件事(1 分鐘)

「1. **完整跑一次**:dashboard 挑一張 pending → 按重訓 → 對照第 8 節的 log → 手動 `UPDATE is_valid` → 重啟 worker → 確認 worker log 的 model_id 換了。走完這一圈你就掌握整條路了。

2. **決定要不要修「不累積」那件事。** 如果重訓會常用,把 payload 改成多張圖是小改動(`TrainEvent`、`execute()` 的迴圈、monitor 端發一個事件而非 N 個),但會改變語意,值得先討論。

3. **加一個候選版本的清理策略。** 最簡單的版本:批准新版本時,把同 major.minor 底下其他 `is_valid=false` 的目錄刪掉。

檔案很少,入口就三個:`adapters/inbound/train_consumer.py`(接線)、`application/use_cases/run_training.py`(全部邏輯,約 180 行)、`adapters/outbound/patchcore/memory_bank_store.py`(MinIO 讀寫)。」

---

## 附錄:一頁流程圖(可印出來)

```
人在 dashboard 勾選 pending 影像
      │  POST /api/retrain(Idempotency-Key;伺服器回查 status='pending')
      ▼  一張圖 = 一個事件
Kafka topic "train"  {"event":"train","model":"patchcore","images":"/images/x.png"}
      │
      ▼  train-service(單一 consumer,循序處理)
┌──────────────────────────────────────────────────────────────┐
│ base = registry.latest()          ← 只看 is_valid=true !     │
│ features = embed(image)           ← 與推論同一個 ONNX        │
│ new_bank = vstack(bank, 隨機 1/98 的向量)                    │
│ index = FAISS(new_bank)                                      │
│ scores = 最近 200 張 status='normal' 對 new_bank 評分         │
│ lower, upper = P75, P99                                      │
│ new_version = base.patch + 1                                 │
│   → memory_bank.npy   (新)                                   │
│   → buffer_zone.txt   (新,或沒驗證影像時複製舊的)           │
│   → onnx + onnx.data  (複製,讓版本目錄自足)                 │
│ register(is_valid = FALSE)        ← 候選,不會自動上線       │
└──────────────────────────────────────────────────────────────┘
      │
      ▼  人工
UPDATE inference_model SET is_valid = true WHERE model_id = '...';
docker compose restart spark-worker-1 spark-worker-2 spark-worker-3
      │                    ↑ 漏掉這步,executor 會繼續用舊模型
      ▼
worker log:Executor ready: PatchCore model_id=<新的>
```
