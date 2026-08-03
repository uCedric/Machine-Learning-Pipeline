# Stage-one 推論交接講稿

> 用途:把 stage-one(PatchCore 異常偵測)交接給接手的人。
> 建議時間:20–25 分鐘 + 問答。標【略】的段落時間不夠可跳過。
> 每一節都附了對應的檔案位置,講到時可以直接打開。

---

## 0. 開場(1 分鐘)

「今天要交接的是 stage-one,也就是異常偵測這一段。它做的事情一句話講完:**一張影像進來,吐出一個異常分數和一個判定(normal / pending / anomaly),外加一張熱力圖**。

但真正需要交接的不是這件事,而是**它跑在哪裡**。因為 stage-one 的運算不在 inference-service 裡面,而是在 Spark worker 上。這個認知落差是接手時最容易出錯的地方,所以我會先講清楚行程的分佈,再走一次資料流,最後講五個你一定會踩到的坑。」

---

## 1. 心智模型:誰在哪裡跑(3 分鐘)

「先建立這張圖,後面所有事情都會好懂。」

```
inference-service 容器              spark-master 容器        spark-worker-1~3 容器
┌────────────────────────┐        ┌──────────────┐        ┌──────────────────────┐
│ python -m bootstrap.   │        │ Master JVM   │        │ Worker JVM (daemon)  │
│   inference            │        │ 只做資源配置 │        │  └ Executor JVM      │
│  ├ Kafka consumer      │◄──────►│ 不跑任何運算 │◄──────►│      └ python3 × ≤3  │
│  ├ Spark driver        │        └──────────────┘        │        ↑ 模型在這裡  │
│  └ stage-two 分群      │                                └──────────────────────┘
└────────────────────────┘
```

要強調的三句話:

1. **spark-master 不跑運算。** 它只負責把 executor 配給 application。driver 在 inference-service 容器裡(client mode)。
2. **ResNet50 在 Python worker 裡,不在 inference-service 裡。** inference-service 這個容器**從來沒有載入過異常模型**。你在它的 log 裡不會看到模型載入。
3. **stage-two 分群反而是在 inference-service 行程內同步跑的。** 這是目前唯一會阻塞 Kafka 消費的東西 —— 記住這點,之後講到效能瓶頸會再回來。

【提示】這裡可以打開 `docker-compose.yml` 對照那幾個服務。

---

## 2. 三層 batch(2 分鐘,**這節不要跳**)

「『batch』在這個系統裡有三個不同意思,搞混會完全看不懂參數。」

| 層級 | 預設 | 參數 | 它控制什麼 |
|---|---|---|---|
| Kafka micro-batch | ≤ 32 筆事件 | `SPARK_MICRO_BATCH_SIZE` | 一次 Spark job、一次 Kafka commit |
| RDD partition | 6 | `SPARK_PARTITIONS` | 切成幾個 task |
| UDF mini-batch | ≤ 8 張 | `SPARK_UDF_BATCH_SIZE` | **backbone 一次前向傳播看到幾張** |

「所以預設值下的實際情形是:32 筆事件 → 切成 6 個 task → 每個 task 約 5 張 → 每個 task 只呼叫模型一次。」

---

## 3. 走一次完整流程(6 分鐘)

「照著程式碼走一遍。檔案是 `inference-service/adapters/outbound/spark/scorer.py`,主要邏輯都在這裡。」

### 3.1 事件累積(driver 端)

`adapters/inbound/inference_consumer.py`

「consumer 不是一筆一筆處理,而是 `poll(32, 2000ms)` —— 最多收 32 筆、最多等 2 秒。收到之後按事件的 `type` 分流,`stage-one` 的整批交給 `RunStageOneBatch`。

**這裡沒有 fallback。** 以前有一個 `SPARK_STAGE_ONE_ENABLED=false` 可以退回單機推論,已經移除了。所以 **spark-master 連不上,inference-service 會在啟動時就失敗**,不會帶著壞掉的叢集假裝正常。這是刻意的。」

### 3.2 送上叢集

`scorer.py` 的 `score()`:

```python
rows = [(bucket, object_key, content_type, size_bytes) ...]   # 只有中繼資料
slices = max(1, min(partitions, len(rows)))
frame = spark.createDataFrame(sc.parallelize(rows, slices), _input_schema())
frame.select("object_key", udf(...)).select(...).collect()     # ← 只有這行真的執行
```

三個要講的點:

- **DataFrame 裡沒有影像,只有位置。** executor 自己去 MinIO 下載。這樣影像不會經過 driver,流量留在叢集內。
- **為什麼是 `parallelize(rows, slices)` 而不是直接 `createDataFrame(rows)`。** 後者會用 `defaultParallelism`,也就是所有 executor 核心數總和 —— 在 8 核機器上是 24。24 個 partition 代表 24 份 ResNet50 副本,每份只算 1~2 張。所以 partition 數必須自己指定。
- **`collect()` 是唯一的 action。** 前面全是惰性的邏輯計畫;整批推論的耗時全部集中在這一行,而 driver 在這裡是阻塞的。

### 3.3 executor 端做什麼

`scorer.py` 的 `_predict_fn` / `_score` / `_fetch`:

```
predict(buckets, keys, content_types, sizes)     ← predict_batch_udf 餵進來的 numpy 陣列
  → 重建 InferenceEvent
  → _fetch:ThreadPoolExecutor(8) 併發下載影像
  → score_batch:堆成 (B,3,224,224) → ONNX → 一次 FAISS 搜尋
  → finalise(每張):buffer zone 判定 → 熱力圖 → MinIO → INSERT inference_results
  → 回傳 {status, anomaly_score, error} 三個陣列
```

「注意熱力圖和結果列都是**在 executor 上寫的**,不是回到 driver 才寫。回 driver 的只有純量 —— 如果回傳 PNG,driver 會變成記憶體炸彈。」

### 3.4 回到 driver

「結果用 `object_key` 對回原本的事件,然後 `RunStageOneBatch` 對每個 `anomaly` 判定發一個 `type=stage-two` 的 Kafka 事件。最後才 commit offset。

**executor 從不碰 Kafka** —— 所以 worker 容器不需要 broker 權限。」

---

## 4. 模型怎麼上到 worker 的(4 分鐘,**這節最重要**)

「這節如果只能記一件事:**driver 不會把程式碼送過去,worker 是自己 import 的。**」

### 4.1 送過去的是什麼

`predict_batch_udf` 的 payload 裡只有:

| 內容 | 序列化方式 |
|---|---|
| pyspark 自己的 wrapper 函式 | by value(bytecode,幾 KB) |
| **我們的 `_predict_fn`** | **by reference** ——「去 import `adapters.outbound.spark.scorer`,拿這個名字」 |
| return schema、batch_size | by value(幾百 bytes) |

「所以 worker 收到指令後執行 `import adapters.outbound.spark.scorer`,從**它自己 image 裡的 `/app`** 讀程式碼(`PYTHONPATH=/app`)。」

### 4.2 由此推出的第一個坑

> **改了 inference-service 的程式碼但沒重建 `spark-app` image,executor 會安靜地跑舊版,而且不會報錯。**

「這是最容易浪費半天的坑。改完程式碼要:

```bash
docker compose build inference-service spark-master
```

`spark-master` 那個 build 產出的就是 `spark-app`,三個 worker 共用同一個 tag。」

### 4.3 模型為什麼只載一次

「模型不能從 driver 送過去 —— `onnxruntime.InferenceSession` 和 FAISS index 都不能 pickle。所以是每個 Python worker 自己建一份,存在模組層級的 `_STATE` 裡。

兩個機制要一起才成立:

- **`_STATE`**(`scorer.py`)—— 模組全域變數,活在 `sys.modules` 裡,跨 task、跨 micro-batch 存活
- **`spark.python.worker.reuse=true`**(在 session 建立時明確釘住)—— 讓 Python 行程本身不被回收

少了後者,每個 task 都會 fork 新行程、重載一次 45MB 的模型。」

【提示】可以順便提一下:如果把 `_predict_fn` 寫成 lambda 或 closure,cloudpickle 就會改成 by value,連 globals 一起複製,`_STATE` 快取會**完全失效**。所以它必須是模組層級的具名函式。

### 4.4 幾份模型副本?

「這題很多人會答錯 —— **不是 partition 數,是 task slot 數。**

```
每個容器的模型副本上限 = SPARK_WORKER_CORES(目前設 3)
```

因為一個併發 task = 一個 Python worker = 一份 ONNX session + FAISS index。partition 數超過 slot 數的部分只是排隊,輪到時直接用已經暖機的行程。

所以 `SPARK_PARTITIONS` 調大很便宜,`SPARK_WORKER_CORES` 才是控制記憶體的旋鈕。」

---

## 5. 失敗語意:什麼會重試、什麼不會(3 分鐘)

「這節接手後一定會用到,因為線上一定會有壞圖。」

| 失敗的東西 | 影響範圍 | 行為 |
|---|---|---|
| 某張圖下載失敗 | 只有那張 | 留下錯誤訊息,其他照跑 |
| 批次前向傳播失敗 | 該 mini-batch | **自動改成逐張重試**以隔離兇手 |
| 某張圖熱力圖/寫入失敗 | 只有那張 | 記錯誤,繼續 |
| task / executor 死掉 | 整個 partition | Spark 重試(預設 4 次),已寫入的會**重複計算** |
| stage-two 發布失敗 | 無 | 記 log;結果列早就存好了 |
| 逃出批次處理的例外 | 整個 micro-batch | **不 commit offset** |

「兩個要特別交代的後果:

**一、失敗的影像會被回報但不會重試。** offset 照樣 commit,所以那張圖就是沒有推論結果。以前有一個 backlog job 可以補掃,已經移除了(你只要 Kafka 事件這一條路)。如果需要補,得從 Postgres 找出沒有 `inference_results` 列的影像重發事件。

**二、重複的結果列是正常的。** `event_id` 每次都是新的 uuid,`ON CONFLICT (event_id)` 對重播完全不會作用。所以查詢時要用 `DISTINCT ON (object_key) ... ORDER BY inferred_at DESC` —— 現有的 stage-two 查詢就是這樣寫的。」

---

## 6. 你會踩的坑(3 分鐘)

「除了前面講的重建 image,還有四個。」

**① 模型升版後 executor 不會自動換。**
`_STATE` 沒有失效機制,而且 `PatchCore` 是 singleton —— 就算重跑 `_executor_state()`,`PatchCore(...)` 也會回傳第一個實例、丟掉新下載的資產。所以在 `inference_model` 把新版本設為 valid 之後,**必須重啟 worker 容器**,否則會繼續用舊模型、而且往 `inference_results` 寫舊的 `model_id`。`refresh` 事件幫不上忙(它現在只剩一行 log)。

**② partition 不保證平均分到三台。** 排程只看哪裡有空閒 slot。6 個 partition 有可能全落在一台,那台就有 3 個 Python worker(受 `SPARK_WORKER_CORES` 限制),另外兩台閒置。要平均得從資源設定下手,Spark 沒有「請散開」的選項。

**③ `SPARK_WORKER_CORES` 是排程上限,不是 CPU 上限。** 那 3 個行程裡的 onnxruntime / FAISS 執行緒仍然可以用滿整台機器。要真的限制 CPU 要用 compose 的 `cpus:`。

**④ ELT 的輸入路徑目前是不一致的。** compose 設 `ELT_INPUT_FOLDER: /app/data`(build 時烤進 image),而 bind mount 是 `./data:/data`。所以往 host 的 `./data` 丟新圖**不重建 image 不會被讀到**。ELT 只是模擬上游的觸發器,但你要餵資料進系統時會第一個撞到這個。改成 `/data` 就好。

---

## 7. 怎麼確認它在正常工作(2 分鐘)

「三個地方看。」

**worker 的 log** —— 這行是最重要的健康指標:

```
Executor ready: PatchCore model_id=... on cpu (pid=1234)
```

「它應該**每個 Python worker 出現一次**(第一批之後大約 3~6 次),然後就再也不出現。如果每個 micro-batch 都出現,代表 `worker.reuse` 沒生效或行程一直在死掉 —— 那你的吞吐量會差一個數量級。」

**driver 的 log**:

```
Dispatching 32 image(s) to Spark across 6 partition(s)
Spark batch complete: 31 scored, 1 failed
```

**Spark master UI**(`localhost:8081`):看 worker 有沒有註冊、每台顯示的核心數應該是 3(不是主機核心數)、以及 application 是否持續佔用著 executor(client mode 的 driver 在服務存活期間會一直佔著)。

**Postgres**:

```sql
SELECT status, count(*) FROM inference_results GROUP BY status;
```

---

## 8. 參數對照(1 分鐘,可當講完後的 handout)

| 變數 | 預設 | 動它會怎樣 |
|---|---|---|
| `SPARK_MICRO_BATCH_SIZE` | 32 | 一次送幾筆。調大 → 吞吐好但 driver 阻塞更久 |
| `SPARK_POLL_TIMEOUT_MS` | 2000 | 湊不滿一批最多等多久 |
| `SPARK_PARTITIONS` | 6 | task 數。調大很便宜(不增加模型副本) |
| `SPARK_UDF_BATCH_SIZE` | 8 | backbone 一次看幾張。調大 → executor 記憶體 ↑ |
| `SPARK_FETCH_WORKERS` | 8 | executor 併發下載數(在 worker 端讀) |
| `SPARK_WORKER_CORES` | 3 | **每台的模型副本上限**,記憶體的真正旋鈕 |
| `MODEL_DEVICE` | cpu | 設 `cuda` 走 GPU(見下節) |

---

## 9. 已知風險與未完成的事(2 分鐘,**誠實交代**)

「這幾件我沒做完或沒驗證過,你接手要知道。」

**① GPU 路徑只有程式碼,沒有硬體驗證過。**
`MODEL_DEVICE=cuda` 的接線做完了(`onnx_providers()` + `_task_device()` 會從 `TaskContext` 取得該 task 分到的 GPU),但這台機器沒有 GPU,所以完全沒跑過。要用還需要:換成 `onnxruntime-gpu` 的 image、compose 給容器 GPU、Spark 宣告 GPU 資源(`spark.executor.resource.gpu.amount` 等三個設定)。注意 `onnxruntime` 和 `onnxruntime-gpu` 是互斥的套件,而 CPU 版**沒有 CUDA provider**,設了也只是安靜地跑在 CPU 上 —— 所以我在載入後加了一行 log 印出實際綁定的 provider,那是你的檢查點。

**② ONNX 的批次軸沒有實測。**
`PatchCore._batch_capable` 會在每個行程第一次批次呼叫時**探測一次**:先看宣告的 batch 維度,若是動態就跑 1 張和 2 張比對輸出列數是否剛好 2 倍。不通過就自動退回逐張前向傳播(FAISS 仍然是整批一次搜尋)。所以它不會壞,但你要在 log 裡確認走的是哪一條:`ONNX backbone batch support: enabled / disabled`。

**③ 下一個瓶頸會是 stage-two,不是 stage-one。**
stage-one 的重運算已經外送了,但 **stage-two 分群是在 inference-service 行程內同步跑的**,而 `collect()` 和分群都佔用同一個 Kafka 消費迴圈。量大之後會以兩種形式出現:stage-one 延遲堆積,或 consumer 超過 `max.poll.interval.ms`(預設 5 分鐘)被踢出 group 觸發 rebalance。真的撞到的話,方向是把 stage-two 也搬離這個行程,或至少獨立成自己的 consumer group。

**④ 橫向擴展目前不能直接做。**
「再開一個 inference-service 分擔 stage-one」在現有設定下有幾個卡點:topic 沒有指定分區數(auto-create 預設 1 個分區,第二個 consumer 會完全閒置)、`container_name` 是固定的(compose 無法 scale)、Spark driver 的 host/port 是寫死的(兩個實例會衝突)、而且沒有設 `spark.cores.max`(第一個 application 會拿走全部核心)。這幾項都可解,但要一起解。

---

## 10. 收尾:接手後建議的前三件事(1 分鐘)

「如果我是你,我會照這個順序做:

1. **跑一次完整流程,盯著 worker 的 log**,確認 `Executor ready` 只出現一次/worker、以及 ONNX 批次探測走哪條路。這會讓你對整條路徑有實感。
2. **修掉 ELT 的輸入路徑**(`ELT_INPUT_FOLDER` 改成 `/data`),否則你餵不進自己的測試資料。
3. **決定模型升版的流程**。目前升版必須重啟 worker,這件事沒有自動化也沒有告警 —— 如果會頻繁換模型,這是第一個該補的洞。

程式碼的入口就三個檔案:`adapters/inbound/inference_consumer.py`(累積與分流)、`application/use_cases/run_stage_one_batch.py`(觸發 stage-two)、`adapters/outbound/spark/scorer.py`(整個 Spark 路徑)。第三個是主體,前兩個各只有幾十行。」

---

## 附錄:一頁流程圖(可印出來)

```
ELT ──► MinIO(images)
    └─► Kafka "inference" (type=stage-one)
              │
              ▼  poll(≤32, 2s)
    ┌─ inference-service:driver ────────────────────────┐
    │  切成 ≤6 partition,只送 (bucket,key,type,size)   │
    │  collect() ← 阻塞                                 │
    └───────────────┬───────────────────────────────────┘
                    ▼  每 partition 一個 task
    ┌─ Spark executor:python worker(每容器 ≤3)──────┐
    │  第一次:_executor_state() 建模型 → _STATE       │
    │  每批 ≤8 張:                                    │
    │    MinIO 下載(8 執行緒併發)                    │
    │    → 堆成 (B,3,224,224) → ONNX ResNet50          │
    │    → 一次 FAISS 搜尋 → buffer zone 判定          │
    │    → 熱力圖 → MinIO                              │
    │    → INSERT inference_results                    │
    │  回傳 (status, score, error)                     │
    └───────────────┬──────────────────────────────────┘
                    ▼
    driver:每個 anomaly → 發 type=stage-two → commit offset
```
