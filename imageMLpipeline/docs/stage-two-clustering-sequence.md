# Stage-two 分群 sequence diagram

`inference-service` 收到 `type=stage-two` 事件後的實際互動。大多數事件只會走到閘門就返回。

```mermaid
sequenceDiagram
    autonumber
    participant K as Kafka<br/>topic inference
    participant C as InferenceConsumer
    participant U as RunStageTwo
    participant RL as ClusteringRunLog<br/>stage_two_run
    participant IS as ClusteringImageSource<br/>inference_results
    participant S3 as ObjectStorage<br/>MinIO
    participant M as ResNet50ClusterModel
    participant REG as ModelRegistry<br/>inference_model
    participant CR as ClusterResultRepository<br/>cluster_results

    K->>C: inference 事件(type=stage-two)
    C->>U: execute(event)

    rect rgb(238, 244, 252)
    Note over U, IS: ① 閘門:兩個累積量,兩條 watermark
    U->>RL: last_watermark("fit")
    RL-->>U: 上次 fit 的 covered_through
    U->>IS: backlog_since(fit_watermark)
    IS-->>U: since_fit, newest_fit
    end

    alt since_fit ≥ REFIT_COUNT(1000)
        rect rgb(252, 246, 236)
        Note over U, CR: ② FIT:全集重新分群並凍結成新版本
        U->>IS: anomalous(DEFECT_SET_SIZE 5000)
        IS-->>U: 缺陷影像位置(最新的 N 筆)
        U->>S3: get_object × N(ThreadPool 16)
        S3-->>U: 影像 bytes

        alt 取得的張數 < MIN_BATCH(15)
            U-->>C: 記 WARNING,watermark 不前進
        else 足夠
            U->>IS: recent_good(patchcore, 200)
            IS-->>U: 良品影像位置
            U->>S3: get_object × 200
            S3-->>U: 影像 bytes

            U->>M: fit(defect_bytes, normal_bytes)
            Note right of M: ① 良品 → prototype(mu, std)<br/>② 每張 → deep 1536 + color 5 + shape 5<br/>③ block_prep → PCA → UMAP → HDBSCAN<br/>④ 第二層沿 ecc 補一刀<br/>⑤ 七個物件凍結成 state<br/>⑥ self.state = live_state(立刻自我採用)
            M-->>U: ClusterFit(assignments, state)

            U->>REG: next_minor("resnet50")
            REG-->>U: 新版本 1.N+1.0(尚未寫入)
            U->>S3: put cluster_state.joblib
            Note right of S3: artifact 先寫,registry 後寫<br/>否則會有指向不存在 prefix 的列
            U->>REG: register(新版本, is_valid=true)
            U->>M: model_id ← 新版本
            U->>CR: save × N(cluster_results)
            U->>RL: record(mode=fit, labelled / noise / mean_prob)
        end
        end

    else 沒有凍結狀態可用
        U-->>C: 記 log「fit 閘門 x/1000,等待」<br/>assign 不可能執行
    else
        rect rgb(240, 248, 240)
        Note over U, CR: ③ ASSIGN:只把新缺陷投影進既有分群
        U->>RL: last_watermark()（任何模式）
        RL-->>U: 上次任一 run 的 covered_through
        U->>IS: backlog_since(run_watermark)
        IS-->>U: since_run, newest_run

        alt since_run ≥ ASSIGN_COUNT(500)
            U->>IS: anomalous_since(watermark, 5000)
            IS-->>U: 只有新到的缺陷影像
            U->>S3: get_object × N
            S3-->>U: 影像 bytes

            U->>M: assign(defect_bytes)
            Note right of M: 全部使用凍結的那一份:<br/>prototype → scalers → PCA<br/>→ UMAP.transform<br/>→ hdbscan.approximate_predict<br/>→ 凍結的第二層切分決策
            M-->>U: assignments(cluster_id, probability)

            U->>CR: save × N(cluster_results)
            U->>RL: record(mode=assign, labelled / noise / mean_prob)
            Note over RL: noise 比例是 drift 訊號<br/>monitor 會拿它跟 fit 的基準比
        else
            U-->>C: 記 log「assign 閘門 x/500,等待」
        end
        end
    end

    U-->>C: 返回
    C->>K: commit offset（整個 micro-batch 處理完後）
```

## 兩條 watermark 為什麼要分開

`assign` 必須推進「任何 run」那條線,否則會重複歸類同一批。但如果它同時推進 fit 那條,
fit 的計數就會每 500 張歸零一次 —— 永遠到不了 1000,分群永遠不會重畫,新的缺陷型態也就
永遠無法形成新 cluster。

```mermaid
sequenceDiagram
    autonumber
    participant A as assign run
    participant W1 as watermark<br/>(任何模式)
    participant W2 as watermark<br/>(僅 fit)
    participant F as fit run

    A->>W1: 前進到 covered_through
    A--xW2: 不動 ← 關鍵
    Note over W2: fit 的計數繼續累積<br/>不會被 assign 歸零
    F->>W1: 前進
    F->>W2: 前進
```
