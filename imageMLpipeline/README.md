Reference to UBER's study:
https://www.uber.com/tw/zh-tw/blog/accelerating-deep-learning/
  -horovod https://www.youtube.com/watch?v=jbbnZIpCu-U
  -Ray™

---

# Spark Image ML Pipeline

An end-to-end image classification pipeline combining **Apache Spark** distributed preprocessing with **PyTorch** model training, orchestrated via a **Chain of Responsibility** design pattern.

---

## Architecture

The pipeline is driven by `main.py`, which chains seven sequential handlers. Each handler reads from and writes to a shared `context` dict, then passes it to the next stage.

```
ParameterIngestion → Validation → DataPreparation → FeatureEngineering
                  → ModelTraining → Evaluation → Deployment
```

| Stage | Handler | Responsibility |
|---|---|---|
| 1 | `ParameterIngestionHandler` | Parse CLI arguments into context |
| 2 | `ValidationHandler` | Validate model, loss, optimizer, and scheduler config |
| 3 | `DataPreparationHandler` | Spark preprocessing — load parquet, apply Pandas UDF, save JPEGs |
| 4 | `FeatureEngineeringHandler` | Build PyTorch DataLoaders with normalization transforms |
| 5 | `ModelTrainingHandler` | Initialize factories and run training loop with optional scheduler |
| 6 | `EvaluationHandler` | Compute final validation loss and accuracy |
| 7 | `DeploymentHandler` | Save model weights and plot training history |

---

## Project Structure

```
imageMLpipeline/
├── main.py                            # Pipeline orchestrator (Chain of Responsibility)
├── run.sh                             # Entrypoint — sets env vars and launches main.py
├── Dockerfile                         # App container (Python + PySpark + PyTorch)
├── Dockerfile.spark                   # Spark master/worker container
├── docker-compose.yml                 # Spark cluster + app service wiring
│
├── src/
│   ├── preprocessing.py               # Spark pipeline — parquet ingestion, Pandas UDF, JPEG output
│   ├── datasource.py                  # DataLoader construction and data directory validation
│   └── trainer.py                     # Trainer class — training loop, validation eval, scheduler step
│
├── models/
│   ├── ResNet18.py                    # Thread-safe singleton ResNet18 model
│   ├── modelFactory.py                # Model factory
│   ├── loss/lossFactory.py            # Loss function factory
│   ├── optim/optimFactory.py          # Optimizer factory
│   └── scheduler/schedulerFactory.py  # LR scheduler factory
│
├── utils/
│   ├── constants.py                   # Supported types, required scheduler params
│   ├── validators.py                  # Scheduler config middleware validator
│   ├── json.py                        # JSON CLI argument parser
│   ├── logger.py                      # Logger setup
│   └── plot.py                        # Training history plots
│
└── .claude/commands/                  # Claude Code skills for extending the pipeline
    ├── add-model.md                   # /add-model — add a new model following factory + singleton pattern
    ├── add-scheduler.md               # /add-scheduler — add a new LR scheduler
    └── add-stage.md                   # /add-stage — add a new CoR stage to the pipeline
```

---

## Design Patterns

- **Chain of Responsibility** (`main.py`) — pipeline stages are decoupled handlers linked at runtime; inserting a new stage requires only a new handler class and a `set_next()` call
- **Factory** (`modelFactory`, `lossFactory`, `optimFactory`, `schedulerFactory`) — object creation is centralized; adding support for a new type only requires updating `constants.py` and the factory registry
- **Singleton** (`ResNet18`) — prevents duplicate model instantiation; raises `RuntimeError` if called again with a conflicting `num_classes`
- **Validation Middleware** (`utils/validators.py`) — scheduler config flows through three distinct layers: parse (`json.py`) → validate (`validators.py`) → construct (factory)

---

## Prerequisites

- Docker installed
- (Optional) Spark Master/Worker running — configured at `spark://spark-master:7077` in `docker-compose.yml`

---

## Running the Pipeline

### Build

```bash
docker compose build
```

### Run

```bash
docker compose up
```

The `app` service executes `run.sh`, which sets training parameters as environment variables and launches `main.py`.

---

## Configuration

Training parameters are set as environment variables in `run.sh` and forwarded as CLI arguments to `main.py`:

| Variable | Default | Argument |
|---|---|---|
| `MODEL_TYPE` | `resnet18` | `--model_type` |
| `LOSS_FUNCTION` | `cross_entropy` | `--loss_function` |
| `OPTIMIZER` | `adam` | `--optimizer` |
| `LEARNING_RATE` | `0.001` | `--learning_rate` |
| `EPOCHS` | `50` | `--epochs` |
| `SCHEDULER` | `{"method":"step_lr","step_size":10,"gamma":0.1}` | `--scheduler` |
| `INPUT_PATH` | `/app/parquet_output/` | `--input_path` |
| `DATA_DIR` | `/app/processed_images` | `--data_dir` |
| `MODEL_PATH` | `/app/models/model.pth` | `--model_path` |

Override any variable via `docker compose run` or the `environment` block in `docker-compose.yml`.

### Scheduler Configuration

The `--scheduler` argument accepts a JSON string. The `method` key selects the scheduler; all remaining keys are forwarded as constructor parameters.

```bash
# StepLR — required: step_size
--scheduler '{"method":"step_lr","step_size":5,"gamma":0.1}'

# CosineAnnealingLR — required: T_max
--scheduler '{"method":"cosine_annealing_lr","T_max":10,"eta_min":0}'

# No scheduler
# omit --scheduler entirely
```

Validation is handled by `utils/validators.py`. Missing required parameters raise a descriptive `ValueError` before model initialization.

---

## Extending the Pipeline

Use the built-in Claude Code skills for consistent extensions:

| Skill | Example | What it does |
|---|---|---|
| `/add-model` | `/add-model ResNet50` | Adds a model following the singleton + factory pattern |
| `/add-scheduler` | `/add-scheduler ReduceLROnPlateau` | Adds a scheduler and registers its required params |
| `/add-stage` | `/add-stage EarlyStoppingHandler` | Adds a CoR stage and wires it into the chain |

---

## Stack

| Component | Technology |
|---|---|
| Distributed preprocessing | Apache Spark, Pandas UDFs, Apache Arrow |
| Image processing | Pillow |
| Model training | PyTorch, torchvision (ResNet18) |
| Containerization | Docker, docker-compose |
| Logging | Python `logging` |

---

## Debugging

If the Spark session fails, `spark_error.log` is written inside the container:

```bash
docker ps
docker exec -it <container_id> /bin/bash
cat spark_error.log
ls -R /app/processed_images
```
