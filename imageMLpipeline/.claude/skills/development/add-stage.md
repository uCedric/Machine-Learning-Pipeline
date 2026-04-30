---
description: Add a new stage to the ML pipeline's Chain of Responsibility in main.py. Usage: /add-stage <StageName>
---

Add a new stage to the ML pipeline's Chain of Responsibility in `main.py`.

The stage to add is: $ARGUMENTS

## Steps

### 1. Create the handler class in `main.py`
- Extend `PipelineHandler`
- Implement `process(self, context: dict) -> dict`
- Read required inputs from `context` by key
- Write outputs back into `context` before returning
- Log progress with a stage number and label: `logger.info("[N/7] Stage name complete.")`
- Do not catch exceptions inside `process` — let them propagate to `main()`

### 2. Wire into `build_pipeline()`
- Instantiate the new handler
- Insert it into the `.set_next()` chain at the correct position
- Update the total stage count in all existing log messages if the chain grows

### 3. Update the README
- Add a row to the Architecture stage table
- Update the chain diagram at the top

## Context Keys Available Per Stage

| After Stage | Keys Added |
|---|---|
| ParameterIngestion | `model_type`, `loss_function`, `optimizer`, `learning_rate`, `scheduler`, `input_path`, `data_dir`, `epochs`, `model_path` |
| Validation | `scheduler_method`, `scheduler_kwargs` |
| DataPreparation | `spark` |
| FeatureEngineering | `train_loader`, `val_loader`, `num_classes` |
| ModelTraining | `model`, `trainer`, `device` |
| Evaluation | `final_loss`, `final_acc` |
