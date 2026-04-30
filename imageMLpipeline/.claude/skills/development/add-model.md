---
description: Add a new PyTorch model to the pipeline using the singleton + factory pattern. Usage: /add-model <ModelName>
---

Add a new PyTorch model to the pipeline following the existing singleton + factory pattern.

The model name to add is: $ARGUMENTS

## Steps

### 1. Create the model file at `models/<ModelName>.py`
- Extend `nn.Module`
- Implement the thread-safe singleton pattern matching `models/ResNet18.py` exactly:
  - `_instance`, `_lock`, `_allow_instantiation` class vars
  - Block direct instantiation via `__init__` guard
  - Expose `get_instance(num_classes, weights, device)` classmethod
  - Raise `RuntimeError` if `get_instance` is called again with a different `num_classes`
- Replace the final fully-connected layer to match `num_classes`
- Implement `forward(self, x)` and `inference(self, x)`

### 2. Register in `utils/constants.py`
- Add entry to `MODEL_DICT`, e.g. `"RESNET50": "resnet50"`

### 3. Register in `models/modelFactory.py`
- Import the new class
- Add to `_REGISTRY` using the `MODEL_DICT` key

### 4. Export from `models/__init__.py`
- Add import line for the new class

## Verification

Confirm the new model is reachable via:
```python
ModelFactory.get_model("<name>", num_classes=N, device="cpu")
```
