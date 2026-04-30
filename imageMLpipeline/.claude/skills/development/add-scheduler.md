---
description: Add a new PyTorch LR scheduler(learning rate scheduler) to the pipeline using the existing factory pattern. Usage: /add-scheduler <SchedulerName>
---

Add a new PyTorch LR scheduler to the pipeline following the existing factory pattern.

The scheduler to add is: $ARGUMENTS

## Steps

### 1. Register in `utils/constants.py`
- Add entry to `SCHEDULER_DICT`, e.g. `"REDUCE_LR_ON_PLATEAU": "reduce_lr_on_plateau"`
- Add entry to `SCHEDULER_REQUIRED_PARAMS` listing params that must be present in the JSON config, e.g. `"reduce_lr_on_plateau": ["mode", "factor"]`

### 2. Register in `models/scheduler/schedulerFactory.py`
- Import the scheduler class from `torch.optim.lr_scheduler`
- Add a lambda entry to `_REGISTRY` using the `SCHEDULER_DICT` key

### 3. Verify validation middleware
Confirm `utils/validators.py` picks up the new entry automatically via `SCHEDULER_DICT.values()` and `SCHEDULER_REQUIRED_PARAMS` — no changes needed there.

### 4. Update `run.sh`
Document a default config comment for the new scheduler if applicable.

## Verification

Confirm the scheduler is usable via:
```bash
--scheduler '{"method":"<key>","<required_param>": <value>}'
```
And that omitting a required param raises a clear `ValueError` from `validate_scheduler_config()`.
