---
name: development
description: Principles and index for extending the imageMLpipeline — models, schedulers, and pipeline stages
---

This folder contains skills for adding new components to the pipeline. Before running any skill, read the relevant file and follow its conventions exactly.

## Available Skills

- [add-model.md](add-model.md) — `/add-model <ModelName>`: Add a PyTorch model using the singleton + factory pattern
- [add-scheduler.md](add-scheduler.md) — `/add-scheduler <SchedulerName>`: Add an LR scheduler using the factory + validation pattern
- [add-stage.md](add-stage.md) — `/add-stage <StageName>`: Add a stage to the Chain of Responsibility in `main.py`

## Principles

**Match existing patterns exactly.** Each skill names a reference file. Replicate its structure, naming, and guards — do not invent alternatives.

**Constants first.** Every new component is declared in `utils/constants.py` before being wired elsewhere. Constants are the single source of truth for valid identifiers.

**Factories resolve everything.** Models and schedulers are instantiated only through their factory `_REGISTRY`. Callers never construct components directly.

**Thread-safe singletons for models.** Models use a locked singleton (`_instance`, `_lock`, `_allow_instantiation`). Calling `get_instance` again with a different `num_classes` raises `RuntimeError` immediately.

**Validation is automatic.** `utils/validators.py` reads `SCHEDULER_DICT` and `SCHEDULER_REQUIRED_PARAMS` directly — adding to constants is sufficient, no parallel validation logic needed.

**Stages share state only through `context`.** Handlers read from and write to the `context` dict. Exceptions propagate to `main()` — never caught inside `process()`.

**Log numbering must stay consistent.** Stage logs follow `[N/Total] Stage name complete.` — if the total changes, update all existing messages.

**Documentation is part of the change.** A new stage is not complete until the README architecture table and chain diagram are updated.
