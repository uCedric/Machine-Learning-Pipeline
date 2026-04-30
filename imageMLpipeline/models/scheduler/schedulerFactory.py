from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils.constants import SCHEDULER_DICT


class SchedulerFactory:
    """Factory class to create learning rate scheduler instances."""

    _REGISTRY = {
        SCHEDULER_DICT["STEP_LR"]: lambda optimizer, **kw: StepLR(optimizer, **kw),
        SCHEDULER_DICT["COSINE_ANNEALING_LR"]: lambda optimizer, **kw: CosineAnnealingLR(optimizer, **kw),
    }

    @staticmethod
    def get_scheduler(scheduler_type, optimizer, **kwargs):
        key = scheduler_type.lower()
        if key not in SchedulerFactory._REGISTRY:
            raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")
        return SchedulerFactory._REGISTRY[key](optimizer, **kwargs)
