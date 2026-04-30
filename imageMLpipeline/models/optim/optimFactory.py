from torch.optim import Adam
from utils.constants import OPTIMIZER_DICT


class OptimFactory:
    """Factory class to create optimizer instances."""

    _REGISTRY = {
        OPTIMIZER_DICT["ADAM"]: lambda **kw: Adam(**kw),
    }

    @staticmethod
    def get_optim(optim_type, **kwargs):
        key = optim_type.lower()
        if key not in OptimFactory._REGISTRY:
            raise ValueError(f"Optimizer type '{optim_type}' is not supported.")
        return OptimFactory._REGISTRY[key](**kwargs)
