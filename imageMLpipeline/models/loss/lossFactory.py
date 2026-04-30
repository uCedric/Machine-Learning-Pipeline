from torch.nn import CrossEntropyLoss
from utils.constants import LOSS_DICT


class LossFactory:
    """Factory class to create loss function instances."""

    _REGISTRY = {
        LOSS_DICT["CROSS_ENTROPY"]: lambda: CrossEntropyLoss(),
    }

    @staticmethod
    def get_loss(loss_type):
        key = loss_type.lower()
        if key not in LossFactory._REGISTRY:
            raise ValueError(f"Loss function type '{loss_type}' is not supported.")
        return LossFactory._REGISTRY[key]()
