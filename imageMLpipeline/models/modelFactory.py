from .ResNet18 import ResNet18
from utils.constants import MODEL_DICT


class ModelFactory:
    """Factory class to create model instances."""

    _REGISTRY = {
        MODEL_DICT["RESNET18"]: ResNet18,
    }

    @staticmethod
    def get_model(model_type, **kwargs):
        key = model_type.lower()
        if key not in ModelFactory._REGISTRY:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        return ModelFactory._REGISTRY[key].get_instance(**kwargs)
