import threading
import torch.nn as nn
from torchvision import models
from utils.logger import get_logger


class ResNet18(nn.Module):
    _instance = None
    _lock = threading.Lock()
    _allow_instantiation = False

    DEFAULT_NUM_CLASSES = 5
    DEFAULT_WEIGHTS = 'DEFAULT'

    def __init__(self, num_classes=None, weights=None):
        if not ResNet18._allow_instantiation:
            raise RuntimeError(
                "Direct instantiation is blocked. Use ResNet18.get_instance(...) instead."
            )

        super(ResNet18, self).__init__()
        self.logger = get_logger(self)

        self.num_classes = num_classes if num_classes is not None else self.DEFAULT_NUM_CLASSES
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS

        self.model = models.resnet18(weights=self.weights)
        self.in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.in_features, self.num_classes)

    @classmethod
    def _get_resnet18(cls, num_classes=None, weights=None, device='cpu'):
        cls._allow_instantiation = True
        try:
            model = cls(num_classes=num_classes, weights=weights)
            model = model.to(device)
        finally:
            cls._allow_instantiation = False
        return model

    @classmethod
    def get_instance(cls, num_classes=None, weights=None, device='cpu'):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._get_resnet18(num_classes, weights, device)
        else:
            expected = cls._instance.num_classes
            if num_classes is not None and num_classes != expected:
                raise RuntimeError(
                    f"Singleton already initialized with num_classes={expected}, "
                    f"got {num_classes}."
                )
        return cls._instance

    def forward(self, x):
        return self.model(x)

    def inference(self, x):
        if len(list(self.model.parameters())) == 0:
            raise RuntimeError(
                "Model parameters/weights are not initialized. "
                "Ensure the model is trained with Trainer.train() or pre-trained."
            )
        return self(x)
