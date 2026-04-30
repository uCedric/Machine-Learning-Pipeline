import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.logger import get_logger


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, device: str = 'cpu', scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.history = None
        self.logger = get_logger(self)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int = 10):
        try:
            self.model.train(True)
            self.logger.info(f"Starting training on {self.device} for {num_epochs} epochs...")
            self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(num_epochs):
                self.model.train()
                running_loss = 0.0
                corrects = 0

                for inputs, labels in train_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(train_dataloader.dataset)
                epoch_acc = corrects.double() / len(train_dataloader.dataset)

                val_loss, val_acc = self.validation_eval(val_dataloader)

                if self.scheduler is not None:
                    self.scheduler.step()

                self.history['train_loss'].append(epoch_loss)
                self.history['train_acc'].append(epoch_acc.item())
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc.item())

                self.logger.info(
                    f'Epoch {epoch}/{num_epochs - 1}: '
                    f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | '
                    f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}'
                )
        finally:
            self.model.train(False)

    def validation_eval(self, loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

        return running_loss / len(loader.dataset), corrects.double() / len(loader.dataset)
