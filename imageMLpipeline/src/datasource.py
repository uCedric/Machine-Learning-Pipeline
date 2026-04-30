import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import logging
import argparse
import time
from utils import plot_history, setup_logger, MODEL_DICT, LOSS_DICT, OPTIMIZER_DICT, parse_scheduler_arg_to_json, validate_scheduler_config
from models import LossFactory, OptimFactory, ModelFactory, Trainer, SchedulerFactory

# Setup logging
logger = setup_logger("main")


def get_data_loaders(data_dir):
    """Prepare data loaders for preprocessed images."""
    # Only convert to Tensor as images are already preprocessed on disk
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    # Split into train/validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader, len(full_dataset.classes)

def check_data_source(data_dir):
    """Step 1: Check the source of image files."""
    logger.info(f"Checking data source at: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found.")

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not classes:
        raise ValueError(f"No label directories found in {data_dir}")

    logger.info(f"Found {len(classes)} classes: {classes}")
    return classes
