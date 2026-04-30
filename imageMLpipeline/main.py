"""
ML Pipeline — Chain of Responsibility

Chain:
  ParameterIngestion → Validation → DataPreparation → FeatureEngineering
  → ModelTraining → Evaluation → Deployment
"""

import os
import sys
import argparse
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Optional

import torch
from pyspark.sql.functions import col

from utils import (
    setup_logger,
    plot_history,
    parse_scheduler_arg_to_json,
    validate_scheduler_config,
    MODEL_DICT, LOSS_DICT, OPTIMIZER_DICT,
)
from models import ModelFactory, LossFactory, OptimFactory, SchedulerFactory, Trainer
from imageMLpipeline.src.datasource import get_data_loaders, check_data_source
from src.preprocessing import get_spark_session, create_dummy_data, process_images_udf, save_to_disk

logger = setup_logger("pipeline")


# ── Abstract Base ─────────────────────────────────────────────────────────────

class PipelineHandler(ABC):
    def __init__(self):
        self._next: Optional["PipelineHandler"] = None

    def set_next(self, handler: "PipelineHandler") -> "PipelineHandler":
        self._next = handler
        return handler

    def handle(self, context: dict) -> dict:
        context = self.process(context)
        if self._next:
            return self._next.handle(context)
        return context

    @abstractmethod
    def process(self, context: dict) -> dict:
        pass


# ── Stage 1: Parameter Ingestion ─────────────────────────────────────────────

class ParameterIngestionHandler(PipelineHandler):
    def process(self, context: dict) -> dict:
        parser = argparse.ArgumentParser(description="ML Pipeline")
        parser.add_argument("--model_type",    type=str,   default="resnet18")
        parser.add_argument("--loss_function", type=str,   default="cross_entropy")
        parser.add_argument("--optimizer",     type=str,   default="adam")
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--scheduler",     type=str,   default=None,
                            help='JSON scheduler config, e.g. \'{"method":"step_lr","step_size":5,"gamma":0.1}\'')
        parser.add_argument("--input_path",    type=str,   default="/app/parquet_output/")
        parser.add_argument("--data_dir",      type=str,   default="/app/processed_images")
        parser.add_argument("--epochs",        type=int,   default=10)
        parser.add_argument("--model_path",    type=str,   default="/app/models/model.pth")

        context.update(vars(parser.parse_args()))
        logger.info("[1/7] Parameter ingestion complete.")
        return context


# ── Stage 2: Validation ───────────────────────────────────────────────────────

class ValidationHandler(PipelineHandler):
    def process(self, context: dict) -> dict:
        model_type    = context.get("model_type", "")
        loss_fn       = context.get("loss_function", "")
        optimizer     = context.get("optimizer", "")
        scheduler_raw = context.get("scheduler")

        if model_type.upper() not in MODEL_DICT:
            raise ValueError(f"Unsupported model: '{model_type}'. Supported: {list(MODEL_DICT.values())}")
        if loss_fn.upper() not in LOSS_DICT:
            raise ValueError(f"Unsupported loss: '{loss_fn}'. Supported: {list(LOSS_DICT.values())}")
        if optimizer.upper() not in OPTIMIZER_DICT:
            raise ValueError(f"Unsupported optimizer: '{optimizer}'. Supported: {list(OPTIMIZER_DICT.values())}")

        scheduler_config = parse_scheduler_arg_to_json(scheduler_raw, arg_name="scheduler")
        if scheduler_config is not None:
            method, kwargs = validate_scheduler_config(scheduler_config)
            context["scheduler_method"] = method
            context["scheduler_kwargs"] = kwargs
        else:
            context["scheduler_method"] = None
            context["scheduler_kwargs"] = {}

        logger.info("[2/7] Validation complete.")
        return context


# ── Stage 3: Data Preparation ─────────────────────────────────────────────────

class DataPreparationHandler(PipelineHandler):
    def process(self, context: dict) -> dict:
        input_path  = context["input_path"]
        output_path = context["data_dir"]

        spark = get_spark_session()
        context["spark"] = spark

        try:
            df_raw = spark.read.parquet(input_path)
            logger.info(f"Loaded partitioned data from {input_path}.")
        except Exception:
            logger.warning("No parquet data found — generating dummy data.")
            create_dummy_data(spark, input_path)
            df_raw = spark.read.parquet(input_path)

        df_raw.createOrReplaceTempView("wafer_images_table")
        df_sql = spark.sql("SELECT path, label, content FROM wafer_images_table")

        processed_df = df_sql.withColumn(
            "processed_image_data", process_images_udf(col("content"))
        )
        processed_df.select("path", "label", "processed_image_data").foreachPartition(save_to_disk)

        logger.info(f"[3/7] Data preparation complete. Images written to {output_path}.")
        return context


# ── Stage 4: Feature Engineering ─────────────────────────────────────────────

class FeatureEngineeringHandler(PipelineHandler):
    def process(self, context: dict) -> dict:
        check_data_source(context["data_dir"])
        train_loader, val_loader, num_classes = get_data_loaders(context["data_dir"])

        context["train_loader"] = train_loader
        context["val_loader"]   = val_loader
        context["num_classes"]  = num_classes

        logger.info(
            f"[4/7] Feature engineering complete. "
            f"{num_classes} classes | "
            f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples."
        )
        return context


# ── Stage 5: Model Training ───────────────────────────────────────────────────

class ModelTrainingHandler(PipelineHandler):
    def process(self, context: dict) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model     = ModelFactory.get_model(context["model_type"], num_classes=context["num_classes"], device=device.type)
        criterion = LossFactory.get_loss(context["loss_function"])
        optimizer = OptimFactory.get_optim(context["optimizer"], params=model.parameters(), lr=context["learning_rate"])
        scheduler = SchedulerFactory.get_scheduler(context["scheduler_method"], optimizer, **context["scheduler_kwargs"]) \
            if context["scheduler_method"] else None

        trainer = Trainer(model, criterion, optimizer, device.type, scheduler=scheduler)
        trainer.train(context["train_loader"], context["val_loader"], num_epochs=context["epochs"])

        context["model"]   = model
        context["trainer"] = trainer
        context["device"]  = device

        logger.info("[5/7] Model training complete.")
        return context


# ── Stage 6: Evaluation ───────────────────────────────────────────────────────

class EvaluationHandler(PipelineHandler):
    def process(self, context: dict) -> dict:
        trainer: Trainer = context["trainer"]
        final_loss, final_acc = trainer.validation_eval(context["val_loader"])

        context["final_loss"] = final_loss
        context["final_acc"]  = final_acc

        logger.info(f"[6/7] Evaluation — Loss: {final_loss:.4f} | Accuracy: {final_acc:.4f}")
        return context


# ── Stage 7: Deployment ───────────────────────────────────────────────────────

class DeploymentHandler(PipelineHandler):
    def process(self, context: dict) -> dict:
        model_path = context["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        torch.save(context["model"].state_dict(), model_path)
        plot_history(context["trainer"].history)

        logger.info(f"[7/7] Deployment complete. Model saved to {model_path}.")
        return context


# ── Pipeline Assembly ─────────────────────────────────────────────────────────

def build_pipeline() -> PipelineHandler:
    ingestion  = ParameterIngestionHandler()
    validation = ValidationHandler()
    data_prep  = DataPreparationHandler()
    features   = FeatureEngineeringHandler()
    training   = ModelTrainingHandler()
    evaluation = EvaluationHandler()
    deployment = DeploymentHandler()

    ingestion \
        .set_next(validation) \
        .set_next(data_prep) \
        .set_next(features) \
        .set_next(training) \
        .set_next(evaluation) \
        .set_next(deployment)

    return ingestion


def main():
    pipeline = build_pipeline()
    try:
        pipeline.handle({})
    except Exception as e:
        logger.error(f"Pipeline failed at: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
