#!/bin/bash
set -e

export PYTHONPATH=/app

# --- ML Training Parameters (override via environment or docker-compose) ---
MODEL_TYPE='resnet18'
LOSS_FUNCTION='cross_entropy'
OPTIMIZER='adam'
LEARNING_RATE=0.001
EPOCHS=50
SCHEDULER='{"method": "step_lr", "step_size": 10, "gamma": 0.1}'
INPUT_PATH='/app/parquet_output/'
DATA_DIR='/app/processed_images'
MODEL_PATH='/app/models/model.pth'

echo "Starting ML Pipeline..."
echo "  model_type    : $MODEL_TYPE"
echo "  loss_function : $LOSS_FUNCTION"
echo "  optimizer     : $OPTIMIZER"
echo "  learning_rate : $LEARNING_RATE"
echo "  epochs        : $EPOCHS"
echo "  scheduler     : ${SCHEDULER:-none}"
echo "  input_path    : $INPUT_PATH"
echo "  data_dir      : $DATA_DIR"
echo "  model_path    : $MODEL_PATH"

python3 /app/main.py \
  --model_type    "$MODEL_TYPE" \
  --loss_function "$LOSS_FUNCTION" \
  --optimizer     "$OPTIMIZER" \
  --learning_rate "$LEARNING_RATE" \
  --epochs        "$EPOCHS" \
  --input_path    "$INPUT_PATH" \
  --data_dir      "$DATA_DIR" \
  --model_path    "$MODEL_PATH" \
  ${SCHEDULER:+--scheduler "$SCHEDULER"}
