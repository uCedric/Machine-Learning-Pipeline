FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./
# torch/torchvision first, from the CPU wheel index (see requirements.txt NOTE):
# the plain PyPI wheels carry the CUDA build plus multi-GB nvidia-* deps this
# CPU-only image never uses. The pinned versions then satisfy requirements.txt.
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY domain/ ./domain/
COPY application/ ./application/
COPY adapters/ ./adapters/
COPY config/ ./config/
COPY bootstrap/ ./bootstrap/

# The command is provided per-service in docker-compose.yml:
#   python -m bootstrap.elt
#   python -m bootstrap.inference
