FROM python:3.11-slim

# The service hosts the Spark *driver* in client mode (stage-one scoring is
# dispatched to the cluster), and a driver needs a JVM even though the model
# runs in Python on the executors. procps supplies the `ps` Spark's launcher
# scripts call.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONPATH=/app \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

WORKDIR /app

COPY requirements.txt ./
# torch/torchvision first, from the CPU wheel index (see requirements.txt NOTE):
# the plain PyPI wheels carry the CUDA build plus multi-GB nvidia-* deps this
# CPU-only image never uses. The pinned versions then satisfy requirements.txt.
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the stage-two ResNet50 ImageNet checkpoint into the image so the
# clustering backbone loads offline at runtime (no download on the first event).
ENV TORCH_HOME=/opt/torch
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT)"

COPY data/ ./data/
COPY domain/ ./domain/
COPY application/ ./application/
COPY adapters/ ./adapters/
COPY config/ ./config/
COPY bootstrap/ ./bootstrap/

# The command is provided per-service in docker-compose.yml:
#   python -m bootstrap.elt         (ELT ingest)
#   python -m bootstrap.inference   (both inference stages, routed by event 'type')
