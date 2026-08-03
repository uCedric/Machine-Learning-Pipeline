# Spark cluster image (master + workers + the stage-one batch job).
#
# Built on python:3.11-slim rather than apache/spark-py:v3.4.0 because the batch
# job imports the inference-service hexagon, and that code needs Python 3.11
# (domain/models.py uses enum.StrEnum). Spark also requires the driver and the
# executors to run the *same* Python version, so the whole cluster moves to 3.11
# together. Spark 3.4 supports Python up to 3.11.
FROM python:3.11-slim

# Spark 3.4 runs on Java 8/11/17; the headless JRE is all the daemons need.
# procps supplies the `ps` that Spark's launcher scripts call.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONPATH=/app \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

# The executors run the *same* PatchCore the inference service runs, so they
# install the *same* pins — one requirements file, no version drift between the
# live path and the batch path. torch/torchvision come from the CPU wheel index
# first (see the NOTE in that file): the plain PyPI wheels carry the CUDA build
# plus multi-GB nvidia-* deps this CPU-only cluster never uses.
COPY inference-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pyspark==3.4.0 pandas==2.2.2 pyarrow==16.1.0

# pip ships the full Spark distribution inside the pyspark package. Symlink it
# to the conventional location so SPARK_HOME and the docker-compose commands
# (/opt/spark/bin/spark-class ...) keep working unchanged.
RUN ln -s /usr/local/lib/python3.11/site-packages/pyspark /opt/spark
ENV SPARK_HOME=/opt/spark \
    PATH=/opt/spark/bin:$PATH \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

# The stage-one hexagon, so executors build the same model from the same code.
COPY inference-service/domain/ ./domain/
COPY inference-service/application/ ./application/
COPY inference-service/adapters/ ./adapters/
COPY inference-service/config/ ./config/
COPY inference-service/bootstrap/ ./bootstrap/

# Retained Spark Structured Streaming demo (not wired into the pipeline).
COPY tasks/ ./tasks/
COPY jobs/ ./jobs/

# The command is provided per-service in docker-compose.yml:
#   spark-class org.apache.spark.deploy.master.Master   (master)
#   spark-class org.apache.spark.deploy.worker.Worker   (workers)
# No driver runs in these containers — inference-service hosts it in client
# mode, and the executors import the hexagon copied above.
