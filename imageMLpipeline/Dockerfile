# Use Python 3.10 slim image (Bullseye to ensure OpenJDK 11 availability)
FROM python:3.10-slim-bullseye

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install OpenJDK 11 and other utilities
RUN apt-get update || (sleep 5 && apt-get update) && \
    mkdir -p /usr/share/man/man1 && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    procps \
    curl \
    vim \
    iputils-ping \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME to OpenJDK 11
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyspark==3.4.0 \
    pyarrow \
    pandas \
    Pillow \
    numpy \
    torch \
    torchvision \
    matplotlib \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the project files
COPY ./src ./src
COPY ./utils ./utils
COPY ./models ./models
COPY main.py .
COPY run.sh .

# Set PYTHONPATH
ENV PYTHONPATH=/app

RUN chmod +x run.sh

CMD ["bash", "run.sh"]
