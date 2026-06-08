FROM apache/spark-py:v3.4.0
USER root

RUN pip install --no-cache-dir Pillow numpy pandas pyarrow

WORKDIR /app

COPY tasks/ /app/tasks/
COPY jobs/ /app/jobs/

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH