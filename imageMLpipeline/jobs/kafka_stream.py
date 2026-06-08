"""
Spark Structured Streaming job — reads from Kafka and prints to console.
Kafka broker  : kafka:29092  (inter-container listener)
Topic         : controlled by KAFKA_TOPIC env var (default: test-topic)
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType

KAFKA_BOOTSTRAP   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC       = os.getenv("KAFKA_TOPIC", "test-topic")
TRIGGER_SECS      = os.getenv("TRIGGER_INTERVAL", "5 seconds")
CHECKPOINT_LOCATION = os.getenv("CHECKPOINT_LOCATION", "/opt/spark/checkpoints/kafka_stream")

def mock_ml_inference(data_string):
    """
    Dummy ML Logic: Predicts '1' if the message contains 'alert', else '0'.
    """
    if not data_string:
        return 0
    return 1 if "alert" in data_string.lower() else 0

spark = (
    SparkSession.builder
    .appName("KafkaSparkConsumer")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"[KafkaSparkConsumer] Subscribing to topic '{KAFKA_TOPIC}' on {KAFKA_BOOTSTRAP}")

# ── Read raw stream from Kafka ────────────────────────────────────────────────
raw = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
)

# ── Decode machine code to RDD (key/value as strings) and enrich with ingest time,  ──────────────────
messages = raw.select(
    col("topic"),
    col("partition"),
    col("offset"),
    col("timestamp").alias("kafka_timestamp"),
    col("key").cast(StringType()).alias("key"),
    col("value").cast(StringType()).alias("value"),
    current_timestamp().alias("processed_at"),
)

# ── ML Inference Stage ───────────────────────────────────────────────────────
# from tasks.inference import mock_ml_inference
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# Register the function as a Spark UDF
predict_udf = udf(mock_ml_inference, IntegerType())


# ── Optional: parse value as JSON if it matches {"event": ..., "data": ...} ──
json_schema = StructType([
    StructField("event", StringType(), True),
    StructField("data",  StringType(), True),
])

parsed = messages.withColumn(
    "json", from_json(col("value"), json_schema)
).select(
    "topic", "partition", "offset",
    "kafka_timestamp", "processed_at",
    "key", "value",
    col("json.event").alias("event"), # .alias() 重新命名列
    col("json.data").alias("data"),
)

# Apply the ML prediction
enriched = parsed.withColumn("prediction", predict_udf(col("data")))

# ── Write to console (swap .format("console") for parquet/delta/jdbc etc.) ───
query = (
    enriched.writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("numRows", 50)
    .option("checkpointLocation", CHECKPOINT_LOCATION)
    .trigger(processingTime=TRIGGER_SECS)
    .start() # trigger the RDD operations from raw => messages => parsed => enriched => console output
)

print("[KafkaSparkConsumer] Streaming query started — waiting for messages …")
query.awaitTermination()
