"""
Spark Image Processing Pipeline using Pandas UDFs and Apache Arrow.

This script demonstrates:
1. Loading partitioned image binary data using Spark SQL.
2. Using Pandas UDFs (which use Apache Arrow) for efficient processing.
3. Fetching from: /home/cedric/imageMLpipeline/parquet_output/
"""

import os
import sys

import time
import logging
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import BinaryType, StructType, StructField, StringType, IntegerType
import pandas as pd
import numpy as np
from PIL import Image
import io

# --- Logging Configuration ---
LOG_FILE = "spark_error.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def stay_alive():
    """Keep the container running even after a failure for debugging."""
    logger.info("Entering idle state to keep container alive. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
        sys.exit(0)

def get_spark_session():
    """Attempt to initialize a Spark session with error logging."""
    try:
        logger.info("Initializing Spark session...")
        session = SparkSession.builder \
            .appName("ImageMLPipeline") \
            .master("spark://spark-master:7077") \
            .config("spark.driver.host", "image-app") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        session.range(1).count()
        logger.info("Spark session successfully established.")
        return session
    except Exception as e:
        error_msg = f"CRITICAL: Failed to initialize Spark session.\nError: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        stay_alive()

# Base schema for the image metadata
BASE_SCHEMA = StructType([
    StructField("id", IntegerType(), False),
    StructField("filename", StringType(), False),
    StructField("content", BinaryType(), False)
])

def create_dummy_data(spark, path: str):
    """Generate partitioned dummy image data and register as a table."""
    labels = ["PLAY", "Nodefect", "BlackParticle", "Oil", "BM_PAR"]
    data = []
    
    for i in range(50):
        label = labels[i % len(labels)]
        group_id = i % 10
        
        img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        
        data.append((i, f"img_{i}.png", byte_arr.getvalue(), label, group_id))
    
    SCHEMA = StructType(BASE_SCHEMA.fields + [
        StructField("label", StringType(), False),
        StructField("group_id", IntegerType(), False)
    ])
    
    df = spark.createDataFrame(data, SCHEMA)
    logger.info(f"Writing dummy data to {path}")
    df.write.mode("overwrite").partitionBy("label", "group_id").parquet(path)
    
    # Register the table so it can be accessed by the main pipeline
    spark.read.parquet(path).createOrReplaceTempView("wafer_images_table")

@pandas_udf(BinaryType())
def process_images_udf(image_series: pd.Series) -> pd.Series:
    """Vectorized image processing using Pandas and Arrow."""
    processed_list = []
    for img_bytes in image_series:
        img = Image.open(io.BytesIO(img_bytes))
        processed_img = img.convert('L') # Grayscale
        output = io.BytesIO()
        processed_img.save(output, format='JPEG')
        processed_list.append(output.getvalue())
    return pd.Series(processed_list)

def save_to_disk(partition):
    """Saves processed images to the local filesystem from Spark executors."""
    import os
    base_output_path = "/app/processed_images"
    for row in partition:
        # Construct the output directory path based on the label
        label_dir = os.path.join(base_output_path, row.label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir, exist_ok=True)
        
        # Get base filename without extension, then add .jpg
        base_name = row.path.split("/")[-1].replace(".jpg", "")
        file_path = os.path.join(label_dir, f"{base_name}.jpg")
        
        with open(file_path, "wb") as f:
            f.write(row.processed_image_data)

def main():
    # Root directory for partitioned parquet data
    input_path = "/app/parquet_output/"
    output_path = "/app/processed_images"

    spark = get_spark_session()

    try:
        # 1. Load data from the existing structured folder
        logger.info(f"Loading partitioned data from {input_path}...")

        # Discover and load the partitioned parquet dataset
        df_raw = spark.read.parquet(input_path)
        df_raw.createOrReplaceTempView("wafer_images_table")

        # 2. Load data using PySpark SQL from the registered table
        logger.info("Executing Spark SQL query to fetch metadata and content...")

        # Fetching via SQL (this utilizes Arrow if config is enabled)
        df_sql = spark.sql("SELECT path, label, content FROM wafer_images_table")
        
        # 3. Apply the Pandas UDF to process images (uses Apache Arrow)
        logger.info("Processing images with Pandas UDF...")
        processed_df = df_sql.withColumn(
            "processed_image_data", 
            process_images_udf(col("content"))
        )
        
        # 4. Save results as individual JPEGs
        logger.info(f"Saving processed images as JPEGs to {output_path} arranged by label...")
        processed_df.select("path", "label", "processed_image_data").foreachPartition(save_to_disk)
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        stay_alive()

if __name__ == "__main__":
    main()
