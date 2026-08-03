"""Spark adapter: stage-one scoring distributed over the cluster's workers.

Imported only by a composition root that intends to use Spark — ``pyspark`` is
installed in the inference-service and spark-app images, and importing this
package pulls it in via :meth:`SparkAnomalyScorer.initialize`.
"""
from adapters.outbound.spark.scorer import SparkAnomalyScorer

__all__ = ["SparkAnomalyScorer"]
