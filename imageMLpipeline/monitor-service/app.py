"""monitor-service: a small FastAPI app that surfaces inference results.

It reads rows from the Postgres ``inference_results`` table (written by the
inference-service) and serves:

* ``GET /``                       — an interactive HTML dashboard (auto-refreshing)
* ``GET /api/results``            — recent results as JSON (optional ``status`` filter)
* ``GET /api/summary``            — verdict counts for the summary cards
* ``GET /api/object/{bucket}/{key}`` — proxies image/heatmap bytes from MinIO
* ``POST /api/retrain``            — publishes a ``train`` event per pending image
* ``GET /healthz``                — liveness probe

The service never writes to Postgres or MinIO; the only side effect is
publishing ``train`` events to Kafka on demand (idempotent per ``Idempotency-Key``).
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from kafka import KafkaProducer
from minio import Minio
from minio.error import S3Error
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monitor-service")

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Only objects in this bucket may be proxied, so the endpoint can't be turned
# into an open relay for arbitrary MinIO keys.
_IMAGES_BUCKET = os.getenv("IMAGES_BUCKET", "images")

# Kafka wiring for the retrain trigger: a ``train`` event is published per
# selected pending image, consumed by the train-service.
_KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
_TRAIN_TOPIC = os.getenv("KAFKA_TRAIN_TOPIC", "train")
_TRAIN_EVENT_TYPE = os.getenv("KAFKA_TRAIN_EVENT_TYPE", "train")

# Request-level idempotency for POST /api/retrain: an in-memory, bounded,
# thread-safe cache of ``Idempotency-Key -> response`` plus an in-flight set so a
# concurrent duplicate of an unfinished request can't double-publish. FastAPI
# runs sync endpoints in a threadpool, so the lock is required.
_IDEMPOTENCY_CACHE_MAX = 1024
_idem_lock = threading.Lock()
_idem_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_idem_inflight: set[str] = set()


def _postgres_dsn() -> str:
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Postgres pool, the MinIO client and the Kafka producer on startup."""
    app.state.pool = ThreadedConnectionPool(1, 10, _postgres_dsn())
    app.state.minio = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=os.getenv("MINIO_SECURE", "false").strip().lower()
        in {"1", "true", "yes", "on"},
    )
    app.state.producer = KafkaProducer(
        bootstrap_servers=_KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
    )
    logger.info("monitor-service ready")
    try:
        yield
    finally:
        app.state.pool.closeall()
        app.state.producer.flush()
        app.state.producer.close()


app = FastAPI(title="Inference Monitor", lifespan=lifespan)


def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a read-only query and return rows as dicts (borrowing from the pool)."""
    pool: ThreadedConnectionPool = app.state.pool
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        pool.putconn(conn)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/results")
def api_results(
    limit: int = Query(50, ge=1, le=500),
    status: str | None = Query(None, pattern="^(normal|pending|anomaly)$"),
) -> list[dict[str, Any]]:
    """Most recent inference results, newest first, optionally filtered by status."""
    if status:
        rows = _query(
            """
            SELECT event_id, image_id, object_key, bucket, anomaly_score,
                   status, heatmap_key, inferred_at
            FROM inference_results
            WHERE status = %s
            ORDER BY inferred_at DESC
            LIMIT %s
            """,
            (status, limit),
        )
    else:
        rows = _query(
            """
            SELECT event_id, image_id, object_key, bucket, anomaly_score,
                   status, heatmap_key, inferred_at
            FROM inference_results
            ORDER BY inferred_at DESC
            LIMIT %s
            """,
            (limit,),
        )
    for row in rows:
        row["event_id"] = str(row["event_id"])
        row["inferred_at"] = row["inferred_at"].isoformat()
    return rows


@app.get("/api/summary")
def api_summary() -> dict[str, int]:
    """Verdict counts, with a key for every status so the UI can render zeros."""
    counts = {"normal": 0, "pending": 0, "anomaly": 0}
    for row in _query("SELECT status, COUNT(*) AS n FROM inference_results GROUP BY status"):
        counts[row["status"]] = int(row["n"])
    counts["total"] = sum(counts.values())
    return counts


@app.get("/api/object/{bucket}/{key:path}")
def api_object(bucket: str, key: str) -> StreamingResponse:
    """Stream an image or heatmap from MinIO so the browser can display it."""
    if bucket != _IMAGES_BUCKET:
        raise HTTPException(status_code=403, detail="bucket not allowed")
    client: Minio = app.state.minio
    try:
        response = client.get_object(bucket, key)
    except S3Error as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _stream() -> Iterator[bytes]:
        try:
            yield from response.stream(32 * 1024)
        finally:
            response.close()
            response.release_conn()

    content_type, _ = mimetypes.guess_type(key)
    return StreamingResponse(_stream(), media_type=content_type or "application/octet-stream")


class RetrainRequest(BaseModel):
    event_ids: list[str]


@app.post("/api/retrain")
def api_retrain(
    req: RetrainRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
) -> dict[str, Any]:
    """Publish a ``train`` event for each selected pending image.

    Idempotent per ``Idempotency-Key``: a repeat with the same key returns the
    cached response and publishes nothing; a concurrent duplicate gets ``409``.
    The server independently re-validates ``status = 'pending'`` so only genuinely
    pending rows are ever published, regardless of what the client sent.
    """
    if not req.event_ids:
        raise HTTPException(status_code=400, detail="no event_ids provided")

    # Idempotency gate: cached repeat -> return as-is; in-flight dup -> 409.
    with _idem_lock:
        if idempotency_key in _idem_cache:
            return _idem_cache[idempotency_key]
        if idempotency_key in _idem_inflight:
            raise HTTPException(status_code=409, detail="request in progress")
        _idem_inflight.add(idempotency_key)
    try:
        rows = _query(
            """
            SELECT r.event_id, r.bucket, r.object_key, m.model_type
            FROM inference_results r
            JOIN inference_model m ON m.model_id = r.model_id
            WHERE r.event_id = ANY(%s::uuid[]) AND r.status = 'pending'
            """,
            (req.event_ids,),
        )
        producer: KafkaProducer = app.state.producer
        published: list[str] = []
        for row in rows:
            event = {
                "event": _TRAIN_EVENT_TYPE,
                "model": row["model_type"],
                "images": f"/{row['bucket']}/{row['object_key']}",
            }
            producer.send(_TRAIN_TOPIC, key=_TRAIN_EVENT_TYPE, value=event).get(timeout=10)
            published.append(str(row["event_id"]))
        result = {
            "requested": len(req.event_ids),
            "published": len(published),
            "event_ids": published,
        }
        with _idem_lock:  # remember the response so an identical resend is a no-op
            _idem_cache[idempotency_key] = result
            _idem_cache.move_to_end(idempotency_key)
            while len(_idem_cache) > _IDEMPOTENCY_CACHE_MAX:
                _idem_cache.popitem(last=False)  # evict oldest
        logger.info("Published %d train event(s) [key=%s]", len(published), idempotency_key)
        return result
    finally:
        with _idem_lock:
            _idem_inflight.discard(idempotency_key)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(request, "index.html")
