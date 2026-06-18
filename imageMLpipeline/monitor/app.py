"""monitor-service: a small FastAPI app that surfaces inference results.

It reads rows from the Postgres ``inference_results`` table (written by the
inference-service) and serves:

* ``GET /``                       — an interactive HTML dashboard (auto-refreshing)
* ``GET /api/results``            — recent results as JSON (optional ``status`` filter)
* ``GET /api/summary``            — verdict counts for the summary cards
* ``GET /api/object/{bucket}/{key}`` — proxies image/heatmap bytes from MinIO
* ``GET /healthz``                — liveness probe

The service is read-only: it never writes to Postgres or MinIO.
"""
from __future__ import annotations

import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from minio import Minio
from minio.error import S3Error
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monitor-service")

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Only objects in this bucket may be proxied, so the endpoint can't be turned
# into an open relay for arbitrary MinIO keys.
_IMAGES_BUCKET = os.getenv("IMAGES_BUCKET", "images")


def _postgres_dsn() -> str:
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Postgres pool and the MinIO client once, on startup."""
    app.state.pool = ThreadedConnectionPool(1, 10, _postgres_dsn())
    app.state.minio = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=os.getenv("MINIO_SECURE", "false").strip().lower()
        in {"1", "true", "yes", "on"},
    )
    logger.info("monitor-service ready")
    try:
        yield
    finally:
        app.state.pool.closeall()


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


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(request, "index.html")
