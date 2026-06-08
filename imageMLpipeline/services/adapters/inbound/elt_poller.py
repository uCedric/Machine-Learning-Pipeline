"""ELT driving adapter: poll a file path and feed images to the IngestImage use case."""
from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from uuid import uuid4

from application.use_cases.ingest_image import IngestImage

logger = logging.getLogger("elt")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


def _content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


class EltPoller:
    def __init__(
        self,
        use_case: IngestImage,
        input_file: str,
        *,
        poll_interval: float,
        run_once: bool,
    ) -> None:
        self._use_case = use_case
        self._input_file = input_file
        self._poll_interval = poll_interval
        self._run_once = run_once

    def _process_once(self) -> int:
        path = Path(self._input_file)
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            return 0
        with open(path, "rb") as f:
            data = f.read()
        key = f"{path.stem}-{uuid4().hex[:8]}{path.suffix.lower()}"
        self._use_case.execute(key=key, data=data, content_type=_content_type(path))
        return 1

    def run(self) -> None:
        logger.info(
            "ELT watching %s (poll=%.1fs, run_once=%s)",
            self._input_file,
            self._poll_interval,
            self._run_once,
        )
        while True:
            count = self._process_once()
            if self._run_once:
                logger.info("run_once set; processed %d image(s); exiting", count)
                break
            time.sleep(self._poll_interval)
