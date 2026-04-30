import json
import logging

logger = logging.getLogger(__name__)


def parse_scheduler_arg_to_json(raw: str, arg_name: str = "argument") -> dict:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"--{arg_name} must be valid JSON: {e}"
        logger.error(msg)
        raise ValueError(msg) from e
    if not isinstance(parsed, dict):
        msg = f"--{arg_name} must be a JSON object (got {type(parsed).__name__})"
        logger.error(msg)
        raise ValueError(msg)
    return parsed
