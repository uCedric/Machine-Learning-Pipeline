import logging
from .constants import SCHEDULER_DICT, SCHEDULER_REQUIRED_PARAMS

logger = logging.getLogger(__name__)


def validate_scheduler_config(config: dict) -> tuple[str, dict]:
    """Validate scheduler config dict and return (method, kwargs).

    Raises ValueError if method is missing/unsupported or required params are absent.
    """
    if "method" not in config:
        msg = "scheduler_config must contain a 'method' key."
        logger.error(msg)
        raise ValueError(msg)

    method = config["method"].lower()
    if method not in SCHEDULER_DICT.values():
        msg = f"Invalid scheduler method: '{method}'. Supported: {list(SCHEDULER_DICT.values())}"
        logger.error(msg)
        raise ValueError(msg)

    required = SCHEDULER_REQUIRED_PARAMS.get(method, [])
    missing = [p for p in required if p not in config]
    if missing:
        msg = f"scheduler_config for '{method}' is missing required params: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    kwargs = {k: v for k, v in config.items() if k != "method"}
    return method, kwargs
