from __future__ import annotations

import inspect
import logging
from datetime import datetime

from .config import config

logger = logging.getLogger("web_search")
logger.setLevel(getattr(logging, config.log_level, logging.INFO))
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        log_dir = config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"web_search_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, config.log_level, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.addHandler(logging.NullHandler())


async def log_info(ctx, message: str, is_debug: bool = False) -> None:
    if is_debug:
        logger.info(message)
    if not ctx:
        return

    info_method = getattr(ctx, "info", None)
    if info_method is None:
        return

    result = info_method(message)
    if inspect.isawaitable(result):
        await result
