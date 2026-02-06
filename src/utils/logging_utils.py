"""Logging utilities for LCComparison2026."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure logging with file and console handlers.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. None disables file logging.
        console: Whether to add console handler.

    Returns:
        Configured root logger for lccomparison.
    """
    logger = logging.getLogger("lccomparison")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
