from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
        name: str,
        log_dir: str = "outputs/logs",
        level: int = logging.INFO,
        console: bool = True,
        filename: Optional[str] = None,
) -> logging.Logger:
    """
    Create a logger that writes to outputs/logs and optionally to console.

    Parameters
    ----------
    name : str
        Logger name (usually the script name).
    log_dir : str
        Directory where log files are stored.
    level : int
        Logging level (e.g., logging.INFO).
    console : bool
        If True, also logs to stdout.
    filename : Optional[str]
        Custom filename. If None, a timestamped filename is used.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # This creates the directory at the path, if it does not yet exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = filename or f"{name}_{ts}.log"
    log_path = Path(log_dir) / log_file

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_path}")
    return logger
