"""
backend/utils/logger.py
-------------------------
Centralized logging setup. Every module imports get_logger(__name__) instead
of configuring logging itself, so log format/destination stays consistent
project-wide and is easy to change in one place.

Logs go to BOTH the console (for live debugging) and logs/app.log (for
later inspection / demoing "production-style" observability).
"""

import logging
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def _configure_root_logger():
    """Configure the root logger once, idempotently."""
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger (e.g. get_logger(__name__)) with project-wide config applied."""
    _configure_root_logger()
    return logging.getLogger(name)
