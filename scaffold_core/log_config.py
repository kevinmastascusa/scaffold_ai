#!/usr/bin/env python3
"""
Centralized Logging Configuration for Scaffold AI
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Try to import colorlog for colored console logging
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Sets up centralized logging for the entire application.

    This function configures the root logger to output to both the console
    and a rotating file. It uses `colorlog` for colored console output
    if the package is available.

    Args:
        log_level: The minimum logging level to capture (e.g., logging.INFO).
        log_dir: The directory to store log files in.
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    log_file = log_path / "scaffold_ai.log"

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console Handler
    if COLORLOG_AVAILABLE:
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + log_format,
            datefmt=date_format,
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
                'ERROR': 'red', 'CRITICAL': 'bold_red',
            }
        )
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format, datefmt=date_format)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # File Handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logging.info(
        "Logging configured: Console and file handlers active."
    ) 