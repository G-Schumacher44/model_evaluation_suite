"""
📜 Module: logging.py
Provides a single, reusable function for configuring the global logger.
Now supports both console and file-based logging.
"""

import logging
from pathlib import Path

def configure_logging(notebook_mode: bool, logging_mode: str, log_path: Path):
    """
    Configures the root logger based on the operational context.

    Args:
        notebook_mode (bool): True if running in a notebook, False for scripts.
        logging_mode (str): One of 'on', 'off', or 'auto'.
        log_path (Path): The file path to save the log file to.
    """
    log_level = logging.INFO
    
    if logging_mode == 'off':
        log_level = logging.CRITICAL + 1
    elif logging_mode == 'auto' and notebook_mode:
        log_level = logging.WARNING
    elif logging_mode == 'on':
        log_level = logging.INFO

    # --- Create separate handlers for console and file ---
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (for screen output)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (for saving logs to a file)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.info(f"Logger configured to level: {logging.getLevelName(log_level)}")
    if log_path:
        logging.info(f"Logs will be saved to: {log_path}")