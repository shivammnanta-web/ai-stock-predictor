"""
Centralized logging configuration for the entire project.
Import `get_logger(__name__)` in any module for consistent formatting.
"""
import logging
import sys


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Creates a consistently-formatted logger.
    
    Args:
        name: Module name (use __name__)
        level: Logging level (default INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
