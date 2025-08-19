"""
Logging Tools - Unified log configuration and management
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional, Union


def setup_logger(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_type: str = "text",
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """
    Set up logging configuration

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path. None means output only to the console.
        format_type: Format type (text, json)
        rotation: Log rotation size
        retention: Log retention time
    """
    # Remove the default handler
    logger.remove()
    
    # Select Format
    if format_type.lower() == "json":
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Adding a console processor
    logger.add(
        sys.stderr,
        format=log_format,
        level=level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler (if log file is specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            format=log_format,
            level=level.upper(),
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        logger.info(f"Log file configuration is complete: {log_path}")
    
    logger.info(f"Logging system initialization complete - Level: {level}")


def get_logger(name: str):
    """Get a named logger"""
    return logger.bind(name=name)
