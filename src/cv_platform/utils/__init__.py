"""
CV Platform 工具模块

包含日志、路径处理、图像处理等工具函数。
"""

from .logger import setup_logger, get_logger

__all__ = [
    'setup_logger',
    'get_logger',
]
