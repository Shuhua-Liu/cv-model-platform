"""
日志工具 - 统一的日志配置和管理
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
    设置日志配置
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None表示只输出到控制台
        format_type: 格式类型 (text, json)
        rotation: 日志轮转大小
        retention: 日志保留时间
    """
    # 移除默认处理器
    logger.remove()
    
    # 选择格式
    if format_type.lower() == "json":
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format=log_format,
        level=level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件处理器（如果指定了日志文件）
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
        
        logger.info(f"日志文件配置完成: {log_path}")
    
    logger.info(f"日志系统初始化完成 - 级别: {level}")


def get_logger(name: str):
    """获取命名日志器"""
    return logger.bind(name=name)
