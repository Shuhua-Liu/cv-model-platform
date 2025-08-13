"""
CV Platform API模块

提供REST API服务，支持模型调用、管理和监控
"""

from .models1.requests import app
from .utils import *

__all__ = [
    'app',
    # 响应模型
    'BaseResponse',
    'SuccessResponse', 
    'ErrorResponse',
    # 模型相关
    'ModelInfo',
    'ModelListResponse',
    'ModelDetailResponse',
    # 检测相关
    'DetectionObject',
    'DetectionResult',
    'DetectionResponse',
    # 分割相关
    'SegmentationResult',
    'SegmentationResponse',
    # 分类相关
    'ClassificationResult',
    'ClassificationResponse',
    # 生成相关
    'GenerationResult',
    'GenerationResponse',
    # 系统相关
    'SystemStatus',
    'HealthResponse',
    'CacheStatsResponse'
]
