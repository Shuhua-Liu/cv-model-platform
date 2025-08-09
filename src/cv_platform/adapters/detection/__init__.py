"""
检测模型适配器
"""

__all__ = []

# 动态导入可用的适配器
try:
    from .ultralytics import UltralyticsAdapter
    __all__.append('UltralyticsAdapter')
except ImportError:
    pass
