"""
分割模型适配器
"""

__all__ = []

# 动态导入可用的适配器
try:
    from .deeplabv3 import DeepLabV3Adapter
    __all__.append('DeepLabV3Adapter')
except ImportError:
    pass

try:
    from .sam import SAMAdapter
    __all__.append('SAMAdapter')
except ImportError:
    pass
