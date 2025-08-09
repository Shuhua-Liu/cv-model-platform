"""
分类模型适配器
"""

__all__ = []

# 动态导入可用的适配器
try:
    from .torchvision import TorchvisionAdapter
    __all__.append('TorchvisionAdapter')
except ImportError:
    pass

try:
    from .timm import TimmAdapter
    __all__.append('TimmAdapter')
except ImportError:
    pass
