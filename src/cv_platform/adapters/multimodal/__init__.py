"""
多模态模型适配器
"""

__all__ = []

# 动态导入可用的适配器
try:
    from .clip import CLIPAdapter
    __all__.append('CLIPAdapter')
except ImportError:
    pass

try:
    from .openclip import OpenCLIPAdapter
    __all__.append('OpenCLIPAdapter')
except ImportError:
    pass

try:
    from .llava import LLaVAAdapter
    __all__.append('LLaVAAdapter')
except ImportError:
    pass