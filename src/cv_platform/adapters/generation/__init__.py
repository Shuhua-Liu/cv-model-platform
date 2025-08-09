"""
生成模型适配器
"""

__all__ = []

# 动态导入可用的适配器
try:
    from .stable_diffusion import StableDiffusionAdapter
    __all__.append('StableDiffusionAdapter')
except ImportError:
    pass

try:
    from .flux import FluxAdapter
    __all__.append('FluxAdapter')
except ImportError:
    pass

try:
    from .controlnet import ControlNetAdapter
    __all__.append('ControlNetAdapter')
except ImportError:
    pass