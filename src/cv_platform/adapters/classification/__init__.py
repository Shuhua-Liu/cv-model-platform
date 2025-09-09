"""
Classification Model Adapter
"""

__all__ = []

# Dynamic import available adapters
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
