"""
Multimodal model adapters
"""

__all__ = []

# Dynamic import available adapters
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
