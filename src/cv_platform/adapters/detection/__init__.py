"""
Detection Model Adapter
"""

__all__ = []

# Dynamic import available adapters
try:
    from .ultralytics import UltralyticsAdapter
    __all__.append('UltralyticsAdapter')
except ImportError:
    pass

try:
    from .detectron2 import Detectron2Adapter
    __all__.append('Detectron2Adapter')
except ImportError:
    pass

# Log the final status
print(f"Detection adapters loaded: {__all__}")
