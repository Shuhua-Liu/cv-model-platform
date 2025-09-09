"""
Feature Extraction Model Adapter
"""

__all__ = []

# Dynamic import available adapters
try:
    from .dinov3 import DINOv3Adapter
    __all__.append('DINOv3Adapter')
except ImportError:
    pass

# Log the final status
print(f"Feature extraction adapters loaded: {__all__}")
