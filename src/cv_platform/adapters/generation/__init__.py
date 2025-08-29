"""
Generation model adapters
"""

__all__ = []

# Dynamic import of available adapters
try:
    from .stable_diffusion import StableDiffusionAdapter
    __all__.append('StableDiffusionAdapter')
    print("✅ StableDiffusionAdapter imported successfully")
except ImportError as e:
    print(f"⚠️ StableDiffusionAdapter imported failed: {e}")

try:
    from .flux import FluxAdapter
    __all__.append('FluxAdapter')
    print("✅ FluxAdapter imported successfully")
except ImportError as e:
    print(f"⚠️ FluxAdapter imported failed: {e}")

try:
    from .controlnet import ControlNetAdapter
    __all__.append('ControlNetAdapter')
    print("✅ ControlNetAdapter imported successfully")
except ImportError as e:
    print(f"⚠️ ControlNetAdapter imported failed: {e}")

# Log the final status
print(f"Generation adapters loaded: {__all__}")
