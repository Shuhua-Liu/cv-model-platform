"""
Generation model adapters
"""

__all__ = []

# Dynamic import available adapters
try:
    from .lama import LaMaAdapter
    __all__.append('LaMaAdapter')
    print("✅ LaMaAdapter imported successfully")
except ImportError as e:
    print(f"⚠️ LaMaAdapter imported failed: {e}")

try:
    from .stable_diffusion_inpainting import StableDiffusionInpaintingAdapter
    __all__.append('StableDiffusionInpaintingAdapter')
    print("✅ StableDiffusionInpaintingAdapter imported successfully")
except ImportError as e:
    print(f"⚠️ StableDiffusionInpaintingAdapter imported failed: {e}")

# Log the final status
print(f"Inpainting adapters loaded: {__all__}")
