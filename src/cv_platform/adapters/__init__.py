"""
CV Platform Adapter Module

Contains various model adapters and the registry.
"""

from .base import (
    BaseModelAdapter,
    DetectionAdapter, 
    SegmentationAdapter,
    ClassificationAdapter,
    GenerationAdapter,
    MultimodalAdapter
)
from .registry import AdapterRegistry, get_registry, register_adapter, create_adapter

__all__ = [
    'BaseModelAdapter',
    'DetectionAdapter',
    'SegmentationAdapter', 
    'ClassificationAdapter',
    'GenerationAdapter',
    'MultimodalAdapter',
    'AdapterRegistry',
    'get_registry',
    'register_adapter',
    'create_adapter',
]