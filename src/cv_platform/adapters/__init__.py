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

# 确保注册表在最后导入，避免循环导入
from .registry import AdapterRegistry, get_registry, register_adapter, create_adapter

# 强制触发适配器注册并验证关键适配器
def _ensure_critical_adapters():
    """确保关键适配器已注册"""
    registry = get_registry()
    
    # 检查关键适配器
    critical_adapters = ['ultralytics']
    missing_adapters = []
    
    for adapter_name in critical_adapters:
        if adapter_name not in registry._adapters:
            missing_adapters.append(adapter_name)
    
    if missing_adapters:
        print(f"⚠️ 发现缺失的关键适配器: {missing_adapters}")
        
        # 尝试强制注册
        for adapter_name in missing_adapters:
            if hasattr(registry, 'force_register_adapter'):
                success = registry.force_register_adapter(adapter_name)
                if success:
                    print(f"✅ 成功强制注册: {adapter_name}")
                else:
                    print(f"❌ 强制注册失败: {adapter_name}")

# 执行关键适配器检查
try:
    _ensure_critical_adapters()
except Exception as e:
    print(f"⚠️ 适配器检查过程中出现异常: {e}")

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