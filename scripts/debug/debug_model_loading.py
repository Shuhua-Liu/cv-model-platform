#!/usr/bin/env python3
"""
Model Loading Debug Script - Diagnose the "unhashable type: dict" error

This script helps identify and fix the issue with model loading.
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_manager import get_model_manager
    from src.cv_platform.core.model_detector import get_model_detector
    from src.cv_platform.adapters.registry import get_registry
    from src.cv_platform.utils.logger import setup_logger
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def debug_model_loading():
    """Debug the model loading issue step by step"""
    setup_logger("DEBUG")
    
    print("🔍 Debugging Model Loading Issue")
    print("=" * 50)
    
    try:
        # Step 1: Check available models
        print("1️⃣ Checking available models...")
        manager = get_model_manager()
        available_models = manager.list_available_models()
        
        print(f"Available models: {len(available_models)}")
        for name, info in available_models.items():
            print(f"  - {name}")
            print(f"    Config: {type(info.get('config', 'None'))}")
            if isinstance(info.get('config'), dict):
                config = info['config']
                print(f"    Type: {config.get('type', 'unknown')}")
                print(f"    Path: {config.get('path', 'unknown')}")
                print(f"    Framework: {config.get('framework', 'unknown')}")
        
        # Step 2: Focus on the problematic model
        problematic_model = "detection_yolo_v8_yolov8n"
        if problematic_model in available_models:
            print(f"\n2️⃣ Analyzing problematic model: {problematic_model}")
            model_info = available_models[problematic_model]
            
            print("Model info structure:")
            print_dict_safely(model_info, indent=2)
            
            # Check config structure
            config = model_info.get('config', {})
            print(f"\nConfig type: {type(config)}")
            if isinstance(config, dict):
                print("Config contents:")
                print_dict_safely(config, indent=2)
            
        # Step 3: Test registry
        print(f"\n3️⃣ Testing adapter registry...")
        registry = get_registry()
        if registry:
            print("Registry available")
            adapters = registry.list_adapters()
            print(f"Available adapters: {list(adapters.keys())}")
        else:
            print("❌ Registry not available")
        
        # Step 4: Test model loading with detailed error handling
        if problematic_model in available_models:
            print(f"\n4️⃣ Attempting to load model with detailed debugging...")
            try:
                debug_model_load(manager, problematic_model, available_models[problematic_model])
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                print("Full traceback:")
                traceback.print_exc()
                
                # Additional debugging
                print(f"\n🔍 Additional debugging info:")
                analyze_error_context(e, available_models[problematic_model])
        
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        traceback.print_exc()


def print_dict_safely(data: Any, indent: int = 0) -> None:
    """Safely print dictionary contents"""
    prefix = "  " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}: (dict)")
                print_dict_safely(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: (list with {len(value)} items)")
                if value and len(value) <= 3:  # Show small lists
                    for i, item in enumerate(value):
                        print(f"{prefix}  [{i}]: {type(item).__name__} = {item}")
            else:
                print(f"{prefix}{key}: {type(value).__name__} = {value}")
    elif isinstance(data, list):
        print(f"{prefix}(list with {len(data)} items)")
        for i, item in enumerate(data[:3]):  # Show first 3 items
            print(f"{prefix}[{i}]: {type(item).__name__}")
            if isinstance(item, dict):
                print_dict_safely(item, indent + 1)
    else:
        print(f"{prefix}{type(data).__name__} = {data}")


def debug_model_load(manager, model_name: str, model_info: Dict[str, Any]) -> None:
    """Debug the model loading process step by step"""
    print(f"🔬 Detailed model loading debug for: {model_name}")
    
    try:
        # Step 1: Extract config
        config = model_info.get('config', {})
        print(f"Step 1 - Config extraction: {type(config)}")
        
        if not isinstance(config, dict):
            print(f"❌ Config is not a dict: {type(config)}")
            return
        
        # Step 2: Check required fields
        print("Step 2 - Checking required config fields:")
        required_fields = ['path', 'type', 'framework']
        for field in required_fields:
            value = config.get(field)
            print(f"  {field}: {type(value).__name__} = {value}")
        
        # Step 3: Test registry access
        print("Step 3 - Testing registry access:")
        registry = get_registry()
        if not registry:
            print("❌ Registry not available")
            return
        
        # Step 4: Test adapter detection
        print("Step 4 - Testing adapter detection:")
        try:
            # Check what the registry is trying to do
            model_path = config.get('path', '')
            print(f"Model path: {model_path}")
            
            # Test auto-detection
            adapter_name = registry.auto_detect_adapter(model_path)
            print(f"Auto-detected adapter: {adapter_name}")
            
        except Exception as e:
            print(f"❌ Adapter detection failed: {e}")
            traceback.print_exc()
            return
        
        # Step 5: Test adapter creation
        print("Step 5 - Testing adapter creation:")
        try:
            # Try to create adapter with minimal params
            test_kwargs = {
                'device': 'cpu',  # Force CPU to avoid GPU issues
                'cache_enabled': False  # Disable cache
            }
            
            print(f"Creating adapter with kwargs: {test_kwargs}")
            
            # This is where the error likely occurs
            adapter = registry.create_adapter(
                model_path=config.get('path', ''),
                **test_kwargs
            )
            
            print(f"✅ Adapter created successfully: {type(adapter)}")
            
        except Exception as e:
            print(f"❌ Adapter creation failed: {e}")
            traceback.print_exc()
            
            # Check what exactly is being passed to create_adapter
            print(f"\n🔍 Analyzing create_adapter call:")
            print(f"model_path type: {type(config.get('path', ''))}")
            print(f"model_path value: {config.get('path', '')}")
            print(f"kwargs keys: {list(test_kwargs.keys())}")
            for k, v in test_kwargs.items():
                print(f"  {k}: {type(v).__name__} = {v}")
    
    except Exception as e:
        print(f"❌ Debug model load failed: {e}")
        traceback.print_exc()


def analyze_error_context(error: Exception, model_info: Dict[str, Any]) -> None:
    """Analyze the error context to identify the cause"""
    print("🔍 Error Context Analysis:")
    
    error_msg = str(error).lower()
    
    if "unhashable type" in error_msg and "dict" in error_msg:
        print("❌ Issue: Attempting to use dict as hash key")
        print("Possible causes:")
        print("1. Model config being passed where string expected")
        print("2. Dictionary being used in set() or as dict key")
        print("3. Tuple containing dict being used as key")
        
        # Check for common problematic patterns
        config = model_info.get('config', {})
        
        print(f"\n🔍 Checking config structure:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  ⚠️ {key} is a dict: {value}")
            elif isinstance(value, list):
                print(f"  📋 {key} is a list with {len(value)} items")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        print(f"    ⚠️ [{i}] contains dict: {item}")
    
    # Check error traceback for more clues
    tb = traceback.format_exc()
    if "registry" in tb.lower():
        print("2. Error likely in registry/adapter creation")
    if "create_adapter" in tb.lower():
        print("3. Error in create_adapter method")
    if "__hash__" in tb.lower():
        print("4. Error in hashing operation")


def suggest_fixes():
    """Suggest potential fixes for the issue"""
    print("\n💡 Suggested Fixes:")
    print("=" * 30)
    
    print("1. Check ModelManager._model_info_to_config() method:")
    print("   - Ensure it returns only serializable values")
    print("   - Check that no dict objects are used as keys")
    
    print("\n2. Check AdapterRegistry.create_adapter() method:")
    print("   - Verify parameter types before processing")
    print("   - Ensure model_path is string, not dict")
    
    print("\n3. Check model configuration:")
    print("   - Verify config.yaml structure")
    print("   - Ensure no nested dicts where strings expected")
    
    print("\n4. Add defensive programming:")
    print("   - Type checking before operations")
    print("   - Convert dicts to strings when needed")
    
    print("\n5. Quick fix suggestions:")
    print("""
   # In ModelManager.load_model():
   if isinstance(model_config, dict):
       # Extract path properly
       model_path = str(model_config.get('path', ''))
   else:
       model_path = str(model_config)
   
   # In create_adapter call:
   adapter = self.registry.create_adapter(
       model_path=model_path,  # Ensure this is string
       **kwargs  # Ensure kwargs contains no dicts as keys
   )
   """)

def check_adapter_registration():
    """检查适配器注册状态"""
    from src.cv_platform.adapters.registry import get_registry
    
    registry = get_registry()
    
    print("🔍 检查适配器注册状态:")
    print("=" * 50)
    
    # 1. 检查已注册的适配器
    adapters_info = registry.list_adapters()
    print(f"已注册的适配器: {list(adapters_info.keys())}")
    
    # 2. 检查 ultralytics 适配器是否存在
    ultralytics_adapter = registry.get_adapter_class('ultralytics')
    print(f"Ultralytics 适配器类: {ultralytics_adapter}")
    
    # 3. 检查框架映射
    print(f"框架映射: {registry._framework_mappings}")
    
    # 4. 测试自动检测
    model_path = "cv_models/detection/yolo/v8/yolov8n.pt"
    detected_adapter = registry.auto_detect_adapter(model_path, None)
    print(f"自动检测结果: {detected_adapter}")
    
    return registry

def manual_register_ultralytics():
    """手动注册 ultralytics 适配器"""
    try:
        from src.cv_platform.adapters.detection.ultralytics import UltralyticsAdapter
        from src.cv_platform.adapters.registry import get_registry
        
        registry = get_registry()
        
        # 手动注册
        registry.register(
            'ultralytics',
            UltralyticsAdapter,
            frameworks=['ultralytics', 'yolo'],
            architectures=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                          'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10x',
                          'yolo11n', 'yolo11s', 'yolo11m']
        )
        
        print("✅ 手动注册 ultralytics 适配器成功")
        return True
        
    except ImportError as e:
        print(f"❌ 无法导入 UltralyticsAdapter: {e}")
        print("请确保安装了 ultralytics: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ 注册失败: {e}")
        return False

def fix_adapter_creation():
    """修复适配器创建"""
    from src.cv_platform.adapters.registry import get_registry
    
    # 1. 检查注册状态
    registry = check_adapter_registration()
    
    # 2. 如果 ultralytics 没有注册，手动注册
    if 'ultralytics' not in registry._adapters:
        print("⚠️  Ultralytics 适配器未注册，尝试手动注册...")
        if not manual_register_ultralytics():
            return False
    
    # 3. 测试创建适配器
    try:
        model_path = "cv_models/detection/yolo/v8/yolov8n.pt"
        
        # 方法1：明确指定适配器名称
        adapter = registry.create_adapter(
            model_path=model_path,
            adapter_name='ultralytics',  # 明确指定
            device='cpu',
            cache_enabled=False
        )
        print("✅ 成功创建适配器 (指定名称)")
        return True
        
    except Exception as e:
        print(f"❌ 创建适配器失败: {e}")
        
        # 方法2：尝试通过框架指定
        try:
            adapter = registry.create_adapter(
                model_path=model_path,
                framework='ultralytics',  # 通过框架指定
                device='cpu',
                cache_enabled=False
            )
            print("✅ 成功创建适配器 (指定框架)")
            return True
        except Exception as e2:
            print(f"❌ 通过框架创建也失败: {e2}")
            return False

if __name__ == "__main__":
    fix_adapter_creation()

# if __name__ == "__main__":
#     debug_model_loading()
#     suggest_fixes()
