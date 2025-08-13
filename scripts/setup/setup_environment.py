#!/usr/bin/env python3
"""
Environment Setup Script - One-click initialization of CV Model Platform environment

Features:
1. Check Python environment
2. Check necessary dependencies
3. Create configuration file
4. Discover local models
5. Verify installation
"""

import sys
import subprocess
import importlib
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_python_version():
    """Check Python version"""
    print("🐍 Check Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python version incompatibility: {version.major}.{version.minor}")
        print("Requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check necessary dependencies"""
    print("\n📦 Check necessary dependencies...")
    
    required_packages = [
        ('yaml', 'pyyaml'),
        ('loguru', 'loguru'),
        ('PIL', 'pillow'),
    ]
    
    optional_packages = [
        ('torch', 'torch'),
        ('cv2', 'opencv-python-headless'),
        ('numpy', 'numpy'),
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check necessary dependencies
    for module_name, package_name in required_packages:
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_required.append(package_name)
            print(f"❌ {package_name}")
    
    # Check optional dependencies
    for module_name, package_name in optional_packages:
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name} (optional)")
        except ImportError:
            missing_optional.append(package_name)
            print(f"⚠️  {package_name} (optional)")
    
    # If the necessary dependencies are missing, try installing
    if missing_required:
        print(f"\n🔧 Try installing the missing required dependencies: {', '.join(missing_required)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_required)
            print("✅ Required dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Dependency installation failed")
            return False
    
    if missing_optional:
        print(f"\n💡 Optional dependencies not installed: {', '.join(missing_optional)}")
        print("These dependencies are not required, but may affect certain functionality.")
    
    return True

def create_config_files():
    """Create a configuration file"""
    print("\n⚙️ Create a configuration file...")
    
    try:
        from src.cv_platform.core.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        config_manager.create_default_configs()
        
        print("✅ Configuration file created successfully.")
        print(f"   - config/models_template.yaml")
        print(f"   - config/platform_template.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration file created failed: {e}")
        return False

def discover_models():
    """Find Local Models"""
    print("\n🔍 Find Local Models...")
    
    try:
        from src.cv_platform.core.model_detector import ModelDetector
        from src.cv_platform.core.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        models_root = config_manager.get_models_root()
        
        if not models_root.exists():
            print(f"⚠️  Model directory doesn't exist: {models_root}")
            print("Please place the model files in this directory, or use the environment variable CV_MODELS_ROOT to specify another path.")
            return True  
        
        detector = ModelDetector(models_root)
        models = detector.detect_models()
        
        print(f"✅ Found {len(models)} models")
        
        if models:
            # Generate model configuration
            config_file = Path("config/models.yaml")
            config = detector.generate_config(models, config_file)
            print(f"✅ Model configuration generated: {config_file}")
            
            # Show summary of models found
            by_type = {}
            for model in models:
                model_type = model.type
                if model_type not in by_type:
                    by_type[model_type] = 0
                by_type[model_type] += 1
            
            print("Types of models found:")
            for model_type, count in by_type.items():
                print(f"     - {model_type}: {count} 个")
        else:
            print("No model files found")
        
        return True
        
    except Exception as e:
        print(f"❌ Model failure detected: {e}")
        return False

def verify_installation():
    """Verify installation"""
    print("\n🧪 Verify installation...")
    
    try:
        # Testing Basics Introduction
        from src.cv_platform.core.config_manager import get_config_manager
        from src.cv_platform.core.model_manager import get_model_manager
        
        print("✅ Core module import successful")
        
        # Test Configuration Manager
        config_manager = get_config_manager()
        models_config = config_manager.get_models_config()
        print("✅ Configuration Manager is working properly.")
        
        # Test Model Manager
        model_manager = get_model_manager()
        available_models = model_manager.list_available_models()
        print(f"✅ Model Manager is working properly. - {len(available_models)} available models")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 CV Model Platform - Environment Settings")
    print("=" * 50)
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Check your Python version
    if check_python_version():
        success_count += 1
    else:
        print("\n❌ Python version check failed, cannot continue")
        return 1
    
    # Step 2: Check dependencies
    if check_dependencies():
        success_count += 1
    else:
        print("\n❌ Dependency check failed, unable to continue")
        return 1
    
    # Step 3: Create a configuration file
    if create_config_files():
        success_count += 1
    
    # Step 4: Discover the model
    if discover_models():
        success_count += 1
    
    # Step 5: Verify installation
    if verify_installation():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Settings results: {success_count}/{total_steps} steps completed")
    
    if success_count >= 4:  # Allow models to fail to be found
        print("🎉 Environment setup successful!")
        print("\n🚀 Next step:")
        print("   1. Run test demo:")
        print("      python examples/basic_usage/detection_demo.py")
        print("   2. List available models:")
        print("      python examples/basic_usage/detection_demo.py --list-models")
        print("   3. View configuration file:")
        print("      cat config/models.yaml")
        
        return 0
    else:
        print("❌ The environment settings are incomplete. Please check the error message.")
        return 1

if __name__ == '__main__':
    exit(main())
