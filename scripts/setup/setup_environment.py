#!/usr/bin/env python3
"""
环境设置脚本 - 一键初始化CV Model Platform环境

功能：
1. 检查Python环境
2. 检查必要依赖
3. 创建配置文件
4. 发现本地模型
5. 验证安装
"""

import sys
import subprocess
import importlib
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python版本不兼容: {version.major}.{version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """检查必要依赖"""
    print("\n📦 检查必要依赖...")
    
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
    
    # 检查必需依赖
    for module_name, package_name in required_packages:
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_required.append(package_name)
            print(f"❌ {package_name}")
    
    # 检查可选依赖
    for module_name, package_name in optional_packages:
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name} (可选)")
        except ImportError:
            missing_optional.append(package_name)
            print(f"⚠️  {package_name} (可选)")
    
    # 如果缺少必需依赖，尝试安装
    if missing_required:
        print(f"\n🔧 尝试安装缺少的必需依赖: {', '.join(missing_required)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_required)
            print("✅ 必需依赖安装成功")
        except subprocess.CalledProcessError:
            print("❌ 依赖安装失败")
            return False
    
    if missing_optional:
        print(f"\n💡 可选依赖未安装: {', '.join(missing_optional)}")
        print("   这些依赖不是必需的，但可能影响某些功能")
    
    return True

def create_config_files():
    """创建配置文件"""
    print("\n⚙️ 创建配置文件...")
    
    try:
        from src.cv_platform.core.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        config_manager.create_default_configs()
        
        print("✅ 配置文件创建成功")
        print(f"   - config/models_template.yaml")
        print(f"   - config/platform_template.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件创建失败: {e}")
        return False

def discover_models():
    """发现本地模型"""
    print("\n🔍 发现本地模型...")
    
    try:
        from src.cv_platform.core.model_detector import ModelDetector
        from src.cv_platform.core.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        models_root = config_manager.get_models_root()
        
        if not models_root.exists():
            print(f"⚠️  模型目录不存在: {models_root}")
            print("   请在该目录下放置模型文件，或使用环境变量 CV_MODELS_ROOT 指定其他路径")
            return True  # 不算失败，只是提醒
        
        detector = ModelDetector(models_root)
        models = detector.detect_models()
        
        print(f"✅ 发现 {len(models)} 个模型")
        
        if models:
            # 生成模型配置
            config_file = Path("config/models.yaml")
            config = detector.generate_config(models, config_file)
            print(f"✅ 模型配置已生成: {config_file}")
            
            # 显示发现的模型摘要
            by_type = {}
            for model in models:
                model_type = model.type
                if model_type not in by_type:
                    by_type[model_type] = 0
                by_type[model_type] += 1
            
            print("   发现的模型类型:")
            for model_type, count in by_type.items():
                print(f"     - {model_type}: {count} 个")
        else:
            print("   未发现模型文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型发现失败: {e}")
        return False

def verify_installation():
    """验证安装"""
    print("\n🧪 验证安装...")
    
    try:
        # 测试基础导入
        from src.cv_platform.core.config_manager import get_config_manager
        from src.cv_platform.core.model_manager import get_model_manager
        
        print("✅ 核心模块导入成功")
        
        # 测试配置管理器
        config_manager = get_config_manager()
        models_config = config_manager.get_models_config()
        print("✅ 配置管理器工作正常")
        
        # 测试模型管理器
        model_manager = get_model_manager()
        available_models = model_manager.list_available_models()
        print(f"✅ 模型管理器工作正常 - {len(available_models)} 个可用模型")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 CV Model Platform - 环境设置")
    print("=" * 50)
    
    success_count = 0
    total_steps = 5
    
    # 步骤1: 检查Python版本
    if check_python_version():
        success_count += 1
    else:
        print("\n❌ Python版本检查失败，无法继续")
        return 1
    
    # 步骤2: 检查依赖
    if check_dependencies():
        success_count += 1
    else:
        print("\n❌ 依赖检查失败，无法继续")
        return 1
    
    # 步骤3: 创建配置文件
    if create_config_files():
        success_count += 1
    
    # 步骤4: 发现模型
    if discover_models():
        success_count += 1
    
    # 步骤5: 验证安装
    if verify_installation():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 设置结果: {success_count}/{total_steps} 步骤完成")
    
    if success_count >= 4:  # 允许模型发现失败
        print("🎉 环境设置成功！")
        print("\n🚀 下一步操作:")
        print("   1. 运行检测演示:")
        print("      python examples/basic_usage/detection_demo.py")
        print("   2. 列出可用模型:")
        print("      python examples/basic_usage/detection_demo.py --list-models")
        print("   3. 查看配置文件:")
        print("      cat config/models.yaml")
        
        return 0
    else:
        print("❌ 环境设置不完整，请检查错误信息")
        return 1

if __name__ == '__main__':
    exit(main())