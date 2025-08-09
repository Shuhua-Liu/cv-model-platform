#!/usr/bin/env python3
"""
简单的模型检测测试脚本
"""

import sys
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, 'src')

def test_imports():
    """测试基础导入"""
    try:
        print("🔍 测试基础导入...")
        from cv_platform.utils.logger import setup_logger
        setup_logger("INFO")
        
        from cv_platform.core.config_manager import get_config_manager
        from cv_platform.core.model_detector import ModelDetector
        print("✅ 基础导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_model_detection(models_root="./cv_models"):
    """测试模型检测功能"""
    try:
        print(f"\n🔍 测试模型检测 - 目录: {models_root}")
        
        from cv_platform.core.model_detector import ModelDetector
        
        models_path = Path(models_root)
        if not models_path.exists():
            print(f"⚠️  模型目录不存在: {models_path}")
            print("请创建模型目录并放置一些模型文件")
            return False
        
        detector = ModelDetector(models_path)
        models = detector.detect_models()
        
        print(f"✅ 发现 {len(models)} 个模型")
        
        if models:
            print("\n📋 发现的模型:")
            for i, model in enumerate(models[:5], 1):  # 只显示前5个
                print(f"  {i}. {model.name}")
                print(f"     类型: {model.type}, 框架: {model.framework}")
                print(f"     大小: {model.size_mb:.1f}MB, 置信度: {model.confidence:.2f}")
                print(f"     路径: {model.path}")
        else:
            print("⚠️  未发现任何模型文件")
            
        return True
        
    except Exception as e:
        print(f"❌ 模型检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_manager():
    """测试配置管理"""
    try:
        print(f"\n🔍 测试配置管理...")
        
        from cv_platform.core.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        
        # 创建默认配置
        config_manager.create_default_configs()
        
        models_root = config_manager.get_models_root()
        cache_dir = config_manager.get_cache_dir()
        
        print(f"✅ 配置管理器工作正常")
        print(f"   模型根目录: {models_root}")
        print(f"   缓存目录: {cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 CV Model Platform 核心功能测试")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # 测试1: 基础导入
    if test_imports():
        success_count += 1
    
    # 测试2: 配置管理
    if test_config_manager():
        success_count += 1
    
    # 测试3: 模型检测
    models_root = sys.argv[1] if len(sys.argv) > 1 else "./cv_models"
    if test_model_detection(models_root):
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有核心功能正常！")
        print("\n🚀 下一步建议:")
        print("   1. 在模型目录放置一些模型文件")
        print("   2. 运行: python scripts/models/detect_models.py")
        print("   3. 检查生成的配置文件: config/models.yaml")
        return 0
    else:
        print("❌ 部分功能存在问题，请检查错误信息")
        return 1

if __name__ == '__main__':
    exit(main())