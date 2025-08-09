#!/usr/bin/env python3
"""
目标检测演示脚本

展示如何使用CV Model Platform进行目标检测
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_manager import get_model_manager
    from src.cv_platform.utils.logger import setup_logger
    setup_logger("INFO")
    from loguru import logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已正确安装依赖并从项目根目录运行")
    sys.exit(1)

def create_test_image():
    """创建一个测试图像"""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # 创建一个简单的测试图像
        width, height = 640, 480
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # 绘制一些简单的形状作为"对象"
        # 绘制矩形
        draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=2)
        draw.rectangle([300, 150, 450, 300], fill='green', outline='black', width=2)
        draw.rectangle([200, 300, 350, 400], fill='blue', outline='black', width=2)
        
        # 绘制圆形
        draw.ellipse([450, 50, 550, 150], fill='yellow', outline='black', width=2)
        
        # 保存测试图像
        test_image_path = Path("test_image.jpg")
        img.save(test_image_path)
        
        logger.info(f"测试图像已创建: {test_image_path}")
        return str(test_image_path)
        
    except ImportError:
        logger.error("PIL未安装，无法创建测试图像")
        return None
    except Exception as e:
        logger.error(f"创建测试图像失败: {e}")
        return None

def test_model_detection(model_name="yolov8n", image_path=None):
    """测试目标检测功能"""
    
    # 获取模型管理器
    logger.info("初始化模型管理器...")
    manager = get_model_manager()
    
    # 列出可用模型
    available_models = manager.list_available_models()
    logger.info(f"发现 {len(available_models)} 个可用模型")
    
    for name, info in available_models.items():
        model_type = info['config'].get('type', 'unknown')
        logger.info(f"  - {name}: {model_type}")
    
    # 检查指定的模型是否可用
    if model_name not in available_models:
        logger.error(f"模型 {model_name} 不可用")
        logger.info("可用的检测模型:")
        detection_models = [name for name, info in available_models.items() 
                          if info['config'].get('type') == 'detection']
        
        if detection_models:
            for name in detection_models:
                logger.info(f"  - {name}")
            model_name = detection_models[0]
            logger.info(f"使用第一个可用的检测模型: {model_name}")
        else:
            logger.error("没有找到任何检测模型")
            return False
    
    # 准备测试图像
    if not image_path:
        image_path = create_test_image()
        if not image_path:
            logger.error("无法创建测试图像")
            return False
    
    test_image = Path(image_path)
    if not test_image.exists():
        logger.error(f"测试图像不存在: {test_image}")
        return False
    
    try:
        # 加载并测试模型
        logger.info(f"加载模型: {model_name}")
        
        # 方法1: 直接使用模型管理器
        logger.info("开始预测...")
        results = manager.predict(model_name, str(test_image))
        
        logger.info(f"检测完成 - 发现 {len(results)} 个对象")
        
        # 显示结果
        if results:
            logger.info("检测结果:")
            for i, detection in enumerate(results, 1):
                class_name = detection['class']
                confidence = detection['confidence']
                bbox = detection['bbox']
                
                logger.info(f"  {i}. {class_name}: {confidence:.3f}")
                logger.info(f"     边界框: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            logger.warning("未检测到任何对象")
        
        # 方法2: 直接使用适配器（可选）
        logger.info("\n测试直接适配器调用...")
        adapter = manager.load_model(model_name)
        direct_results = adapter.predict(str(test_image))
        
        logger.info(f"直接调用结果: {len(direct_results)} 个对象")
        
        # 获取模型信息
        model_info = adapter.get_model_info()
        logger.info("模型信息:")
        logger.info(f"  适配器: {model_info.get('adapter_class', 'unknown')}")
        logger.info(f"  设备: {model_info.get('device', 'unknown')}")
        logger.info(f"  已加载: {model_info.get('is_loaded', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='CV Model Platform 目标检测演示')
    
    parser.add_argument('--model', '-m',
                      type=str,
                      default='yolov8n',
                      help='要使用的模型名称')
    
    parser.add_argument('--image', '-i',
                      type=str,
                      help='测试图像路径（如果不提供将创建测试图像）')
    
    parser.add_argument('--list-models', '-l',
                      action='store_true',
                      help='列出所有可用模型')
    
    parser.add_argument('--verbose', '-v',
                      action='store_true',
                      help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logger("DEBUG")
    
    try:
        if args.list_models:
            # 只列出模型
            logger.info("获取可用模型列表...")
            manager = get_model_manager()
            available_models = manager.list_available_models()
            
            print("\n📋 可用模型:")
            print("=" * 60)
            
            for name, info in available_models.items():
                config = info['config']
                model_type = config.get('type', 'unknown')
                framework = config.get('framework', 'unknown')
                source = info.get('source', 'unknown')
                
                print(f"🔧 {name}")
                print(f"   类型: {model_type}")
                print(f"   框架: {framework}")
                print(f"   来源: {source}")
                print(f"   路径: {config.get('path', 'unknown')}")
                print()
            
            return 0
        
        # 运行检测测试
        print("🚀 CV Model Platform - 目标检测演示")
        print("=" * 50)
        
        success = test_model_detection(args.model, args.image)
        
        if success:
            print("\n✅ 检测演示完成！")
            print("\n🎉 CV Model Platform 工作正常")
            print("\n🚀 接下来可以尝试:")
            print("   1. 使用自己的图像: python examples/basic_usage/detection_demo.py -i your_image.jpg")
            print("   2. 尝试其他模型: python examples/basic_usage/detection_demo.py -m model_name")
            print("   3. 查看所有模型: python examples/basic_usage/detection_demo.py --list-models")
            return 0
        else:
            print("\n❌ 检测演示失败")
            return 1
            
    except KeyboardInterrupt:
        print("\n用户取消操作")
        return 0
    except Exception as e:
        logger.error(f"程序异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
