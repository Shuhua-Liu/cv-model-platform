#!/usr/bin/env python3
"""
Image Segmentation Demo Script

Shows how to use the CV Model Platform for image segmentation.
Supports DeepLabV3 and SAM models.
"""

import sys
import argparse
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_manager import get_model_manager
    from src.cv_platform.utils.logger import setup_logger
    setup_logger("INFO")
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the dependencies are installed correctly and run from the project root directory")
    sys.exit(1)

def create_test_image():
    """Create a test image"""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Creating a test image that is more suitable for segmentation
        width, height = 640, 480
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes as segmentation objects
        # Background
        draw.rectangle([0, 0, width, height], fill='lightblue')
        
        # Main object
        draw.ellipse([150, 100, 350, 300], fill='red', outline='darkred', width=3)
        draw.rectangle([400, 150, 580, 350], fill='green', outline='darkgreen', width=3)
        draw.polygon([(50, 350), (150, 250), (250, 350), (150, 450)], fill='yellow', outline='orange', width=3)
        
        # Small object
        draw.ellipse([450, 50, 550, 150], fill='purple', outline='darkviolet', width=2)
        draw.rectangle([50, 50, 120, 120], fill='orange', outline='darkorange', width=2)
        
        # Save the test image
        test_image_path = Path("test_segmentation_image.jpg")
        img.save(test_image_path)
        
        logger.info(f"Segmentation test images were created: {test_image_path}")
        return str(test_image_path)
        
    except ImportError:
        logger.error("PIL is not installed, cannot create test image")
        return None
    except Exception as e:
        logger.error(f"Creating a test image failed: {e}")
        return None

def test_deeplabv3_segmentation(model_name, image_path):
    """Testing DeepLabV3 segmentation"""
    try:
        logger.info(f"Testing DeepLabV3 segmentation: {model_name}")
        
        manager = get_model_manager()
        
        # Load model
        logger.info("Loading the DeepLabV3 model...")
        results = manager.predict(model_name, image_path, threshold=0.5)
        
        logger.info("DeepLabV3 segmentation completed")
        
        # Show results
        if 'masks' in results and len(results['masks']) > 0:
            logger.info(f"{len(results['masks'])} segmented regions found")
            
            for i, (class_id, class_name, score, area) in enumerate(zip(
                results.get('class_ids', []),
                results.get('class_names', []),
                results.get('scores', []),
                results.get('areas', [])
            )):
                logger.info(f"  {i+1}. {class_name} (ID: {class_id})")
                logger.info(f"     Confidence: {score:.3f}, Area: {area:.0f} pixels")
        else:
            logger.warning("No segmentation region found")
        
        # Try visualization
        try:
            adapter = manager.load_model(model_name)
            vis_result = adapter.visualize_results(
                image_path, 
                results, 
                save_path="deeplabv3_result.jpg"
            )
            logger.info("DeepLabV3 visualization results have been saved: deeplabv3_result.jpg")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"DeepLabV3 segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sam_segmentation(model_name, image_path, mode="automatic"):
    """Testing SAM segmentation"""
    try:
        logger.info(f"Testing SAM segmentation: {model_name} (Mode: {mode})")
        
        manager = get_model_manager()
        adapter = manager.load_model(model_name)
        
        if mode == "automatic":
            # Automatic segmentation
            logger.info("Perform SAM automatic segmentation...")
            results = adapter.predict(image_path, mode="automatic")
            
        elif mode == "point":
            # Click to segment - click in the center of the image
            logger.info("Perform SAM click split...")
            results = adapter.predict_point(image_path, point=(320, 240), label=1)
            
        elif mode == "box":
            # Frame selection segmentation - select the central area of the image
            logger.info("Perform SAM frame selection segmentation...")
            results = adapter.predict_box(image_path, box=(200, 150, 450, 350))
            
        else:
            raise ValueError(f"Unsupported SAM mode: {mode}")
        
        logger.info("SAM segmentation completed")
        
        # Show results
        if 'masks' in results and len(results['masks']) > 0:
            logger.info(f"{len(results['masks'])} segmentation masks found")
            
            scores = results.get('scores', [])
            areas = results.get('areas', [])
            
            for i, (score, area) in enumerate(zip(scores, areas)):
                logger.info(f"  Mask {i+1}: score: {score:.3f}, area: {area:.0f} pixels")
                
            # Show statistics
            if scores:
                logger.info(f"Average scores: {sum(scores)/len(scores):.3f}")
                logger.info(f"Total area: {sum(areas):.0f} pixels")
        else:
            logger.warning("Segmentation mask not found")
        
        # Visualization
        try:
            vis_result = adapter.visualize_results(
                image_path, 
                results, 
                save_path=f"sam_{mode}_result.jpg"
            )
            logger.info(f"SAM visualization results have been saved: sam_{mode}_result.jpg")
        except Exception as e:
            logger.warning(f"Visualization failure: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"SAM Segmentation Test Failure: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='CV Model Platform 图像分割演示')
    
    parser.add_argument('--model', '-m',
                      type=str,
                      help='要使用的模型名称')
    
    parser.add_argument('--image', '-i',
                      type=str,
                      help='测试图像路径（如果不提供将创建测试图像）')
    
    parser.add_argument('--mode',
                      type=str,
                      choices=['automatic', 'point', 'box'],
                      default='automatic',
                      help='SAM Segmentation Mode')
    
    parser.add_argument('--list-models', '-l',
                      action='store_true',
                      help='列出所有可用的分割模型')
    
    parser.add_argument('--test-all',
                      action='store_true',
                      help='测试所有可用的分割模型')
    
    parser.add_argument('--verbose', '-v',
                      action='store_true',
                      help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logger("DEBUG")
    
    try:
        manager = get_model_manager()
        available_models = manager.list_available_models()
        
        # 筛选分割模型
        segmentation_models = {
            name: info for name, info in available_models.items()
            if info['config'].get('type') == 'segmentation'
        }
        
        if args.list_models:
            print("\n📋 可用的分割模型:")
            print("=" * 60)
            
            if not segmentation_models:
                print("⚠️  未找到分割模型")
                print("\n💡 建议:")
                print("   1. 确保模型目录包含分割模型文件")
                print("   2. 运行模型发现脚本更新配置")
                return 0
            
            for name, info in segmentation_models.items():
                config = info['config']
                framework = config.get('framework', 'unknown')
                architecture = config.get('architecture', 'unknown')
                
                print(f"🔧 {name}")
                print(f"   Architecture: {architecture}")
                print(f"   Framework: {framework}")
                print(f"   Path: {config.get('path', 'unknown')}")
                
                # 检查依赖
                if framework == 'torchvision':
                    print("   依赖: torchvision (通常已安装)")
                elif framework == 'segment_anything':
                    print("   依赖: segment-anything (需要安装)")
                print()
            
            return 0
        
        if not segmentation_models:
            print("❌ 未找到任何分割模型")
            print("请运行: python examples/basic_usage/segmentation_demo.py --list-models")
            return 1
        
        # 准备测试图像
        image_path = args.image
        if not image_path:
            image_path = create_test_image()
            if not image_path:
                logger.error("无法创建测试图像")
                return 1
        
        test_image = Path(image_path)
        if not test_image.exists():
            logger.error(f"测试图像不存在: {test_image}")
            return 1
        
        print("🚀 CV Model Platform - 图像分割演示")
        print("=" * 50)
        
        success_count = 0
        total_tests = 0
        
        if args.test_all:
            # 测试所有分割模型
            for model_name, info in segmentation_models.items():
                framework = info['config'].get('framework', 'unknown')
                
                print(f"\n🧪 测试模型: {model_name}")
                print("-" * 30)
                
                total_tests += 1
                
                if framework == 'torchvision':
                    success = test_deeplabv3_segmentation(model_name, image_path)
                elif framework == 'segment_anything':
                    success = test_sam_segmentation(model_name, image_path, args.mode)
                else:
                    logger.warning(f"未知框架: {framework}")
                    success = False
                
                if success:
                    success_count += 1
                
        else:
            # 测试指定模型或第一个可用模型
            if args.model:
                if args.model not in segmentation_models:
                    logger.error(f"模型 {args.model} 不可用")
                    logger.info("可用的分割模型:")
                    for name in segmentation_models.keys():
                        logger.info(f"  - {name}")
                    return 1
                model_name = args.model
            else:
                model_name = next(iter(segmentation_models.keys()))
                logger.info(f"使用第一个可用的分割模型: {model_name}")
            
            framework = segmentation_models[model_name]['config'].get('framework', 'unknown')
            
            total_tests = 1
            
            if framework == 'torchvision':
                success = test_deeplabv3_segmentation(model_name, image_path)
            elif framework == 'segment_anything':
                success = test_sam_segmentation(model_name, image_path, args.mode)
            else:
                logger.error(f"不支持的框架: {framework}")
                success = False
            
            if success:
                success_count = 1
        
        print("\n" + "=" * 50)
        print(f"📊 测试结果: {success_count}/{total_tests} 通过")
        
        if success_count > 0:
            print("🎉 分割演示完成！")
            print("\n🚀 接下来可以尝试:")
            print("   1. 使用自己的图像: python examples/basic_usage/segmentation_demo.py -i your_image.jpg")
            print("   2. 尝试SAM交互模式: python examples/basic_usage/segmentation_demo.py --mode point")
            print("   3. 测试所有模型: python examples/basic_usage/segmentation_demo.py --test-all")
            return 0 if success_count == total_tests else 1
        else:
            print("❌ 分割演示失败")
            print("\n💡 可能的解决方案:")
            print("   1. 安装缺少的依赖: pip install segment-anything")
            print("   2. 检查模型文件路径")
            print("   3. 运行: python examples/basic_usage/segmentation_demo.py --list-models")
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
