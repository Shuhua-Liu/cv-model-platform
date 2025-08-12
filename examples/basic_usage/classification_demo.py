#!/usr/bin/env python3
"""
图像分类演示脚本

展示如何使用CV Model Platform进行图像分类
支持ResNet、EfficientNet、ViT等分类模型
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径

project_root = Path(**file**).parent.parent.parent
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
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np


        # 创建一个包含明显特征的测试图像
        width, height = 224, 224  # 分类模型常用尺寸
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # 创建一个简单的"狗"形状（圆形头部 + 椭圆身体）
        # 头部
        head_center = (width // 2, height // 3)
        head_radius = 40
        draw.ellipse([
            head_center[0] - head_radius, head_center[1] - head_radius,
            head_center[0] + head_radius, head_center[1] + head_radius
        ], fill='brown', outline='black', width=2)
        
        # 身体
        body_center = (width // 2, height * 2 // 3)
        body_width, body_height = 60, 40
        draw.ellipse([
            body_center[0] - body_width, body_center[1] - body_height,
            body_center[0] + body_width, body_center[1] + body_height
        ], fill='brown', outline='black', width=2)
        
        # 眼睛
        eye_size = 5
        left_eye = (head_center[0] - 15, head_center[1] - 10)
        right_eye = (head_center[0] + 15, head_center[1] - 10)
        draw.ellipse([left_eye[0] - eye_size, left_eye[1] - eye_size,
                    left_eye[0] + eye_size, left_eye[1] + eye_size], fill='black')
        draw.ellipse([right_eye[0] - eye_size, right_eye[1] - eye_size,
                    right_eye[0] + eye_size, right_eye[1] + eye_size], fill='black')
        
        # 鼻子
        nose = (head_center[0], head_center[1] + 5)
        draw.ellipse([nose[0] - 3, nose[1] - 2, nose[0] + 3, nose[1] + 2], fill='black')
        
        # 腿
        leg_positions = [
            (body_center[0] - 30, body_center[1] + 25),  # 左前腿
            (body_center[0] - 10, body_center[1] + 25),  # 左后腿
            (body_center[0] + 10, body_center[1] + 25),  # 右前腿
            (body_center[0] + 30, body_center[1] + 25),  # 右后腿
        ]
        
        for leg_pos in leg_positions:
            draw.rectangle([
                leg_pos[0] - 5, leg_pos[1], 
                leg_pos[0] + 5, leg_pos[1] + 20
            ], fill='brown', outline='black')
        
        # 添加一些背景纹理
        for _ in range(20):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            color = (np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255))
            draw.point((x, y), fill=color)
        
        # 保存测试图像
        test_image_path = Path("test_classification_image.jpg")
        img.save(test_image_path)
        
        logger.info(f"分类测试图像已创建: {test_image_path}")
        return str(test_image_path)
    
    except ImportError:
        logger.error("PIL未安装，无法创建测试图像")
        return None
    except Exception as e:
        logger.error(f"创建测试图像失败: {e}")
        return None


def test_classification(model_name, image_path, top_k=5):
"""测试图像分类功能"""
    try:
    logger.info(f"测试图像分类: {model_name}")


        manager = get_model_manager()
        
        # 加载模型
        logger.info("加载分类模型...")
        results = manager.predict(model_name, image_path, top_k=top_k)
        
        logger.info("分类完成")
        
        # 显示结果
        if 'predictions' in results and len(results['predictions']) > 0:
            predictions = results['predictions']
            top_class = results.get('top_class', predictions[0]['class'])
            top_confidence = results.get('top_confidence', predictions[0]['confidence'])
            
            logger.info(f"🎯 最佳预测: {top_class} (置信度: {top_confidence:.3f})")
            logger.info(f"📊 前 {len(predictions)} 个预测结果:")
            
            for i, pred in enumerate(predictions, 1):
                class_name = pred['class']
                confidence = pred['confidence']
                class_id = pred.get('class_id', 'N/A')
                
                # 添加信心程度指示
                if confidence > 0.7:
                    confidence_icon = "🟢"
                elif confidence > 0.3:
                    confidence_icon = "🟡"
                else:
                    confidence_icon = "🔴"
                
                logger.info(f" {i}. {confidence_icon} {class_name}")
                logger.info(f" 置信度: {confidence:.3f} | 类别ID: {class_id}")
        else:
            logger.warning("未获得分类结果")
        
        # 尝试可视化（如果适配器支持）
        try:
            adapter = manager.load_model(model_name)
            if hasattr(adapter, 'visualize_results'):
                vis_result = adapter.visualize_results(
                    image_path, 
                    results, 
                    save_path="classification_result.jpg"
                )
                logger.info("分类可视化结果已保存: classification_result.jpg")
        except Exception as e:
            logger.debug(f"可视化失败（这是正常的，因为分类模型通常不需要可视化）: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"分类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_classification(model_name, image_paths, top_k=3):
"""测试批量分类"""
    try:
    logger.info(f"测试批量分类: {model_name}")
    logger.info(f"批量大小: {len(image_paths)}")


        manager = get_model_manager()
        adapter = manager.load_model(model_name)
        
        # 检查是否支持批量预测
        if hasattr(adapter, 'predict_batch'):
            logger.info("使用批量预测接口...")
            batch_results = adapter.predict_batch(image_paths, top_k=top_k)
        else:
            logger.info("逐个预测...")
            batch_results = []
            for img_path in image_paths:
                result = adapter.predict(img_path, top_k=top_k)
                batch_results.append(result)
        
        # 显示批量结果
        logger.info("📊 批量分类结果:")
        for i, (img_path, result) in enumerate(zip(image_paths, batch_results), 1):
            if 'predictions' in result and result['predictions']:
                top_pred = result['predictions'][0]
                logger.info(f"   {i}. {Path(img_path).name}: {top_pred['class']} ({top_pred['confidence']:.3f})")
            else:
                logger.info(f"   {i}. {Path(img_path).name}: 分类失败")
        
        return True
        
    except Exception as e:
        logger.error(f"批量分类测试失败: {e}")
        return False


def test_model_comparison(models, image_path, top_k=3):
"""比较不同模型的分类结果"""
    try:
    logger.info("🔄 模型对比测试")
    logger.info(f"参与对比的模型: {’, ’.join(models)}")


        manager = get_model_manager()
        comparison_results = {}
        
        for model_name in models:
            try:
                logger.info(f"测试模型: {model_name}")
                result = manager.predict(model_name, image_path, top_k=top_k)
                comparison_results[model_name] = result
            except Exception as e:
                logger.warning(f"模型 {model_name} 测试失败: {e}")
                comparison_results[model_name] = None
        
        # 显示对比结果
        logger.info("📋 模型对比结果:")
        print("=" * 80)
        print(f"{'模型名称':<20} {'最佳预测':<25} {'置信度':<10} {'前3预测'}")
        print("-" * 80)
        
        for model_name, result in comparison_results.items():
            if result and 'predictions' in result and result['predictions']:
                top_pred = result['predictions'][0]
                top_3 = [p['class'][:10] for p in result['predictions'][:3]]
                print(f"{model_name:<20} {top_pred['class'][:24]:<25} {top_pred['confidence']:<10.3f} {', '.join(top_3)}")
            else:
                print(f"{model_name:<20} {'失败':<25} {'N/A':<10} {'N/A'}")
        
        print("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"模型对比失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=‘CV Model Platform 图像分类演示’)


    parser.add_argument('--model', '-m',
                    type=str,
                    help='要使用的模型名称')

    parser.add_argument('--image', '-i',
                    type=str,
                    help='测试图像路径（如果不提供将创建测试图像）')

    parser.add_argument('--top-k', '-k',
                    type=int,
                    default=5,
                    help='返回前K个预测结果 (默认: 5)')

    parser.add_argument('--list-models', '-l',
                    action='store_true',
                    help='列出所有可用的分类模型')

    parser.add_argument('--test-all',
                    action='store_true',
                    help='测试所有可用的分类模型')

    parser.add_argument('--batch-test',
                    action='store_true',
                    help='测试批量分类（需要多个图像）')

    parser.add_argument('--compare',
                    action='store_true',
                    help='比较多个模型的分类结果')

    parser.add_argument('--images',
                    type=str,
                    nargs='+',
                    help='多个图像路径（用于批量测试）')

    parser.add_argument('--verbose', '-v',
                    action='store_true',
                    help='详细输出')

    args = parser.parse_args()

    if args.verbose:
        setup_logger("DEBUG")

    try:
        manager = get_model_manager()
        available_models = manager.list_available_models()
        
        # 筛选分类模型
        classification_models = {
            name: info for name, info in available_models.items()
            if info['config'].get('type') == 'classification'
        }
        
        if args.list_models:
            print("\n📋 可用的分类模型:")
            print("=" * 60)
            
            if not classification_models:
                print("⚠️  未找到分类模型")
                print("\n💡 建议:")
                print("   1. 确保模型目录包含分类模型文件")
                print("   2. 运行模型发现脚本更新配置")
                print("   3. 分类模型通常使用torchvision预训练模型")
                return 0
            
            for name, info in classification_models.items():
                config = info['config']
                framework = config.get('framework', 'unknown')
                architecture = config.get('architecture', 'unknown')
                
                print(f"🔧 {name}")
                print(f"   架构: {architecture}")
                print(f"   框架: {framework}")
                print(f"   路径: {config.get('path', 'unknown')}")
                
                # 检查依赖
                if framework == 'torchvision':
                    print("   依赖: torchvision (通常已安装)")
                elif framework == 'timm':
                    print("   依赖: timm (需要安装: pip install timm)")
                elif framework == 'transformers':
                    print("   依赖: transformers (需要安装: pip install transformers)")
                print()
            
            return 0
        
        if not classification_models:
            print("❌ 未找到任何分类模型")
            print("请运行: python examples/basic_usage/classification_demo.py --list-models")
            return 1
        
        # 准备测试图像
        if args.batch_test and args.images:
            image_paths = args.images
            for img_path in image_paths:
                if not Path(img_path).exists():
                    logger.error(f"图像文件不存在: {img_path}")
                    return 1
        else:
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
        
        print("🚀 CV Model Platform - 图像分类演示")
        print("=" * 50)
        
        success_count = 0
        total_tests = 0
        
        if args.test_all:
            # 测试所有分类模型
            for model_name in classification_models.keys():
                print(f"\n🧪 测试模型: {model_name}")
                print("-" * 30)
                
                total_tests += 1
                
                if test_classification(model_name, image_path, args.top_k):
                    success_count += 1
        
        elif args.batch_test:
            # 批量测试
            if not args.images:
                logger.error("批量测试需要提供多个图像路径 --images")
                return 1
            
            model_name = args.model
            if not model_name:
                model_name = next(iter(classification_models.keys()))
                logger.info(f"使用第一个可用的分类模型: {model_name}")
            
            total_tests = 1
            if test_batch_classification(model_name, args.images, args.top_k):
                success_count = 1
        
        elif args.compare:
            # 模型对比
            model_names = list(classification_models.keys())[:3]  # 最多比较3个模型
            if len(model_names) < 2:
                logger.warning("至少需要2个模型才能进行对比")
                return 1
            
            total_tests = 1
            if test_model_comparison(model_names, image_path, args.top_k):
                success_count = 1
        
        else:
            # 测试指定模型或第一个可用模型
            if args.model:
                if args.model not in classification_models:
                    logger.error(f"模型 {args.model} 不可用")
                    logger.info("可用的分类模型:")
                    for name in classification_models.keys():
                        logger.info(f"  - {name}")
                    return 1
                model_name = args.model
            else:
                model_name = next(iter(classification_models.keys()))
                logger.info(f"使用第一个可用的分类模型: {model_name}")
            
            total_tests = 1
            if test_classification(model_name, image_path, args.top_k):
                success_count = 1
        
        print("\n" + "=" * 50)
        print(f"📊 测试结果: {success_count}/{total_tests} 通过")
        
        if success_count > 0:
            print("🎉 分类演示完成！")
            print("\n🚀 接下来可以尝试:")
            print("   1. 使用自己的图像: python examples/basic_usage/classification_demo.py -i your_image.jpg")
            print("   2. 调整top-k值: python examples/basic_usage/classification_demo.py -k 10")
            print("   3. 批量测试: python examples/basic_usage/classification_demo.py --batch-test --images img1.jpg img2.jpg")
            print("   4. 模型对比: python examples/basic_usage/classification_demo.py --compare")
            print("   5. 测试所有模型: python examples/basic_usage/classification_demo.py --test-all")
            return 0 if success_count == total_tests else 1
        else:
            print("❌ 分类演示失败")
            print("\n💡 可能的解决方案:")
            print("   1. 安装缺少的依赖: pip install timm transformers")
            print("   2. 检查模型文件路径")
            print("   3. 确保有可用的分类模型")
            print("   4. 运行: python examples/basic_usage/classification_demo.py --list-models")
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