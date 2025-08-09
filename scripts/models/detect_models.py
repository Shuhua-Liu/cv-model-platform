#!/usr/bin/env python3
"""
Model Auto-Discovery Script - Scans and generates model configuration files

Usage:
    python scripts/models/detect_models.py
    python scripts/models/detect_models.py --models-root H:/cv_models
    python scripts/models/detect_models.py --output config/models.yaml --summary
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_detector import ModelDetector
    from src.cv_platform.core.config_manager import get_config_manager
    from src.cv_platform.utils.logger import setup_logger
    setup_logger("INFO")
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure this script is run from the project root directory, or that the cv-model-platform package is correctly installed.")
    print("The missing file(s) might be:")
    print("- src/cv_platform/__init__.py")
    print("- src/cv_platform/core/__init__.py")
    print("- src/cv_platform/utils/__init__.py")
    sys.exit(1)


def print_summary(models, total_size_mb: float):
    """打印模型发现摘要"""
    print("\n" + "="*60)
    print("📊 模型发现摘要")
    print("="*60)
    
    print(f"🔍 发现模型总数: {len(models)}")
    print(f"💾 总文件大小: {total_size_mb/1024:.2f} GB")
    
    # 按类型统计
    by_type = {}
    by_framework = {}
    by_confidence = {'high': 0, 'medium': 0, 'low': 0}
    
    for model in models:
        # 按类型
        model_type = model.type
        if model_type not in by_type:
            by_type[model_type] = []
        by_type[model_type].append(model)
        
        # 按框架
        framework = model.framework
        if framework not in by_framework:
            by_framework[framework] = []
        by_framework[framework].append(model)
        
        # 按置信度
        if model.confidence > 0.7:
            by_confidence['high'] += 1
        elif model.confidence > 0.3:
            by_confidence['medium'] += 1
        else:
            by_confidence['low'] += 1
    
    # 显示按类型统计
    print(f"\n📁 按类型统计:")
    for model_type, type_models in by_type.items():
        count = len(type_models)
        size = sum(m.size_mb for m in type_models)
        print(f"  {model_type:15s}: {count:3d} 个模型, {size:8.1f} MB")
    
    # 显示按框架统计
    print(f"\n🔧 按框架统计:")
    for framework, framework_models in by_framework.items():
        count = len(framework_models)
        size = sum(m.size_mb for m in framework_models)
        print(f"  {framework:15s}: {count:3d} 个模型, {size:8.1f} MB")
    
    # 显示置信度统计
    print(f"\n🎯 检测置信度:")
    print(f"  高置信度 (>0.7): {by_confidence['high']:3d} 个")
    print(f"  中置信度 (0.3-0.7): {by_confidence['medium']:3d} 个") 
    print(f"  低置信度 (<0.3): {by_confidence['low']:3d} 个")


def print_detailed_results(models):
    """打印详细的模型信息"""
    print("\n" + "="*80)
    print("📋 发现的模型详情")
    print("="*80)
    
    # 按置信度排序
    models.sort(key=lambda x: x.confidence, reverse=True)
    
    for i, model in enumerate(models, 1):
        confidence_icon = "🟢" if model.confidence > 0.7 else "🟡" if model.confidence > 0.3 else "🔴"
        
        print(f"\n{i:2d}. {confidence_icon} {model.name}")
        print(f"    类型: {model.type:12s} | 框架: {model.framework:12s} | 架构: {model.architecture}")
        print(f"    路径: {model.path}")
        print(f"    大小: {model.size_mb:8.1f} MB | 格式: {model.format:10s} | 置信度: {model.confidence:.2f}")
        
        if model.metadata:
            metadata_str = []
            for key, value in model.metadata.items():
                if key in ['total_parameters', 'file_hash']:
                    if key == 'total_parameters' and isinstance(value, (int, float)):
                        metadata_str.append(f"{key}: {value/1e6:.1f}M params")
                    else:
                        metadata_str.append(f"{key}: {value}")
            
            if metadata_str:
                print(f"    元数据: {' | '.join(metadata_str[:3])}")


def main():
    parser = argparse.ArgumentParser(description='扫描并发现本地CV模型文件')
    
    parser.add_argument('--models-root', 
                      type=str,
                      default=None,
                      help='模型根目录路径 (默认使用配置文件中的路径)')
    
    parser.add_argument('--output', '-o',
                      type=str,
                      default=None,
                      help='输出配置文件路径 (默认: config/models.yaml)')
    
    parser.add_argument('--summary', '-s',
                      action='store_true',
                      help='显示发现摘要')
    
    parser.add_argument('--detailed', '-d',
                      action='store_true', 
                      help='显示详细模型信息')
    
    parser.add_argument('--min-size',
                      type=float,
                      default=0.1,
                      help='最小文件大小(MB) (默认: 0.1)')
    
    parser.add_argument('--max-size',
                      type=float,
                      default=50000,
                      help='最大文件大小(MB) (默认: 50000)')
    
    parser.add_argument('--include',
                      type=str,
                      nargs='+',
                      help='包含的文件名模式')
    
    parser.add_argument('--exclude',
                      type=str,
                      nargs='+', 
                      help='排除的文件名模式')
    
    parser.add_argument('--rescan',
                      action='store_true',
                      help='重新扫描（忽略缓存）')
    
    parser.add_argument('--verbose', '-v',
                      action='store_true',
                      help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    try:
        # 确定模型根目录
        if args.models_root:
            models_root = Path(args.models_root)
        else:
            config_manager = get_config_manager()
            models_root = config_manager.get_models_root()
        
        if not models_root.exists():
            print(f"❌ 模型根目录不存在: {models_root}")
            print("请检查路径或使用 --models-root 参数指定正确的路径")
            return 1
        
        print(f"🔍 扫描模型目录: {models_root}")
        
        # 创建检测器
        detector = ModelDetector(models_root)
        
        # 执行模型发现
        models = detector.detect_models(
            include_patterns=args.include,
            exclude_patterns=args.exclude,
            min_size_mb=args.min_size,
            max_size_mb=args.max_size
        )
        
        if not models:
            print("⚠️  未发现任何模型文件")
            print("\n💡 建议检查:")
            print("   1. 模型文件是否存在于指定目录")
            print("   2. 文件格式是否支持 (.pt, .pth, .safetensors, .onnx等)")
            print("   3. 文件大小是否在指定范围内")
            return 0
        
        # 计算总大小
        total_size_mb = sum(model.size_mb for model in models)
        
        # 显示结果
        if args.summary or not args.detailed:
            print_summary(models, total_size_mb)
        
        if args.detailed:
            print_detailed_results(models)
        
        # 生成配置文件
        output_file = args.output
        if output_file is None:
            # 默认输出路径
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            output_file = config_dir / "models.yaml"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成并保存配置
        config = detector.generate_config(models, output_path)
        
        print(f"\n✅ 模型配置已生成: {output_path}")
        print(f"📄 包含 {len(config['models'])} 个模型配置")
        
        # 显示推荐的下一步操作
        print(f"\n🚀 下一步操作:")
        print(f"   1. 检查生成的配置文件: {output_path}")
        print(f"   2. 根据需要调整模型参数")
        print(f"   3. 运行测试脚本验证模型加载:")
        print(f"      python examples/basic_usage/detection_demo.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"模型发现失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())