#!/usr/bin/env python3
"""
Model Auto-Discovery Script - 兼容当前ModelDetector版本

用法:
    python scripts/models/detect_models.py
    python scripts/models/detect_models.py --models-root ./cv_models
    python scripts/models/detect_models.py --output config/models.yaml --summary
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_detector import ModelDetector, ModelInfo
    from src.cv_platform.core.config_manager import get_config_manager
    from src.cv_platform.utils.logger import setup_logger
    setup_logger("INFO")
    from loguru import logger
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保此脚本从项目根目录运行，或者cv-model-platform包已正确安装。")
    sys.exit(1)


def apply_filters(models: List[ModelInfo], 
                 include_patterns: List[str] = None,
                 exclude_patterns: List[str] = None,
                 min_size_mb: float = 0.1,
                 max_size_mb: float = 50000.0) -> List[ModelInfo]:
    """
    手动应用过滤器到模型列表
    
    Args:
        models: 原始模型列表
        include_patterns: 包含模式列表
        exclude_patterns: 排除模式列表
        min_size_mb: 最小文件大小(MB)
        max_size_mb: 最大文件大小(MB)
        
    Returns:
        过滤后的模型列表
    """
    filtered_models = []
    
    for model in models:
        # 大小过滤
        if not (min_size_mb <= model.file_size_mb <= max_size_mb):
            continue
        
        # 包含模式过滤
        if include_patterns:
            if not any(pattern.lower() in model.name.lower() or 
                      pattern.lower() in str(model.path).lower() 
                      for pattern in include_patterns):
                continue
        
        # 排除模式过滤
        if exclude_patterns:
            if any(pattern.lower() in model.name.lower() or 
                  pattern.lower() in str(model.path).lower() 
                  for pattern in exclude_patterns):
                continue
        
        filtered_models.append(model)
    
    return filtered_models


def print_summary(models: List[ModelInfo], total_size_mb: float):
    """打印模型发现摘要"""
    
    print(f"\n📊 模型发现摘要")
    print("=" * 50)
    print(f"总计发现: {len(models)} 个模型")
    print(f"总大小: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    if not models:
        return
    
    # 按类型统计
    by_type = {}
    for model in models:
        model_type = model.type
        if model_type not in by_type:
            by_type[model_type] = {'count': 0, 'size': 0}
        by_type[model_type]['count'] += 1
        by_type[model_type]['size'] += model.file_size_mb
    
    print(f"\n📁 按类型分布:")
    for model_type, stats in by_type.items():
        print(f"  {model_type:<15}: {stats['count']} 个模型, {stats['size']:.1f} MB")
    
    # 按框架统计
    by_framework = {}
    for model in models:
        framework = model.framework
        if framework not in by_framework:
            by_framework[framework] = 0
        by_framework[framework] += 1
    
    print(f"\n🔧 按框架分布:")
    for framework, count in by_framework.items():
        print(f"  {framework:<15}: {count} 个模型")
    
    # 大小分布
    size_ranges = {
        '< 10MB': [0, 10],
        '10-100MB': [10, 100], 
        '100MB-1GB': [100, 1024],
        '> 1GB': [1024, float('inf')]
    }
    
    print(f"\n📏 按大小分布:")
    for range_name, (min_size, max_size) in size_ranges.items():
        count = len([m for m in models if min_size <= m.file_size_mb < max_size])
        if count > 0:
            print(f"  {range_name:<15}: {count} 个模型")
    
    # 最大和最小模型
    if models:
        largest = max(models, key=lambda m: m.file_size_mb)
        smallest = min(models, key=lambda m: m.file_size_mb)
        
        print(f"\n🏆 模型信息:")
        print(f"  最大模型: {largest.name} ({largest.file_size_mb:.1f} MB)")
        print(f"  最小模型: {smallest.name} ({smallest.file_size_mb:.1f} MB)")


def print_detailed_results(models: List[ModelInfo]):
    """打印详细的模型发现结果"""
    
    print(f"\n📋 详细模型列表")
    print("=" * 100)
    print(f"{'名称':<25} {'类型':<12} {'框架':<15} {'大小':<10} {'置信度':<8} {'路径'}")
    print("-" * 100)
    
    for model in sorted(models, key=lambda m: (m.type, m.name)):
        size_str = f"{model.file_size_mb:.1f}MB"
        confidence_str = f"{model.confidence:.2f}"
        path_str = str(model.path)
        
        # 截断过长的路径
        if len(path_str) > 40:
            path_str = "..." + path_str[-37:]
        
        print(f"{model.name:<25} {model.type:<12} {model.framework:<15} "
              f"{size_str:<10} {confidence_str:<8} {path_str}")


def validate_models(models: List[ModelInfo]) -> Dict[str, List[str]]:
    """验证发现的模型"""
    
    issues = {
        'warnings': [],
        'errors': [],
        'suggestions': []
    }
    
    # 检查文件存在性
    for model in models:
        if not model.path.exists():
            issues['errors'].append(f"模型文件不存在: {model.path}")
    
    # 检查置信度
    low_confidence_models = [m for m in models if m.confidence < 0.7]
    if low_confidence_models:
        issues['warnings'].append(f"发现 {len(low_confidence_models)} 个低置信度模型")
        for model in low_confidence_models[:3]:  # 只显示前3个
            issues['warnings'].append(f"  - {model.name} (置信度: {model.confidence:.2f})")
    
    # 检查未知类型
    unknown_models = [m for m in models if m.type == 'unknown']
    if unknown_models:
        issues['warnings'].append(f"发现 {len(unknown_models)} 个未知类型模型")
        for model in unknown_models[:3]:  # 只显示前3个
            issues['warnings'].append(f"  - {model.name}")
    
    # 检查重复名称
    names = [m.name for m in models]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    if duplicates:
        issues['warnings'].append(f"发现重复名称: {', '.join(duplicates)}")
    
    # 建议
    if len(models) == 0:
        issues['suggestions'].append("考虑将模型文件放置在支持的目录结构中")
        issues['suggestions'].append("支持的文件格式: .pt, .pth, .ckpt, .safetensors, .onnx 等")
    
    return issues


def interactive_model_selection(models: List[ModelInfo]) -> List[ModelInfo]:
    """交互式模型选择"""
    
    if not models:
        return models
    
    print(f"\n🔍 发现 {len(models)} 个模型，请选择要包含在配置中的模型:")
    print("(输入模型编号，用空格分隔，或输入 'all' 选择全部，'q' 退出)")
    
    for i, model in enumerate(models, 1):
        status = "✅" if model.confidence > 0.8 else "⚠️" if model.confidence > 0.6 else "❌"
        print(f"  {i:2d}. {status} {model.name:<25} ({model.type}, {model.file_size_mb:.1f}MB)")
    
    while True:
        try:
            selection = input("\n请选择: ").strip()
            
            if selection.lower() == 'q':
                print("❌ 用户取消操作")
                return []
            
            if selection.lower() == 'all':
                print(f"✅ 已选择全部 {len(models)} 个模型")
                return models
            
            if not selection:
                print("✅ 未选择任何模型")
                return []
            
            indices = [int(x) for x in selection.split()]
            selected_models = [models[i-1] for i in indices if 1 <= i <= len(models)]
            
            print(f"✅ 已选择 {len(selected_models)} 个模型")
            return selected_models
            
        except (ValueError, IndexError):
            print("❌ 无效输入，请输入有效的模型编号")


def test_generated_config(config_path: Path) -> bool:
    """
    测试生成的配置文件是否有效
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        测试是否通过
    """
    try:
        import yaml
        
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"  📄 配置文件格式: ✅ 有效YAML")
        
        # 检查基本结构
        if 'models_root' not in config:
            print(f"  ❌ 缺少 models_root 字段")
            return False
        
        if 'models' not in config:
            print(f"  ❌ 缺少 models 字段")
            return False
        
        print(f"  📋 配置结构: ✅ 完整")
        
        # 检查每个模型配置
        models_root = Path(config['models_root'])
        valid_models = 0
        total_models = len(config['models'])
        
        for model_name, model_config in config['models'].items():
            # 检查必需字段
            required_fields = ['type', 'path', 'framework', 'device']
            missing_fields = [field for field in required_fields if field not in model_config]
            
            if missing_fields:
                print(f"  ⚠️ 模型 {model_name} 缺少字段: {missing_fields}")
                continue
            
            # 检查文件路径
            model_path = model_config['path']
            if '{models_root}' in model_path:
                actual_path = Path(model_path.replace('{models_root}', str(models_root)))
            else:
                actual_path = Path(model_path)
            
            if actual_path.exists():
                valid_models += 1
            else:
                print(f"  ⚠️ 模型文件不存在: {actual_path}")
        
        print(f"  📁 文件检查: {valid_models}/{total_models} 个模型文件存在")
        
        # 尝试加载配置到ConfigManager
        try:
            from src.cv_platform.core.config_manager import ConfigManager
            temp_config_manager = ConfigManager()
            print(f"  🔧 ConfigManager: ✅ 可以加载")
        except Exception as e:
            print(f"  🔧 ConfigManager: ⚠️ 加载失败 - {e}")
        
        return valid_models > 0
        
    except Exception as e:
        print(f"  ❌ 配置文件测试失败: {e}")
        return False


def generate_model_config(models: List[ModelInfo], models_root: Path) -> Dict[str, Any]:
    """
    手动生成模型配置
    
    Args:
        models: 模型列表
        models_root: 模型根目录
        
    Returns:
        配置字典
    """
    config = {
        'models_root': str(models_root),
        'models': {},
        'metadata': {
            'generated_by': 'detect_models.py',
            'generated_at': __import__('time').time(),
            'total_models': len(models),
            'source_directory': str(models_root)
        }
    }
    
    for model in models:
        # 生成相对路径
        try:
            relative_path = model.path.relative_to(models_root)
            model_path = "{models_root}/" + str(relative_path).replace('\\', '/')
        except ValueError:
            # 如果无法生成相对路径，使用绝对路径
            model_path = str(model.path)
        
        # 基础配置
        model_config = {
            'type': model.type,
            'path': model_path,
            'framework': model.framework,
            'architecture': model.architecture,
            'device': 'auto'
        }
        
        # 添加类型特定的配置
        if model.type == 'detection':
            model_config.update({
                'batch_size': 4,
                'confidence': 0.25,
                'nms_threshold': 0.45,
                'max_det': 300
            })
        elif model.type == 'segmentation':
            model_config.update({
                'batch_size': 1,
                'points_per_side': 32,
                'pred_iou_thresh': 0.88,
                'stability_score_thresh': 0.95
            })
        elif model.type == 'classification':
            model_config.update({
                'batch_size': 8,
                'top_k': 5,
                'pretrained': True
            })
        elif model.type == 'multimodal':
            model_config.update({
                'batch_size': 8,
                'max_text_length': 77,
                'temperature': 0.07
            })
        elif model.type == 'generation':
            model_config.update({
                'batch_size': 1,
                'num_inference_steps': 20,
                'guidance_scale': 7.5,
                'enable_memory_efficient_attention': True
            })
        
        # 添加元数据
        if hasattr(model, 'metadata') and model.metadata:
            model_config['metadata'] = model.metadata
        
        config['models'][model.name] = model_config
    
    return config


def save_config_to_file(config: Dict[str, Any], output_path: Path):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    import yaml
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
        return True
    except Exception as e:
        print(f"❌ 保存配置文件失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='扫描并生成模型配置文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 使用默认设置扫描
  %(prog)s --models-root ./cv_models          # 指定模型目录
  %(prog)s --include yolo sam                 # 只包含YOLO和SAM模型
  %(prog)s --exclude test debug               # 排除包含test或debug的模型
  %(prog)s --min-size 10 --max-size 1000     # 只包含10MB-1GB的模型
  %(prog)s --output config/my_models.yaml    # 指定输出文件
  %(prog)s --interactive                     # 交互式选择模型
        """
    )
    
    parser.add_argument('--models-root', '-r',
                      type=str,
                      help='模型根目录路径')
    
    parser.add_argument('--output', '-o',
                      type=str,
                      help='输出配置文件路径 (默认: config/models.yaml)')
    
    parser.add_argument('--include',
                      nargs='+',
                      help='包含模式 (只包含匹配的模型)')
    
    parser.add_argument('--exclude',
                      nargs='+', 
                      help='排除模式 (排除匹配的模型)')
    
    parser.add_argument('--min-size',
                      type=float,
                      default=0.1,
                      help='最小文件大小 (MB) (默认: 0.1)')
    
    parser.add_argument('--max-size',
                      type=float,
                      default=50000.0,
                      help='最大文件大小 (MB) (默认: 50000)')
    
    parser.add_argument('--summary',
                      action='store_true',
                      help='显示摘要统计信息')
    
    parser.add_argument('--detailed',
                      action='store_true',
                      help='显示详细模型列表')
    
    parser.add_argument('--interactive', '-i',
                      action='store_true',
                      help='交互式选择要包含的模型')
    
    parser.add_argument('--validate',
                      action='store_true',
                      help='验证发现的模型')
    
    parser.add_argument('--force-rescan',
                      action='store_true',
                      help='强制重新扫描(忽略缓存)')
    
    parser.add_argument('--test-config',
                      action='store_true',
                      help='测试生成的配置文件是否有效')
    
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
            try:
                config_manager = get_config_manager()
                models_root = config_manager.get_models_root()
            except Exception as e:
                logger.warning(f"无法获取配置管理器: {e}")
                models_root = Path("./cv_models")
        
        if not models_root.exists():
            print(f"❌ 模型根目录不存在: {models_root}")
            print("请检查路径或使用 --models-root 参数指定正确的路径")
            print("\n💡 提示:")
            print(f"   1. 创建目录: mkdir -p {models_root}")
            print(f"   2. 或指定现有目录: --models-root /path/to/your/models")
            return 1
        
        print(f"🔍 扫描模型目录: {models_root}")
        
        # 创建检测器
        detector = ModelDetector(models_root)
        
        # 执行模型发现 - 使用当前API
        models = detector.detect_models(force_rescan=args.force_rescan)
        
        if not models:
            print("⚠️  未发现任何模型文件")
            print("\n💡 建议检查:")
            print("   1. 模型文件是否存在于指定目录")
            print("   2. 文件格式是否支持 (.pt, .pth, .safetensors, .onnx等)")
            print("   3. 文件大小是否合理")
            
            # 显示目录内容
            try:
                files = list(models_root.rglob("*"))
                if files:
                    print(f"\n📁 目录 {models_root} 包含 {len(files)} 个文件:")
                    for file in files[:10]:  # 只显示前10个
                        if file.is_file():
                            size_mb = file.stat().st_size / (1024*1024)
                            print(f"   📄 {file.name} ({size_mb:.1f}MB)")
                    if len(files) > 10:
                        print(f"   ... 还有 {len(files)-10} 个文件")
                else:
                    print(f"\n📁 目录 {models_root} 为空")
            except Exception as e:
                print(f"   无法列出目录内容: {e}")
            
            return 0
        
        print(f"✅ 原始扫描发现 {len(models)} 个模型")
        
        # 应用过滤器
        filtered_models = apply_filters(
            models,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
            min_size_mb=args.min_size,
            max_size_mb=args.max_size
        )
        
        if len(filtered_models) != len(models):
            print(f"🔽 过滤后剩余 {len(filtered_models)} 个模型")
        
        models = filtered_models
        
        if not models:
            print("❌ 过滤后没有模型剩余")
            print("💡 尝试调整过滤条件")
            return 0
        
        # 验证模型
        if args.validate:
            print(f"\n🔍 验证发现的模型...")
            issues = validate_models(models)
            
            if issues['errors']:
                print(f"❌ 错误:")
                for error in issues['errors']:
                    print(f"  - {error}")
            
            if issues['warnings']:
                print(f"⚠️ 警告:")
                for warning in issues['warnings']:
                    print(f"  - {warning}")
            
            if issues['suggestions']:
                print(f"💡 建议:")
                for suggestion in issues['suggestions']:
                    print(f"  - {suggestion}")
        
        # 交互式选择
        if args.interactive:
            models = interactive_model_selection(models)
            if not models:
                print("❌ 未选择任何模型")
                return 0
        
        # 计算总大小
        total_size_mb = sum(model.file_size_mb for model in models)
        
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
        try:
            print(f"\n📝 生成配置文件...")
            config = generate_model_config(models, models_root)
            
            if save_config_to_file(config, output_path):
                print(f"✅ 模型配置已生成: {output_path}")
                print(f"📄 包含 {len(config.get('models', {}))} 个模型配置")
                
                # 显示生成的配置摘要
                print(f"\n📋 配置摘要:")
                print(f"  模型根目录: {config['models_root']}")
                print(f"  配置的模型:")
                for model_name, model_config in config['models'].items():
                    print(f"    - {model_name:<20} ({model_config['type']}, {model_config['framework']})")
                
                # 显示推荐的下一步操作
                print(f"\n🚀 下一步操作:")
                print(f"   1. 检查生成的配置文件:")
                print(f"      cat {output_path}")
                print(f"   2. 启动API服务器:")
                print(f"      python src/cv_platform/api/main.py")
                print(f"   3. 测试模型列表:")
                print(f"      curl http://localhost:8000/api/v1/models")
                print(f"   4. 检查系统健康:")
                print(f"      curl http://localhost:8000/api/v1/health")
                
                # 测试配置文件
                if args.test_config:
                    print(f"\n🧪 测试配置文件...")
                    if test_generated_config(output_path):
                        print("✅ 配置文件测试通过")
                    else:
                        print("⚠️ 配置文件测试发现问题")
                
            else:
                return 1
                
        except Exception as e:
            print(f"❌ 生成配置文件失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"模型发现失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())