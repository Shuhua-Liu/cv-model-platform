#!/usr/bin/env python3
"""
Model Auto-Discovery Script - Enhanced model detection with intelligent grouping

This script uses the existing ModelDetector to find available models and provides:
1. Smart HuggingFace model directory detection
2. Intelligent model grouping and deduplication
3. Enhanced filtering and analysis
4. Configuration file generation

Usage:
    python scripts/models/detect_models.py
    python scripts/models/detect_models.py --models-root ~/cv_models
    python scripts/models/detect_models.py --output config/models.yaml --summary
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_detector import ModelDetector, ModelInfo, get_model_detector
    from src.cv_platform.core.config_manager import get_config_manager
    from src.cv_platform.utils.logger import setup_logger
    setup_logger("INFO")
    from loguru import logger
    
    # Check for PyYAML
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed. Configuration file generation may fail.")
        print("Install with: pip install PyYAML")
        yaml = None
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure this script is run from the project root directory.")
    sys.exit(1)


class IntelligentModelFilter:
    """Enhanced filtering and grouping for detected models"""
    
    # HuggingFace model indicators
    HF_CONFIG_FILES = {
        'config.json', 'model_index.json', 'pytorch_model.bin.index.json'
    }
    
    # Known component patterns to group/deduplicate
    COMPONENT_PATTERNS = {
        'stable_diffusion': ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler'],
        'clip': ['vision_model', 'text_model'],
        'blip': ['vision_model', 'text_decoder'],
        'detectron2': ['model_final', 'config']
    }
    
    def __init__(self, models_root: Path):
        """Initialize the intelligent filter"""
        self.models_root = Path(models_root)
        self.processed_paths: Set[Path] = set()
        
    def filter_models(self, raw_models: List[ModelInfo]) -> List[ModelInfo]:
        """
        Apply intelligent filtering to remove redundant models
        
        Args:
            raw_models: Raw list of detected models
            
        Returns:
            Filtered list with duplicates and components removed
        """
        logger.info(f"Filtering {len(raw_models)} raw models...")
        
        # Step 1: Group models by directory to detect HuggingFace structures
        models_by_dir = self._group_models_by_directory(raw_models)
        
        # Step 2: Identify HuggingFace model directories
        hf_models = self._identify_huggingface_models(models_by_dir)
        
        # Step 3: Filter out component files from HuggingFace models
        filtered_models = self._filter_component_files(raw_models, hf_models)
        
        # Step 4: Remove low-confidence duplicates
        final_models = self._remove_duplicates(filtered_models)
        
        logger.info(f"Filtered to {len(final_models)} unique models")
        return final_models
    
    def _group_models_by_directory(self, models: List[ModelInfo]) -> Dict[Path, List[ModelInfo]]:
        """Group models by their parent directory"""
        models_by_dir = defaultdict(list)
        
        for model in models:
            parent_dir = model.path.parent
            models_by_dir[parent_dir].append(model)
        
        return dict(models_by_dir)
    
    def _identify_huggingface_models(self, models_by_dir: Dict[Path, List[ModelInfo]]) -> Set[Path]:
        """Identify directories that contain HuggingFace models"""
        hf_directories = set()
        
        for directory, models in models_by_dir.items():
            # Check for HuggingFace config files
            has_config = any(
                (directory / config_file).exists() 
                for config_file in self.HF_CONFIG_FILES
            )
            
            # Check for multiple component-like files
            component_count = len([
                model for model in models 
                if any(component in model.name.lower() 
                      for patterns in self.COMPONENT_PATTERNS.values() 
                      for component in patterns)
            ])
            
            # Check for typical HuggingFace structure
            has_model_files = len(models) > 1
            has_pytorch_bin = any('pytorch_model' in model.name for model in models)
            has_safetensors = any('model.safetensors' in model.name for model in models)
            
            if (has_config or 
                (component_count >= 2) or 
                (has_model_files and (has_pytorch_bin or has_safetensors))):
                
                hf_directories.add(directory)
                logger.debug(f"Identified HuggingFace model directory: {directory}")
        
        return hf_directories
    
    def _filter_component_files(self, models: List[ModelInfo], hf_directories: Set[Path]) -> List[ModelInfo]:
        """Filter out individual component files from HuggingFace models"""
        filtered_models = []
        
        for model in models:
            model_dir = model.path.parent
            
            # If this is inside a HuggingFace directory, create a representative model
            if model_dir in hf_directories:
                # Check if we already processed this directory
                if model_dir not in self.processed_paths:
                    # Create a directory-level model entry
                    dir_model = self._create_directory_model(model_dir, models)
                    if dir_model:
                        filtered_models.append(dir_model)
                    self.processed_paths.add(model_dir)
            else:
                # Regular single-file model
                filtered_models.append(model)
        
        return filtered_models
    
    def _create_directory_model(self, directory: Path, all_models: List[ModelInfo]) -> Optional[ModelInfo]:
        """Create a representative model for a HuggingFace directory"""
        # Get all models in this directory
        dir_models = [m for m in all_models if m.path.parent == directory]
        
        if not dir_models:
            return None
        
        # Use the first model as a template and modify it
        template_model = dir_models[0]
        
        # Calculate total directory size
        total_size = sum(m.file_size_mb for m in dir_models)
        
        # Determine model characteristics from directory name and config
        model_type, framework, architecture = self._analyze_directory_model(directory)
        
        # Create enhanced metadata
        metadata = {
            'is_huggingface_directory': True,
            'component_files': [m.name for m in dir_models],
            'component_count': len(dir_models),
            'directory_path': str(directory),
            'config_files': [
                str(f) for f in directory.glob('*.json') 
                if f.name in self.HF_CONFIG_FILES
            ]
        }
        
        # Try to read config for additional info
        config_path = directory / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    metadata['hf_config'] = config_data
                    
                    # Extract additional info from config
                    if '_class_name' in config_data:
                        metadata['model_class'] = config_data['_class_name']
                    if 'architectures' in config_data:
                        metadata['architectures'] = config_data['architectures']
                        
            except Exception as e:
                logger.debug(f"Could not read config from {config_path}: {e}")
        
        # Create the directory model
        return ModelInfo(
            name=directory.name,
            path=directory,  # Point to directory instead of file
            type=model_type,
            framework=framework,
            architecture=architecture,
            confidence=0.95,  # High confidence for HuggingFace models
            file_size_mb=total_size,
            last_modified=max(m.last_modified for m in dir_models),
            metadata=metadata
        )
    
    def _analyze_directory_model(self, directory: Path) -> Tuple[str, str, str]:
        """Analyze directory to determine model characteristics"""
        dir_name = directory.name.lower()
        
        # Check for known patterns
        if any(pattern in dir_name for pattern in ['stable-diffusion', 'sd-', 'sdxl']):
            return 'generation', 'diffusers', 'stable_diffusion'
        elif 'clip' in dir_name:
            return 'multimodal', 'transformers', 'clip'
        elif 'blip' in dir_name:
            return 'multimodal', 'transformers', 'blip'
        elif any(pattern in dir_name for pattern in ['llava', 'multimodal']):
            return 'multimodal', 'transformers', 'multimodal'
        elif any(pattern in dir_name for pattern in ['bert', 'roberta', 'gpt']):
            return 'text', 'transformers', 'language_model'
        
        # Check parent directory structure for hints
        parent_parts = [p.lower() for p in directory.parts]
        if 'generation' in parent_parts:
            return 'generation', 'huggingface', 'unknown'
        elif 'multimodal' in parent_parts:
            return 'multimodal', 'huggingface', 'unknown'
        elif 'classification' in parent_parts:
            return 'classification', 'huggingface', 'unknown'
        
        return 'unknown', 'huggingface', 'unknown'
    
    def _remove_duplicates(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Remove duplicate models based on name and characteristics"""
        seen_models = {}
        unique_models = []
        
        for model in models:
            # Create a key based on name and characteristics
            key = (model.name.lower(), model.type, model.framework)
            
            if key not in seen_models:
                seen_models[key] = model
                unique_models.append(model)
            else:
                # Keep the one with higher confidence
                existing = seen_models[key]
                if model.confidence > existing.confidence:
                    # Replace in both dict and list
                    seen_models[key] = model
                    unique_models[unique_models.index(existing)] = model
        
        return unique_models


def generate_models_config(models: List[ModelInfo], output_path: Path) -> Dict[str, Any]:
    """
    Generate YAML configuration from detected models
    
    Args:
        models: List of detected models
        output_path: Path to save the configuration file
    
    Returns:
        Generated configuration dictionary
    """
    if not yaml:
        raise ImportError("PyYAML is required for configuration generation")
    
    config = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_models': len(models),
            'detection_method': 'enhanced_detector',
            'version': '2.0',
            'generator': 'cv_platform_detect_models'
        },
        'models': {}
    }
    
    # Group models by type for statistics
    models_by_type = defaultdict(list)
    for model in models:
        models_by_type[model.type].append(model)
    
    # Generate configuration for each model
    for model in models:
        # Create a clean model name for config key
        config_name = model.name
        
        # Remove file extensions for cleaner names
        if config_name.endswith(('.pt', '.pth', '.safetensors', '.ckpt', '.onnx', '.bin')):
            config_name = Path(config_name).stem
        
        # Ensure unique config names
        base_name = config_name
        counter = 1
        while config_name in config['models']:
            config_name = f"{base_name}_{counter}"
            counter += 1
        
        # Build model configuration
        model_config = {
            'path': str(model.path),
            'type': model.type,
            'framework': model.framework,
            'architecture': model.architecture,
            'device': 'auto',  # Let the system decide
            'enabled': True,
            'metadata': {
                'size_mb': round(model.file_size_mb, 2),
                'confidence': round(model.confidence, 2),
                'last_modified': model.last_modified,
                'is_huggingface': model.metadata.get('is_huggingface_directory', False)
            }
        }
        
        # Add type-specific configurations
        if model.type == 'detection':
            model_config.update({
                'confidence_threshold': 0.25,
                'nms_threshold': 0.45,
                'batch_size': 1,
                'input_size': [640, 640]
            })
        elif model.type == 'segmentation':
            if 'sam' in model.architecture.lower():
                model_config.update({
                    'points_per_side': 32,
                    'pred_iou_thresh': 0.88,
                    'stability_score_thresh': 0.95,
                    'batch_size': 1
                })
            else:
                model_config.update({
                    'batch_size': 1,
                    'input_size': [512, 512]
                })
        elif model.type == 'generation':
            model_config.update({
                'num_inference_steps': 20,
                'guidance_scale': 7.5,
                'enable_memory_efficient_attention': True,
                'batch_size': 1
            })
        elif model.type == 'classification':
            model_config.update({
                'top_k': 5,
                'batch_size': 4,
                'input_size': [224, 224]
            })
        elif model.type == 'multimodal':
            model_config.update({
                'batch_size': 1,
                'max_text_length': 77
            })
        
        # Add HuggingFace specific metadata
        if model.metadata.get('is_huggingface_directory'):
            model_config['metadata'].update({
                'component_files': model.metadata.get('component_files', []),
                'component_count': model.metadata.get('component_count', 0)
            })
            
            # Add config info if available
            if 'hf_config' in model.metadata:
                hf_config = model.metadata['hf_config']
                if 'model_type' in hf_config:
                    model_config['metadata']['model_type'] = hf_config['model_type']
                if '_class_name' in hf_config:
                    model_config['metadata']['model_class'] = hf_config['_class_name']
        
        # Add file-specific metadata
        model_config['metadata'].update({
            'file_extension': model.path.suffix if model.path.is_file() else 'directory',
            'relative_path': str(model.path.relative_to(model.path.parent.parent)) if len(model.path.parts) > 1 else str(model.path)
        })
        
        config['models'][config_name] = model_config
    
    # Add summary statistics
    config['summary'] = {
        'by_type': {model_type: len(type_models) for model_type, type_models in models_by_type.items()},
        'by_framework': {},
        'total_size_mb': round(sum(m.file_size_mb for m in models), 2),
        'average_size_mb': round(sum(m.file_size_mb for m in models) / len(models), 2) if models else 0,
        'high_confidence_models': len([m for m in models if m.confidence >= 0.9]),
        'huggingface_models': len([m for m in models if m.metadata.get('is_huggingface_directory', False)])
    }
    
    # Framework statistics
    framework_counts = defaultdict(int)
    for model in models:
        framework_counts[model.framework] += 1
    config['summary']['by_framework'] = dict(framework_counts)
    
    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        logger.info(f"Configuration saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise
    
    return config


def print_summary(models: List[ModelInfo], total_size_mb: float) -> None:
    """Print summary of detected models"""
    print(f"\n📊 Enhanced Model Detection Summary")
    print("=" * 60)
    print(f"Total models found: {len(models)}")
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"Average size: {total_size_mb/len(models):.1f} MB per model" if models else "N/A")
    
    # Count high-confidence models
    high_conf_models = [m for m in models if m.confidence >= 0.9]
    print(f"High confidence models (≥90%): {len(high_conf_models)}")
    
    # Count HuggingFace models
    hf_models = [m for m in models if m.metadata.get('is_huggingface_directory', False)]
    print(f"HuggingFace directory models: {len(hf_models)}")
    
    # Group by type
    by_type = defaultdict(list)
    for model in models:
        by_type[model.type].append(model)
    
    print(f"\nModels by type:")
    for model_type, type_models in sorted(by_type.items()):
        type_size = sum(m.file_size_mb for m in type_models)
        avg_conf = sum(m.confidence for m in type_models) / len(type_models)
        print(f"  📁 {model_type:12s}: {len(type_models):2d} models ({type_size:6.1f} MB, avg conf: {avg_conf:.2f})")
        
        # Show breakdown by framework
        by_framework = defaultdict(int)
        for model in type_models:
            by_framework[model.framework] += 1
        
        for framework, count in sorted(by_framework.items()):
            print(f"      └─ {framework:10s}: {count:2d} models")
    
    # Group by framework
    by_framework = defaultdict(int)
    for model in models:
        by_framework[model.framework] += 1
    
    print(f"\nModels by framework:")
    for framework, count in sorted(by_framework.items()):
        print(f"  🔧 {framework:12s}: {count:2d} models")


def print_detailed_results(models: List[ModelInfo]) -> None:
    """Print detailed information about detected models"""
    print(f"\n📋 Detailed Model Information")
    print("=" * 80)
    
    for i, model in enumerate(models, 1):
        # Confidence icon
        if model.confidence >= 0.9:
            confidence_icon = "🟢"
        elif model.confidence >= 0.7:
            confidence_icon = "🟡"
        else:
            confidence_icon = "🔴"
        
        # HuggingFace icon
        hf_icon = "🤗" if model.metadata.get('is_huggingface_directory', False) else ""
        
        print(f"\n{i:2d}. {confidence_icon} {hf_icon} {model.name}")
        print(f"    Type: {model.type:12s} | Framework: {model.framework:12s} | Architecture: {model.architecture}")
        print(f"    Path: {model.path}")
        print(f"    Size: {model.file_size_mb:8.1f} MB | Confidence: {model.confidence:.2f}")
        
        # Show HuggingFace specific info
        if model.metadata.get('is_huggingface_directory'):
            component_count = model.metadata.get('component_count', 0)
            print(f"    🤗 HuggingFace directory with {component_count} component files")
            
            # Show some component files
            components = model.metadata.get('component_files', [])
            if components:
                shown_components = components[:3]
                if len(components) > 3:
                    shown_components.append(f"... +{len(components)-3} more")
                print(f"    Components: {', '.join(shown_components)}")
        
        # Show additional metadata
        metadata_items = []
        if model.metadata.get('model_class'):
            metadata_items.append(f"class: {model.metadata['model_class']}")
        if model.metadata.get('file_extension'):
            metadata_items.append(f"format: {model.metadata['file_extension']}")
        
        if metadata_items:
            print(f"    Metadata: {' | '.join(metadata_items)}")


def main():
    """Main function with enhanced argument parsing"""
    parser = argparse.ArgumentParser(
        description='Enhanced CV model detection with intelligent filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/models/detect_models.py
  python scripts/models/detect_models.py --models-root ~/cv_models --summary
  python scripts/models/detect_models.py --detailed --min-confidence 0.8
  python scripts/models/detect_models.py --filter-components --output my_models.yaml
        """
    )
    
    parser.add_argument('--models-root', 
                      type=str,
                      default=None,
                      help='Model root directory path (default: from config)')
    
    parser.add_argument('--output', '-o',
                      type=str,
                      default=None,
                      help='Output configuration file path (default: config/models.yaml)')
    
    parser.add_argument('--summary', '-s',
                      action='store_true',
                      help='Show detection summary only')
    
    parser.add_argument('--detailed', '-d',
                      action='store_true', 
                      help='Show detailed model information')
    
    parser.add_argument('--min-confidence',
                      type=float,
                      default=0.5,
                      help='Minimum confidence threshold (default: 0.5)')
    
    parser.add_argument('--filter-components',
                      action='store_true',
                      help='Apply intelligent component filtering for HuggingFace models')
    
    parser.add_argument('--force-rescan',
                      action='store_true',
                      help='Force rescan (ignore cache)')
    
    parser.add_argument('--export-format',
                      choices=['yaml', 'json', 'csv'],
                      default='yaml',
                      help='Export format for results')
    
    parser.add_argument('--verbose', '-v',
                      action='store_true',
                      help='Verbose output')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    try:
        # Determine models root directory
        if args.models_root:
            models_root = Path(args.models_root)
        else:
            config_manager = get_config_manager()
            models_root = config_manager.get_models_root()
        
        if not models_root.exists():
            print(f"❌ Models root directory does not exist: {models_root}")
            print("Please check the path or use --models-root to specify the correct path")
            return 1
        
        print(f"🔍 Scanning models directory: {models_root}")
        
        # Create model detector
        detector = get_model_detector(models_root)
        
        # Perform model detection
        start_time = time.time()
        raw_models = detector.detect_models(force_rescan=args.force_rescan)
        detection_time = time.time() - start_time
        
        if not raw_models:
            print("⚠️  No models found")
            print("\n💡 Suggestions:")
            print("   1. Check if model files exist in the specified directory")
            print("   2. Ensure file formats are supported (.pt, .pth, .safetensors, .onnx, etc.)")
            print("   3. Check minimum confidence threshold")
            print("   4. For HuggingFace models, ensure they contain proper structure")
            return 0
        
        print(f"📊 Raw detection completed in {detection_time:.2f}s - found {len(raw_models)} files")
        
        # Apply intelligent filtering if requested
        if args.filter_components:
            print("🧠 Applying intelligent component filtering...")
            filter_start = time.time()
            
            intelligent_filter = IntelligentModelFilter(models_root)
            filtered_models = intelligent_filter.filter_models(raw_models)
            
            filter_time = time.time() - filter_start
            print(f"✨ Filtering completed in {filter_time:.2f}s - {len(filtered_models)} unique models")
            
            models = filtered_models
        else:
            models = raw_models
        
        # Apply confidence filtering
        if args.min_confidence > 0:
            before_count = len(models)
            models = [m for m in models if m.confidence >= args.min_confidence]
            if len(models) < before_count:
                print(f"🎯 Confidence filtering: {before_count} → {len(models)} models (≥{args.min_confidence:.1f})")
        
        if not models:
            print("⚠️  No models meet the specified criteria")
            return 0
        
        # Calculate total size
        total_size_mb = sum(model.file_size_mb for model in models)
        
        # Display results
        if args.summary or not args.detailed:
            print_summary(models, total_size_mb)
        
        if args.detailed:
            print_detailed_results(models)
        
        # Generate configuration file
        output_file = args.output
        if output_file is None:
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            output_file = config_dir / "models.yaml"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate configuration
        try:
            if args.export_format == 'yaml':
                config = generate_models_config(models, output_path)
                print(f"\n✅ Model configuration generated: {output_path}")
                print(f"📄 Contains {len(config['models'])} model configurations")
            else:
                # Use ModelDetector's export functionality
                export_data = detector.export_detection_results(args.export_format)
                
                if args.export_format == 'json':
                    output_path = output_path.with_suffix('.json')
                    with open(output_path, 'w') as f:
                        f.write(export_data)
                elif args.export_format == 'csv':
                    output_path = output_path.with_suffix('.csv')
                    with open(output_path, 'w') as f:
                        f.write(export_data)
                
                print(f"\n✅ Results exported to: {output_path}")
                
        except Exception as e:
            logger.error(f"Could not generate output file: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        # Show recommended next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review the generated file: {output_path}")
        print(f"   2. Adjust model parameters as needed")
        print(f"   3. Test model loading:")
        print(f"      python examples/basic_usage/detection_demo.py")
        print(f"   4. Start the API server:")
        print(f"      python scripts/start_api.py")
        
        # Show detection summary from ModelDetector
        if args.verbose:
            detection_summary = detector.get_detection_summary()
            print(f"\n🔧 Detection Summary:")
            print(f"   Total scan time: {detection_time:.2f}s")
            print(f"   Models root: {detection_summary['models_root']}")
            print(f"   Last scan: {time.ctime(detection_summary['last_scan_time'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Model detection failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())