#!/usr/bin/env python3
"""
基于文件夹的模型检测修复方案
🎯 问题场景:
用户从HuggingFace下载了完整的模型文件夹，比如:
- generation/stable_diffusion/sd_2_1/ (包含unet, vae, text_encoder等)
- generation/flux/
- generation/controlnet/
🔧 目标:
- 检测到的模型应该是 "sd_2_1"，而不是 "sd_2_1_unet", "sd_2_1_vae" 等
- 每个HuggingFace模型文件夹只对应一个检测结果
- 路径指向文件夹，而不是单个文件
🛠️ 解决方案:
1. 检测HuggingFace模型文件夹结构
2. 以文件夹为单位进行模型识别
3. 过滤掉文件夹内的组件文件
"""
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import time
from loguru import logger

# 添加项目根目录到路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

def detect_huggingface_model_folders(models_root: Path) -> List[Dict]:
    """
    检测HuggingFace模型文件夹
    Args:
        models_root: 模型根目录
    Returns:
        检测到的模型文件夹列表
    """
    print("🔍 检测HuggingFace模型文件夹...")
    # HuggingFace模型文件夹的特征
    hf_indicators = {
        'config_files': ['config.json', 'model_index.json'],
        'model_files': ['unet', 'vae', 'text_encoder', 'safety_checker'],
        'tokenizer_files': ['tokenizer_config.json', 'tokenizer.json'],
        'scheduler_files': ['scheduler_config.json']
    }
    detected_folders = []
    # 扫描generation目录
    generation_dir = models_root / 'generation'
    if not generation_dir.exists():
        print(f"⚠️ generation目录不存在: {generation_dir}")
        return detected_folders
    # 递归查找可能的模型文件夹
    for potential_folder in generation_dir.rglob("*"):
        if not potential_folder.is_dir():
            continue
        # 检查是否是HuggingFace模型文件夹
        folder_info = analyze_model_folder(potential_folder, hf_indicators)
        if folder_info:
            detected_folders.append(folder_info)
            print(f"   ✅ 发现模型文件夹: {folder_info['name']}")
    return detected_folders

def analyze_model_folder(folder_path: Path, hf_indicators: Dict) -> Optional[Dict]:
    """
    分析文件夹是否是HuggingFace模型
    Args:
        folder_path: 文件夹路径
        hf_indicators: HuggingFace指示器
    Returns:
        模型信息字典或None
    """
    try:
        # 获取文件夹中的所有文件
        files = [f.name for f in folder_path.iterdir() if f.is_file()]
        subfolders = [f.name for f in folder_path.iterdir() if f.is_dir()]
        # 检查HuggingFace特征
        has_config = any(config in files for config in hf_indicators['config_files'])
        has_model_components = any(component in subfolders for component in hf_indicators['model_files'])
        has_model_files = any(f.endswith(('.safetensors', '.pt', '.pth', '.bin')) for f in files)
        # 计算置信度
        confidence = 0.0
        framework = 'diffusers'
        architecture = 'unknown'
        # 基于文件夹名称判断架构
        folder_name = folder_path.name.lower()
        # Stable Diffusion系列
        if any(pattern in folder_name for pattern in ['sd_1_5', 'sd1_5', 'stable-diffusion-v1-5']):
            architecture = 'stable_diffusion_1_5'
            confidence = 0.9
        elif any(pattern in folder_name for pattern in ['sd_2_1', 'sd2_1', 'stable-diffusion-2-1']):
            architecture = 'stable_diffusion_2_1'
            confidence = 0.9
        elif any(pattern in folder_name for pattern in ['sdxl', 'sd_xl', 'stable-diffusion-xl']):
            architecture = 'stable_diffusion_xl'
            confidence = 0.9
        elif 'unclip' in folder_name:
            architecture = 'stable_diffusion_unclip'
            confidence = 0.9
        # FLUX系列
        elif 'flux' in folder_name:
            architecture = 'flux'
            confidence = 0.9
        # ControlNet
        elif 'controlnet' in folder_name:
            architecture = 'controlnet'
            confidence = 0.9
        # 通用stable diffusion
        elif any(pattern in folder_name for pattern in ['stable', 'diffusion']):
            architecture = 'stable_diffusion'
            confidence = 0.7
        # 检查HuggingFace结构特征
        if has_config:
            confidence += 0.2
        if has_model_components:
            confidence += 0.3
        if has_model_files:
            confidence += 0.2
        # 必须满足最低置信度要求
        if confidence < 0.6:
            return None
        # 计算文件夹大小
        total_size_mb = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file()) / (1024 * 1024)
        return {
            'name': folder_path.name,
            'path': folder_path,
            'type': 'generation',
            'framework': framework,
            'architecture': architecture,
            'confidence': confidence,
            'folder_size_mb': total_size_mb,
            'is_folder_model': True,
            'components': {
                'files': files,
                'subfolders': subfolders,
                'has_config': has_config,
                'has_model_components': has_model_components
            }
        }
    except Exception as e:
        print(f"   ❌ 分析文件夹失败 {folder_path}: {e}")
        return None
    
def patch_model_detector_for_folders():
    """为ModelDetector应用基于文件夹的检测补丁"""
    print("🔧 应用基于文件夹的模型检测补丁...")
    try:
        from src.cv_platform.core.model_detector import ModelDetector, ModelInfo
        # 备份原方法
        original_perform_scan = ModelDetector._perform_model_scan
        original_analyze_file = ModelDetector._analyze_model_file
        def patched_perform_model_scan(self) -> int:
            """修复后的模型扫描 - 优先检测文件夹模型"""
            if self._scan_in_progress:
                logger.warning("Model scan already in progress")
                return len(self._detected_models)
            self._scan_in_progress = True
            try:
                print(f"🔍 开始扫描模型 (文件夹优先): {self.models_root}")
                detected_models = []
                processed_folders = set()  # 记录已处理的文件夹
                # 1. 首先检测HuggingFace文件夹模型
                folder_models = detect_huggingface_model_folders(self.models_root)
                for folder_info in folder_models:
                    model_info = ModelInfo(
                        name=folder_info['name'],
                        path=folder_info['path'],
                        type=folder_info['type'],
                        framework=folder_info['framework'],
                        architecture=folder_info['architecture'],
                        confidence=folder_info['confidence'],
                        file_size_mb=folder_info['folder_size_mb'],
                        last_modified=folder_info['path'].stat().st_mtime,
                        metadata={
                            'is_folder_model': True,
                            'components': folder_info['components']
                        }
                    )
                    detected_models.append(model_info)
                    processed_folders.add(folder_info['path'])
                    print(f"   📁 文件夹模型: {folder_info['name']} ({folder_info['architecture']})")
                # 2. 然后检测单文件模型（排除已处理文件夹中的文件）
                scanned_files = 0
                for root_path in self._walk_model_directories():
                    for file_path in root_path.rglob("*"):
                        if not file_path.is_file():
                            continue
                        # 检查文件是否在已处理的文件夹中
                        if any(folder in file_path.parents for folder in processed_folders):
                            continue  # 跳过文件夹模型中的文件
                        if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                            scanned_files += 1
                            try:
                                model_info = original_analyze_file(self, file_path)
                                if model_info and model_info.confidence > 0.5:
                                    detected_models.append(model_info)
                                    print(f"   📄 单文件模型: {model_info.name} ({model_info.architecture})")
                            except Exception as e:
                                print(f"   ❌ 分析文件失败 {file_path}: {e}")
                # 更新状态
                self._detected_models = detected_models
                self._last_scan_time = time.time()
                print(f"✅ 扫描完成: {len(folder_models)} 文件夹模型, {scanned_files} 文件扫描, {len(detected_models)} 总模型")
                return len(detected_models)
            finally:
                self._scan_in_progress = False
        # 应用补丁
        ModelDetector._perform_model_scan = patched_perform_model_scan
        print("✅ 基于文件夹的检测补丁已应用")
        print("🎯 改进内容:")
        print("   1. 优先检测HuggingFace模型文件夹")
        print("   2. 每个文件夹只生成一个模型记录")
        print("   3. 避免文件夹内组件文件的重复检测")
        print("   4. 保留单文件模型的检测能力")
        return True
    except Exception as e:
        print(f"❌ 补丁应用失败: {e}")
        return False
    
def test_folder_based_detection():
    """测试基于文件夹的模型检测"""
    print("\n🧪 测试基于文件夹的模型检测...")
    try:
        from src.cv_platform.core.model_detector import ModelDetector
        from src.cv_platform.core.config_manager import get_config_manager
        config_manager = get_config_manager()
        models_root = config_manager.get_models_root()
        # 创建检测器
        detector = ModelDetector(models_root)
        # 执行检测
        models = detector.detect_models(force_rescan=True)
        # 分析结果
        generation_models = [m for m in models if m.type == 'generation']
        folder_models = [m for m in generation_models if m.metadata.get('is_folder_model', False)]
        file_models = [m for m in generation_models if not m.metadata.get('is_folder_model', False)]
        print(f"🎯 检测结果:")
        print(f"   总模型数: {len(models)}")
        print(f"   生成模型: {len(generation_models)}")
        print(f"   文件夹模型: {len(folder_models)}")
        print(f"   单文件模型: {len(file_models)}")
        if folder_models:
            print(f"\n📁 文件夹模型详情:")
            for i, model in enumerate(folder_models, 1):
                components = model.metadata.get('components', {})
                print(f"   {i}. {model.name}")
                print(f"      架构: {model.architecture}")
                print(f"      大小: {model.file_size_mb:.1f}MB")
                print(f"      路径: {model.path}")
                print(f"      组件: {len(components.get('subfolders', []))} 子文件夹, {len(components.get('files', []))} 文件")
                print()
        if file_models:
            print(f"\n📄 单文件模型:")
            for model in file_models:
                print(f"   - {model.name} ({model.file_size_mb:.1f}MB)")
        return len(generation_models)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return -1
    
def create_model_config_for_folders():
    """为文件夹模型创建配置"""
    print("\n📝 生成文件夹模型配置示例...")
    config_example = """# config/models.yaml - 文件夹模型配置示例
    models:
    # HuggingFace文件夹模型
    sd_2_1:
    path: "generation/stable_diffusion/sd_2_1"  # 指向文件夹
    type: generation
    framework: diffusers
    architecture: stable_diffusion_2_1
    device: auto
    sd_2_1_unclip:
    path: "generation/stable_diffusion/sd_2_1_unclip"
    type: generation
    framework: diffusers
    architecture: stable_diffusion_unclip
    device: auto
    sdxl:
    path: "generation/stable_diffusion/sdxl"
    type: generation
    framework: diffusers
    architecture: stable_diffusion_xl
    device: auto
    flux_dev:
    path: "generation/flux"
    type: generation
    framework: diffusers
    architecture: flux
    device: auto
    controlnet_canny:
    path: "generation/controlnet"
    type: generation
    framework: diffusers
    architecture: controlnet
    device: auto
    # 使用方法：
    # 1. 模型加载时，适配器会自动从文件夹中找到需要的组件
    # 2. 路径指向文件夹，而不是特定的.safetensors文件
    # 3. diffusers库会自动处理文件夹结构
    """
    print(config_example)

def main():
    """主修复函数"""
    print("🚨 基于文件夹的模型检测修复工具")
    print("=" * 60)
    print("问题: HuggingFace文件夹被拆分成多个组件模型")
    print("目标: 每个文件夹对应一个模型 (如 sd_2_1)")
    print("方案: 优先检测文件夹，过滤组件文件\n")
    # 1. 应用修复补丁
    patch_success = patch_model_detector_for_folders()
    if not patch_success:
        print("❌ 修复补丁应用失败")
        return False
    # 2. 测试修复结果
    generation_count = test_folder_based_detection()
    if generation_count == -1:
        print("❌ 测试失败")
        return False
    # 3. 生成配置示例
    create_model_config_for_folders()
    # 4. 评估修复效果
    print(f"\n📈 修复效果评估:")
    if generation_count <= 10:
        print(f"✅ 优秀: 生成模型数量控制在 {generation_count} 个")
        print("   文件夹模型检测成功，每个HuggingFace模型对应一个记录")
    elif generation_count <= 20:
        print(f"✅ 良好: 生成模型数量为 {generation_count} 个")
        print("   大幅改善，但可能还有一些单文件模型")
    else:
        print(f"⚠️ 需要进一步优化: 仍有 {generation_count} 个模型")
    print(f"\n💡 使用建议:")
    print("=" * 40)
    print("1. 模型路径现在指向文件夹:")
    print("   generation/stable_diffusion/sd_2_1/")
    print("   而不是具体的 .safetensors 文件")
    print()
    print("2. 在使用时，diffusers会自动:")
    print("   - 加载 unet/diffusion_pytorch_model.safetensors")
    print("   - 加载 vae/diffusion_pytorch_model.safetensors")
    print("   - 加载 text_encoder/pytorch_model.safetensors")
    print("   - 以及其他必要组件")
    print()
    print("3. 配置文件示例已生成，可以参考使用")
    return generation_count <= 15

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 修复被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 修复过程中出现未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
