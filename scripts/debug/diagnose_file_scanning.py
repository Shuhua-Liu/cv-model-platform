#!/usr/bin/env python3
"""
诊断文件扫描问题 - 为什么找不到 sd_2_1
🎯 关键步骤诊断：
1. 检查目录是否存在
2. 检查文件是否存在
3. 检查文件扩展名是否支持
4. 检查置信度阈值过滤
5. 检查_analyze_model_file逻辑
"""
import sys
from pathlib import Path
# 添加项目根目录到路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
def check_directory_structure(models_root: Path):
   """检查目录结构"""
   print("📁 检查目录结构...")
   generation_dir = models_root / 'generation'
   if not generation_dir.exists():
       print(f"❌ generation目录不存在: {generation_dir}")
       return {}
   print(f"✅ generation目录存在: {generation_dir}")
   # 查找所有包含sd的目录
   target_patterns = ['sd_2_1', 'sd2_1', 'stable_diffusion', 'stable-diffusion']
   found_dirs = {}
   for pattern in target_patterns:
       matching_dirs = list(generation_dir.rglob(f"*{pattern}*"))
       if matching_dirs:
           for dir_path in matching_dirs:
               if dir_path.is_dir():
                   found_dirs[pattern] = dir_path
                   print(f"   🎯 找到匹配目录 '{pattern}': {dir_path}")
   return found_dirs
def check_files_in_directories(found_dirs: dict):
   """检查目录中的文件"""
   print("\n📄 检查目录中的文件...")
   # 支持的扩展名（来自ModelDetector）
   SUPPORTED_EXTENSIONS = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.pkl'}
   all_files = []
   for pattern, directory in found_dirs.items():
       print(f"\n🔍 检查目录: {directory}")
       # 获取所有文件
       files = list(directory.rglob("*"))
       model_files = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
       print(f"   📊 总文件数: {len(files)}")
       print(f"   🎯 模型文件数: {len(model_files)}")
       if model_files:
           print(f"   📋 前10个模型文件:")
           for i, file_path in enumerate(model_files[:10], 1):
               size_mb = file_path.stat().st_size / (1024 * 1024)
               print(f"      {i}. {file_path.name} ({size_mb:.1f}MB)")
               all_files.append(file_path)
       else:
           print(f"   ❌ 未找到支持的模型文件")
   return all_files
def test_model_detection_logic(models_root: Path, all_files: list):
   """测试模型检测逻辑"""
   print(f"\n🧪 测试模型检测逻辑...")
   try:
       from src.cv_platform.core.model_detector import ModelDetector
       # 创建检测器
       detector = ModelDetector(models_root)
       print(f"🔧 检测器配置:")
       print(f"   models_root: {detector.models_root}")
       print(f"   支持的扩展名: {detector.SUPPORTED_EXTENSIONS}")
       # 手动测试每个文件
       detected_models = []
       skipped_files = []
       for file_path in all_files[:20]:  # 只测试前20个文件
           print(f"\n📝 测试文件: {file_path.name}")
           # 检查扩展名
           if file_path.suffix.lower() not in detector.SUPPORTED_EXTENSIONS:
               print(f"   🚫 扩展名不支持: {file_path.suffix}")
               skipped_files.append((file_path, "不支持的扩展名"))
               continue
           try:
               # 调用_analyze_model_file
               model_info = detector._analyze_model_file(file_path)
               if model_info:
                   print(f"   ✅ 检测成功:")
                   print(f"      名称: {model_info.name}")
                   print(f"      类型: {model_info.type}")
                   print(f"      框架: {model_info.framework}")
                   print(f"      架构: {model_info.architecture}")
                   print(f"      置信度: {model_info.confidence:.2f}")
                   # 检查置信度阈值
                   if model_info.confidence > 0.5:
                       detected_models.append(model_info)
                       print(f"      🎯 通过置信度阈值")
                   else:
                       print(f"      🚫 置信度过低 ({model_info.confidence:.2f} ≤ 0.5)")
                       skipped_files.append((file_path, f"置信度过低: {model_info.confidence:.2f}"))
               else:
                   print(f"   ❌ 检测失败: _analyze_model_file返回None")
                   skipped_files.append((file_path, "_analyze_model_file返回None"))
           except Exception as e:
               print(f"   💥 检测异常: {e}")
               skipped_files.append((file_path, f"异常: {e}"))
       # 汇总结果
       print(f"\n📊 检测结果汇总:")
       print(f"   检测成功: {len(detected_models)}")
       print(f"   跳过文件: {len(skipped_files)}")
       # 查找目标模型
       target_models = []
       for model in detected_models:
           if any(target in model.name.lower() for target in ['sd_2_1', 'sd2_1']):
               target_models.append(model)
       print(f"   目标模型: {len(target_models)}")
       if target_models:
           print(f"\n🎯 找到的目标模型:")
           for model in target_models:
               print(f"      - {model.name} ({model.confidence:.2f})")
       else:
           print(f"\n❌ 未找到目标模型 (sd_2_1)")
           # 分析跳过的文件
           print(f"\n🔍 分析跳过的文件:")
           sd_related_skipped = []
           for file_path, reason in skipped_files:
               if 'sd_2_1' in str(file_path).lower():
                   sd_related_skipped.append((file_path, reason))
           if sd_related_skipped:
               print(f"   发现 {len(sd_related_skipped)} 个被跳过的sd_2_1相关文件:")
               for file_path, reason in sd_related_skipped:
                   print(f"      - {file_path}: {reason}")
           else:
               print(f"   未发现sd_2_1相关的被跳过文件")
       return detected_models, skipped_files
   except Exception as e:
       print(f"💥 测试失败: {e}")
       import traceback
       traceback.print_exc()
       return [], []
def test_walk_directories(models_root: Path):
   """测试目录遍历逻辑"""
   print(f"\n🚶 测试目录遍历逻辑...")
   try:
       from src.cv_platform.core.model_detector import ModelDetector
       detector = ModelDetector(models_root)
       # 调用_walk_model_directories
       directories_to_scan = detector._walk_model_directories()
       print(f"🗂️ 要扫描的目录:")
       for i, directory in enumerate(directories_to_scan, 1):
           print(f"   {i}. {directory}")
       # 统计每个目录下的文件
       total_files = 0
       supported_files = 0
       for directory in directories_to_scan:
           if not directory.exists():
               print(f"   ❌ 目录不存在: {directory}")
               continue
           files = list(directory.rglob("*"))
           dir_files = [f for f in files if f.is_file()]
           dir_supported = [f for f in dir_files if f.suffix.lower() in detector.SUPPORTED_EXTENSIONS]
           total_files += len(dir_files)
           supported_files += len(dir_supported)
           if 'sd_2_1' in str(directory).lower():
               print(f"   🎯 关键目录 {directory}:")
               print(f"      总文件: {len(dir_files)}")
               print(f"      支持的文件: {len(dir_supported)}")
               if dir_supported:
                   print(f"      文件列表:")
                   for file_path in dir_supported[:5]:
                       print(f"         - {file_path.name}")
       print(f"\n📊 遍历统计:")
       print(f"   扫描目录数: {len(directories_to_scan)}")
       print(f"   总文件数: {total_files}")
       print(f"   支持的文件: {supported_files}")
       return directories_to_scan
   except Exception as e:
       print(f"💥 目录遍历测试失败: {e}")
       return []
def provide_fix_suggestions(found_dirs: dict, detected_models: list, skipped_files: list):
   """提供修复建议"""
   print(f"\n💡 修复建议:")
   print("=" * 50)
   if not found_dirs:
       print("1. 🔍 目录问题:")
       print("   - 检查sd_2_1目录是否存在于generation目录下")
       print("   - 确认目录路径是否正确")
       print("   - 检查目录权限")
   if found_dirs and not detected_models:
       print("1. 📄 文件问题:")
       print("   - 检查目录下是否有支持的模型文件 (.pt, .pth, .safetensors等)")
       print("   - 确认文件不是空文件或损坏文件")
       # 分析跳过原因
       skip_reasons = {}
       for _, reason in skipped_files:
           skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
       if skip_reasons:
           print("2. 🚫 跳过原因统计:")
           for reason, count in skip_reasons.items():
               print(f"   - {reason}: {count} 个文件")
   print(f"\n🔧 可能的修复方法:")
   print("1. 降低置信度阈值:")
   print("   在 ModelDetector._perform_model_scan() 中")
   print("   将 'confidence > 0.5' 改为 'confidence > 0.3'")
   print()
   print("2. 增加调试日志:")
   print("   在 _analyze_model_file 中添加 logger.debug 输出")
   print()
   print("3. 检查_generate_model_name逻辑:")
   print("   确保生成的模型名称包含'sd_2_1'")
def main():
   """主诊断函数"""
   print("🚨 诊断文件扫描问题")
   print("=" * 60)
   print("问题: 最初扫描就找不到 sd_2_1 模型")
   print("目标: 诊断文件扫描的每个步骤\n")
   try:
       # 获取模型根目录
       from src.cv_platform.core.config_manager import get_config_manager
       config_manager = get_config_manager()
       models_root = config_manager.get_models_root()
       print(f"📁 模型根目录: {models_root}")
       # 1. 检查目录结构
       found_dirs = check_directory_structure(models_root)
       if not found_dirs:
           print("❌ 未找到任何相关目录，请检查目录结构")
           return
       # 2. 检查文件
       all_files = check_files_in_directories(found_dirs)
       if not all_files:
           print("❌ 未找到任何模型文件")
           return
       # 3. 测试目录遍历
       directories_to_scan = test_walk_directories(models_root)
       # 4. 测试模型检测逻辑
       detected_models, skipped_files = test_model_detection_logic(models_root, all_files)
       # 5. 提供修复建议
       provide_fix_suggestions(found_dirs, detected_models, skipped_files)
       print(f"\n🎯 诊断完成!")
   except Exception as e:
       print(f"💥 诊断失败: {e}")
       import traceback
       traceback.print_exc()
if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       print("\n\n👋 诊断被用户中断")
   except Exception as e:
       print(f"\n\n💥 诊断过程中出现未预期的错误: {e}")
       import traceback
       traceback.print_exc()
