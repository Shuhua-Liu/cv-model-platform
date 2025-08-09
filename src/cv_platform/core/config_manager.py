"""
配置管理器 - 处理平台的所有配置需求

支持配置优先级：环境变量 > 用户配置文件 > 默认模板
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config文件夹
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置文件路径
        self.user_models_config = self.config_dir / "models.yaml"
        self.user_platform_config = self.config_dir / "platform.yaml" 
        self.template_models_config = self.config_dir / "models_template.yaml"
        self.template_platform_config = self.config_dir / "platform_template.yaml"
        
        # 内存中的配置
        self._models_config = {}
        self._platform_config = {}
        
        # 加载配置
        self._load_configs()
    
    def _load_configs(self):
        """加载所有配置文件"""
        logger.info("加载配置文件...")
        
        # 加载模型配置
        self._models_config = self._load_models_config()
        
        # 加载平台配置
        self._platform_config = self._load_platform_config()
        
        logger.info(f"配置加载完成 - 发现 {len(self._models_config.get('models', {}))} 个模型配置")
    
    def _load_models_config(self) -> Dict[str, Any]:
        """加载模型配置，支持优先级覆盖"""
        config = {}
        
        # 1. 首先加载模板配置（默认值）
        template_config = self._load_yaml_file(self.template_models_config)
        if template_config:
            config.update(template_config)
            logger.debug(f"已加载模型模板配置: {self.template_models_config}")
        
        # 2. 然后加载用户配置（覆盖默认值）
        user_config = self._load_yaml_file(self.user_models_config)
        if user_config:
            config = self._merge_configs(config, user_config)
            logger.debug(f"已加载用户模型配置: {self.user_models_config}")
        
        # 3. 最后应用环境变量（最高优先级）
        config = self._apply_env_overrides(config, "models")
        
        return config
    
    def _load_platform_config(self) -> Dict[str, Any]:
        """加载平台配置"""
        config = {}
        
        # 1. 加载模板配置
        template_config = self._load_yaml_file(self.template_platform_config)
        if template_config:
            config.update(template_config)
        
        # 2. 加载用户配置
        user_config = self._load_yaml_file(self.user_platform_config)
        if user_config:
            config = self._merge_configs(config, user_config)
        
        # 3. 应用环境变量
        config = self._apply_env_overrides(config, "platform")
        
        return config
    
    def _load_yaml_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """安全加载YAML文件"""
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"无法加载配置文件 {file_path}: {e}")
            return None
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        env_prefix = "CV_"
        
        # 通用环境变量映射
        env_mappings = {
            "CV_MODELS_ROOT": ("models_root",),
            "CV_CACHE_DIR": ("cache_dir",),
            "CV_LOG_LEVEL": ("logging", "level"),
            "CV_API_HOST": ("api", "host"), 
            "CV_API_PORT": ("api", "port"),
            "CV_GPU_DEVICES": ("gpu", "devices"),
            "CV_MAX_BATCH_SIZE": ("inference", "max_batch_size"),
        }
        
        # 特定模型的环境变量 (CV_MODEL_{MODEL_NAME}_PATH)
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                if env_key in env_mappings:
                    # 处理预定义的环境变量
                    keys = env_mappings[env_key]
                    self._set_nested_value(config, keys, env_value)
                    logger.debug(f"应用环境变量覆盖: {env_key} = {env_value}")
                
                elif env_key.startswith("CV_MODEL_") and env_key.endswith("_PATH"):
                    # 处理模型路径覆盖: CV_MODEL_YOLOV8_PATH
                    model_name = env_key[9:-5].lower()  # 移除 CV_MODEL_ 和 _PATH
                    if "models" not in config:
                        config["models"] = {}
                    if model_name not in config["models"]:
                        config["models"][model_name] = {}
                    config["models"][model_name]["path"] = env_value
                    logger.debug(f"应用模型路径覆盖: {model_name} = {env_value}")
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], keys: tuple, value: str):
        """设置嵌套字典的值"""
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 尝试转换数据类型
        final_key = keys[-1]
        if value.lower() in ('true', 'false'):
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        elif value.replace('.', '', 1).isdigit():
            current[final_key] = float(value)
        else:
            current[final_key] = value
    
    def get_models_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self._models_config.copy()
    
    def get_platform_config(self) -> Dict[str, Any]:
        """获取平台配置"""
        return self._platform_config.copy()
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取特定模型的配置"""
        models = self._models_config.get("models", {})
        return models.get(model_name)
    
    def get_models_root(self) -> Path:
        """获取模型根目录"""
        # 优先级：环境变量 > 配置文件 > 默认值
        models_root = (
            os.environ.get("CV_MODELS_ROOT") or
            self._platform_config.get("models_root") or
            self._models_config.get("models_root") or
            "./cv_models"  # 默认使用相对路径
        )
        return Path(models_root)
    
    def get_cache_dir(self) -> Path:
        """获取缓存目录"""
        cache_dir = (
            os.environ.get("CV_CACHE_DIR") or
            self._platform_config.get("cache_dir") or
            ".cv_platform_cache"
        )
        return Path(cache_dir)
    
    def save_user_config(self, config_type: str, config: Dict[str, Any]):
        """保存用户配置到文件"""
        if config_type == "models":
            config_file = self.user_models_config
        elif config_type == "platform":
            config_file = self.user_platform_config
        else:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {config_file}")
        except Exception as e:
            logger.error(f"保存配置失败 {config_file}: {e}")
            raise
    
    def create_default_configs(self):
        """创建默认配置文件模板"""
        # 创建模型配置模板
        models_template = {
            "models_root": "./cv_models",  # 使用相对路径
            "models": {
                "yolov8n": {
                    "type": "detection",
                    "path": "{models_root}/detection/yolo/v8/yolov8n.pt",
                    "device": "auto",
                    "batch_size": 4,
                    "confidence": 0.25,
                    "nms_threshold": 0.45
                },
                "sam_vit_b": {
                    "type": "segmentation", 
                    "path": "{models_root}/segmentation/sam/vit_b/sam_vit_b_01ec64.pth",
                    "device": "auto",
                    "points_per_side": 32
                },
                "resnet50": {
                    "type": "classification",
                    "path": "torchvision://resnet50",  # 使用torchvision预训练
                    "device": "auto",
                    "pretrained": True
                },
                "stable_diffusion": {
                    "type": "generation",
                    "path": "{models_root}/generation/stable_diffusion/sd_1_5/",
                    "device": "auto",
                    "enable_memory_efficient_attention": True
                }
            }
        }
        
        # 创建平台配置模板
        platform_template = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "max_request_size": "100MB"
            },
            "cache": {
                "enabled": True,
                "max_size": "4GB",
                "ttl": 3600
            },
            "logging": {
                "level": "INFO",
                "format": "text"
            },
            "inference": {
                "max_batch_size": 8,
                "timeout": 300
            },
            "gpu": {
                "devices": "auto",
                "memory_fraction": 0.8
            }
        }
        
        # 保存模板文件
        if not self.template_models_config.exists():
            with open(self.template_models_config, 'w', encoding='utf-8') as f:
                yaml.dump(models_template, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"创建模型配置模板: {self.template_models_config}")
        
        if not self.template_platform_config.exists():
            with open(self.template_platform_config, 'w', encoding='utf-8') as f:
                yaml.dump(platform_template, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"创建平台配置模板: {self.template_platform_config}")
    
    def reload_configs(self):
        """重新加载配置"""
        logger.info("重新加载配置...")
        self._load_configs()
    
    def validate_config(self) -> Dict[str, list]:
        """验证配置有效性"""
        errors = {"models": [], "platform": []}
        
        # 验证模型根目录
        models_root = self.get_models_root()
        if not models_root.exists():
            errors["models"].append(f"模型根目录不存在: {models_root}")
        
        # 验证每个模型配置
        models = self._models_config.get("models", {})
        for model_name, model_config in models.items():
            if not isinstance(model_config, dict):
                errors["models"].append(f"模型 {model_name} 配置格式错误")
                continue
                
            if "path" not in model_config:
                errors["models"].append(f"模型 {model_name} 缺少路径配置")
                continue
            
            # 展开路径模板
            model_path = model_config["path"].format(models_root=models_root)
            model_path = Path(model_path)
            
            if not model_path.exists() and not model_config["path"].startswith("torchvision://"):
                errors["models"].append(f"模型文件不存在: {model_path}")
        
        return errors


# 全局配置管理器实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager