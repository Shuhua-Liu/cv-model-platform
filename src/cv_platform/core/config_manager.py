"""
Configuration Manager - Handles all configuration requirements for the platform

Supports configuration priority: Environment variables > User profiles > Default templates
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger


class ConfigManager:
    """Configuration Manager Class"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize Configuration Manager
        
        Args:
            config_dir: Configuration file directory, default is the config folder under the project root directory.
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file path
        self.user_models_config = self.config_dir / "models.yaml"
        self.user_platform_config = self.config_dir / "platform.yaml" 
        self.template_models_config = self.config_dir / "models_template.yaml"
        self.template_platform_config = self.config_dir / "platform_template.yaml"
        
        # Configuration in memory
        self._models_config = {}
        self._platform_config = {}
        
        # Load configuration
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files"""
        logger.info("Load configuration file...")
        
        # Load model configuration
        self._models_config = self._load_models_config()
        
        # Load platform configuration
        self._platform_config = self._load_platform_config()
        
        logger.info(f"Configuration loading complete - {len(self._models_config.get('models', {}))} model configurations found")
    
    def _load_models_config(self) -> Dict[str, Any]:
        """Load model configuration, support priority override"""
        config = {}
        
        # 1. First load the template configuration (default values)
        template_config = self._load_yaml_file(self.template_models_config)
        if template_config:
            config.update(template_config)
            logger.debug(f"Model template configuration loaded: {self.template_models_config}")
        
        # 2. Then load the user configuration (overwriting the default values).
        user_config = self._load_yaml_file(self.user_models_config)
        if user_config:
            config = self._merge_configs(config, user_config)
            logger.debug(f"User model configuration loaded: {self.user_models_config}")
        
        # 3. Finally, apply environment variables (highest priority)
        config = self._apply_env_overrides(config, "models")
        
        return config
    
    def _load_platform_config(self) -> Dict[str, Any]:
        """Load platform configuration"""
        config = {}
        
        # 1. Load template configuration
        template_config = self._load_yaml_file(self.template_platform_config)
        if template_config:
            config.update(template_config)
        
        # 2. Load user configuration
        user_config = self._load_yaml_file(self.user_platform_config)
        if user_config:
            config = self._merge_configs(config, user_config)
        
        # 3. Application environment variables
        config = self._apply_env_overrides(config, "platform")
        
        return config
    
    def _load_yaml_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Securely load YAML files"""
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Unable to load configuration file {file_path}: {e}")
            return None
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep Merge Configuration Dictionary"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """Override application environment variables"""
        env_prefix = "CV_"
        
        # General environment variable mapping
        env_mappings = {
            "CV_MODELS_ROOT": ("models_root",),
            "CV_CACHE_DIR": ("cache_dir",),
            "CV_LOG_LEVEL": ("logging", "level"),
            "CV_API_HOST": ("api", "host"), 
            "CV_API_PORT": ("api", "port"),
            "CV_GPU_DEVICES": ("gpu", "devices"),
            "CV_MAX_BATCH_SIZE": ("inference", "max_batch_size"),
        }
        
        # Environment variables for specific models (CV_MODEL_{MODEL_NAME}_PATH)
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                if env_key in env_mappings:
                    # Handling predefined environment variables
                    keys = env_mappings[env_key]
                    self._set_nested_value(config, keys, env_value)
                    logger.debug(f"Override application environment variables: {env_key} = {env_value}")
                
                elif env_key.startswith("CV_MODEL_") and env_key.endswith("_PATH"):
                    # Handling model path coverage: CV_MODEL_YOLOV8_PATH
                    model_name = env_key[9:-5].lower()  # remove CV_MODEL_ 和 _PATH
                    if "models" not in config:
                        config["models"] = {}
                    if model_name not in config["models"]:
                        config["models"][model_name] = {}
                    config["models"][model_name]["path"] = env_value
                    logger.debug(f"Application model path coverage: {model_name} = {env_value}")
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], keys: tuple, value: str):
        """Set the value of the nested dictionary"""
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Try converting the data type.
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
        """Get model configuration"""
        return self._models_config.copy()
    
    def get_platform_config(self) -> Dict[str, Any]:
        """Get Platform Configuration"""
        return self._platform_config.copy()
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the configuration for a specific model"""
        models = self._models_config.get("models", {})
        return models.get(model_name)
    
    def get_models_root(self) -> Path:
        """Get the root directory of the model"""
        # Priority: Environment variables > Configuration files > Default values
        models_root = (
            os.environ.get("CV_MODELS_ROOT") or
            self._platform_config.get("models_root") or
            self._models_config.get("models_root") or
            "./cv_models"  # Default to using relative paths
        )
        return Path(models_root)
    
    def get_cache_dir(self) -> Path:
        """Get cache directory"""
        cache_dir = (
            os.environ.get("CV_CACHE_DIR") or
            self._platform_config.get("cache_dir") or
            ".cv_platform_cache"
        )
        return Path(cache_dir)
    
    def save_user_config(self, config_type: str, config: Dict[str, Any]):
        """Save user configuration to file"""
        if config_type == "models":
            config_file = self.user_models_config
        elif config_type == "platform":
            config_file = self.user_platform_config
        else:
            raise ValueError(f"Unsupported configuration types: {config_type}")
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to: {config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration {config_file}: {e}")
            raise
    
    def create_default_configs(self):
        """Create a default configuration file template"""
        # Create model configuration template
        models_template = {
            "models_root": "./cv_models",  # Use relative paths
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
                    "path": "{models_root}/classification/resnet/resnet50-11ad3fa6.pth",  
                    "device": "auto",
                    "pretrained": False
                },
                "stable_diffusion": {
                    "type": "generation",
                    "path": "{models_root}/generation/stable_diffusion/sd_2_1/",
                    "device": "auto",
                    "enable_memory_efficient_attention": True
                }
            }
        }
        
        # Create platform configuration template
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
        
        # Save template file
        if not self.template_models_config.exists():
            with open(self.template_models_config, 'w', encoding='utf-8') as f:
                yaml.dump(models_template, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Create model configuration template: {self.template_models_config}")
        
        if not self.template_platform_config.exists():
            with open(self.template_platform_config, 'w', encoding='utf-8') as f:
                yaml.dump(platform_template, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Create platform configuration template: {self.template_platform_config}")
    
    def reload_configs(self):
        """Reload configuration"""
        logger.info("Reload configuration...")
        self._load_configs()
    
    def validate_config(self) -> Dict[str, list]:
        """Verify configuration validity"""
        errors = {"models": [], "platform": []}
        
        # Verify model root directory
        models_root = self.get_models_root()
        if not models_root.exists():
            errors["models"].append(f"The model root directory does not exist: {models_root}")
        
        # Validate each model configuration
        models = self._models_config.get("models", {})
        for model_name, model_config in models.items():
            if not isinstance(model_config, dict):
                errors["models"].append(f"Model {model_name} configuration format error")
                continue
                
            if "path" not in model_config:
                errors["models"].append(f"Model {model_name} is missing path configuration.")
                continue
            
            # Expand path template
            model_path = model_config["path"].format(models_root=models_root)
            model_path = Path(model_path)
            
            if not model_path.exists() and not model_config["path"].startswith("torchvision://"):
                errors["models"].append(f"Model file does not exist: {model_path}")
        
        return errors


# Global Configuration Manager Instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
    
