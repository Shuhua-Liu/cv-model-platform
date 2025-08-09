"""
OpenCLIP 多模态适配器 - 支持OpenCLIP模型

支持的模型：
- OpenCLIP: ViT-B-32, ViT-B-16, ViT-L-14, ConvNeXt, CoCa, EVA等
- 各种预训练数据集: LAION-2B, LAION-400M, OpenAI等
"""

import time
from typing import Dict, Any, Union, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from loguru import logger

from ..base import MultimodalAdapter

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("open_clip_torch未安装，OpenCLIP适配器将不可用")


class OpenCLIPAdapter(MultimodalAdapter):
    """OpenCLIP多模态模型适配器"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 pretrained: str = "openai",  # openai, laion2b_s34b_b79k, etc.
                 batch_size: int = 32,
                 **kwargs):
        """
        初始化OpenCLIP适配器
        
        Args:
            model_path: 模型名称或文件路径
            device: 计算设备
            pretrained: 预训练数据集名称
            batch_size: 批处理大小
        """
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("需要安装open_clip_torch: pip install open_clip_torch")
        
        super().__init__(model_path, device, **kwargs)
        
        self.pretrained = pretrained
        self.batch_size = batch_size
        
        # 确定模型名称
        self.model_name = self._determine_model_name()
        
        # 模型组件
        self.model = None
        self.tokenizer = None
        self.preprocess_fn = None
    
    def _determine_model_name(self) -> str:
        """确定OpenCLIP模型名称"""
        path_str = str(self.model_path)
        
        # 如果是文件路径，从文件名推断
        if self.model_path.is_file():
            return path_str
        
        # OpenCLIP模型名称映射
        name_mappings = {
            'vit-b-32': 'ViT-B-32',
            'vit-b-16': 'ViT-B-16',
            'vit-l-14': 'ViT-L-14',
            'vit-l-14-336': 'ViT-L-14-336',
            'vit-h-14': 'ViT-H-14',
            'vit-g-14': 'ViT-g-14',
            'convnext-base': 'convnext_base',
            'convnext-large': 'convnext_large',
            'coca-vit-b-32': 'coca_ViT-B-32',
            'coca-vit-l-14': 'coca_ViT-L-14',
            'eva-vit-g-14': 'EVA02-CLIP-B-16',
        }
        
        path_lower = path_str.lower()
        for key, value in name_mappings.items():
            if key in path_lower:
                return value
        
        # 如果没有匹配，直接返回路径
        return path_str
    
    def load_model(self) -> None:
        """加载OpenCLIP模型"""
        try:
            logger.info(f"加载OpenCLIP模型: {self.model_name} (预训练: {self.pretrained})")
            
            # 检查模型是否可用
            available_models = open_clip.list_models()
            if self.model_name not in available_models:
                logger.warning(f"模型 {self.model_name} 不在可用列表中，尝试加载...")
            
            # 创建模型和预处理
            self.model, _, self.preprocess_fn = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            
            # 获取tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # 设置为评估模式
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"OpenCLIP模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"OpenCLIP模型加载失败: {e}")
            
            # 尝试使用默认模型
            try:
                logger.info("尝试使用默认ViT-B-32模型...")
                self.model, _, self.preprocess_fn = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='openai',
                    device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.model.eval()
                self.model_name = 'ViT-B-32'
                self.pretrained = 'openai'
                self.is_loaded = True
                logger.info("默认模型加载成功")
            except Exception as e2:
                logger.error(f"默认模型加载也失败: {e2}")
                raise e
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """预处理图像"""
        # 加载图像
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            raise ValueError(f"不支持的图像格式: {type(image)}")
        
        # 应用预处理
        return self.preprocess_fn(img)
    
    def preprocess_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """预处理文本"""
        if isinstance(text, str):
            text = [text]
        
        return self.tokenizer(text)
    
    def predict(self,
                image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
                text: Optional[Union[str, List[str]]] = None,
                mode: str = "similarity",
                **kwargs) -> Dict[str, Any]:
        """
        执行多模态推理
        
        Args:
            image: 输入图像
            text: 输入文本
            mode: 推理模式 (similarity, image_embedding, text_embedding, zero_shot)
            
        Returns:
            推理结果字典
        """
        if not self.is_loaded:
            self.load_model()
        
        if image is None and text is None:
            raise ValueError("必须提供图像或文本输入")
        
        try:
            start_time = time.time()
            
            if mode == "similarity":
                result = self._compute_similarity(image, text)
            elif mode == "image_embedding":
                result = self._encode_image(image)
            elif mode == "text_embedding":
                result = self._encode_text(text)
            elif mode == "zero_shot":
                result = self._zero_shot_classification(image, text)
            else:
                raise ValueError(f"不支持的模式: {mode}")
            
            inference_time = time.time() - start_time
            
            # 添加元数据
            result['metadata'] = {
                'inference_time': inference_time,
                'mode': mode,
                'model_name': self.model_name,
                'pretrained': self.pretrained
            }
            
            logger.debug(f"OpenCLIP推理完成 - 模式: {mode}, 耗时: {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"OpenCLIP预测失败: {e}")
            raise
    
    def _compute_similarity(self, 
                           image: Union[str, Path, Image.Image, np.ndarray],
                           text: Union[str, List[str]]) -> Dict[str, Any]:
        """计算图像和文本之间的相似度"""
        if image is None or text is None:
            raise ValueError("相似度计算需要同时提供图像和文本")
        
        # 预处理
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        text_tensor = self.preprocess_text(text).to(self.device)
        
        with torch.no_grad():
            # 编码
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)
            
            # 归一化
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # 计算相似度
            similarity = torch.matmul(image_features, text_features.T)
            
            # 转换为numpy
            similarity_scores = similarity.cpu().numpy()
        
        # 处理文本列表
        if isinstance(text, str):
            text = [text]
        
        # 构建结果
        results = []
        for i, txt in enumerate(text):
            results.append({
                'text': txt,
                'similarity': float(similarity_scores[0, i])
            })
        
        return {
            'similarities': results,
            'max_similarity': float(similarity_scores.max()),
            'best_match': text[similarity_scores.argmax()]
        }
    
    def _encode_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """编码图像为向量"""
        if image is None:
            raise ValueError("图像编码需要提供图像输入")
        
        # 预处理
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
            # 转换为numpy
            embedding = image_features.cpu().numpy()
        
        return {
            'embedding': embedding.squeeze(),
            'embedding_dim': embedding.shape[-1],
            'norm': float(torch.norm(image_features).cpu())
        }
    
    def _encode_text(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """编码文本为向量"""
        if text is None:
            raise ValueError("文本编码需要提供文本输入")
        
        # 预处理
        text_tensor = self.preprocess_text(text).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tensor)
            text_features = F.normalize(text_features, dim=-1)
            
            # 转换为numpy
            embeddings = text_features.cpu().numpy()
        
        # 处理单个文本和文本列表
        if isinstance(text, str):
            return {
                'embedding': embeddings.squeeze(),
                'embedding_dim': embeddings.shape[-1],
                'text': text
            }
        else:
            results = []
            for i, txt in enumerate(text):
                results.append({
                    'text': txt,
                    'embedding': embeddings[i],
                    'embedding_dim': embeddings.shape[-1]
                })
            return {'text_embeddings': results}
    
    def _zero_shot_classification(self, 
                                 image: Union[str, Path, Image.Image, np.ndarray],
                                 class_names: List[str]) -> Dict[str, Any]:
        """零样本图像分类"""
        if image is None or not class_names:
            raise ValueError("零样本分类需要提供图像和类别名称")
        
        # 构建文本提示
        text_prompts = [f"a photo of a {class_name}" for class_name in class_names]
        
        # 计算相似度
        similarity_result = self._compute_similarity(image, text_prompts)
        
        # 应用softmax获得概率
        similarities = [item['similarity'] for item in similarity_result['similarities']]
        similarities_tensor = torch.tensor(similarities)
        probabilities = F.softmax(similarities_tensor, dim=0).numpy()
        
        # 构建分类结果
        predictions = []
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            predictions.append({
                'class': class_name,
                'class_id': i,
                'confidence': float(prob),
                'similarity': similarities[i]
            })
        
        # 按置信度排序
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predictions': predictions,
            'top_class': predictions[0]['class'],
            'top_confidence': predictions[0]['confidence'],
            'class_names': class_names
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'multimodal',
            'framework': 'open_clip',
            'architecture': 'openclip',
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'batch_size': self.batch_size,
            'supported_modes': ['similarity', 'image_embedding', 'text_embedding', 'zero_shot']
        })
        
        if self.is_loaded:
            try:
                # 获取可用模型列表
                available_models = open_clip.list_models()
                info['available_models'] = available_models[:10]  # 只显示前10个
                
                # 获取预训练数据集
                available_pretrained = open_clip.list_pretrained(self.model_name)
                info['available_pretrained'] = list(available_pretrained.keys())[:5]  # 只显示前5个
                
            except Exception as e:
                logger.debug(f"获取OpenCLIP模型详细信息失败: {e}")
        
        return info
    
    def list_available_models(self) -> List[str]:
        """列出所有可用的OpenCLIP模型"""
        try:
            return open_clip.list_models()
        except Exception as e:
            logger.error(f"获取可用模型列表失败: {e}")
            return []
    
    def list_pretrained_for_model(self, model_name: str = None) -> Dict[str, Any]:
        """列出特定模型的预训练权重"""
        if model_name is None:
            model_name = self.model_name
        
        try:
            pretrained_info = open_clip.list_pretrained(model_name)
            return pretrained_info
        except Exception as e:
            logger.error(f"获取预训练权重信息失败: {e}")
            return {}
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """模型预热"""
        if not self.is_loaded:
            self.load_model()
        
        # 创建dummy输入进行预热
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        dummy_text = "a test image for warmup"
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                _ = self.predict(image=dummy_image, text=dummy_text, mode="similarity")
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
            except Exception as e:
                logger.warning(f"预热运行 {i+1} 失败: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"OpenCLIP模型预热完成 - 平均耗时: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
    
    def unload_model(self) -> None:
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.preprocess_fn = None
        self.is_loaded = False
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("OpenCLIP模型已卸载")