"""
OpenAI CLIP 多模态适配器 - 支持原版OpenAI CLIP模型

支持的模型：
- OpenAI CLIP: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px, RN50, RN101等
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
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

if not any([CLIP_AVAILABLE, TRANSFORMERS_AVAILABLE]):
    logger.warning("未安装CLIP相关库，CLIP适配器将不可用")


class CLIPAdapter(MultimodalAdapter):
    """CLIP多模态模型适配器"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 model_source: str = "auto",  # auto, openai, huggingface
                 batch_size: int = 32,
                 **kwargs):
        """
        初始化CLIP适配器
        
        Args:
            model_path: 模型文件路径或模型名称
            device: 计算设备
            model_source: 模型来源 (auto, openai, huggingface)
            batch_size: 批处理大小
        """
        super().__init__(model_path, device, **kwargs)
        
        self.model_source = model_source
        self.batch_size = batch_size
        
        # 确定实际使用的模型源
        self.actual_source = self._determine_model_source()
        
        # 模型组件
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.preprocess_fn = None
        
        # 确定模型名称
        self.model_name = self._determine_model_name()
    
    def _determine_model_source(self) -> str:
        """确定模型来源"""
        if self.model_source != "auto":
            return self.model_source
        
        path_str = str(self.model_path).lower()
        
        # 检查是否是HuggingFace格式
        if (self.model_path.is_dir() and 
            (self.model_path / "config.json").exists()):
            return "huggingface"
        
        # 检查是否是标准CLIP模型名
        if any(name in path_str for name in ['vit-b', 'vit-l', 'rn50', 'rn101']):
            if CLIP_AVAILABLE:
                return "openai"
            elif TRANSFORMERS_AVAILABLE:
                return "huggingface"
        
        # 默认选择
        if CLIP_AVAILABLE:
            return "openai"
        elif TRANSFORMERS_AVAILABLE:
            return "huggingface"
        else:
            raise ImportError("未安装任何CLIP库")
    
    def _determine_model_name(self) -> str:
        """确定模型名称"""
        path_str = str(self.model_path)
        
        # 如果是文件路径，从文件名推断
        if self.model_path.is_file() or not self.model_path.exists():
            return path_str
        
        # 常见的CLIP模型名称映射
        name_mappings = {
            'vit-b-32': 'ViT-B/32',
            'vit-b-16': 'ViT-B/16', 
            'vit-l-14': 'ViT-L/14',
            'vit-l-14-336': 'ViT-L/14@336px',
            'rn50': 'RN50',
            'rn101': 'RN101',
            'rn50x4': 'RN50x4',
            'rn50x16': 'RN50x16',
            'rn50x64': 'RN50x64'
        }
        
        path_lower = path_str.lower()
        for key, value in name_mappings.items():
            if key in path_lower:
                return value
        
        # 默认返回原始路径
        return path_str
    
    def load_model(self) -> None:
        """加载CLIP模型"""
        try:
            logger.info(f"加载CLIP模型: {self.model_name} (来源: {self.actual_source})")
            
            if self.actual_source == "openai":
                self._load_openai_clip()
            elif self.actual_source == "huggingface":
                self._load_huggingface_clip()
            else:
                raise ValueError(f"不支持的模型来源: {self.actual_source}")
            
            # 移动到指定设备
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            logger.info(f"CLIP模型加载成功 - 来源: {self.actual_source}")
            
        except Exception as e:
            logger.error(f"CLIP模型加载失败: {e}")
            raise
    
    def _load_openai_clip(self) -> None:
        """加载OpenAI CLIP模型"""
        if not CLIP_AVAILABLE:
            raise ImportError("需要安装clip: pip install git+https://github.com/openai/CLIP.git")
        
        try:
            self.model, self.preprocess_fn = clip.load(self.model_name, device=self.device)
            logger.info(f"OpenAI CLIP模型加载成功: {self.model_name}")
        except Exception as e:
            # 尝试本地文件加载
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model, self.preprocess_fn = clip.load("ViT-B/32", device=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info("从本地文件加载OpenAI CLIP模型")
            else:
                raise e
    
    def _load_huggingface_clip(self) -> None:
        """加载HuggingFace CLIP模型"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装transformers: pip install transformers")
        
        try:
            model_path_str = str(self.model_path)
            
            # 尝试不同的模型标识符
            possible_names = [
                model_path_str,
                f"openai/clip-{self.model_name.lower().replace('/', '-')}",
                "openai/clip-vit-base-patch32"  # 默认模型
            ]
            
            for name in possible_names:
                try:
                    self.processor = CLIPProcessor.from_pretrained(name)
                    self.model = CLIPModel.from_pretrained(name)
                    logger.info(f"HuggingFace CLIP模型加载成功: {name}")
                    break
                except Exception as e:
                    logger.debug(f"尝试加载 {name} 失败: {e}")
                    continue
            else:
                raise ValueError("无法加载任何HuggingFace CLIP模型")
                
        except Exception as e:
            logger.error(f"HuggingFace CLIP模型加载失败: {e}")
            raise
    
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
        
        # 根据模型来源进行预处理
        if self.actual_source == "openai":
            return self.preprocess_fn(img)
        elif self.actual_source == "huggingface":
            inputs = self.processor(images=img, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        else:
            raise ValueError(f"不支持的模型来源: {self.actual_source}")
    
    def preprocess_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """预处理文本"""
        if isinstance(text, str):
            text = [text]
        
        # 根据模型来源进行预处理
        if self.actual_source == "openai":
            return clip.tokenize(text)
        elif self.actual_source == "huggingface":
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            return inputs['input_ids']
        else:
            raise ValueError(f"不支持的模型来源: {self.actual_source}")
    
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
                'model_source': self.actual_source,
                'model_name': self.model_name
            }
            
            logger.debug(f"CLIP推理完成 - 模式: {mode}, 耗时: {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"CLIP预测失败: {e}")
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
            if self.actual_source == "openai":
                # 编码
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tensor)
                
                # 归一化
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # 计算相似度
                similarity = torch.matmul(image_features, text_features.T)
                
            elif self.actual_source == "huggingface":
                # HuggingFace CLIP
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                text_inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                
                outputs = self.model(**image_inputs, **text_inputs)
                similarity = outputs.logits_per_image
            
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
            if self.actual_source == "openai":
                image_features = self.model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)
            elif self.actual_source == "huggingface":
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**image_inputs)
                image_features = F.normalize(outputs, dim=-1)
            
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
            if self.actual_source == "openai":
                text_features = self.model.encode_text(text_tensor)
                text_features = F.normalize(text_features, dim=-1)
            elif self.actual_source == "huggingface":
                text_inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(outputs, dim=-1)
            
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
    
    def predict_batch(self,
                     images: Optional[List[Union[str, Path, Image.Image, np.ndarray]]] = None,
                     texts: Optional[List[str]] = None,
                     mode: str = "similarity",
                     **kwargs) -> List[Dict[str, Any]]:
        """批量预测"""
        if not self.is_loaded:
            self.load_model()
        
        results = []
        
        if mode == "similarity" and images and texts:
            # 图像-文本相似度批量计算
            for image in images:
                result = self.predict(image=image, text=texts, mode=mode, **kwargs)
                results.append(result)
        
        elif mode == "image_embedding" and images:
            # 批量图像编码
            for image in images:
                result = self.predict(image=image, mode=mode, **kwargs)
                results.append(result)
        
        elif mode == "text_embedding" and texts:
            # 批量文本编码
            for text in texts:
                result = self.predict(text=text, mode=mode, **kwargs)
                results.append(result)
        
        else:
            raise ValueError("批量预测的输入与模式不匹配")
        
        return results
    
    def find_similar_images(self,
                           query_image: Union[str, Path, Image.Image, np.ndarray],
                           candidate_images: List[Union[str, Path, Image.Image, np.ndarray]],
                           top_k: int = 5) -> Dict[str, Any]:
        """查找相似图像"""
        # 编码查询图像
        query_result = self.predict(image=query_image, mode="image_embedding")
        query_embedding = query_result['embedding']
        
        # 编码候选图像
        candidate_embeddings = []
        for img in candidate_images:
            result = self.predict(image=img, mode="image_embedding")
            candidate_embeddings.append(result['embedding'])
        
        # 计算相似度
        candidate_embeddings = np.stack(candidate_embeddings)
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # 获取top-k
        top_k = min(top_k, len(candidate_images))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 构建结果
        similar_images = []
        for idx in top_indices:
            similar_images.append({
                'image_index': int(idx),
                'similarity': float(similarities[idx]),
                'image_path': str(candidate_images[idx]) if isinstance(candidate_images[idx], (str, Path)) else f"image_{idx}"
            })
        
        return {
            'query_embedding': query_embedding,
            'similar_images': similar_images,
            'total_candidates': len(candidate_images)
        }
    
    def search_with_text(self,
                        text_query: str,
                        candidate_images: List[Union[str, Path, Image.Image, np.ndarray]],
                        top_k: int = 5) -> Dict[str, Any]:
        """使用文本搜索图像"""
        # 编码文本查询
        text_result = self.predict(text=text_query, mode="text_embedding")
        text_embedding = text_result['embedding']
        
        # 编码候选图像
        image_embeddings = []
        for img in candidate_images:
            result = self.predict(image=img, mode="image_embedding")
            image_embeddings.append(result['embedding'])
        
        # 计算相似度
        image_embeddings = np.stack(image_embeddings)
        similarities = np.dot(image_embeddings, text_embedding)
        
        # 获取top-k
        top_k = min(top_k, len(candidate_images))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 构建结果
        matching_images = []
        for idx in top_indices:
            matching_images.append({
                'image_index': int(idx),
                'similarity': float(similarities[idx]),
                'image_path': str(candidate_images[idx]) if isinstance(candidate_images[idx], (str, Path)) else f"image_{idx}"
            })
        
        return {
            'text_query': text_query,
            'text_embedding': text_embedding,
            'matching_images': matching_images,
            'total_candidates': len(candidate_images)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'multimodal',
            'framework': self.actual_source,
            'architecture': 'clip',
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'supported_modes': ['similarity', 'image_embedding', 'text_embedding', 'zero_shot']
        })
        
        if self.is_loaded:
            try:
                # 获取嵌入维度
                if self.actual_source == "openai":
                    if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'output_dim'):
                        info['embedding_dim'] = self.model.visual.output_dim
                    elif hasattr(self.model, 'embed_dim'):
                        info['embedding_dim'] = self.model.embed_dim
                elif self.actual_source == "huggingface":
                    info['embedding_dim'] = self.model.config.projection_dim
                
                # 获取输入分辨率
                if self.actual_source == "openai":
                    if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'input_resolution'):
                        info['input_resolution'] = self.model.visual.input_resolution
                elif self.actual_source == "huggingface":
                    info['input_resolution'] = self.model.config.vision_config.image_size
                
            except Exception as e:
                logger.debug(f"获取CLIP模型详细信息失败: {e}")
        
        return info
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """模型预热"""
        if not self.is_loaded:
            self.load_model()
        
        # 创建dummy输入进行预热
        dummy_image = Image.new('RGB', (224, 224), color='red')
        dummy_text = "a test image"
        
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
            
            logger.info(f"CLIP模型预热完成 - 平均耗时: {avg_time:.3f}s")
            
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
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.preprocess_fn = None
        self.is_loaded = False
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("CLIP模型已卸载")