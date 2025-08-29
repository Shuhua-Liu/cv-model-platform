"""
OpenAI CLIP Multimodal Adapter -  Supports original OpenAI CLIP model

Supported Models:
- OpenAI CLIP: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px, RN50, RN101, etc.
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
    logger.warning("CLIP adapter will be unavailable if the CLIP-related libraries are not installed.")


class CLIPAdapter(MultimodalAdapter):
    """CLIP Multimodal Model Adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 model_source: str = "auto",  # auto, openai, huggingface
                 batch_size: int = 32,
                 **kwargs):
        """
        Initialize CLIP Adapter
        
        Args:
            model_path: Model name or file path
            device: Computing device
            model_source: Pre-trained dataset name (auto, openai, huggingface)
            batch_size: Batch size
        """
        super().__init__(model_path, device, **kwargs)
        
        self.model_source = model_source
        self.batch_size = batch_size
        
        # Determine model source
        self.actual_source = self._determine_model_source()
        
        # Model Components
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.preprocess_fn = None
        
        # Determine model name
        self.model_name = self._determine_model_name()
    
    def _determine_model_source(self) -> str:
        """Determine model source"""
        if self.model_source != "auto":
            return self.model_source
        
        path_str = str(self.model_path).lower()
        
        # Check if HuggingFace format
        if (self.model_path.is_dir() and 
            (self.model_path / "config.json").exists()):
            return "huggingface"
        
        # Check if standard CLIP model name
        if any(name in path_str for name in ['vit-b', 'vit-l', 'rn50', 'rn101']):
            if CLIP_AVAILABLE:
                return "openai"
            elif TRANSFORMERS_AVAILABLE:
                return "huggingface"
        
        # Default option
        if CLIP_AVAILABLE:
            return "openai"
        elif TRANSFORMERS_AVAILABLE:
            return "huggingface"
        else:
            raise ImportError("No CLIP libraries are installed.")
    
    def _determine_model_name(self) -> str:
        """Determine model name"""
        path_str = str(self.model_path)
        
        # If it is a file path, infer from the filename.
        if self.model_path.is_file() or not self.model_path.exists():
            return path_str
        
        # Common CLIP Model Name Mapping
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
        
        # Return original path by default
        return path_str
    
    def load_model(self) -> None:
        """Load CLIP model"""
        try:
            logger.info(f"Load CLIP model: {self.model_name} (Source: {self.actual_source})")
            
            if self.actual_source == "openai":
                self._load_openai_clip()
            elif self.actual_source == "huggingface":
                self._load_huggingface_clip()
            else:
                raise ValueError(f"Unsupported model sources: {self.actual_source}")
            
            # Move to the specified device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            logger.info(f"CLIP model loaded successfully - Source: {self.actual_source}")
            
        except Exception as e:
            logger.error(f"CLIP model loaded failed: {e}")
            raise
    
    def _load_openai_clip(self) -> None:
        """Load OpenAI CLIP model"""
        if not CLIP_AVAILABLE:
            raise ImportError("Require install clip: pip install git+https://github.com/openai/CLIP.git")
        
        try:
            self.model, self.preprocess_fn = clip.load(self.model_name, device=self.device)
            logger.info(f"OpenAI CLIP model loaded successfully: {self.model_name}")
        except Exception as e:
            # Attempt to load local files
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model, self.preprocess_fn = clip.load("ViT-B/32", device=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info("Load the OpenAI CLIP model from a local file")
            else:
                raise e
    
    def _load_huggingface_clip(self) -> None:
        """Load HuggingFace CLIP model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Require install transformers: pip install transformers")
        
        try:
            model_path_str = str(self.model_path)
            
            # Try different model identifiers
            possible_names = [
                model_path_str,
                f"openai/clip-{self.model_name.lower().replace('/', '-')}",
                "openai/clip-vit-base-patch32"  # Default model
            ]
            
            for name in possible_names:
                try:
                    self.processor = CLIPProcessor.from_pretrained(name)
                    self.model = CLIPModel.from_pretrained(name)
                    logger.info(f"HuggingFace CLIP model loaded successfully: {name}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load {name}: {e}")
                    continue
            else:
                raise ValueError("Unable to load any HuggingFace CLIP models")
                
        except Exception as e:
            logger.error(f"HuggingFace CLIP model loading failed: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image"""
        # Load images
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            raise ValueError(f"Unsupported image formats: {type(image)}")
        
        # Perform preprocessing based on the model source
        if self.actual_source == "openai":
            return self.preprocess_fn(img)
        elif self.actual_source == "huggingface":
            inputs = self.processor(images=img, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        else:
            raise ValueError(f"Unsupported model sources: {self.actual_source}")
    
    def preprocess_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Preprocess text"""
        if isinstance(text, str):
            text = [text]
        
        # Perform preprocessing based on the model source
        if self.actual_source == "openai":
            return clip.tokenize(text)
        elif self.actual_source == "huggingface":
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            return inputs['input_ids']
        else:
            raise ValueError(f"Unsupported model sources: {self.actual_source}")
    
    def predict(self,
                image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
                text: Optional[Union[str, List[str]]] = None,
                mode: str = "similarity",
                **kwargs) -> Dict[str, Any]:
        """
        Perform multimodal inference
        
        Args:
            image: Input image
            text: Input text
            mode: Inference mode (similarity, image_embedding, text_embedding, zero_shot)
            
        Returns:
            Inference result dictionary
        """
        if not self.is_loaded:
            self.load_model()
        
        if image is None and text is None:
            raise ValueError("Image or text input must be provided.")
        
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
                raise ValueError(f"Unsupported modes: {mode}")
            
            inference_time = time.time() - start_time
            
            # Add metadata
            result['metadata'] = {
                'inference_time': inference_time,
                'mode': mode,
                'model_source': self.actual_source,
                'model_name': self.model_name
            }
            
            logger.debug(f"CLIP inference completed - Mode: {mode}, Duration:: {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"CLIP prediction failed: {e}")
            raise
    
    def _compute_similarity(self, 
                           image: Union[str, Path, Image.Image, np.ndarray],
                           text: Union[str, List[str]]) -> Dict[str, Any]:
        """Calculate the similarity between images and text"""
        if image is None or text is None:
            raise ValueError("Similarity calculations require both images and text to be provided simultaneously.")
        
        # Preprocess
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        text_tensor = self.preprocess_text(text).to(self.device)
        
        with torch.no_grad():
            if self.actual_source == "openai":
                # Encode
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tensor)
                
                # Normalization
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Calculate similarity
                similarity = torch.matmul(image_features, text_features.T)
                
            elif self.actual_source == "huggingface":
                # HuggingFace CLIP
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                text_inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                
                outputs = self.model(**image_inputs, **text_inputs)
                similarity = outputs.logits_per_image
            
            # Convert to numpy
            similarity_scores = similarity.cpu().numpy()
        
        # Processing Text Lists
        if isinstance(text, str):
            text = [text]
        
        # Construction Results
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
        """Encode images as vectors"""
        if image is None:
            raise ValueError("Image encoding requires image input.")
        
        # Preprocess
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.actual_source == "openai":
                image_features = self.model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)
            elif self.actual_source == "huggingface":
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**image_inputs)
                image_features = F.normalize(outputs, dim=-1)
            
            # Convert to numpy
            embedding = image_features.cpu().numpy()
        
        return {
            'embedding': embedding.squeeze(),
            'embedding_dim': embedding.shape[-1],
            'norm': float(torch.norm(image_features).cpu())
        }
    
    def _encode_text(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """Encode text as vectors"""
        if text is None:
            raise ValueError("Text encoding requires text input")
        
        # Preprocess
        text_tensor = self.preprocess_text(text).to(self.device)
        
        with torch.no_grad():
            if self.actual_source == "openai":
                text_features = self.model.encode_text(text_tensor)
                text_features = F.normalize(text_features, dim=-1)
            elif self.actual_source == "huggingface":
                text_inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(outputs, dim=-1)
            
            # Convert to numpy
            embeddings = text_features.cpu().numpy()
        
        # Processing individual texts and lists of texts
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
        """Zero-shot Image Classification"""
        if image is None or not class_names:
            raise ValueError("Zero-shot classification requires the provision of images and category names.")
        
        # Build Text Prompts
        text_prompts = [f"a photo of a {class_name}" for class_name in class_names]
        
        # Calculate similarity
        similarity_result = self._compute_similarity(image, text_prompts)
        
        # Apply softmax to obtain probabilities
        similarities = [item['similarity'] for item in similarity_result['similarities']]
        similarities_tensor = torch.tensor(similarities)
        probabilities = F.softmax(similarities_tensor, dim=0).numpy()
        
        # Construct classification results
        predictions = []
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            predictions.append({
                'class': class_name,
                'class_id': i,
                'confidence': float(prob),
                'similarity': similarities[i]
            })
        
        # Sort by confidence level
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
        """Batch Prediction"""
        if not self.is_loaded:
            self.load_model()
        
        results = []
        
        if mode == "similarity" and images and texts:
            # Batch Image-Text Similarity Calculation
            for image in images:
                result = self.predict(image=image, text=texts, mode=mode, **kwargs)
                results.append(result)
        
        elif mode == "image_embedding" and images:
            # Batch Image Encoding
            for image in images:
                result = self.predict(image=image, mode=mode, **kwargs)
                results.append(result)
        
        elif mode == "text_embedding" and texts:
            # Batch Text Encoding
            for text in texts:
                result = self.predict(text=text, mode=mode, **kwargs)
                results.append(result)
        
        else:
            raise ValueError("Batch prediction input does not match the pattern")
        
        return results
    
    def find_similar_images(self,
                           query_image: Union[str, Path, Image.Image, np.ndarray],
                           candidate_images: List[Union[str, Path, Image.Image, np.ndarray]],
                           top_k: int = 5) -> Dict[str, Any]:
        """Find similar images"""
        # Encoding Query Image
        query_result = self.predict(image=query_image, mode="image_embedding")
        query_embedding = query_result['embedding']
        
        # Encoding Candidate Images
        candidate_embeddings = []
        for img in candidate_images:
            result = self.predict(image=img, mode="image_embedding")
            candidate_embeddings.append(result['embedding'])
        
        # Calculate similarity
        candidate_embeddings = np.stack(candidate_embeddings)
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get top-k
        top_k = min(top_k, len(candidate_images))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Construct Results
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
        """Search images using text"""
        # Encoded Text Query
        text_result = self.predict(text=text_query, mode="text_embedding")
        text_embedding = text_result['embedding']
        
        # Encoding Candidate Images
        image_embeddings = []
        for img in candidate_images:
            result = self.predict(image=img, mode="image_embedding")
            image_embeddings.append(result['embedding'])
        
        # Calculate similarity
        image_embeddings = np.stack(image_embeddings)
        similarities = np.dot(image_embeddings, text_embedding)
        
        # Get top-k
        top_k = min(top_k, len(candidate_images))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Construct Results
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
        """Get model details"""
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
                # Obtain Embedding Dimensions
                if self.actual_source == "openai":
                    if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'output_dim'):
                        info['embedding_dim'] = self.model.visual.output_dim
                    elif hasattr(self.model, 'embed_dim'):
                        info['embedding_dim'] = self.model.embed_dim
                elif self.actual_source == "huggingface":
                    info['embedding_dim'] = self.model.config.projection_dim
                
                # Retrieve input resolution
                if self.actual_source == "openai":
                    if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'input_resolution'):
                        info['input_resolution'] = self.model.visual.input_resolution
                elif self.actual_source == "huggingface":
                    info['input_resolution'] = self.model.config.vision_config.image_size
                
            except Exception as e:
                logger.debug(f"Failed to retrieve detailed information about the CLIP model: {e}")
        
        return info
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """Model Warmup"""
        if not self.is_loaded:
            self.load_model()
        
        # Create dummy inputs for preheating
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
                logger.warning(f"Warm-up operation {i+1} failed: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"CLIP model warmup complete - Average time taken: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
    
    def unload_model(self) -> None:
        """Unload model to free up memory"""
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
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("CLIP model has been unloaded")
