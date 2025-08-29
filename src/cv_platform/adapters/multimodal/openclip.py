"""
OpenCLIP Multimodal Adapter - Supports OpenCLIP Models

Supported Models:
    - OpenCLIP: ViT-B-32, ViT-B-16, ViT-L-14, ConvNeXt, CoCa, EVA, etc.
    - Various pre-training datasets: LAION-2B, LAION-400M, OpenAI, etc.
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
    logger.warning("open_clip_torch is not installed; the OpenCLIP adapter will be unavailable.")


class OpenCLIPAdapter(MultimodalAdapter):
    """OpenCLIP Multimodal Model Adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 pretrained: str = "openai",  # openai, laion2b_s34b_b79k, etc.
                 batch_size: int = 32,
                 **kwargs):
        """
        Initialize OpenCLIP Adapter
        
        Args:
            model_path: Model name or file path
            device: Computing device
            pretrained: Pre-trained dataset name
            batch_size: Batch size
        """
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("Require install open_clip_torch: pip install open_clip_torch")
        
        super().__init__(model_path, device, **kwargs)
        
        self.pretrained = pretrained
        self.batch_size = batch_size
        
        # Determine the model name
        self.model_name = self._determine_model_name()
        
        # Model Components
        self.model = None
        self.tokenizer = None
        self.preprocess_fn = None
    
    def _determine_model_name(self) -> str:
        """Determine the OpenCLIP model name"""
        path_str = str(self.model_path)
        
        # If it is a file path, infer from the filename.
        if self.model_path.is_file():
            return path_str
        
        # OpenCLIP Model Name Mapping
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
        
        # If no match is found, return the path directly.
        return path_str
    
    def load_model(self) -> None:
        """Load OpenCLIP model"""
        try:
            logger.info(f"Load OpenCLIP model: {self.model_name} (Pre-training: {self.pretrained})")
            
            # Check if the model is available
            available_models = open_clip.list_models()
            if self.model_name not in available_models:
                logger.warning(f"The model {self.model_name} is not in the available list. Attempting to load....")
            
            # Model Creation and Preprocessing
            self.model, _, self.preprocess_fn = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            
            # Obtain tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"OpenCLIP model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"OpenCLIP model loaded failed: {e}")
            
            # Try using the default model
            try:
                logger.info("Try using the default ViT-B-32 model...")
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
                logger.info("Default model loaded successfully")
            except Exception as e2:
                logger.error(f"Default model loading also failed.: {e2}")
                raise e
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocessed image"""
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            raise ValueError(f"Unsupported image formats: {type(image)}")
        
        # Apply Preprocessing
        return self.preprocess_fn(img)
    
    def preprocess_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Preprocess text"""
        if isinstance(text, str):
            text = [text]
        
        return self.tokenizer(text)
    
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
                'model_name': self.model_name,
                'pretrained': self.pretrained
            }
            
            logger.debug(f"OpenCLIP inference completed - Mode: {mode}, Duration: {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"OpenCLIP prediction failed: {e}")
            raise
    
    def _compute_similarity(self, 
                           image: Union[str, Path, Image.Image, np.ndarray],
                           text: Union[str, List[str]]) -> Dict[str, Any]:
        """Calculate the similarity between images and text"""
        if image is None or text is None:
            raise ValueError("Similarity calculations require both images and text to be provided simultaneously.")
        
        # Pre-process
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        text_tensor = self.preprocess_text(text).to(self.device)
        
        with torch.no_grad():
            # Encode
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)
            
            # Normalization
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Calculate similarity
            similarity = torch.matmul(image_features, text_features.T)
            
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
        
        # Preprocessing
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
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
            raise ValueError("Text encoding requires text input.")
        
        # Preprocessing
        text_tensor = self.preprocess_text(text).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tensor)
            text_features = F.normalize(text_features, dim=-1)
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model details"""
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
                # Retrieve the list of available models
                available_models = open_clip.list_models()
                info['available_models'] = available_models[:10]  # Show only the first 10
                
                # Acquire pre-trained datasets
                available_pretrained = open_clip.list_pretrained(self.model_name)
                info['available_pretrained'] = list(available_pretrained.keys())[:5]  # Show only the first 5
                
            except Exception as e:
                logger.debug(f"Failed to retrieve detailed information about the OpenCLIP model: {e}")
        
        return info
    
    def list_available_models(self) -> List[str]:
        """List all available OpenCLIP models"""
        try:
            return open_clip.list_models()
        except Exception as e:
            logger.error(f"Failed to retrieve the list of available models: {e}")
            return []
    
    def list_pretrained_for_model(self, model_name: str = None) -> Dict[str, Any]:
        """List the pre-trained weights for a specific model"""
        if model_name is None:
            model_name = self.model_name
        
        try:
            pretrained_info = open_clip.list_pretrained(model_name)
            return pretrained_info
        except Exception as e:
            logger.error(f"Failed to retrieve pre-trained weight information: {e}")
            return {}
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """Model warmup"""
        if not self.is_loaded:
            self.load_model()
        
        # Create dummy input to warm up
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
                logger.warning(f"Warmup operation {i+1} failed: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"OpenCLIP model warmup completed - Average duration: {avg_time:.3f}s")
            
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
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.preprocess_fn = None
        self.is_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("OpenCLIP model has been unloaded")
