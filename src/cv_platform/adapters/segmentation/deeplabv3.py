"""
DeepLabV3 Segmentation Adapter - Supports the DeepLabV3 model from torchvision 

Supported models:
- deeplabv3_resnet50
- deeplabv3_resnet101  
- deeplabv3_mobilenet_v3_large
"""

import time
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from loguru import logger

from ..base import SegmentationAdapter

try:
    import torchvision.transforms as transforms
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision is not installed, the DeepLabV3 adapter will not be available.")


class DeepLabV3Adapter(SegmentationAdapter):
    """DeepLabV3 model adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 num_classes: int = 21,  # PASCAL VOC num of class
                 batch_size: int = 1,
                 **kwargs):
        """
        Initialize the DeepLabV3 adapter
        
        Args:
            model_path: Path to the model file
            device: Device to use for computation
            num_classes: Number of classes (21 for PASCAL VOC, 81 for COCO)
            batch_size: Batch size
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Need to install torchvision: pip install torchvision")
        
        super().__init__(model_path, device, **kwargs)
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Data preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # DeepLabV3 usually uses 512x512.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # PASCAL VOC class names
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # If it's the COCO dataset, there would be more classes
        if num_classes > 21:
            self.class_names = [f'class_{i}' for i in range(num_classes)]
    
    def load_model(self) -> None:
        """Load the DeepLabV3 model"""
        try:
            logger.info(f"Load the DeepLabV3 model: {self.model_path}")
            
            # Determine the specific model architecture based on the file name
            model_name = self.model_path.name.lower()
            
            # First, try to directly load the user's weight file
            if self.model_path.exists():
                logger.info("Detected local model file, attempting to load directly...")
                
                try:
                    # Directly load the entire model state
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    
                    # If it's a complete model, use it directly
                    if hasattr(checkpoint, 'state_dict') or not isinstance(checkpoint, dict):
                        self.model = checkpoint
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to(self.device)
                        self.model.eval()
                        self.is_loaded = True
                        logger.info("Successfully loaded the complete model directly")
                        return
                except Exception as e:
                    logger.info(f"Direct loading failed, try using the torchvision architecture: {e}")
            
            # If loading directly fails, use the torchvision architecture
            logger.info("Create a model using the torchvision architecture...")
            
            if 'resnet101' in model_name:
                self.model = models.segmentation.deeplabv3_resnet101(
                    weights=None,  # Do not use pre-trained weights
                    num_classes=self.num_classes
                )
                logger.info("Create DeepLabV3-ResNet101 architecture")
            elif 'resnet50' in model_name:
                self.model = models.segmentation.deeplabv3_resnet50(
                    weights=None,
                    num_classes=self.num_classes
                )
                logger.info("Create DeepLabV3-ResNet50 architecture")
            elif 'mobilenet' in model_name:
                self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
                    weights=None,
                    num_classes=self.num_classes
                )
                logger.info("Create DeepLabV3-MobileNetV3 architecture")
            else:
                # Default use ResNet-101
                self.model = models.segmentation.deeplabv3_resnet101(
                    weights=None,
                    num_classes=self.num_classes
                )
                logger.info("Create the default DeepLabV3-ResNet101 architecture")

            # Load user weight file
            if self.model_path.exists():
                logger.info("Load user weight file...")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Attempt to load the status dictionary
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    logger.info("Successfully loaded user weights (strict mode)")
                except RuntimeError as e:
                    logger.warning(f"Strict mode loading failed, try non-strict mode: {e}")
                    try:
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info("Successfully loaded user weights (non-strict mode)")
                    except RuntimeError as e:
                        logger.error(f"Weight loading failed: {e}")
                        logger.info("Use randomly initialized weights")
            else:
                logger.warning(f"Model file does not exist: {self.model_path}")
                logger.info("Use randomly initialized weights")

            # Move to specified device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"DeepLabV3 model loaded successfully - Number of classes: {self.num_classes}")
            
        except Exception as e:
            logger.error(f"DeepLabV3 model loading failed: {e}")
            raise
    
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Preprocess input data"""
        if isinstance(input_data, (str, Path)):
            # File path
            image = Image.open(input_data).convert('RGB')
        elif isinstance(input_data, Image.Image):
            # PIL image
            image = input_data.convert('RGB')
        elif isinstance(input_data, np.ndarray):
            # numpy array
            if input_data.ndim == 3 and input_data.shape[2] == 3:
                image = Image.fromarray(input_data)
            else:
                raise ValueError(f"Unsupported numpy array format: {input_data.shape}")
        else:
            raise ValueError(f"Unsupported input format: {type(input_data)}")

        # Save original size for post-processing
        self.original_size = image.size  # (width, height)

        # Apply preprocessing transforms
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                threshold: float = 0.5,
                return_probabilities: bool = False,
                **kwargs) -> Dict[str, Any]:
        """
        Perform image segmentation
        
        Args:
            image: input image
            threshold: segmentation threshold
            return_probabilities: whether to return probability maps

        Returns:
            Segmentation result dictionary
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Preprocess input
            input_tensor = self.preprocess(image)
            
            # Execute reasoning
            start_time = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # DeepLabV3 returns a dictionary that contains the 'out' key
                if isinstance(output, dict):
                    logits = output['out']
                else:
                    logits = output
                
                # Using softmax to obtain probabilities
                probabilities = F.softmax(logits, dim=1)
                
                # Get predicted category
                predictions = torch.argmax(probabilities, dim=1)
                
            inference_time = time.time() - start_time
            
            # Post-processing results
            processed_results = self.postprocess(
                predictions, 
                probabilities if return_probabilities else None,
                threshold=threshold,
                inference_time=inference_time
            )

            logger.debug(f"DeepLabV3 segmentation completed - Time taken: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"DeepLabV3 prediction failed: {e}")
            raise
    
    def postprocess(self, 
                   predictions: torch.Tensor,
                   probabilities: torch.Tensor = None,
                   threshold: float = 0.5,
                   **kwargs) -> Dict[str, Any]:
        """Post-processing DeepLabV3 output"""
        try:
            # Move to CPU and convert to numpy
            pred_mask = predictions.squeeze(0).cpu().numpy()  # [H, W]
            
            if probabilities is not None:
                prob_maps = probabilities.squeeze(0).cpu().numpy()  # [C, H, W]
            else:
                prob_maps = None
            
            # Adjust to the original image size
            from PIL import Image as PILImage
            pred_mask_pil = PILImage.fromarray(pred_mask.astype(np.uint8))
            pred_mask_resized = pred_mask_pil.resize(self.original_size, PILImage.NEAREST)
            pred_mask_final = np.array(pred_mask_resized)
            
            # Generate segmentation masks (one mask for each category)
            masks = []
            scores = []
            areas = []
            bboxes = []
            
            unique_classes = np.unique(pred_mask_final)
            
            for class_id in unique_classes:
                if class_id == 0:  # Skip the background type
                    continue
                
                # Create a binary mask for this category
                class_mask = (pred_mask_final == class_id).astype(np.uint8)
                
                if np.sum(class_mask) == 0:
                    continue
                
                masks.append(class_mask)
                
                # Calculate the average confidence of this category
                if prob_maps is not None:
                    # Adjust the size of the probability graph
                    prob_class = prob_maps[class_id]
                    prob_pil = PILImage.fromarray((prob_class * 255).astype(np.uint8))
                    prob_resized = prob_pil.resize(self.original_size, PILImage.BILINEAR)
                    prob_final = np.array(prob_resized) / 255.0
                    
                    avg_score = np.mean(prob_final[class_mask > 0])
                    scores.append(float(avg_score))
                else:
                    scores.append(1.0)  # If there is no probability graph, set it to 1.0

                # Calculate area
                area = np.sum(class_mask)
                areas.append(float(area))

                # Calculate bounding box
                y_indices, x_indices = np.where(class_mask > 0)
                if len(y_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                else:
                    bboxes.append([0.0, 0.0, 0.0, 0.0])
            
            # Construction Results
            result = {
                'masks': np.array(masks) if masks else np.empty((0, *pred_mask_final.shape)),
                'scores': scores,
                'areas': areas,
                'bboxes': bboxes,
                'class_ids': [int(cls) for cls in unique_classes if cls != 0],
                'class_names': [self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}' 
                              for cls in unique_classes if cls != 0],
                'prediction_mask': pred_mask_final,  # Complete prediction mask
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'original_size': self.original_size,
                    'model_input_size': (512, 512),
                    'num_classes': self.num_classes,
                    'threshold': threshold
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"DeepLabV3 post-processing failed: {e}")
            raise
    
    def visualize_results(self, 
                         image: Union[str, Path, Image.Image, np.ndarray],
                         results: Dict[str, Any],
                         save_path: str = None,
                         alpha: float = 0.6) -> Image.Image:
        """Visual segmentation results"""
        try:
            # Load the original image
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert('RGB')
            else:
                raise ValueError(f"Unsupported image format: {type(image)}")
            
            # Ensure the image size matches
            img = img.resize(self.original_size)
            img_array = np.array(img)
            
            # Obtain the prediction mask
            pred_mask = results['prediction_mask']
            
            # Create a color segmentation map
            colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
            
            # Assign colors to each category
            colors = [
                [0, 0, 0],      # Background
                [128, 0, 0],    # Airplane
                [0, 128, 0],    # Bicycle
                [128, 128, 0],  # Bird
                [0, 0, 128],    # Boat
                [128, 0, 128],  # Bottle
                [0, 128, 128],  # Bus
                [128, 128, 128], # Car
                [64, 0, 0],     # Cat
                [192, 0, 0],    # Chair
                [64, 128, 0],   # Cow
                [192, 128, 0],  # Dining table
                [64, 0, 128],   # Dog
                [192, 0, 128],  # Horse
                [64, 128, 128], # Motorbike
                [192, 128, 128], # Person
                [0, 64, 0],     # Potted plant
                [128, 64, 0],   # Sheep
                [0, 192, 0],    # Sofa
                [128, 192, 0],  # Train
                [0, 64, 128],   # TV/Monitor
            ]
            
            # Apply Colors
            for class_id in np.unique(pred_mask):
                if class_id < len(colors):
                    colored_mask[pred_mask == class_id] = colors[class_id]

            # Blend the original image and the segmentation map
            blended = (1 - alpha) * img_array + alpha * colored_mask
            blended = blended.astype(np.uint8)
            
            result_img = Image.fromarray(blended)
            
            # Save image
            if save_path:
                result_img.save(save_path)
                logger.info(f"The segmentation visualization result has been saved: {save_path}")
            
            return result_img
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Return to the original image
            if isinstance(image, Image.Image):
                return image
            else:
                return Image.open(image) if isinstance(image, (str, Path)) else Image.fromarray(image)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'segmentation',
            'framework': 'torchvision',
            'architecture': 'deeplabv3',
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'input_size': (512, 512),
            'class_names': self.class_names[:10] if len(self.class_names) > 10 else self.class_names
        })
        
        if self.is_loaded:
            try:
                # Get the number of model parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params
                })
                
            except Exception as e:
                logger.debug(f"Failed to obtain model details: {e}")
        
        return info
