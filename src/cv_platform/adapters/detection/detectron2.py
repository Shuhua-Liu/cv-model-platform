"""
Detectron2 Model Adapter

This adapter integrates Facebook Research's Detectron2 framework for object detection, instance segmentation, and keypoint detection.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from loguru import logger
from PIL import Image
import cv2

try:

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.transforms as T

    DETECTRON2_AVAILABLE = True

except ImportError:

    DETECTRON2_AVAILABLE = False

    logger.warning("Detectron2 not available. Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")

from ..base import BaseModelAdapter, DetectionAdapter, SegmentationAdapter


class Detectron2Adapter(DetectionAdapter, SegmentationAdapter):
    """
    Detectron2 framework adapter supporting detection and segmentation models
    """
    # Supported model configurations from model zoo

    MODEL_CONFIGS = {
        # Detection Models
        "faster_rcnn_r50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "faster_rcnn_r101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "retinanet_r50": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "fcos_r50": "COCO-Detection/fcos_R_50_FPN_1x.yaml",

        # Instance Segmentation Models
        "mask_rcnn_r50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "mask_rcnn_r101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",

        # Panoptic Segmentation Models
        "panoptic_fpn_r50": "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",
        "mask2former_r50": "COCO-PanopticSegmentation/mask2former_R50_bs16_50ep.yaml",

        # Keypoint Detection Models
        "keypoint_rcnn_r50": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    }

    def __init__(self, model_path: str = None, device: str = "auto", **kwargs):
        """Initialize the Detectron2 adapter"""

        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 is required but not installed")

        super().__init__(model_path, device, **kwargs)

        self.predictor = None
        self.cfg = None
        self.metadata = None
        self.model_type = None  # 'detection' or 'segmentation'

    def load_model(self, **kwargs) -> bool:
        """
        Load a Detectron2 model

        Args:
            **kwargs: Additional configuration parameters
                - confidence_threshold: Confidence threshold for predictions
                - nms_threshold: NMS threshold for detection
                - weights: Custom weights file path
        """
        try:

            self.cfg = get_cfg()
            model_path_str = str(self.model_path)
            # Handle different model path types
            if self._is_model_zoo_config(model_path_str):
                # Model zoo config (e.g., "faster_rcnn_r50")
                config_file = self.MODEL_CONFIGS.get(model_path_str)
                if not config_file:
                    raise ValueError(f"Unknown model config: {model_path_str}")

                self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

                # Determine model type from config name
                if any(x in model_path_str for x in ['mask_rcnn', 'mask2former', 'panoptic']):
                    self.model_type = 'segmentation'
                else:
                    self.model_type = 'detection'
            elif str(self.model_path).endswith('.yaml'):
                # Custom config file
                self.cfg.merge_from_file(str(self.model_path))
                if 'weights' in kwargs:
                    self.cfg.MODEL.WEIGHTS = kwargs['weights']
                # Try to determine model type from config
                self.model_type = kwargs.get('model_type', 'detection')
            else:
                # Assume it's a checkpoint file, try to infer config
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")

                # Try to infer config from model name
                config_name = self._infer_config_from_path(model_path_str)
                if config_name:
                    config_file = self.MODEL_CONFIGS[config_name]
                    self.cfg.merge_from_file(model_zoo.get_config_file(config_file))

                    # Determine model type
                    if any(x in config_name for x in ['mask_rcnn', 'mask2former', 'panoptic']):
                        self.model_type = 'segmentation'
                    else:
                        self.model_type = 'detection'
                else:
                    raise ValueError(f"Cannot infer model configuration for: {model_path_str}")

                self.cfg.MODEL.WEIGHTS = str(self.model_path)

            # Set device
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cfg.MODEL.DEVICE = device

            # Apply additional configurations
            if 'confidence_threshold' in kwargs:
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = kwargs['confidence_threshold']
            elif 'confidence' in kwargs:
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = kwargs['confidence']
            if 'nms_threshold' in kwargs:
                self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = kwargs['nms_threshold']

            # Create predictor
            self.predictor = DefaultPredictor(self.cfg)

            # Get metadata for visualization
            if self.cfg.DATASETS.TEST:
                dataset_name = self.cfg.DATASETS.TEST[0]
                self.metadata = MetadataCatalog.get(dataset_name)
            else:
                # Default to COCO metadata
                self.metadata = MetadataCatalog.get("coco_2017_val")

            self.is_loaded = True
            logger.info(f"✅ Detectron2 model loaded: {self.model_path.name}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load Detectron2 model: {e}")
            self.is_loaded = False

            return False

    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                confidence: float = None,
                **kwargs) -> List[Dict[str, Any]]:
        """
        Perform prediction on an image
        Args:
            image: Input image
            confidence: Confidence threshold (optional, uses model default if None)
            **kwargs: Additional parameters
        Returns:
            List of detection/segmentation results
        """

        if not self.is_loaded or not self.predictor:
            raise RuntimeError("Model not loaded")
        try:
            # Load and preprocess image
            if isinstance(image, (str, Path)):
                image_array = cv2.imread(str(image))
                if image_array is None:
                    raise ValueError(f"Could not load image from {image}")
            elif isinstance(image, Image.Image):
                image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                # Assume it's already in BGR format (OpenCV standard)
                image_array = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Update confidence threshold if provided
            if confidence is not None:
                if confidence != self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
                    # Need to recreate predictor with new config
                    self.predictor = DefaultPredictor(self.cfg)

            # Run prediction
            outputs = self.predictor(image_array)
            # Convert outputs to standard format
            if self.model_type == 'segmentation':
                return self._convert_segmentation_outputs(outputs, image_array.shape)
            else:
                return self._convert_detection_outputs(outputs, image_array.shape)
        except Exception as e:
            logger.error(f"❌ Detectron2 prediction failed: {e}")
            raise RuntimeError(f"Detectron2 prediction failed: {e}")

    def unload_model(self) -> bool:
        """Unload the model to free memory"""
        try:
            if self.predictor:
                del self.predictor
                self.predictor = None
            if self.cfg:
                self.cfg = None
            
            self.metadata = None
            self.is_loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("✅ Detectron2 model unloaded")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to unload Detectron2 model: {e}")

            return False

    def get_model_info(self) -> Dict[str, Any]:

        """Get comprehensive model information"""
        info = super().get_model_info()

        # Add Detectron2-specific information
        info.update({
            "framework": "detectron2",
            "adapter_class": self.__class__.__name__,
            "model_type": self.model_type,
            "supports_segmentation": self.model_type == 'segmentation',
            "supports_detection": True,  # All detectron2 models support detection
        })

        if self.cfg:
            info.update({
                "meta_architecture": self.cfg.MODEL.META_ARCHITECTURE,
                "backbone": getattr(self.cfg.MODEL, 'BACKBONE', {}).get('NAME', 'unknown'),
                "confidence_threshold": self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                "nms_threshold": self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            })

        if self.metadata and hasattr(self.metadata, 'thing_classes'):
            info["num_classes"] = len(self.metadata.thing_classes)
            info["class_names"] = self.metadata.thing_classes[:10]  # First 10 classes

        return info

    def _is_model_zoo_config(self, model_path: str) -> bool:
        """Check if model_path is a model zoo configuration name"""
        return model_path in self.MODEL_CONFIGS

    def _infer_config_from_path(self, model_path: str) -> Optional[str]:
        """Infer config name from model file path"""
        model_path_lower = model_path.lower()
        # Try to match common patterns
        for config_name in self.MODEL_CONFIGS:
            config_lower = config_name.lower()
            if config_lower.replace('_', '') in model_path_lower.replace('_', '').replace('-', ''):
                return config_name

        # Default fallbacks based on common naming patterns
        if 'faster' in model_path_lower and 'rcnn' in model_path_lower:
            if 'r101' in model_path_lower or 'resnet101' in model_path_lower:
                return 'faster_rcnn_r101'
            return 'faster_rcnn_r50'

        elif 'mask' in model_path_lower and 'rcnn' in model_path_lower:
            if 'r101' in model_path_lower or 'resnet101' in model_path_lower:
                return 'mask_rcnn_r101'
            return 'mask_rcnn_r50'

        elif 'retinanet' in model_path_lower:
            return 'retinanet_r50'

        elif 'fcos' in model_path_lower:
            return 'fcos_r50'

        elif 'mask2former' in model_path_lower:
            return 'mask2former_r50'

        elif 'keypoint' in model_path_lower and 'rcnn' in model_path_lower:
            return 'keypoint_rcnn_r50'

        return None

    def _convert_detection_outputs(self, 
                                 outputs: Dict[str, Any], 
                                 image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Convert Detectron2 detection outputs to standard format"""
        instances = outputs["instances"].to("cpu")
        results = []
        height, width = image_shape[:2]

        # Get predictions
        if len(instances) == 0:
            return results

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        # Convert each detection
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            result = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(scores[i]),
                "class_id": int(classes[i]),
                "class": self._get_class_name(int(classes[i])),
                "area": float((x2 - x1) * (y2 - y1))
            }
            results.append(result)
        return results

    def _convert_segmentation_outputs(self,
                                    outputs: Dict[str, Any], 
                                    image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Convert Detectron2 segmentation outputs to standard format"""
        instances = outputs["instances"].to("cpu")
        results = []
        height, width = image_shape[:2]

        if len(instances) == 0:
            return results

        # Get predictions
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

        # Convert each detection/segmentation
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            
            result = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(scores[i]),
                "class_id": int(classes[i]),
                "class": self._get_class_name(int(classes[i])),
                "area": float((x2 - x1) * (y2 - y1))
            }

            # Add mask if available
            if masks is not None and i < len(masks):
                result["mask"] = masks[i].astype(bool)
                # Calculate mask area
                result["mask_area"] = float(np.sum(masks[i]))

            results.append(result)

        return results

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        if self.metadata and hasattr(self.metadata, 'thing_classes'):
            if 0 <= class_id < len(self.metadata.thing_classes):
                return self.metadata.thing_classes[class_id]
        return f"class_{class_id}"


    # Registration function
    def register_detectron2_adapter():
        """Register Detectron2 adapter with the system"""
        if DETECTRON2_AVAILABLE:
            from ..registry import register_adapter

            register_adapter(
                name="detectron2",
                adapter_class=Detectron2Adapter,
                frameworks=["detectron2"],
                architectures=["faster_rcnn", "mask_rcnn", "retinanet", "fcos", 
                            "mask2former", "panoptic_fpn", "keypoint_rcnn"]
            )
            logger.info("✅ Detectron2 adapter registered")
        else:
            logger.warning("⚠️ Detectron2 adapter not registered - detectron2 not installed")

    # Auto-register when module is imported
    try:
        register_detectron2_adapter()
    except Exception as e:
        logger.warning(f"Failed to auto-register Detectron2 adapter: {e}")

