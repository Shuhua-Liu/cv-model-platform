"""
SAM (Segment Anything Model) Adapter - Implements abstract methods and improves error handling

Features:
1. Implements the abstract method postprocess from the base class
2. Maintains original functionality
3. Adds better error handling

Supported models:
- sam_vit_h (ViT-Huge)
- sam_vit_l (ViT-Large)
- sam_vit_b (ViT-Base)
- mobile_sam
"""

import time
from typing import Dict, Any, Union, List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from loguru import logger

from ..base import SegmentationAdapter

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("segment-anything is not installed, SAMAdapter will not be available.")


class SAMAdapter(SegmentationAdapter):
    """Segment Anything Model Adapter"""

    def __init__(self,
                 model_path: Union[str, Path],
                 device: str = "auto",
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95,
                 crop_n_layers: int = 0,
                 crop_n_points_downscale_factor: int = 1,
                 min_mask_region_area: int = 0,
                 **kwargs):
        """
        Initialize SAMAdapter

        Args:
            model_path: Model file path
            device: Compute device
            points_per_side: Number of points per side (for automatic mask generation)
            pred_iou_thresh: Prediction IoU threshold
            stability_score_thresh: Stability score threshold
            crop_n_layers: Number of crop layers
            crop_n_points_downscale_factor: Crop points downscale factor
            min_mask_region_area: Minimum mask region area
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything is required: pip install git+https://github.com/facebookresearch/segment-anything.git")

        super().__init__(model_path, device, **kwargs)

        # SAM parameters
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area

        # Determine model type
        self.model_type = self._determine_model_type()

        # SAM predictor and mask generator
        self.predictor = None
        self.mask_generator = None

    def _determine_model_type(self) -> str:
        """Determine SAM model type from filename"""
        filename = self.model_path.name.lower()

        if 'vit_h' in filename or 'huge' in filename:
            return 'vit_h'
        elif 'vit_l' in filename or 'large' in filename:
            return 'vit_l'
        elif 'vit_b' in filename or 'base' in filename:
            return 'vit_b'
        elif 'mobile' in filename:
            return 'vit_b'  # Mobile SAM is based on ViT-B
        else:
            # Default to ViT-B
            logger.warning(f"Cannot determine SAM model type from filename: {filename}, defaulting to vit_b")
            return 'vit_b'

    def load_model(self) -> None:
        """Load SAM model"""
        try:
            logger.info(f"Loading SAM model: {self.model_path} (type: {self.model_type})")

            # Load SAM model
            self.model = sam_model_registry[self.model_type](checkpoint=str(self.model_path))
            self.model.to(device=self.device)

            # Create predictor (for interactive segmentation)
            self.predictor = SamPredictor(self.model)

            # Create automatic mask generator (for full image segmentation)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                min_mask_region_area=self.min_mask_region_area,
            )

            self.is_loaded = True
            logger.info(f"SAM model loaded successfully - type: {self.model_type}")

        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise

    def preprocess(self, input_data: Any) -> np.ndarray:
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
                image = input_data
            else:
                raise ValueError(f"Unsupported numpy array format: {input_data.shape}")
        else:
            raise ValueError(f"Unsupported input format: {type(input_data)}")

        # Convert to numpy array (RGB format)
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Save original size
        self.original_size = image_array.shape[:2]  # (height, width)

        return image_array

    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                mode: str = "automatic",
                points: Optional[List[List[float]]] = None,
                point_labels: Optional[List[int]] = None,
                boxes: Optional[List[List[float]]] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask_output: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        Perform image segmentation

        Args:
            image: Input image
            mode: Segmentation mode ("automatic" or "interactive")
            points: Prompt point coordinates [[x1, y1], [x2, y2], ...]
            point_labels: Point labels [1, 0, 1, ...] (1 for foreground, 0 for background)
            boxes: Bounding boxes [[x1, y1, x2, y2], ...]
            mask_input: Input mask
            multimask_output: Whether to output multiple masks

        Returns:
            Segmentation result dictionary
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Preprocess input
            image_array = self.preprocess(image)

            start_time = time.time()

            if mode == "automatic":
                # Automatic segmentation mode
                masks = self.mask_generator.generate(image_array)
                inference_time = time.time() - start_time

                # Postprocess automatic segmentation results
                processed_results = self._postprocess_automatic(
                    masks,
                    inference_time=inference_time
                )

            elif mode == "interactive":
                # Interactive segmentation mode
                self.predictor.set_image(image_array)

                # Convert input format
                input_points = np.array(points) if points else None
                input_labels = np.array(point_labels) if point_labels else None
                input_boxes = np.array(boxes) if boxes else None

                # Perform prediction
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_boxes[0] if input_boxes is not None and len(input_boxes) > 0 else None,
                    mask_input=mask_input,
                    multimask_output=multimask_output,
                )

                inference_time = time.time() - start_time

                # Postprocess interactive segmentation results
                processed_results = self._postprocess_interactive(
                    masks,
                    scores,
                    logits,
                    inference_time=inference_time
                )

            else:
                raise ValueError(f"Unsupported segmentation mode: {mode}")

            logger.debug(f"SAM segmentation completed - mode: {mode}, time: {inference_time:.3f}s")
            return processed_results

        except Exception as e:
            logger.error(f"SAM prediction failed: {e}")
            raise

    def postprocess(self, raw_output: Any, **kwargs) -> Dict[str, Any]:
        """
        Postprocess SAM output - implements base class abstract method

        This method is for compatibility with BaseModelAdapter's abstract method requirements.
        The actual postprocessing logic is implemented in _postprocess_automatic and _postprocess_interactive.
        """
        if isinstance(raw_output, dict):
            # If already a processed dict, return directly
            return raw_output
        elif isinstance(raw_output, list):
            # If raw output from SAM automatic mode (mask list)
            return self._postprocess_automatic(raw_output, **kwargs)
        elif isinstance(raw_output, tuple) and len(raw_output) == 3:
            # If raw output from SAM interactive mode (masks, scores, logits)
            masks, scores, logits = raw_output
            return self._postprocess_interactive(masks, scores, logits, **kwargs)
        else:
            # Unknown format, return empty result
            logger.warning(f"Unknown SAM output format: {type(raw_output)}")
            return {
                'masks': np.empty((0, 0, 0)),
                'scores': [],
                'areas': [],
                'bboxes': [],
                'metadata': {
                    'error': 'Unknown output format',
                    **kwargs
                }
            }

    def _postprocess_interactive(self,
                               masks: np.ndarray,
                               scores: np.ndarray,
                               logits: np.ndarray,
                               **kwargs) -> Dict[str, Any]:
        """Postprocess interactive segmentation results"""
        try:
            # masks: [N, H, W], scores: [N], logits: [N, H, W]

            mask_arrays = []
            score_list = []
            areas = []
            bboxes = []

            for i in range(len(masks)):
                mask = masks[i].astype(np.uint8)
                mask_arrays.append(mask)

                score_list.append(float(scores[i]))

                # Calculate area
                area = np.sum(mask)
                areas.append(float(area))

                # Calculate bounding box
                if area > 0:
                    y_indices, x_indices = np.where(mask > 0)
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                else:
                    bboxes.append([0.0, 0.0, 0.0, 0.0])

            result = {
                'masks': np.array(mask_arrays) if mask_arrays else np.empty((0, *self.original_size)),
                'scores': score_list,
                'areas': areas,
                'bboxes': bboxes,
                'logits': logits,  # Keep logits for further processing
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'mode': 'interactive',
                    'num_masks': len(masks),
                    'original_size': self.original_size
                }
            }

            return result

        except Exception as e:
            logger.error(f"SAM interactive postprocessing failed: {e}")
            raise

    def _postprocess_automatic(self, masks: List[Dict], **kwargs) -> Dict[str, Any]:
        """Postprocess automatic segmentation results"""
        try:
            if not masks:
                return {
                    'masks': np.empty((0, *self.original_size)),
                    'scores': [],
                    'areas': [],
                    'bboxes': [],
                    'metadata': {
                        'inference_time': kwargs.get('inference_time', 0),
                        'mode': 'automatic',
                        'num_masks': 0
                    }
                }

            # Sort by stability score
            masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)

            # Extract data
            mask_arrays = []
            scores = []
            areas = []
            bboxes = []

            for mask_data in masks:
                mask = mask_data['segmentation']
                mask_arrays.append(mask.astype(np.uint8))

                scores.append(float(mask_data['stability_score']))
                areas.append(float(mask_data['area']))

                # Convert bounding box format
                bbox = mask_data['bbox']  # [x, y, w, h]
                bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bboxes.append([float(x) for x in bbox_xyxy])

            result = {
                'masks': np.array(mask_arrays),
                'scores': scores,
                'areas': areas,
                'bboxes': bboxes,
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'mode': 'automatic',
                    'num_masks': len(masks),
                    'original_size': self.original_size,
                    'points_per_side': self.points_per_side
                }
            }

            return result

        except Exception as e:
            logger.error(f"SAM automatic postprocessing failed: {e}")
            raise

    def predict_point(self,
                     image: Union[str, Path, Image.Image, np.ndarray],
                     point: Tuple[float, float],
                     label: int = 1,
                     **kwargs) -> Dict[str, Any]:
        """Segment with a single point prompt"""
        return self.predict(
            image=image,
            mode="interactive",
            points=[list(point)],
            point_labels=[label],
            **kwargs
        )

    def predict_box(self,
                   image: Union[str, Path, Image.Image, np.ndarray],
                   box: Tuple[float, float, float, float],
                   **kwargs) -> Dict[str, Any]:
        """Segment with a bounding box prompt"""
        return self.predict(
            image=image,
            mode="interactive",
            boxes=[list(box)],
            **kwargs
        )

    def predict_points(self,
                      image: Union[str, Path, Image.Image, np.ndarray],
                      points: List[Tuple[float, float]],
                      labels: List[int],
                      **kwargs) -> Dict[str, Any]:
        """Segment with multiple point prompts"""
        points_list = [list(p) for p in points]
        return self.predict(
            image=image,
            mode="interactive",
            points=points_list,
            point_labels=labels,
            **kwargs
        )

    def visualize_results(self,
                         image: Union[str, Path, Image.Image, np.ndarray],
                         results: Dict[str, Any],
                         save_path: str = None,
                         show_points: bool = False,
                         show_boxes: bool = False,
                         alpha: float = 0.6) -> Image.Image:
        """Visualize SAM segmentation results"""
        try:
            # Load original image
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert('RGB')
            else:
                raise ValueError(f"Unsupported image format: {type(image)}")

            img_array = np.array(img)

            # Get masks
            masks = results['masks']

            if len(masks) == 0:
                logger.warning("No segmentation masks found")
                return img

            # Create colored mask
            overlay = img_array.copy()

            # Assign different color to each mask
            colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
                [255, 128, 128], # Light Red
                [128, 255, 128], # Light Green
            ]

            for i, mask in enumerate(masks):
                if np.sum(mask) == 0:
                    continue

                color = colors[i % len(colors)]

                # Apply color to mask area
                overlay[mask > 0] = color

            # Blend original image and mask
            blended = (1 - alpha) * img_array + alpha * overlay
            result_img = Image.fromarray(blended.astype(np.uint8))

            # If need to show points or boxes, add drawing logic here
            if show_points or show_boxes:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(result_img)
                # Drawing logic for points and boxes can be added here
                # Need to get points and boxes info from metadata or elsewhere
                pass

            # Save image
            if save_path:
                result_img.save(save_path)
                logger.info(f"SAM visualization result saved: {save_path}")

            return result_img

        except Exception as e:
            logger.error(f"SAM visualization failed: {e}")
            # Return original image
            if isinstance(image, Image.Image):
                return image
            else:
                return Image.open(image) if isinstance(image, (str, Path)) else Image.fromarray(image)

    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """Model warmup"""
        if not self.is_loaded:
            self.load_model()

        # Create dummy input for warmup
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                # Use automatic mode for warmup (reduce points for speed)
                original_points = self.points_per_side
                self.points_per_side = 8  # Temporarily reduce points

                # Recreate mask generator
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.model,
                    points_per_side=self.points_per_side,
                    pred_iou_thresh=self.pred_iou_thresh,
                    stability_score_thresh=self.stability_score_thresh,
                    crop_n_layers=0,  # Simplified
                    min_mask_region_area=100,
                )

                results = self.predict(dummy_image, mode="automatic")
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)

                # Restore original setting
                self.points_per_side = original_points

            except Exception as e:
                logger.warning(f"SAM warmup run {i+1} failed: {e}")

        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)

            logger.info(f"SAM model warmup completed - avg time: {avg_time:.3f}s")

            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }

        return super().warmup(num_runs)

    def predict_batch(self,
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     mode: str = "automatic",
                     **kwargs) -> List[Dict[str, Any]]:
        """Batch prediction"""
        if not self.is_loaded:
            self.load_model()

        results = []
        for image in images:
            try:
                result = self.predict(image, mode=mode, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for an image: {e}")
                # Add empty result to keep list length consistent
                results.append({
                    'masks': np.empty((0, 0, 0)),
                    'scores': [],
                    'areas': [],
                    'bboxes': [],
                    'metadata': {'error': str(e)}
                })

        return results

    def set_automatic_config(self, preset: str = "default"):
        """Set preset config for automatic segmentation"""
        presets = {
            "fast": {
                "points_per_side": 16,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.9,
                "crop_n_layers": 0,
                "min_mask_region_area": 500
            },
            "default": {
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
                "crop_n_layers": 0,
                "min_mask_region_area": 0
            },
            "quality": {
                "points_per_side": 64,
                "pred_iou_thresh": 0.92,
                "stability_score_thresh": 0.97,
                "crop_n_layers": 1,
                "min_mask_region_area": 100
            }
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available presets: {list(presets.keys())}")

        config = presets[preset]

        # Update parameters
        self.points_per_side = config["points_per_side"]
        self.pred_iou_thresh = config["pred_iou_thresh"]
        self.stability_score_thresh = config["stability_score_thresh"]
        self.crop_n_layers = config["crop_n_layers"]
        self.min_mask_region_area = config["min_mask_region_area"]

        # Recreate mask generator
        if self.is_loaded:
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                min_mask_region_area=self.min_mask_region_area,
            )

        logger.info(f"SAM config set to: {preset}")

    def get_mask_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get mask statistics"""
        masks = results.get('masks', [])
        scores = results.get('scores', [])
        areas = results.get('areas', [])

        if len(masks) == 0:
            return {
                'num_masks': 0,
                'total_area': 0,
                'avg_area': 0,
                'avg_score': 0,
                'coverage_ratio': 0
            }

        total_area = sum(areas)
        avg_area = total_area / len(areas)
        avg_score = sum(scores) / len(scores) if scores else 0

        # Calculate coverage ratio (assuming image size)
        image_area = self.original_size[0] * self.original_size[1] if hasattr(self, 'original_size') else 1
        coverage_ratio = total_area / image_area

        return {
            'num_masks': len(masks),
            'total_area': total_area,
            'avg_area': avg_area,
            'avg_score': avg_score,
            'coverage_ratio': coverage_ratio,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0
        }

    def filter_masks(self,
                    results: Dict[str, Any],
                    min_area: int = 100,
                    min_score: float = 0.8,
                    max_masks: int = None) -> Dict[str, Any]:
        """Filter mask results"""
        masks = results.get('masks', [])
        scores = results.get('scores', [])
        areas = results.get('areas', [])
        bboxes = results.get('bboxes', [])

        if len(masks) == 0:
            return results

        # Create filter indices
        valid_indices = []
        for i, (area, score) in enumerate(zip(areas, scores)):
            if area >= min_area and score >= min_score:
                valid_indices.append(i)

        # If max_masks specified, sort by score and take top N
        if max_masks and len(valid_indices) > max_masks:
            scored_indices = [(i, scores[i]) for i in valid_indices]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            valid_indices = [i for i, _ in scored_indices[:max_masks]]

        # Apply filtering
        filtered_results = results.copy()
        filtered_results['masks'] = masks[valid_indices] if len(valid_indices) > 0 else np.empty((0, *masks.shape[1:]))
        filtered_results['scores'] = [scores[i] for i in valid_indices]
        filtered_results['areas'] = [areas[i] for i in valid_indices]
        filtered_results['bboxes'] = [bboxes[i] for i in valid_indices]

        # Update metadata
        if 'metadata' not in filtered_results:
            filtered_results['metadata'] = {}
        filtered_results['metadata']['filtered'] = True
        filtered_results['metadata']['original_num_masks'] = len(masks)
        filtered_results['metadata']['filtered_num_masks'] = len(valid_indices)
        filtered_results['metadata']['filter_params'] = {
            'min_area': min_area,
            'min_score': min_score,
            'max_masks': max_masks
        }

        logger.info(f"Mask filtering: {len(masks)} -> {len(valid_indices)}")
        return filtered_results

    def merge_masks(self, results: Dict[str, Any], overlap_threshold: float = 0.8) -> Dict[str, Any]:
        """Merge overlapping masks"""
        masks = results.get('masks', [])

        if len(masks) <= 1:
            return results

        try:
            # Calculate IoU between masks
            def calculate_iou(mask1, mask2):
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                return intersection / union if union > 0 else 0

            # Find mask pairs to merge
            to_merge = []
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    iou = calculate_iou(masks[i], masks[j])
                    if iou > overlap_threshold:
                        to_merge.append((i, j, iou))

            if not to_merge:
                return results

            # Perform merge (simplified: only merge the pair with highest IoU)
            to_merge.sort(key=lambda x: x[2], reverse=True)
            i, j, _ = to_merge[0]

            # Create merged result
            merged_results = results.copy()
            merged_mask = np.logical_or(masks[i], masks[j]).astype(np.uint8)

            # Update arrays
            new_masks = []
            new_scores = []
            new_areas = []
            new_bboxes = []

            scores = results.get('scores', [])
            areas = results.get('areas', [])
            bboxes = results.get('bboxes', [])

            for k in range(len(masks)):
                if k == i:
                    # Use the merged mask
                    new_masks.append(merged_mask)
                    new_scores.append(max(scores[i], scores[j]) if k < len(scores) else 1.0)
                    new_areas.append(float(np.sum(merged_mask)))

                    # Calculate merged bounding box
                    y_indices, x_indices = np.where(merged_mask > 0)
                    if len(y_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        new_bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                    else:
                        new_bboxes.append([0.0, 0.0, 0.0, 0.0])
                elif k != j:  # Skip merged mask
                    new_masks.append(masks[k])
                    new_scores.append(scores[k] if k < len(scores) else 1.0)
                    new_areas.append(areas[k] if k < len(areas) else float(np.sum(masks[k])))
                    new_bboxes.append(bboxes[k] if k < len(bboxes) else [0.0, 0.0, 0.0, 0.0])

            merged_results['masks'] = np.array(new_masks)
            merged_results['scores'] = new_scores
            merged_results['areas'] = new_areas
            merged_results['bboxes'] = new_bboxes

            if 'metadata' not in merged_results:
                merged_results['metadata'] = {}
            merged_results['metadata']['merged'] = True
            merged_results['metadata']['original_num_masks'] = len(masks)
            merged_results['metadata']['merged_num_masks'] = len(new_masks)

            logger.info(f"Mask merge: {len(masks)} -> {len(new_masks)}")
            return merged_results

        except Exception as e:
            logger.warning(f"Mask merge failed: {e}")
            return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model detailed info"""
        info = super().get_model_info()

        info.update({
            'model_type': 'segmentation',
            'framework': 'segment_anything',
            'architecture': f'sam_{self.model_type}',
            'points_per_side': self.points_per_side,
            'pred_iou_thresh': self.pred_iou_thresh,
            'stability_score_thresh': self.stability_score_thresh,
            'supported_modes': ['automatic', 'interactive'],
            'input_prompts': ['points', 'boxes', 'masks']
        })

        if self.is_loaded:
            try:
                # Get model parameter count
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params
                })

            except Exception as e:
                logger.debug(f"Failed to get SAM model detailed info: {e}")

        return info
