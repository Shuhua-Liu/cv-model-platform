#!/usr/bin/env python3
"""
Enhanced Detection Demo - Compatible with updated ModelManager and ModelDetector

This demo shows how to use the enhanced CV platform with BaseManager inheritance
for object detection tasks using various models like YOLO.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_manager import get_model_manager
    from src.cv_platform.core.model_detector import get_model_detector
    from src.cv_platform.core.config_manager import get_config_manager
    from src.cv_platform.utils.logger import setup_logger
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the project is properly installed or run from project root")
    sys.exit(1)


def create_test_image(output_path: str = "test_image.jpg") -> Optional[str]:
    """
    Create a simple test image for detection demo
    
    Args:
        output_path: Path to save the test image
        
    Returns:
        Path to created image or None if failed
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image with some geometric shapes
        width, height = 640, 480
        image = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(image)
        
        # Draw some rectangles (simulating objects to detect)
        objects = [
            {"bbox": [100, 100, 200, 200], "color": "red", "label": "object1"},
            {"bbox": [300, 150, 450, 280], "color": "green", "label": "object2"},
            {"bbox": [200, 300, 350, 400], "color": "blue", "label": "object3"}
        ]
        
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            draw.rectangle([x1, y1, x2, y2], fill=obj["color"], outline="black", width=3)
            
            # Add label
            try:
                font = ImageFont.load_default()
                draw.text((x1, y1-20), obj["label"], fill="black", font=font)
            except:
                draw.text((x1, y1-20), obj["label"], fill="black")
        
        # Add title
        draw.text((10, 10), "Test Image for Object Detection", fill="black")
        
        image.save(output_path, quality=95)
        logger.info(f"Test image created: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create test image: {e}")
        return None


def check_system_health() -> bool:
    """
    Check system health using BaseManager capabilities
    
    Returns:
        True if system is healthy, False otherwise
    """
    logger.info("🔍 Checking system health...")
    
    try:
        # Get manager instances
        manager = get_model_manager()
        detector = get_model_detector()
        config_manager = get_config_manager()
        
        # Check ModelManager health
        manager_health = manager.perform_health_check()
        logger.info(f"ModelManager Status: {manager_health.status.value} - {manager_health.message}")
        
        if manager_health.details:
            logger.info(f"  Models available: {manager_health.details.get('models_available', 'unknown')}")
            logger.info(f"  Cache enabled: {manager_health.details.get('cache_stats', {}).get('cached_models', 'unknown')}")
        
        # Check ModelDetector health
        detector_health = detector.perform_health_check()
        logger.info(f"ModelDetector Status: {detector_health.status.value} - {detector_health.message}")
        
        if detector_health.details:
            logger.info(f"  Models detected: {detector_health.details.get('models_detected', 'unknown')}")
            logger.info(f"  Last scan: {detector_health.details.get('last_scan_age_minutes', 'unknown'):.1f} minutes ago")
        
        # Check if we have detection models available
        available_models = manager.list_available_models()
        detection_models = [
            name for name, info in available_models.items() 
            if info.get('config', {}).get('type') == 'detection'
        ]
        
        if not detection_models:
            logger.warning("⚠️ No detection models found")
            return False
        
        logger.info(f"✅ System health check passed - {len(detection_models)} detection models available")
        return True
        
    except Exception as e:
        logger.error(f"❌ System health check failed: {e}")
        return False


def demonstrate_model_discovery():
    """Demonstrate the enhanced model discovery capabilities"""
    logger.info("🔍 Demonstrating model discovery...")
    
    try:
        # Get detector instance
        detector = get_model_detector()
        
        # Show detection summary
        summary = detector.get_detection_summary()
        logger.info(f"Detection Summary:")
        logger.info(f"  Total models: {summary.get('total_models', 0)}")
        logger.info(f"  Models by type: {summary.get('models_by_type', {})}")
        logger.info(f"  Models by framework: {summary.get('models_by_framework', {})}")
        logger.info(f"  Total size: {summary.get('total_size_mb', 0):.1f} MB")
        
        # Force a rescan to demonstrate real-time discovery
        logger.info("Performing fresh model scan...")
        models = detector.detect_models(force_rescan=True)
        
        logger.info(f"Fresh scan found {len(models)} models:")
        for model in models[:5]:  # Show first 5 models
            logger.info(f"  - {model.name} ({model.type}/{model.framework}, {model.confidence:.2f} confidence)")
        
        if len(models) > 5:
            logger.info(f"  ... and {len(models) - 5} more models")
        
        return models
        
    except Exception as e:
        logger.error(f"Model discovery demonstration failed: {e}")
        return []


def demonstrate_model_loading_and_caching():
    """Demonstrate model loading and caching capabilities"""
    logger.info("🚀 Demonstrating model loading and caching...")
    
    try:
        manager = get_model_manager()
        
        # Get available detection models
        available_models = manager.list_available_models()
        detection_models = [
            name for name, info in available_models.items() 
            if info.get('config', {}).get('type') == 'detection'
        ]
        
        if not detection_models:
            logger.warning("No detection models available for loading demo")
            return None
        
        model_name = detection_models[0]
        logger.info(f"Loading model: {model_name}")
        
        # First load (should load from disk)
        start_time = time.time()
        try:
            adapter = manager.load_model(model_name)
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
            
            # Get model info
            model_info = adapter.get_model_info()
            logger.info(f"Model Info:")
            logger.info(f"  Adapter: {model_info.get('adapter_class', 'unknown')}")
            logger.info(f"  Device: {model_info.get('device', 'unknown')}")
            logger.info(f"  File size: {model_info.get('file_size_mb', 0):.1f} MB")
            
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Model loading demonstration failed: {e}")
        return None


def run_detection_test(adapter, image_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Run detection test on the given image
    
    Args:
        adapter: Loaded model adapter
        image_path: Path to test image
        
    Returns:
        Detection results or None if failed
    """
    logger.info(f"🎯 Running detection on: {image_path}")
    
    try:
        # Check if image exists
        if not Path(image_path).exists():
            logger.error(f"Test image not found: {image_path}")
            return None
        
        # Run detection
        start_time = time.time()
        results = adapter.predict(image_path)
        inference_time = time.time() - start_time
        
        logger.info(f"✅ Detection completed in {inference_time:.3f}s")
        logger.info(f"Found {len(results)} objects:")
        
        # Display results
        for i, detection in enumerate(results, 1):
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            
            logger.info(f"  {i}. {class_name}: {confidence:.3f}")
            logger.info(f"     Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        return results
        
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        return None


def demonstrate_cache_performance():
    """Demonstrate cache performance benefits"""
    logger.info("⚡ Demonstrating cache performance...")
    
    try:
        manager = get_model_manager()
        
        # Get cache statistics
        cache_stats = manager.get_cache_stats()
        logger.info(f"Cache Stats:")
        
        if cache_stats.get('cache_disabled'):
            logger.info("  Cache is disabled")
            return
        
        logger.info(f"  Cached models: {cache_stats.get('entry_count', 0)}")
        logger.info(f"  Cache utilization: {cache_stats.get('cache_efficiency', {}).get('utilization', 0):.1%}")
        logger.info(f"  Total cache size: {cache_stats.get('size_mb', 0):.1f} MB")
        
        # Show cached models
        cached_models = cache_stats.get('models', [])
        if cached_models:
            logger.info(f"  Cached models: {', '.join(cached_models)}")
        
    except Exception as e:
        logger.error(f"Cache demonstration failed: {e}")


def demonstrate_system_status():
    """Demonstrate comprehensive system status"""
    logger.info("📊 System Status Overview...")
    
    try:
        manager = get_model_manager()
        
        # Get system status
        system_status = manager.get_system_status()
        
        logger.info("System Status:")
        models_info = system_status.get('models', {})
        logger.info(f"  Total models: {models_info.get('total', 0)}")
        logger.info(f"  Cached models: {models_info.get('cached', 0)}")
        
        cache_info = system_status.get('cache', {})
        logger.info(f"  Cache enabled: {cache_info.get('enabled', False)}")
        logger.info(f"  Cache size: {cache_info.get('size', 0)}/{cache_info.get('max_size', 0)}")
        
        components_info = system_status.get('components', {})
        logger.info("  Components:")
        logger.info(f"    Config Manager: {'✅' if components_info.get('config_manager') else '❌'}")
        logger.info(f"    Registry: {'✅' if components_info.get('registry') else '❌'}")
        logger.info(f"    Detector: {'✅' if components_info.get('detector') else '❌'}")
        
        # GPU info if available
        gpu_info = system_status.get('gpu', [])
        if gpu_info:
            logger.info("  GPU Information:")
            for gpu in gpu_info:
                logger.info(f"    GPU {gpu['device_id']}: {gpu['name']}")
                logger.info(f"      Memory: {gpu['memory_allocated']:.1f}/{gpu['memory_total']:.1f} GB")
        
        # Performance summary
        perf_summary = manager.get_performance_summary()
        if perf_summary and 'error' not in perf_summary:
            metrics = perf_summary.get('metrics', {})
            logger.info("  Performance Metrics:")
            for metric_name, metric_info in metrics.items():
                value = metric_info.get('value', 'N/A')
                logger.info(f"    {metric_name}: {value}")
        
    except Exception as e:
        logger.error(f"System status demonstration failed: {e}")


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='Enhanced Object Detection Demo')
    parser.add_argument('--image', '-i', type=str, help='Path to test image')
    parser.add_argument('--model', '-m', type=str, help='Specific model to use')
    parser.add_argument('--create-test-image', action='store_true', 
                       help='Create a test image if none provided')
    parser.add_argument('--skip-health-check', action='store_true',
                       help='Skip initial health check')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level)
    
    logger.info("🚀 Enhanced Object Detection Demo")
    logger.info("=" * 60)
    
    try:
        # Step 1: System Health Check
        if not args.skip_health_check:
            if not check_system_health():
                logger.error("❌ System health check failed. Please check your setup.")
                return 1
        
        # Step 2: Demonstrate model discovery
        discovered_models = demonstrate_model_discovery()
        
        # Step 3: Prepare test image
        image_path = args.image
        if not image_path:
            if args.create_test_image:
                image_path = create_test_image()
                if not image_path:
                    logger.error("Failed to create test image")
                    return 1
            else:
                logger.error("No test image provided. Use --image or --create-test-image")
                return 1
        
        # Step 4: Load and test model
        adapter = demonstrate_model_loading_and_caching()
        if not adapter:
            logger.error("Failed to load any detection model")
            return 1
        
        # Step 5: Run detection test
        results = run_detection_test(adapter, image_path)
        if results is None:
            logger.error("Detection test failed")
            return 1
        
        # Step 6: Demonstrate cache performance
        demonstrate_cache_performance()
        
        # Step 7: Show system status
        demonstrate_system_status()
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Demo completed successfully!")
        logger.info(f"📊 Results summary:")
        logger.info(f"  Models discovered: {len(discovered_models)}")
        logger.info(f"  Objects detected: {len(results) if results else 0}")
        logger.info(f"  Test image: {image_path}")
        
        logger.info("\n🚀 Next steps:")
        logger.info("  1. Try with your own images")
        logger.info("  2. Experiment with different models") 
        logger.info("  3. Check the API documentation")
        logger.info("  4. Start the web interface or API server")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())