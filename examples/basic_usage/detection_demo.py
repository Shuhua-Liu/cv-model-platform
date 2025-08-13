#!/usr/bin/env python3
"""
Object Detection Demo Script

Shows how to perform object detection using the CV Model Platform.
"""

import sys
import argparse
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_manager import get_model_manager
    from src.cv_platform.utils.logger import setup_logger
    setup_logger("INFO")
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the dependencies are installed correctly and run from the project root directory")
    sys.exit(1)

def create_test_image():
    """Create a test image"""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple test image
        width, height = 640, 480
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some simple shapes as "objects"
        # Draw a rectangle
        draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=2)
        draw.rectangle([300, 150, 450, 300], fill='green', outline='black', width=2)
        draw.rectangle([200, 300, 350, 400], fill='blue', outline='black', width=2)
        
        # Draw a circle
        draw.ellipse([450, 50, 550, 150], fill='yellow', outline='black', width=2)
        
        # Save the test image
        test_image_path = Path("test_image.jpg")
        img.save(test_image_path)
        
        logger.info(f"Test image created: {test_image_path}")
        return str(test_image_path)
        
    except ImportError:
        logger.error("PIL is not installed, cannot create test image")
        return None
    except Exception as e:
        logger.error(f"Creating a test image failed: {e}")
        return None

def test_model_detection(model_name="yolov8n", image_path=None):
    """Testing the object detection function"""
    
    # Get the model manager
    logger.info("Initialize the model manager...")
    manager = get_model_manager()
    
    # List available models
    available_models = manager.list_available_models()
    logger.info(f"Found {len(available_models)} available models")
    
    for name, info in available_models.items():
        model_type = info['config'].get('type', 'unknown')
        logger.info(f"  - {name}: {model_type}")
    
    # Checks whether the specified model is available
    if model_name not in available_models:
        logger.error(f"Model {model_name} is not available")
        logger.info("Available detection models:")
        detection_models = [name for name, info in available_models.items() 
                          if info['config'].get('type') == 'detection']
        
        if detection_models:
            for name in detection_models:
                logger.info(f"  - {name}")
            model_name = detection_models[0]
            logger.info(f"Use the first available detection model: {model_name}")
        else:
            logger.error("No detection models found")
            return False
    
    # Prepare test images
    if not image_path:
        image_path = create_test_image()
        if not image_path:
            logger.error("Unable to create test image")
            return False
    
    test_image = Path(image_path)
    if not test_image.exists():
        logger.error(f"The test image does not exist: {test_image}")
        return False
    
    try:
        # Load and test the model
        logger.info(f"Loading the model: {model_name}")
        
        # Method 1: Using the Model Manager Directly
        logger.info("Start forecasting...")
        results = manager.predict(model_name, str(test_image))
        
        logger.info(f"Detection complete - {len(results)} objects found")
        
        # Show result
        if results:
            logger.info("Test results:")
            for i, detection in enumerate(results, 1):
                class_name = detection['class']
                confidence = detection['confidence']
                bbox = detection['bbox']
                
                logger.info(f"  {i}. {class_name}: {confidence:.3f}")
                logger.info(f"     Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            logger.warning("No objects detected")
        
        # Method 2: Use the adapter directly (optional)
        logger.info("\nTesting direct adapter calls...")
        adapter = manager.load_model(model_name)
        direct_results = adapter.predict(str(test_image))
        
        logger.info(f"Direct call results: {len(direct_results)} objects")
        
        # Get model info
        model_info = adapter.get_model_info()
        logger.info("Model Info:")
        logger.info(f"  Adapter: {model_info.get('adapter_class', 'unknown')}")
        logger.info(f"  Device: {model_info.get('device', 'unknown')}")
        logger.info(f"  Loaded: {model_info.get('is_loaded', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='CV Model Platform object detection demo')
    
    parser.add_argument('--model', '-m',
                      type=str,
                      default='yolov8n',
                      help='The name of the model to use')
    
    parser.add_argument('--image', '-i',
                      type=str,
                      help='Test image path (if not provided a test image will be created)')
    
    parser.add_argument('--list-models', '-l',
                      action='store_true',
                      help='List all available models')
    
    parser.add_argument('--verbose', '-v',
                      action='store_true',
                      help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logger("DEBUG")
    
    try:
        if args.list_models:
            # List only models
            logger.info("Get a list of available models...")
            manager = get_model_manager()
            available_models = manager.list_available_models()
            
            print("\n📋 Available models:")
            print("=" * 60)
            
            for name, info in available_models.items():
                config = info['config']
                model_type = config.get('type', 'unknown')
                framework = config.get('framework', 'unknown')
                source = info.get('source', 'unknown')
                
                print(f"🔧 {name}")
                print(f"   Type: {model_type}")
                print(f"   Framework: {framework}")
                print(f"   Source: {source}")
                print(f"   Path: {config.get('path', 'unknown')}")
                print()
            
            return 0
        
        # Run detection tests
        print("🚀 CV Model Platform - Object Detection Demo")
        print("=" * 50)
        
        success = test_model_detection(args.model, args.image)
        
        if success:
            print("\n✅ Detection demonstration completed！")
            print("\n🎉 CV Model Platform works properly")
            print("\n🚀 Next you can try:")
            print("   1. Use your own images: python examples/basic_usage/detection_demo.py -i your_image.jpg")
            print("   2. Try other models: python examples/basic_usage/detection_demo.py -m model_name")
            print("   3. View all models: python examples/basic_usage/detection_demo.py --list-models")
            return 0
        else:
            print("\n❌ Detection demonstration failed")
            return 1
            
    except KeyboardInterrupt:
        print("\nUser cancels the operation")
        return 0
    except Exception as e:
        logger.error(f"Program exception: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
