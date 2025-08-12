#!/usr/bin/env python3
"""
Image Classification Demo Script

Demonstrates how to use CV Model Platform for image classification
Supports classification models such as ResNet, EfficientNet, ViT, etc.
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
    print("Please ensure that dependencies are correctly installed and run from the project root directory.")
    sys.exit(1)

def create_test_image():
    """Create a test image"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np


        # Create a test image with distinctive features.
        width, height = 224, 224  # Common sizes for classification models
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create a simple “dog” shape (round head + oval body)
        # Head
        head_center = (width // 2, height // 3)
        head_radius = 40
        draw.ellipse([
            head_center[0] - head_radius, head_center[1] - head_radius,
            head_center[0] + head_radius, head_center[1] + head_radius
        ], fill='brown', outline='black', width=2)
        
        # Body
        body_center = (width // 2, height * 2 // 3)
        body_width, body_height = 60, 40
        draw.ellipse([
            body_center[0] - body_width, body_center[1] - body_height,
            body_center[0] + body_width, body_center[1] + body_height
        ], fill='brown', outline='black', width=2)
        
        # Eye
        eye_size = 5
        left_eye = (head_center[0] - 15, head_center[1] - 10)
        right_eye = (head_center[0] + 15, head_center[1] - 10)
        draw.ellipse([left_eye[0] - eye_size, left_eye[1] - eye_size,
                    left_eye[0] + eye_size, left_eye[1] + eye_size], fill='black')
        draw.ellipse([right_eye[0] - eye_size, right_eye[1] - eye_size,
                    right_eye[0] + eye_size, right_eye[1] + eye_size], fill='black')
        
        # Nose
        nose = (head_center[0], head_center[1] + 5)
        draw.ellipse([nose[0] - 3, nose[1] - 2, nose[0] + 3, nose[1] + 2], fill='black')
        
        # Leg
        leg_positions = [
            (body_center[0] - 30, body_center[1] + 25),  
            (body_center[0] - 10, body_center[1] + 25),  
            (body_center[0] + 10, body_center[1] + 25),  
            (body_center[0] + 30, body_center[1] + 25),  
        ]
        
        for leg_pos in leg_positions:
            draw.rectangle([
                leg_pos[0] - 5, leg_pos[1], 
                leg_pos[0] + 5, leg_pos[1] + 20
            ], fill='brown', outline='black')
        
        # Add some background textures
        for _ in range(20):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            color = (np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255))
            draw.point((x, y), fill=color)
        
        # Save test image
        test_image_path = Path("test_classification_image.jpg")
        img.save(test_image_path)
        
        logger.info(f"Classification test images have been created: {test_image_path}")
        return str(test_image_path)
    
    except ImportError:
        logger.error("PIL is not installed, so test images cannot be created.")
        return None
    except Exception as e:
        logger.error(f"Failed to create test image: {e}")
        return None


def test_classification(model_name, image_path, top_k=5):
    """Test image classification function"""
    try:
        logger.info(f"Test image classification: {model_name}")


        manager = get_model_manager()
        
        # Load model
        logger.info("Load classification model...")
        results = manager.predict(model_name, image_path, top_k=top_k)
        
        logger.info("Classification completed")
        
        # Show results
        if 'predictions' in results and len(results['predictions']) > 0:
            predictions = results['predictions']
            top_class = results.get('top_class', predictions[0]['class'])
            top_confidence = results.get('top_confidence', predictions[0]['confidence'])
            
            logger.info(f"🎯 Best prediction: {top_class} (confidence: {top_confidence:.3f})")
            logger.info(f"📊 The first {len(predictions)} prediction results:")
            
            for i, pred in enumerate(predictions, 1):
                class_name = pred['class']
                confidence = pred['confidence']
                class_id = pred.get('class_id', 'N/A')
                
                # Add confidence level indicator
                if confidence > 0.7:
                    confidence_icon = "🟢"
                elif confidence > 0.3:
                    confidence_icon = "🟡"
                else:
                    confidence_icon = "🔴"
                
                logger.info(f"{i}. {confidence_icon} {class_name}")
                logger.info(f"Confidence: {confidence:.3f} | Class ID: {class_id}")
        else:
            logger.warning("No classification results found")
        
        # Try visualization (if supported by the adapter)
        try:
            adapter = manager.load_model(model_name)
            if hasattr(adapter, 'visualize_results'):
                vis_result = adapter.visualize_results(
                    image_path, 
                    results, 
                    save_path="classification_result.jpg"
                )
                logger.info("Classification visualization results saved: classification_result.jpg")
        except Exception as e:
            logger.debug(f"Visualization failed (this is normal, as classification models typically do not require visualization): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_classification(model_name, image_paths, top_k=3):
    """Test batch classification"""
    try:
        logger.info(f"Test batch classification: {model_name}")
        logger.info(f"Batch size: {len(image_paths)}")


        manager = get_model_manager()
        adapter = manager.load_model(model_name)
        
        # Check if batch prediction is supported
        if hasattr(adapter, 'predict_batch'):
            logger.info("Use the batch prediction interface...")
            batch_results = adapter.predict_batch(image_paths, top_k=top_k)
        else:
            logger.info("Predict individually...")
            batch_results = []
            for img_path in image_paths:
                result = adapter.predict(img_path, top_k=top_k)
                batch_results.append(result)
        
        # Show batch results
        logger.info("📊 Batch classification results:")
        for i, (img_path, result) in enumerate(zip(image_paths, batch_results), 1):
            if 'predictions' in result and result['predictions']:
                top_pred = result['predictions'][0]
                logger.info(f"   {i}. {Path(img_path).name}: {top_pred['class']} ({top_pred['confidence']:.3f})")
            else:
                logger.info(f"   {i}. {Path(img_path).name}: Classification failed")
        
        return True
        
    except Exception as e:
        logger.error(f"Batch classification test failed: {e}")
        return False


def test_model_comparison(models, image_path, top_k=3):
    """Compare the classification results of different models"""
    try:
        logger.info("🔄 Model comparison test")
        logger.info(f"Models included in the comparison: {','.join(models)}")


        manager = get_model_manager()
        comparison_results = {}
        
        for model_name in models:
            try:
                logger.info(f"Test model: {model_name}")
                result = manager.predict(model_name, image_path, top_k=top_k)
                comparison_results[model_name] = result
            except Exception as e:
                logger.warning(f"Model {model_name} test failed: {e}")
                comparison_results[model_name] = None
        
        # Show comparison results
        logger.info("📋 Model comparison results:")
        print("=" * 80)
        print(f"{'Model name':<20} {'Best prediction':<25} {'Confidence':<10} {'Top 3 predictions'}")
        print("-" * 80)
        
        for model_name, result in comparison_results.items():
            if result and 'predictions' in result and result['predictions']:
                top_pred = result['predictions'][0]
                top_3 = [p['class'][:10] for p in result['predictions'][:3]]
                print(f"{model_name:<20} {top_pred['class'][:24]:<25} {top_pred['confidence']:<10.3f} {', '.join(top_3)}")
            else:
                print(f"{model_name:<20} {'Failed':<25} {'N/A':<10} {'N/A'}")
        
        print("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='CV Model Platform Image Classification Demo')


    parser.add_argument('--model', '-m',
                    type=str,
                    help='Model name to be used')

    parser.add_argument('--image', '-i',
                    type=str,
                    help='Test image path (if not provided, a test image will be created)')

    parser.add_argument('--top-k', '-k',
                    type=int,
                    default=5,
                    help='Return the previous K prediction results (default: 5)')

    parser.add_argument('--list-models', '-l',
                    action='store_true',
                    help='List all available classification models')

    parser.add_argument('--test-all',
                    action='store_true',
                    help='Test all available classification models')

    parser.add_argument('--batch-test',
                    action='store_true',
                    help='Test batch classification (requires multiple images)')

    parser.add_argument('--compare',
                    action='store_true',
                    help='Compare the classification results of multiple models')

    parser.add_argument('--images',
                    type=str,
                    nargs='+',
                    help='Multiple image paths (for batch testing)')

    parser.add_argument('--verbose', '-v',
                    action='store_true',
                    help='Detailed output')

    args = parser.parse_args()

    if args.verbose:
        setup_logger("DEBUG")

    try:
        manager = get_model_manager()
        available_models = manager.list_available_models()
        
        # Screening classification model
        classification_models = {
            name: info for name, info in available_models.items()
            if info['config'].get('type') == 'classification'
        }
        
        if args.list_models:
            print("\n📋 Available classification models:")
            print("=" * 60)
            
            if not classification_models:
                print("⚠️ No classification model found")
                print("\n💡 Recommendation:")
                print("1. Ensure that the model directory contains the classification model file.")
                print("2. Run model discovery script to update configuration")
                print("3. Classification models typically use pre-trained models from torchvision.")
                return 0
            
            for name, info in classification_models.items():
                config = info['config']
                framework = config.get('framework', 'unknown')
                architecture = config.get('architecture', 'unknown')
                
                print(f"🔧 {name}")
                print(f"Architecture: {architecture}")
                print(f"Framework: {framework}")
                print(f"Path: {config.get('path', 'unknown')}")
                
                # Check dependencies
                if framework == 'torchvision':
                    print("Dependencies: torchvision (usually already installed)")
                elif framework == 'timm':
                    print("Dependencies: timm (Requires installation: pip install timm)")
                elif framework == 'transformers':
                    print("Dependencies: transformers (Requires installation: pip install transformers)")
                print()
            
            return 0
        
        if not classification_models:
            print("❌  No classification models found")
            print("Please run: python examples/basic_usage/classification_demo.py --list-models")
            return 1
        
        # Prepare test images
        if args.batch_test and args.images:
            image_paths = args.images
            for img_path in image_paths:
                if not Path(img_path).exists():
                    logger.error(f"Image file does not exist: {img_path}")
                    return 1
        else:
            image_path = args.image
            if not image_path:
                image_path = create_test_image()
                if not image_path:
                    logger.error("Unable to create test image")
                    return 1
            
            test_image = Path(image_path)
            if not test_image.exists():
                logger.error(f"Test image does not exist: {test_image}")
                return 1
        
        print("🚀 CV Model Platform - Image Classification Demo")
        print("=" * 50)
        
        success_count = 0
        total_tests = 0
        
        if args.test_all:
            # Test all classification models
            for model_name in classification_models.keys():
                print(f"\n🧪 Test model: {model_name}")
                print("-" * 30)
                
                total_tests += 1
                
                if test_classification(model_name, image_path, args.top_k):
                    success_count += 1
        
        elif args.batch_test:
            # Batch testing
            if not args.images:
                logger.error("Batch testing requires multiple image paths to be provided. --images")
                return 1
            
            model_name = args.model
            if not model_name:
                model_name = next(iter(classification_models.keys()))
                logger.info(f"Use the first available classification model: {model_name}")
            
            total_tests = 1
            if test_batch_classification(model_name, args.images, args.top_k):
                success_count = 1
        
        elif args.compare:
            # Model comparison
            model_names = list(classification_models.keys())[:3]  # Compare up to 3 models
            if len(model_names) < 2:
                logger.warning("At least two models are required for comparison.")
                return 1
            
            total_tests = 1
            if test_model_comparison(model_names, image_path, args.top_k):
                success_count = 1
        
        else:
            # Test the specified model or the first available model.
            if args.model:
                if args.model not in classification_models:
                    logger.error(f"Model {args.model} is not available.")
                    logger.info("Available classification models:")
                    for name in classification_models.keys():
                        logger.info(f"  - {name}")
                    return 1
                model_name = args.model
            else:
                model_name = next(iter(classification_models.keys()))
                logger.info(f"Use the first available classification model: {model_name}")
            
            total_tests = 1
            if test_classification(model_name, image_path, args.top_k):
                success_count = 1
        
        print("\n" + "=" * 50)
        print(f"📊 Test results: {success_count}/{total_tests} passed")
        
        if success_count > 0:
            print("🎉 Classification demo complete!")
            print("\n🚀 Next, try the following:")
            print("1. Use your own images: python examples/basic_usage/classification_demo.py -i your_image.jpg")
            print("2. Adjust the top-k value: python examples/basic_usage/classification_demo.py -k 10")
            print("3. Batch testing: python examples/basic_usage/classification_demo.py --batch-test --images img1.jpg img2.jpg")
            print("4. Model comparison: python examples/basic_usage/classification_demo.py --compare")
            print("5. Test all models: python examples/basic_usage/classification_demo.py --test-all")
            return 0 if success_count == total_tests else 1
        else:
            print("❌ Classification demonstration failed")
            print("\n💡 Possible solutions:")
            print("1. Install any missing dependencies: pip install timm transformers")
            print("2. Check the model file path")
            print("3. Ensure that classification models are available.")
            print("4. Run: python examples/basic_usage/classification_demo.py --list-models")
            return 1
            
    except KeyboardInterrupt:
        print("\nUser cancel operation")
        return 0
    except Exception as e:
        logger.error(f"Program exception: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
