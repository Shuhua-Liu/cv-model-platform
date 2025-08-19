#!/usr/bin/env python3
"""
Simple API Client Example

Shows how to use the requests library to call the CV Model Platform API.
"""

import requests
import json

# API server address
API_BASE = "http://localhost:8000"

def demo_detection():
    """Detection Demo"""
    print("🔍 Object Detection Demo")
    
    # Upload images for detection
    with open("test_image.jpg", "rb") as f:
        response = requests.post(
            f"{API_BASE}/detect/yolov8n",
            files={"image": f},
            data={
                "confidence": 0.25,
                "nms_threshold": 0.45
            }
        )
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            detections = result["data"]["detections"]
            print(f"✅ {len(detections)} objects detected:")
            for obj in detections:
                print(f"   - {obj['class_name']}: {obj['confidence']:.3f}")
        else:
            print(f"❌ Detection failure: {result['message']}")
    else:
        print(f"❌ HTTP Errors: {response.status_code}")

def demo_segmentation():
    """Segmentation Demo"""
    print("\n🎨 Image Segmentation Demo")
    
    with open("test_image.jpg", "rb") as f:
        response = requests.post(
            f"{API_BASE}/segment/deeplabv3_resnet101",
            files={"image": f},
            data={
                "mode": "automatic",
                "threshold": 0.5,
                "save_visualization": True
            }
        )
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            seg_data = result["data"]["segmentation"]
            print(f"✅ Generate {seg_data['num_masks']} masks")
            print(f"   Coverage: {seg_data['coverage_ratio']:.2%}")
            if seg_data.get("result_url"):
                print(f"   Visualization: {API_BASE}{seg_data['result_url']}")
        else:
            print(f"❌ Segmentation failed: {result['message']}")
    else:
        print(f"❌ HTTP Errors: {response.status_code}")

def demo_model_list():
    """Model List Demo"""
    print("\n📋 Model List Demo")
    
    response = requests.get(f"{API_BASE}/models")
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            models = result["data"]
            print(f"✅ Available models ({len(models)}):")
            for model in models:
                print(f"   📦 {model['name']}: {model['type']} ({model['framework']})")
        else:
            print(f"❌ Failed to obtain: {result['message']}")
    else:
        print(f"❌ HTTP Errors: {response.status_code}")

def demo_health_check():
    """Health Check Demo"""
    print("\n💓 Health Check Demo")
    
    response = requests.get(f"{API_BASE}/health")
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            status = result["data"]
            print("✅ System status is normal")
            print(f"   Number of models: {status['models']['total']}")
            print(f"   Cache Model: {status['models']['cached']}")
            print(f"   CPU usage: {status['system']['cpu_percent']:.1f}%")
            print(f"   Memory usage: {status['system']['memory_percent']:.1f}%")
        else:
            print(f"❌ Health check failure: {result['message']}")
    else:
        print(f"❌ HTTP Errors: {response.status_code}")

def main():
    """Main"""
    print("🚀 CV Model Platform API Simple Client Demo")
    print("=" * 50)
    
    try:
        # Health check
        demo_health_check()
        
        # Model list
        demo_model_list()
        
        # Detection demo
        demo_detection()
        
        # Segmentation demo
        demo_segmentation()
        
    except requests.exceptions.ConnectionError:
        print("❌ Unable to connect to the API server")
        print("Please make sure the API server is started: python scripts/start_api.py")
    except Exception as e:
        print(f"❌ Error during presentation: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demonstration completed!")

if __name__ == "__main__":
    main()