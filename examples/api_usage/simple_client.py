#!/usr/bin/env python3
"""
简单API客户端示例

展示如何使用requests库调用CV Model Platform API
"""

import requests
import json

# API服务器地址
API_BASE = "http://localhost:8000"

def demo_detection():
    """检测演示"""
    print("🔍 目标检测演示")
    
    # 上传图像进行检测
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
            print(f"✅ 检测到 {len(detections)} 个对象:")
            for obj in detections:
                print(f"   - {obj['class_name']}: {obj['confidence']:.3f}")
        else:
            print(f"❌ 检测失败: {result['message']}")
    else:
        print(f"❌ HTTP错误: {response.status_code}")

def demo_segmentation():
    """分割演示"""
    print("\n🎨 图像分割演示")
    
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
            print(f"✅ 生成 {seg_data['num_masks']} 个掩码")
            print(f"   覆盖率: {seg_data['coverage_ratio']:.2%}")
            if seg_data.get("result_url"):
                print(f"   可视化: {API_BASE}{seg_data['result_url']}")
        else:
            print(f"❌ 分割失败: {result['message']}")
    else:
        print(f"❌ HTTP错误: {response.status_code}")

def demo_model_list():
    """模型列表演示"""
    print("\n📋 模型列表演示")
    
    response = requests.get(f"{API_BASE}/models")
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            models = result["data"]
            print(f"✅ 可用模型 ({len(models)} 个):")
            for model in models:
                print(f"   📦 {model['name']}: {model['type']} ({model['framework']})")
        else:
            print(f"❌ 获取失败: {result['message']}")
    else:
        print(f"❌ HTTP错误: {response.status_code}")

def demo_health_check():
    """健康检查演示"""
    print("\n💓 健康检查演示")
    
    response = requests.get(f"{API_BASE}/health")
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            status = result["data"]
            print("✅ 系统状态正常")
            print(f"   模型数量: {status['models']['total']}")
            print(f"   缓存模型: {status['models']['cached']}")
            print(f"   CPU使用: {status['system']['cpu_percent']:.1f}%")
            print(f"   内存使用: {status['system']['memory_percent']:.1f}%")
        else:
            print(f"❌ 健康检查失败: {result['message']}")
    else:
        print(f"❌ HTTP错误: {response.status_code}")

def main():
    """主函数"""
    print("🚀 CV Model Platform API 简单客户端演示")
    print("=" * 50)
    
    try:
        # 健康检查
        demo_health_check()
        
        # 模型列表
        demo_model_list()
        
        # 检测演示
        demo_detection()
        
        # 分割演示
        demo_segmentation()
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器")
        print("请确保API服务器已启动: python scripts/start_api.py")
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 演示完成!")

if __name__ == "__main__":
    main()