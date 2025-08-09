#!/usr/bin/env python3
"""
API测试客户端

演示如何使用CV Model Platform的REST API
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, Any

class CVPlatformAPIClient:
    """CV Platform API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化API客户端
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """获取模型列表"""
        response = self.session.get(f"{self.base_url}/models")
        return response.json()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        response = self.session.get(f"{self.base_url}/models/{model_name}")
        return response.json()
    
    def detect_objects(self, model_name: str, image_path: str, 
                      confidence: float = 0.25, nms_threshold: float = 0.45) -> Dict[str, Any]:
        """目标检测"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'confidence': confidence,
                'nms_threshold': nms_threshold
            }
            response = self.session.post(
                f"{self.base_url}/detect/{model_name}", 
                files=files, 
                data=data
            )
        return response.json()
    
    def segment_image(self, model_name: str, image_path: str,
                     mode: str = "automatic", threshold: float = 0.5,
                     save_visualization: bool = False) -> Dict[str, Any]:
        """图像分割"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'mode': mode,
                'threshold': threshold,
                'save_visualization': save_visualization
            }
            response = self.session.post(
                f"{self.base_url}/segment/{model_name}",
                files=files,
                data=data
            )
        return response.json()
    
    def classify_image(self, model_name: str, image_path: str, 
                      top_k: int = 5) -> Dict[str, Any]:
        """图像分类"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            params = {'top_k': top_k}
            response = self.session.post(
                f"{self.base_url}/classify/{model_name}",
                files=files,
                params=params
            )
        return response.json()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        response = self.session.get(f"{self.base_url}/cache/stats")
        return response.json()
    
    def clear_cache(self) -> Dict[str, Any]:
        """清空缓存"""
        response = self.session.post(f"{self.base_url}/cache/clear")
        return response.json()

def test_api_connection(client: CVPlatformAPIClient):
    """测试API连接"""
    print("🔗 测试API连接...")
    
    try:
        response = client.health_check()
        if response.get('success'):
            print("✅ API连接正常")
            return True
        else:
            print(f"❌ API连接失败: {response.get('message')}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API服务器: {e}")
        print("请确保API服务器已启动: python scripts/start_api.py")
        return False

def test_model_listing(client: CVPlatformAPIClient):
    """测试模型列表"""
    print("\n📋 测试模型列表...")
    
    try:
        response = client.list_models()
        if response.get('success'):
            models = response['data']
            print(f"✅ 发现 {len(models)} 个模型:")
            
            for model in models:
                print(f"   📦 {model['name']}: {model['type']} ({model['framework']})")
            
            return models
        else:
            print(f"❌ 获取模型列表失败: {response.get('message')}")
            return []
    except Exception as e:
        print(f"❌ 测试模型列表失败: {e}")
        return []

def test_detection(client: CVPlatformAPIClient, models: list):
    """测试目标检测"""
    print("\n🔍 测试目标检测...")
    
    # 寻找检测模型
    detection_models = [m for m in models if m['type'] == 'detection']
    
    if not detection_models:
        print("⚠️  未找到检测模型，跳过检测测试")
        return
    
    model_name = detection_models[0]['name']
    print(f"使用模型: {model_name}")
    
    # 检查测试图像
    test_image = Path("test_image.jpg")
    if not test_image.exists():
        print("⚠️  测试图像不存在，跳过检测测试")
        return
    
    try:
        start_time = time.time()
        response = client.detect_objects(model_name, str(test_image))
        elapsed = time.time() - start_time
        
        if response.get('success'):
            data = response['data']
            detections = data['detections']
            print(f"✅ 检测完成 - 耗时: {elapsed:.2f}s")
            print(f"   发现 {len(detections)} 个对象:")
            
            for i, detection in enumerate(detections[:5], 1):  # 只显示前5个
                print(f"   {i}. {detection['class_name']}: {detection['confidence']:.3f}")
        else:
            print(f"❌ 检测失败: {response.get('message')}")
    
    except Exception as e:
        print(f"❌ 检测测试失败: {e}")

def test_segmentation(client: CVPlatformAPIClient, models: list):
    """测试图像分割"""
    print("\n🎨 测试图像分割...")
    
    # 寻找分割模型
    segmentation_models = [m for m in models if m['type'] == 'segmentation']
    
    if not segmentation_models:
        print("⚠️  未找到分割模型，跳过分割测试")
        return
    
    model_name = segmentation_models[0]['name']
    print(f"使用模型: {model_name}")
    
    # 检查测试图像
    test_image = Path("test_segmentation_image.jpg")
    if not test_image.exists():
        test_image = Path("test_image.jpg")
        if not test_image.exists():
            print("⚠️  测试图像不存在，跳过分割测试")
            return
    
    try:
        start_time = time.time()
        response = client.segment_image(
            model_name, 
            str(test_image), 
            save_visualization=True
        )
        elapsed = time.time() - start_time
        
        if response.get('success'):
            data = response['data']
            segmentation = data['segmentation']
            print(f"✅ 分割完成 - 耗时: {elapsed:.2f}s")
            print(f"   生成 {segmentation['num_masks']} 个掩码")
            print(f"   覆盖率: {segmentation['coverage_ratio']:.2%}")
            
            if segmentation.get('result_url'):
                print(f"   可视化结果: {client.base_url}{segmentation['result_url']}")
        else:
            print(f"❌ 分割失败: {response.get('message')}")
    
    except Exception as e:
        print(f"❌ 分割测试失败: {e}")

def test_model_info(client: CVPlatformAPIClient, models: list):
    """测试模型详细信息"""
    print("\n📊 测试模型详细信息...")
    
    if not models:
        print("⚠️  没有可用模型")
        return
    
    model_name = models[0]['name']
    print(f"获取模型信息: {model_name}")
    
    try:
        response = client.get_model_info(model_name)
        
        if response.get('success'):
            info = response['data']
            print("✅ 模型信息获取成功:")
            print(f"   设备: {info.get('device')}")
            print(f"   是否已加载: {info.get('is_loaded')}")
            
            if 'total_parameters' in info:
                params = info['total_parameters']
                print(f"   参数数量: {params/1e6:.1f}M")
            
        else:
            print(f"❌ 获取模型信息失败: {response.get('message')}")
    
    except Exception as e:
        print(f"❌ 模型信息测试失败: {e}")

def test_cache_management(client: CVPlatformAPIClient):
    """测试缓存管理"""
    print("\n💾 测试缓存管理...")
    
    try:
        # 获取缓存统计
        response = client.get_cache_stats()
        
        if response.get('success'):
            stats = response['data']
            print("✅ 缓存统计:")
            print(f"   缓存模型数: {stats.get('cached_models', 0)}")
            print(f"   缓存已启用: {stats.get('cache_enabled', False)}")
        else:
            print(f"❌ 获取缓存统计失败: {response.get('message')}")
    
    except Exception as e:
        print(f"❌ 缓存管理测试失败: {e}")

def main():
    """主测试函数"""
    print("🚀 CV Model Platform API 测试")
    print("=" * 50)
    
    # 创建客户端
    client = CVPlatformAPIClient("http://localhost:8000")
    
    # 测试连接
    if not test_api_connection(client):
        return
    
    # 测试模型列表
    models = test_model_listing(client)
    
    if models:
        # 测试模型信息
        test_model_info(client, models)
        
        # 测试检测
        test_detection(client, models)
        
        # 测试分割
        test_segmentation(client, models)
        
        # 测试缓存
        test_cache_management(client)
    
    print("\n" + "=" * 50)
    print("🎉 API测试完成!")
    print("\n💡 更多API使用方法:")
    print("   1. 访问API文档: http://localhost:8000/docs")
    print("   2. 查看ReDoc文档: http://localhost:8000/redoc")
    print("   3. 使用Postman或curl进行测试")

if __name__ == '__main__':
    main()