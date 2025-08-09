# CV Model Platform REST API 使用指南

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动API服务器
```bash
# 使用默认配置启动
python scripts/start_api.py

# 自定义配置启动
python scripts/start_api.py --host 0.0.0.0 --port 8000 --workers 4

# 开发模式（热重载）
python scripts/start_api.py --reload --log-level debug
```

### 3. 访问API文档
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API根路径**: http://localhost:8000/

## 📋 API端点概览

### 系统管理
| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | API信息和端点列表 |
| `/health` | GET | 健康检查和系统状态 |
| `/models` | GET | 获取所有可用模型 |
| `/models/{name}` | GET | 获取特定模型详细信息 |

### 模型调用
| 端点 | 方法 | 描述 |
|------|------|------|
| `/detect/{model_name}` | POST | 目标检测 |
| `/segment/{model_name}` | POST | 图像分割 |
| `/classify/{model_name}` | POST | 图像分类 |
| `/generate/{model_name}` | POST | 图像生成 |

### 缓存管理
| 端点 | 方法 | 描述 |
|------|------|------|
| `/cache/stats` | GET | 获取缓存统计 |
| `/cache/clear` | POST | 清空模型缓存 |

## 🔍 使用示例

### Python客户端
```python
import requests

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 获取模型列表
response = requests.get("http://localhost:8000/models")
models = response.json()["data"]

# 目标检测
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect/yolov8n",
        files={"image": f},
        data={"confidence": 0.25}
    )
    result = response.json()
```

### cURL示例
```bash
# 健康检查
curl http://localhost:8000/health

# 获取模型列表
curl http://localhost:8000/models

# 目标检测
curl -X POST \
  -F "image=@test_image.jpg" \
  -F "confidence=0.25" \
  http://localhost:8000/detect/yolov8n

# 图像分割
curl -X POST \
  -F "image=@test_image.jpg" \
  -F "mode=automatic" \
  -F "save_visualization=true" \
  http://localhost:8000/segment/deeplabv3_resnet101
```

### JavaScript示例
```javascript
// 使用FormData上传图像
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('confidence', '0.25');

fetch('http://localhost:8000/detect/yolov8n', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('检测结果:', data.data.detections);
    }
});
```

## 📊 响应格式

### 成功响应
```json
{
    "success": true,
    "message": "操作成功",
    "data": { ... },
    "execution_time": 1.23,
    "request_id": "abc123"
}
```

### 错误响应
```json
{
    "success": false,
    "message": "错误描述",
    "request_id": "abc123"
}
```

## 🔍 目标检测API

### 请求
```http
POST /detect/{model_name}
Content-Type: multipart/form-data

image: [图像文件]
confidence: 0.25 (可选，默认0.25)
nms_threshold: 0.45 (可选，默认0.45)
```

### 响应
```json
{
    "success": true,
    "data": {
        "detections": [
            {
                "bbox": [x1, y1, x2, y2],
                "class_name": "person",
                "class_id": 0,
                "confidence": 0.85,
                "area": 1200.5
            }
        ],
        "total_objects": 1,
        "model_name": "yolov8n",
        "parameters": {
            "confidence": 0.25,
            "nms_threshold": 0.45
        }
    }
}
```

## 🎨 图像分割API

### 请求
```http
POST /segment/{model_name}
Content-Type: multipart/form-data

image: [图像文件]
mode: "automatic" (automatic/point/box)
threshold: 0.5 (可选)
save_visualization: false (可选)
points: "[[x1,y1],[x2,y2]]" (point模式时需要)
point_labels: "[1,0]" (point模式时需要)
box: "[x1,y1,x2,y2]" (box模式时需要)
```

### 响应
```json
{
    "success": true,
    "data": {
        "segmentation": {
            "num_masks": 5,
            "total_area": 15234.0,
            "avg_score": 0.89,
            "coverage_ratio": 0.35,
            "result_url": "/static/segment_result_abc123.jpg"
        },
        "model_name": "sam_vit_b",
        "mode": "automatic"
    }
}
```

## 📊 图像分类API

### 请求
```http
POST /classify/{model_name}
Content-Type: multipart/form-data

image: [图像文件]
top_k: 5 (可选，默认5)
```

### 响应
```json
{
    "success": true,
    "data": {
        "classification": {
            "predictions": [
                {
                    "class": "Egyptian cat",
                    "class_id": 285,
                    "confidence": 0.95
                }
            ],
            "top_class": "Egyptian cat",
            "top_confidence": 0.95
        }
    }
}
```

## 🚀 部署配置

### 生产环境配置
```bash
# 使用Gunicorn部署
pip install gunicorn

# 启动多进程服务
gunicorn src.cv_platform.api.rest_api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300

# 使用supervisor管理进程
# 配置文件示例见 deploy/supervisor.conf
```

### Docker部署
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

COPY . .
EXPOSE 8000

CMD ["python", "scripts/start_api.py", "--host", "0.0.0.0"]
```

### Nginx反向代理
```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## 🔒 安全考虑

### 文件上传限制
- 最大文件大小：100MB
- 支持格式：JPEG, PNG, WebP
- 文件类型验证：基于MIME类型

### 访问控制
```python
# 可以添加认证中间件
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.middleware("http")
async def authenticate(request: Request, call_next):
    # 添加API密钥验证逻辑
    pass
```

### 速率限制
```python
# 可以使用slowapi进行速率限制
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/detect/{model_name}")
@limiter.limit("10/minute")
async def detect_objects(request: Request, ...):
    pass
```

## 📈 监控和日志

### 性能监控
- 每个请求的执行时间会记录在响应中
- 可以集成Prometheus进行指标收集
- 使用`/health`端点进行健康监控

### 日志配置
```python
# 配置详细日志
setup_logger("DEBUG")  # 开发环境
setup_logger("INFO")   # 生产环境
```

### 错误处理
- 所有异常都会被捕获并返回友好的错误信息
- 详细的错误日志会记录到服务器日志中
- 支持自定义错误处理器

## 🧪 测试

### 运行API测试
```bash
# 启动API服务器（终端1）
python scripts/start_api.py

# 运行测试客户端（终端2）
python examples/api_usage/test_api.py

# 运行简单示例
python examples/api_usage/simple_client.py
```

### 单元测试
```bash
# 使用pytest进行API测试
pytest tests/api/ -v
```

## 🔧 故障排除

### 常见问题

1. **API服务器无法启动**
   - 检查端口是否被占用
   - 确认所有依赖已安装
   - 查看错误日志

2. **模型加载失败**
   - 检查模型文件路径
   - 确认模型格式正确
   - 查看模型配置文件

3. **上传文件失败**
   - 检查文件大小是否超限
   - 确认文件格式支持
   - 查看磁盘空间

### 调试模式
```bash
# 启用详细日志和热重载
python scripts/start_api.py --reload --log-level debug
```

## 📚 更多资源

- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [Uvicorn部署指南](https://www.uvicorn.org/deployment/)
- [API最佳实践](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design)

