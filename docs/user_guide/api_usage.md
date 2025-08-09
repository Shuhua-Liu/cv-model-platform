# CV Model Platform REST API Usage Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
# Start with default configuration
python scripts/start_api.py

# Start with custom configuration
python scripts/start_api.py --host 0.0.0.0 --port 8000 --workers 4

# Development mode (hot-reload)
python scripts/start_api.py --reload --log-level debug
```

### 3. Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

## 📋 API Endpoint Overview

### System Management
| Endpoint | Method | Description |
|------|------|------|
| `/` | GET | API information and endpoint list |
| `/health` | GET | Health check and system status |
| `/models` | GET | Get all available models |
| `/models/{name}` | GET | Get details for a specific model |

### Model Invocation
| Endpoint | Method | Description |
|------|------|------|
| `/detect/{model_name}` | POST | Object Detection |
| `/segment/{model_name}` | POST | Image Segmentation |
| `/classify/{model_name}` | POST | Image Classification |
| `/generate/{model_name}` | POST | Image Generation |

### Cache Management
| Endpoint | Method | Description |
|------|------|------|
| `/cache/stats` | GET | Get cache statistics |
| `/cache/clear` | POST | Clear the model cache |

## 🔍 Usage Examples

### Python Client
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get model list
response = requests.get("http://localhost:8000/models")
models = response.json()["data"]

# Object Detection
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect/yolov8n",
        files={"image": f},
        data={"confidence": 0.25}
    )
    result = response.json()
```

### cURL Example
```bash
# Health check
curl http://localhost:8000/health

# Get model list
curl http://localhost:8000/models

# Object Detection
curl -X POST \
  -F "image=@test_image.jpg" \
  -F "confidence=0.25" \
  http://localhost:8000/detect/yolov8n

# Image Segmentation
curl -X POST \
  -F "image=@test_image.jpg" \
  -F "mode=automatic" \
  -F "save_visualization=true" \
  http://localhost:8000/segment/deeplabv3_resnet101
```

### JavaScript Example
```javascript
// Using FormData to upload an image
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
        console.log('Detection results:', data.data.detections);
    }
});
```

## 📊 Response Format

### Success Response
```json
{
    "success": true,
    "message": "Operation successful",
    "data": { ... },
    "execution_time": 1.23,
    "request_id": "abc123"
}
```

### Error Response
```json
{
    "success": false,
    "message": "Error Description",
    "request_id": "abc123"
}
```

## 🔍 Object Detection API

### Request
```http
POST /detect/{model_name}
Content-Type: multipart/form-data

image: [Image File]
confidence: 0.25 (optional, default 0.25)
nms_threshold: 0.45 (optional, default 0.45)
```

### Response
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

## 🎨 Image Segmentation API

### Request
```http
POST /segment/{model_name}
Content-Type: multipart/form-data

image: [Image File]
mode: "automatic" (automatic/point/box)
threshold: 0.5 (optional)
save_visualization: false (optional)
points: "[[x1,y1],[x2,y2]]" (required for point mode)
point_labels: "[1,0]" (required for point mode)
box: "[x1,y1,x2,y2]" (required for box mode)
```

### Response
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

## 📊 Image Classification API

### Request
```http
POST /classify/{model_name}
Content-Type: multipart/form-data

image: [Image File]
top_k: 5 (optional, default 5)
```

### Response
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

## 🚀 Deployment Configuration

### Production Environment Configuration
```bash
# Deploy with Gunicorn
pip install gunicorn

# Start multi-process service
gunicorn src.cv_platform.api.rest_api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300

# Manage processes with supervisor
# See deploy/supervisor.conf for an example configuration
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "scripts/start_api.py", "--host", "0.0.0.0"]
```

### Nginx Reverse Proxy
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

## 🔒 Security Considerations

### File Upload Limits
- Max file size: 100MB
- Supported formats: JPEG, PNG, WebP
- File type validation: Based on MIME type

### Access Control
```python
# Authentication middleware can be added
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.middleware("http")
async def authenticate(request: Request, call_next):
    # Add API key validation logic here
    pass
```

### Rate Limiting
```python
# slowapi can be used for rate limiting
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

## 📈 Monitoring and Logging

### Performance Monitoring
- Execution time for each request is recorded in the response
- Can be integrated with Prometheus for metrics collection
- Use the `/health` endpoint for health monitoring

### Logging Configuration
```python
# Configure detailed logging
setup_logger("DEBUG")  # Development environment
setup_logger("INFO")   # Production environment
```

### Error Handling
- All exceptions are caught and return a user-friendly error message
- Detailed error logs are recorded in the server logs
- Supports custom error handlers

## 🧪 Testing

### Running API Tests
```bash
# Start the API server (Terminal 1)
python scripts/start_api.py

# Run the test client (Terminal 2)
python examples/api_usage/test_api.py

# Run simple examples
python examples/api_usage/simple_client.py
```

### Unit Tests
```bash
# Use pytest for API testing
pytest tests/api/ -v
```

## 🔧 Troubleshooting

### Common Issues

1. **API server fails to start**
   - Check if the port is already in use
   - Confirm all dependencies are installed
   - Check the error logs

2. **Model fails to load**
   - Check the model file path
   - Confirm the model format is correct
   - Check the model configuration file

3. **File upload fails**
   - Check if the file size exceeds the limit
   - Confirm the file format is supported
   - Check available disk space

### Debug Mode
```bash
# Enable detailed logging and hot-reload
python scripts/start_api.py --reload --log-level debug
```

## 📚 更多资源

- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Deployment Guide](https://www.uvicorn.org/deployment/)
- [API Best Practices](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design)

