"""
CV Model Platform REST API

基于FastAPI的RESTful API服务，提供统一的模型调用接口
"""

import os
import sys
import time
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.cv_platform.core.model_manager import get_model_manager
    from src.cv_platform.utils.logger import setup_logger
    from src.cv_platform.core.config_manager import get_config_manager
except ImportError as e:
    logger.error(f"导入CV Platform模块失败: {e}")
    sys.exit(1)

# 全局变量
model_manager = None
temp_dir = Path("temp_api_files")
temp_dir.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global model_manager
    logger.info("初始化CV Model Platform API...")
    
    try:
        model_manager = get_model_manager()
        logger.info("模型管理器初始化成功")
    except Exception as e:
        logger.error(f"模型管理器初始化失败: {e}")
        sys.exit(1)
    
    yield
    
    # 关闭时清理
    logger.info("清理API资源...")
    if model_manager:
        model_manager.clear_cache()

# 创建FastAPI应用
app = FastAPI(
    title="CV Model Platform API",
    description="统一的计算机视觉模型服务API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（用于临时结果文件）
app.mount("/static", StaticFiles(directory=str(temp_dir)), name="static")

# =============================================================================
# 数据模型定义
# =============================================================================

class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    type: str
    framework: str
    architecture: str
    device: str
    is_loaded: bool
    file_size_mb: Optional[float] = None

class DetectionResult(BaseModel):
    """检测结果"""
    bbox: List[float] = Field(description="边界框坐标 [x1, y1, x2, y2]")
    class_name: str = Field(description="类别名称") 
    class_id: int = Field(description="类别ID")
    confidence: float = Field(description="置信度")
    area: float = Field(description="区域面积")

class SegmentationResult(BaseModel):
    """分割结果"""
    num_masks: int = Field(description="掩码数量")
    total_area: float = Field(description="总面积")
    avg_score: float = Field(description="平均分数")
    coverage_ratio: float = Field(description="覆盖率")
    result_url: Optional[str] = Field(description="结果文件URL")

class GenerationResult(BaseModel):
    """生成结果"""
    image_url: str = Field(description="生成图像URL")
    prompt: str = Field(description="输入提示词")
    steps: int = Field(description="推理步数")
    guidance_scale: float = Field(description="引导尺度")

class APIResponse(BaseModel):
    """API响应格式"""
    success: bool
    message: str
    data: Optional[Any] = None
    execution_time: Optional[float] = None
    request_id: Optional[str] = None

# =============================================================================
# 工具函数
# =============================================================================

def generate_request_id() -> str:
    """生成请求ID"""
    return str(uuid.uuid4())[:8]

def save_temp_file(file: UploadFile, prefix: str = "temp") -> Path:
    """保存临时文件"""
    file_ext = Path(file.filename).suffix if file.filename else ".jpg"
    temp_file = temp_dir / f"{prefix}_{uuid.uuid4().hex}{file_ext}"
    
    with open(temp_file, "wb") as f:
        f.write(file.file.read())
    
    return temp_file

async def cleanup_temp_file(file_path: Path, delay: int = 300):
    """延迟清理临时文件"""
    await asyncio.sleep(delay)
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"清理临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"清理临时文件失败: {e}")

def create_success_response(data: Any, message: str = "操作成功", 
                          execution_time: float = None, request_id: str = None) -> APIResponse:
    """创建成功响应"""
    return APIResponse(
        success=True,
        message=message,
        data=data,
        execution_time=execution_time,
        request_id=request_id
    )

def create_error_response(message: str, request_id: str = None) -> APIResponse:
    """创建错误响应"""
    return APIResponse(
        success=False,
        message=message,
        request_id=request_id
    )

# =============================================================================
# API路由定义
# =============================================================================

@app.get("/", response_model=APIResponse)
async def root():
    """根路径 - API信息"""
    return create_success_response({
        "name": "CV Model Platform API",
        "version": "0.1.0",
        "description": "统一的计算机视觉模型服务",
        "endpoints": {
            "models": "/models",
            "detect": "/detect/{model_name}",
            "segment": "/segment/{model_name}",
            "classify": "/classify/{model_name}",
            "generate": "/generate/{model_name}",
            "health": "/health",
            "docs": "/docs"
        }
    })

@app.get("/health", response_model=APIResponse)
async def health_check():
    """健康检查"""
    try:
        status = model_manager.get_system_status()
        return create_success_response(status, "系统正常")
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统异常: {str(e)}")

@app.get("/models", response_model=APIResponse)
async def list_models():
    """获取所有可用模型列表"""
    try:
        models = model_manager.list_available_models()
        
        model_list = []
        for name, info in models.items():
            config = info['config']
            model_info = ModelInfo(
                name=name,
                type=config.get('type', 'unknown'),
                framework=config.get('framework', 'unknown'),
                architecture=config.get('architecture', 'unknown'),
                device=config.get('device', 'auto'),
                is_loaded=False  # 这里可以检查缓存状态
            )
            
            # 尝试获取文件大小
            if 'model_info' in info:
                model_info.file_size_mb = info['model_info'].size_mb
            
            model_list.append(model_info)
        
        return create_success_response(model_list)
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

@app.get("/models/{model_name}", response_model=APIResponse)
async def get_model_info(model_name: str):
    """获取特定模型信息"""
    try:
        models = model_manager.list_available_models()
        
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        
        # 加载模型获取详细信息
        adapter = model_manager.load_model(model_name)
        detailed_info = adapter.get_model_info()
        
        return create_success_response(detailed_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")

@app.post("/detect/{model_name}", response_model=APIResponse)
async def detect_objects(
    model_name: str,
    image: UploadFile = File(..., description="输入图像文件"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    nms_threshold: float = Query(0.45, ge=0.0, le=1.0, description="NMS阈值"),
    background_tasks: BackgroundTasks = None
):
    """目标检测API"""
    request_id = generate_request_id()
    start_time = time.time()
    temp_file = None
    
    try:
        # 检查模型是否存在
        models = model_manager.list_available_models()
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"检测模型 {model_name} 不存在")
        
        # 检查模型类型
        model_type = models[model_name]['config'].get('type')
        if model_type != 'detection':
            raise HTTPException(status_code=400, detail=f"模型 {model_name} 不是检测模型，类型: {model_type}")
        
        # 保存临时文件
        temp_file = save_temp_file(image, "detect")
        
        # 执行检测
        results = model_manager.predict(
            model_name, 
            str(temp_file),
            confidence=confidence,
            nms_threshold=nms_threshold
        )
        
        # 转换结果格式
        detection_results = [
            DetectionResult(
                bbox=result['bbox'],
                class_name=result['class'],
                class_id=result['class_id'],
                confidence=result['confidence'],
                area=result['area']
            )
            for result in results
        ]
        
        execution_time = time.time() - start_time
        
        # 添加后台任务清理临时文件
        if background_tasks and temp_file:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return create_success_response(
            {
                "detections": detection_results,
                "total_objects": len(detection_results),
                "model_name": model_name,
                "parameters": {
                    "confidence": confidence,
                    "nms_threshold": nms_threshold
                }
            },
            f"检测完成，发现 {len(detection_results)} 个对象",
            execution_time,
            request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")
    finally:
        # 如果没有后台任务，立即清理
        if temp_file and not background_tasks:
            try:
                temp_file.unlink()
            except:
                pass

@app.post("/segment/{model_name}", response_model=APIResponse)
async def segment_image(
    model_name: str,
    image: UploadFile = File(..., description="输入图像文件"),
    mode: str = Form("automatic", description="分割模式: automatic, point, box"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="分割阈值"),
    points: Optional[str] = Form(None, description="点坐标 JSON格式: [[x1,y1],[x2,y2]]"),
    point_labels: Optional[str] = Form(None, description="点标签 JSON格式: [1,0,1]"),
    box: Optional[str] = Form(None, description="边界框 JSON格式: [x1,y1,x2,y2]"),
    save_visualization: bool = Form(False, description="是否保存可视化结果"),
    background_tasks: BackgroundTasks = None
):
    """图像分割API"""
    import json
    
    request_id = generate_request_id()
    start_time = time.time()
    temp_file = None
    
    try:
        # 检查模型
        models = model_manager.list_available_models()
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"分割模型 {model_name} 不存在")
        
        model_type = models[model_name]['config'].get('type')
        if model_type != 'segmentation':
            raise HTTPException(status_code=400, detail=f"模型 {model_name} 不是分割模型，类型: {model_type}")
        
        # 保存临时文件
        temp_file = save_temp_file(image, "segment")
        
        # 加载模型
        adapter = model_manager.load_model(model_name)
        
        # 根据模型框架选择预测方式
        framework = models[model_name]['config'].get('framework')
        
        if framework == 'segment_anything':
            # SAM模型
            predict_kwargs = {'mode': mode}
            
            if mode == 'point' and points and point_labels:
                try:
                    points_list = json.loads(points)
                    labels_list = json.loads(point_labels)
                    predict_kwargs.update({
                        'points': points_list,
                        'point_labels': labels_list
                    })
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="点坐标或标签格式错误")
            
            elif mode == 'box' and box:
                try:
                    box_coords = json.loads(box)
                    predict_kwargs.update({'boxes': [box_coords]})
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="边界框格式错误")
            
            results = adapter.predict(str(temp_file), **predict_kwargs)
            
        else:
            # 其他分割模型（如DeepLabV3）
            results = adapter.predict(str(temp_file), threshold=threshold)
        
        # 处理可视化
        result_url = None
        if save_visualization:
            vis_filename = f"segment_result_{request_id}.jpg"
            vis_path = temp_dir / vis_filename
            
            try:
                adapter.visualize_results(str(temp_file), results, save_path=str(vis_path))
                result_url = f"/static/{vis_filename}"
            except Exception as e:
                logger.warning(f"可视化失败: {e}")
        
        # 构建响应
        metadata = results.get('metadata', {})
        segmentation_result = SegmentationResult(
            num_masks=metadata.get('num_masks', len(results.get('masks', []))),
            total_area=metadata.get('total_area', sum(results.get('areas', []))),
            avg_score=metadata.get('avg_score', sum(results.get('scores', [])) / len(results.get('scores', [])) if results.get('scores') else 0),
            coverage_ratio=metadata.get('coverage_ratio', 0),
            result_url=result_url
        )
        
        execution_time = time.time() - start_time
        
        # 清理任务
        if background_tasks and temp_file:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return create_success_response(
            {
                "segmentation": segmentation_result,
                "model_name": model_name,
                "mode": mode,
                "parameters": {
                    "threshold": threshold,
                    "framework": framework
                },
                "metadata": metadata
            },
            f"分割完成，生成 {segmentation_result.num_masks} 个掩码",
            execution_time,
            request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分割失败: {e}")
        raise HTTPException(status_code=500, detail=f"分割失败: {str(e)}")
    finally:
        if temp_file and not background_tasks:
            try:
                temp_file.unlink()
            except:
                pass

@app.post("/classify/{model_name}", response_model=APIResponse)
async def classify_image(
    model_name: str,
    image: UploadFile = File(..., description="输入图像文件"),
    top_k: int = Query(5, ge=1, le=100, description="返回前K个结果"),
    background_tasks: BackgroundTasks = None
):
    """图像分类API"""
    request_id = generate_request_id()
    start_time = time.time()
    temp_file = None
    
    try:
        # 检查模型
        models = model_manager.list_available_models()
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"分类模型 {model_name} 不存在")
        
        model_type = models[model_name]['config'].get('type')
        if model_type != 'classification':
            raise HTTPException(status_code=400, detail=f"模型 {model_name} 不是分类模型，类型: {model_type}")
        
        # 保存临时文件
        temp_file = save_temp_file(image, "classify")
        
        # 执行分类
        results = model_manager.predict(model_name, str(temp_file), top_k=top_k)
        
        execution_time = time.time() - start_time
        
        # 清理任务
        if background_tasks and temp_file:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return create_success_response(
            {
                "classification": results,
                "model_name": model_name,
                "parameters": {"top_k": top_k}
            },
            "分类完成",
            execution_time,
            request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")
    finally:
        if temp_file and not background_tasks:
            try:
                temp_file.unlink()
            except:
                pass

@app.get("/cache/stats", response_model=APIResponse)
async def get_cache_stats():
    """获取缓存统计信息"""
    try:
        stats = model_manager.get_cache_stats()
        return create_success_response(stats)
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")

@app.post("/cache/clear", response_model=APIResponse)
async def clear_cache():
    """清空模型缓存"""
    try:
        model_manager.clear_cache()
        return create_success_response(None, "缓存已清空")
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {str(e)}")

# =============================================================================
# 错误处理
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=create_error_response("资源未找到").__dict__
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=create_error_response("内部服务器错误").__dict__
    )

# =============================================================================
# 主函数
# =============================================================================

def main():
    """启动API服务器"""
    # 设置日志
    setup_logger("INFO")
    
    # 获取配置
    config_manager = get_config_manager()
    platform_config = config_manager.get_platform_config()
    api_config = platform_config.get('api', {})
    
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    workers = api_config.get('workers', 1)
    
    logger.info(f"启动CV Model Platform API服务器...")
    logger.info(f"服务地址: http://{host}:{port}")
    logger.info(f"API文档: http://{host}:{port}/docs")
    
    # 启动服务器
    uvicorn.run(
        "cv_platform.api.rest_api:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
