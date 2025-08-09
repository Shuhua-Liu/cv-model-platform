"""
API数据模式定义

使用Pydantic定义所有API请求和响应的数据结构
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

# =============================================================================
# 枚举定义
# =============================================================================

class ModelType(str, Enum):
    """模型类型枚举"""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    MULTIMODAL = "multimodal"

class SegmentationMode(str, Enum):
    """分割模式枚举"""
    AUTOMATIC = "automatic"
    POINT = "point"
    BOX = "box"
    INTERACTIVE = "interactive"

class ImageFormat(str, Enum):
    """图像格式枚举"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"

# =============================================================================
# 基础模型
# =============================================================================

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(description="操作是否成功")
    message: str = Field(description="响应消息")
    execution_time: Optional[float] = Field(None, description="执行时间(秒)")
    request_id: Optional[str] = Field(None, description="请求ID")

class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    error_code: Optional[str] = Field(None, description="错误代码")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")

class SuccessResponse(BaseResponse):
    """成功响应模型"""
    success: bool = True
    data: Any = Field(description="响应数据")

# =============================================================================
# 模型相关
# =============================================================================

class ModelInfo(BaseModel):
    """模型信息"""
    name: str = Field(description="模型名称")
    type: ModelType = Field(description="模型类型")
    framework: str = Field(description="模型框架")
    architecture: str = Field(description="模型架构")
    device: str = Field(description="运行设备")
    is_loaded: bool = Field(description="是否已加载")
    file_size_mb: Optional[float] = Field(None, description="文件大小(MB)")
    total_parameters: Optional[int] = Field(None, description="总参数数量")

class ModelListResponse(SuccessResponse):
    """模型列表响应"""
    data: List[ModelInfo]

class ModelDetailResponse(SuccessResponse):
    """模型详情响应"""
    data: Dict[str, Any]

# =============================================================================
# 检测相关
# =============================================================================

class BoundingBox(BaseModel):
    """边界框"""
    x1: float = Field(description="左上角X坐标")
    y1: float = Field(description="左上角Y坐标") 
    x2: float = Field(description="右下角X坐标")
    y2: float = Field(description="右下角Y坐标")

class DetectionObject(BaseModel):
    """检测对象"""
    bbox: List[float] = Field(description="边界框坐标 [x1, y1, x2, y2]")
    class_name: str = Field(description="类别名称")
    class_id: int = Field(description="类别ID")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    area: float = Field(ge=0.0, description="区域面积")

class DetectionRequest(BaseModel):
    """检测请求参数"""
    confidence: float = Field(0.25, ge=0.0, le=1.0, description="置信度阈值")
    nms_threshold: float = Field(0.45, ge=0.0, le=1.0, description="NMS阈值")
    max_objects: Optional[int] = Field(None, ge=1, description="最大检测对象数")

class DetectionResult(BaseModel):
    """检测结果"""
    detections: List[DetectionObject] = Field(description="检测对象列表")
    total_objects: int = Field(ge=0, description="总对象数")
    model_name: str = Field(description="使用的模型名称")
    parameters: DetectionRequest = Field(description="请求参数")

class DetectionResponse(SuccessResponse):
    """检测响应"""
    data: DetectionResult

# =============================================================================
# 分割相关
# =============================================================================

class Point(BaseModel):
    """点坐标"""
    x: float = Field(description="X坐标")
    y: float = Field(description="Y坐标")

class SegmentationRequest(BaseModel):
    """分割请求参数"""
    mode: SegmentationMode = Field(SegmentationMode.AUTOMATIC, description="分割模式")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="分割阈值")
    points: Optional[List[List[float]]] = Field(None, description="提示点坐标")
    point_labels: Optional[List[int]] = Field(None, description="点标签")
    box: Optional[List[float]] = Field(None, description="边界框 [x1,y1,x2,y2]")
    save_visualization: bool = Field(False, description="是否保存可视化结果")
    min_area: Optional[int] = Field(None, ge=0, description="最小掩码面积")
    max_masks: Optional[int] = Field(None, ge=1, description="最大掩码数量")

class MaskInfo(BaseModel):
    """掩码信息"""
    area: float = Field(ge=0.0, description="掩码面积")
    bbox: List[float] = Field(description="边界框 [x1, y1, x2, y2]")
    score: float = Field(ge=0.0, le=1.0, description="质量分数")

class SegmentationResult(BaseModel):
    """分割结果"""
    num_masks: int = Field(ge=0, description="掩码数量")
    masks_info: List[MaskInfo] = Field(description="掩码信息列表")
    total_area: float = Field(ge=0.0, description="总面积")
    avg_score: float = Field(ge=0.0, le=1.0, description="平均分数")
    coverage_ratio: float = Field(ge=0.0, le=1.0, description="覆盖率")
    visualization_url: Optional[str] = Field(None, description="可视化结果URL")
    model_name: str = Field(description="使用的模型名称")
    parameters: SegmentationRequest = Field(description="请求参数")

class SegmentationResponse(SuccessResponse):
    """分割响应"""
    data: SegmentationResult

# =============================================================================
# 分类相关
# =============================================================================

class ClassificationItem(BaseModel):
    """分类项"""
    class_name: str = Field(description="类别名称")
    class_id: int = Field(description="类别ID")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")

class ClassificationRequest(BaseModel):
    """分类请求参数"""
    top_k: int = Field(5, ge=1, le=100, description="返回前K个结果")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="置信度阈值")

class ClassificationResult(BaseModel):
    """分类结果"""
    predictions: List[ClassificationItem] = Field(description="预测结果列表")
    top_class: str = Field(description="最高置信度类别")
    top_confidence: float = Field(ge=0.0, le=1.0, description="最高置信度")
    model_name: str = Field(description="使用的模型名称")
    parameters: ClassificationRequest = Field(description="请求参数")

class ClassificationResponse(SuccessResponse):
    """分类响应"""
    data: ClassificationResult

# =============================================================================
# 生成相关
# =============================================================================

class GenerationRequest(BaseModel):
    """生成请求参数"""
    prompt: str = Field(description="正向提示词")
    negative_prompt: Optional[str] = Field(None, description="负向提示词")
    num_steps: int = Field(20, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="引导尺度")
    width: int = Field(512, ge=64, le=2048, description="图像宽度")
    height: int = Field(512, ge=64, le=2048, description="图像高度")
    seed: Optional[int] = Field(None, description="随机种子")
    num_images: int = Field(1, ge=1, le=4, description="生成图像数量")

class GeneratedImage(BaseModel):
    """生成的图像"""
    url: str = Field(description="图像URL")
    seed: Optional[int] = Field(None, description="使用的随机种子")
    filename: str = Field(description="文件名")

class GenerationResult(BaseModel):
    """生成结果"""
    images: List[GeneratedImage] = Field(description="生成的图像列表")
    model_name: str = Field(description="使用的模型名称")
    parameters: GenerationRequest = Field(description="请求参数")

class GenerationResponse(SuccessResponse):
    """生成响应"""
    data: GenerationResult

# =============================================================================
# 系统相关
# =============================================================================

class SystemStatus(BaseModel):
    """系统状态"""
    models: Dict[str, Any] = Field(description="模型状态")
    system: Dict[str, Any] = Field(description="系统资源")
    torch: Dict[str, Any] = Field(description="PyTorch信息")
    gpu: Optional[List[Dict[str, Any]]] = Field(None, description="GPU信息")

class CacheStats(BaseModel):
    """缓存统计"""
    cached_models: int = Field(description="缓存模型数量")
    max_size: int = Field(description="最大缓存大小")
    cache_enabled: bool = Field(description="缓存是否启用")
    models: Dict[str, Dict[str, Any]] = Field(description="缓存模型详情")

class HealthResponse(SuccessResponse):
    """健康检查响应"""
    data: SystemStatus

class CacheStatsResponse(SuccessResponse):
    """缓存统计响应"""
    data: CacheStats

# =============================================================================
# 批处理相关
# =============================================================================

class BatchRequest(BaseModel):
    """批处理请求"""
    model_name: str = Field(description="模型名称")
    images: List[str] = Field(description="图像URL或base64编码列表")
    parameters: Optional[Dict[str, Any]] = Field(None, description="模型参数")

class BatchResult(BaseModel):
    """批处理结果"""
    results: List[Any] = Field(description="批处理结果列表")
    success_count: int = Field(description="成功处理数量")
    total_count: int = Field(description="总处理数量")
    failed_indices: List[int] = Field(description="失败的索引列表")

class BatchResponse(SuccessResponse):
    """批处理响应"""
    data: BatchResult

# =============================================================================
# 文件相关
# =============================================================================

class FileInfo(BaseModel):
    """文件信息"""
    filename: str = Field(description="文件名")
    size: int = Field(description="文件大小(字节)")
    content_type: str = Field(description="文件类型")
    upload_time: str = Field(description="上传时间")

class UploadResponse(SuccessResponse):
    """上传响应"""
    data: FileInfo

# =============================================================================
# WebSocket相关
# =============================================================================

class WSMessage(BaseModel):
    """WebSocket消息"""
    type: str = Field(description="消息类型")
    data: Any = Field(description="消息数据")
    timestamp: str = Field(description="时间戳")

class WSRequest(BaseModel):
    """WebSocket请求"""
    action: str = Field(description="操作类型")
    model_name: Optional[str] = Field(None, description="模型名称")
    parameters: Optional[Dict[str, Any]] = Field(None, description="参数")

class WSResponse(BaseModel):
    """WebSocket响应"""
    success: bool = Field(description="是否成功")
    message: str = Field(description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    progress: Optional[float] = Field(None, description="进度(0-1)")

# =============================================================================
# 配置相关
# =============================================================================

class ModelConfig(BaseModel):
    """模型配置"""
    type: ModelType = Field(description="模型类型")
    path: str = Field(description="模型路径")
    device: str = Field(description="运行设备")
    framework: str = Field(description="模型框架")
    parameters: Optional[Dict[str, Any]] = Field(None, description="模型参数")

class PlatformConfig(BaseModel):
    """平台配置"""
    api: Dict[str, Any] = Field(description="API配置")
    cache: Dict[str, Any] = Field(description="缓存配置")
    logging: Dict[str, Any] = Field(description="日志配置")
    models_root: str = Field(description="模型根目录")

class ConfigResponse(SuccessResponse):
    """配置响应"""
    data: Union[ModelConfig, PlatformConfig, Dict[str, Any]]

# =============================================================================
# 验证相关
# =============================================================================

class ValidationResult(BaseModel):
    """验证结果"""
    model_name: str = Field(description="模型名称")
    status: str = Field(description="验证状态")
    errors: List[str] = Field(description="错误列表")
    warnings: List[str] = Field(description="警告列表")

class ValidationResponse(SuccessResponse):
    """验证响应"""
    data: List[ValidationResult]

# =============================================================================
# 统计相关
# =============================================================================

class ModelUsageStats(BaseModel):
    """模型使用统计"""
    model_name: str = Field(description="模型名称")
    request_count: int = Field(description="请求次数")
    avg_response_time: float = Field(description="平均响应时间")
    success_rate: float = Field(description="成功率")
    last_used: str = Field(description="最后使用时间")

class PlatformStats(BaseModel):
    """平台统计"""
    total_requests: int = Field(description="总请求数")
    active_models: int = Field(description="活跃模型数")
    avg_response_time: float = Field(description="平均响应时间")
    uptime: str = Field(description="运行时间")
    model_usage: List[ModelUsageStats] = Field(description="模型使用统计")

class StatsResponse(SuccessResponse):
    """统计响应"""
    data: PlatformStats
