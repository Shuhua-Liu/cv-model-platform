"""
File Management Router

File location: src/cv_platform/api/routers/files.py

Handles file upload, download, management, and batch processing operations
with comprehensive validation and security features.
"""

import os
import uuid
import mimetypes
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
import aiofiles
from PIL import Image
import io

from ..models.responses import APIResponse
from ..models.requests import FileUploadMetadata
from ..dependencies.auth import get_current_user, verify_permissions
from ..dependencies.components import get_scheduler, get_model_manager

# Create router
router = APIRouter()

# Configuration
UPLOAD_DIR = Path("temp_api_files/uploads")
RESULTS_DIR = Path("temp_api_files/results")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'}
ALLOWED_ARCHIVE_TYPES = {'application/zip', 'application/x-zip-compressed'}

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=APIResponse)
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    description: Optional[str] = Form(None, description="File description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    auto_process: bool = Form(False, description="Automatically process with default model"),
    model_name: Optional[str] = Form(None, description="Model to use for auto-processing"),
    current_user: dict = Depends(verify_permissions(["write"])),
    background_tasks: BackgroundTasks = None,
    scheduler = Depends(get_scheduler)
):
    """
    Upload a single file with optional auto-processing
    
    Args:
        file: File to upload
        description: Optional file description
        tags: Comma-separated list of tags
        auto_process: Whether to automatically process the file
        model_name: Model to use for auto-processing
        current_user: Current authenticated user with write permissions
        background_tasks: FastAPI background tasks
        scheduler: Task scheduler dependency
        
    Returns:
        Upload result with file information and optional processing task ID
    """
    try:
        # Validate file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Validate file type
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
        if not content_type:
            raise HTTPException(
                status_code=400,
                detail="Could not determine file type"
            )
        
        # Generate unique file ID and path
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix if file.filename else ""
        safe_filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Process tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # Create file metadata
        file_metadata = {
            "file_id": file_id,
            "original_filename": file.filename,
            "safe_filename": safe_filename,
            "content_type": content_type,
            "file_size_bytes": file_size,
            "description": description,
            "tags": tag_list,
            "uploaded_by": current_user.get('user_id'),
            "upload_timestamp": time.time(),
            "file_path": str(file_path),
            "status": "uploaded"
        }
        
        # Validate and process image if applicable
        image_info = None
        if content_type in ALLOWED_IMAGE_TYPES:
            try:
                image_info = await _analyze_image(file_path)
                file_metadata.update(image_info)
            except Exception as e:
                logger.warning(f"Failed to analyze image {file_id}: {e}")
        
        # Auto-process if requested
        processing_task_id = None
        if auto_process:
            if not model_name:
                # Try to infer model based on file type
                model_name = _infer_model_for_file(content_type, image_info)
            
            if model_name:
                try:
                    # Submit processing task
                    processing_task_id = scheduler.submit_task(
                        model_name=model_name,
                        method="predict",
                        kwargs={"image_path": str(file_path)},
                        metadata={
                            "file_id": file_id,
                            "auto_processed": True,
                            "user_id": current_user.get('user_id')
                        }
                    )
                    
                    file_metadata["processing_task_id"] = processing_task_id
                    file_metadata["status"] = "processing"
                    
                except Exception as e:
                    logger.error(f"Failed to submit auto-processing task: {e}")
                    file_metadata["auto_process_error"] = str(e)
        
        # Store metadata (in production, use a database)
        await _store_file_metadata(file_id, file_metadata)
        
        return APIResponse(
            success=True,
            message=f"File uploaded successfully",
            data={
                "file_id": file_id,
                "filename": file.filename,
                "size_bytes": file_size,
                "content_type": content_type,
                "auto_processing": auto_process,
                "processing_task_id": processing_task_id,
                "image_info": image_info,
                "download_url": f"/api/v1/files/{file_id}/download"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {e}"
        )


@router.post("/upload/batch", response_model=APIResponse)
async def upload_batch_files(
    files: List[UploadFile] = File(..., description="Multiple files to upload"),
    auto_process: bool = Form(False, description="Auto-process all uploaded files"),
    model_name: Optional[str] = Form(None, description="Model for batch processing"),
    current_user: dict = Depends(verify_permissions(["write"])),
    background_tasks: BackgroundTasks = None,
    scheduler = Depends(get_scheduler)
):
    """
    Upload multiple files in batch with optional processing
    
    Args:
        files: List of files to upload
        auto_process: Whether to auto-process all files
        model_name: Model to use for batch processing
        current_user: Current authenticated user
        background_tasks: FastAPI background tasks
        scheduler: Task scheduler dependency
        
    Returns:
        Batch upload results
    """
    try:
        if len(files) > 50:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 50 files per batch."
            )
        
        upload_results = []
        failed_uploads = []
        processing_tasks = []
        
        for file in files:
            try:
                # Upload each file individually
                result = await upload_file(
                    file=file,
                    description=f"Batch upload - {file.filename}",
                    tags="batch_upload",
                    auto_process=auto_process,
                    model_name=model_name,
                    current_user=current_user,
                    background_tasks=background_tasks,
                    scheduler=scheduler
                )
                
                upload_results.append(result.data)
                
                if result.data.get("processing_task_id"):
                    processing_tasks.append(result.data["processing_task_id"])
                    
            except Exception as e:
                failed_uploads.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        # Create batch processing task if needed
        batch_task_id = None
        if processing_tasks and len(processing_tasks) > 1:
            try:
                batch_task_id = await _create_batch_processing_task(
                    processing_tasks, current_user.get('user_id'), scheduler
                )
            except Exception as e:
                logger.warning(f"Failed to create batch processing task: {e}")
        
        return APIResponse(
            success=len(failed_uploads) == 0,
            message=f"Batch upload completed - {len(upload_results)} successful, {len(failed_uploads)} failed",
            data={
                "successful_uploads": upload_results,
                "failed_uploads": failed_uploads,
                "batch_size": len(files),
                "success_count": len(upload_results),
                "failure_count": len(failed_uploads),
                "processing_tasks": processing_tasks,
                "batch_processing_task_id": batch_task_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed batch upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed batch upload: {e}"
        )


@router.get("/{file_id}", response_model=APIResponse)
async def get_file_info(
    file_id: str,
    include_processing_results: bool = Query(False, description="Include processing results if available"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get file information and metadata
    
    Args:
        file_id: Unique file identifier
        include_processing_results: Whether to include processing results
        current_user: Current authenticated user
        
    Returns:
        File information and metadata
    """
    try:
        # Get file metadata
        file_metadata = await _get_file_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found"
            )
        
        # Check if user has access to this file
        if (file_metadata.get("uploaded_by") != current_user.get("user_id") and 
            "admin" not in current_user.get("permissions", [])):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this file"
            )
        
        # Get processing results if requested
        processing_results = None
        if include_processing_results and file_metadata.get("processing_task_id"):
            try:
                from ..dependencies.components import get_scheduler
                scheduler = await get_scheduler()
                task_result = scheduler.get_task_result(
                    file_metadata["processing_task_id"], 
                    timeout=0.1
                )
                if task_result:
                    processing_results = {
                        "task_id": file_metadata["processing_task_id"],
                        "status": task_result.status.value,
                        "result": task_result.result,
                        "execution_time": task_result.execution_time
                    }
            except Exception as e:
                logger.warning(f"Failed to get processing results: {e}")
        
        return APIResponse(
            success=True,
            message=f"File information for {file_id}",
            data={
                **file_metadata,
                "processing_results": processing_results,
                "download_url": f"/api/v1/files/{file_id}/download",
                "delete_url": f"/api/v1/files/{file_id}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file info {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file info: {e}"
        )


@router.get("/{file_id}/download")
async def download_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Download a file by ID
    
    Args:
        file_id: Unique file identifier
        current_user: Current authenticated user
        
    Returns:
        File download response
    """
    try:
        # Get file metadata
        file_metadata = await _get_file_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found"
            )
        
        # Check access permissions
        if (file_metadata.get("uploaded_by") != current_user.get("user_id") and 
            "admin" not in current_user.get("permissions", [])):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this file"
            )
        
        # Check if file exists on disk
        file_path = Path(file_metadata["file_path"])
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="File data not found on storage"
            )
        
        # Return file
        return FileResponse(
            path=str(file_path),
            filename=file_metadata.get("original_filename", file_metadata["safe_filename"]),
            media_type=file_metadata.get("content_type", "application/octet-stream")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file: {e}"
        )


@router.delete("/{file_id}", response_model=APIResponse)
async def delete_file(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a file and its metadata
    
    Args:
        file_id: Unique file identifier
        current_user: Current authenticated user
        
    Returns:
        Deletion result
    """
    try:
        # Get file metadata
        file_metadata = await _get_file_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found"
            )
        
        # Check permissions
        if (file_metadata.get("uploaded_by") != current_user.get("user_id") and 
            "admin" not in current_user.get("permissions", [])):
            raise HTTPException(
                status_code=403,
                detail="Permission denied to delete this file"
            )
        
        # Delete file from storage
        file_path = Path(file_metadata["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Delete metadata
        await _delete_file_metadata(file_id)
        
        return APIResponse(
            success=True,
            message=f"File {file_id} deleted successfully",
            data={
                "file_id": file_id,
                "deleted_by": current_user.get("user_id"),
                "deletion_timestamp": time.time()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {e}"
        )


@router.get("/", response_model=APIResponse)
async def list_files(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of files"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    include_processing_status: bool = Query(False, description="Include processing status"),
    current_user: dict = Depends(get_current_user)
):
    """
    List files with filtering and pagination
    
    Args:
        user_id: Filter by user ID (admin only for other users)
        content_type: Filter by MIME content type
        tags: Filter by tags
        limit: Maximum number of results
        offset: Number of results to skip
        include_processing_status: Include processing status
        current_user: Current authenticated user
        
    Returns:
        Paginated list of files
    """
    try:
        # Check permissions for user filter
        if user_id and user_id != current_user.get("user_id"):
            if "admin" not in current_user.get("permissions", []):
                raise HTTPException(
                    status_code=403,
                    detail="Admin permissions required to view other users' files"
                )
        
        # If not admin and no user_id specified, filter by current user
        if not user_id and "admin" not in current_user.get("permissions", []):
            user_id = current_user.get("user_id")
        
        # Get file list (in production, this would query a database)
        all_files = await _list_files_from_storage(
            user_id=user_id,
            content_type=content_type,
            tags=tags.split(',') if tags else None,
            limit=limit,
            offset=offset
        )
        
        # Add processing status if requested
        if include_processing_status:
            try:
                from ..dependencies.components import get_scheduler
                scheduler = await get_scheduler()
                
                for file_info in all_files["files"]:
                    if file_info.get("processing_task_id"):
                        task_result = scheduler.get_task_result(
                            file_info["processing_task_id"], 
                            timeout=0.1
                        )
                        if task_result:
                            file_info["processing_status"] = task_result.status.value
                        else:
                            file_info["processing_status"] = "pending"
                            
            except Exception as e:
                logger.warning(f"Failed to get processing status: {e}")
        
        return APIResponse(
            success=True,
            message=f"Found {len(all_files['files'])} files",
            data=all_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {e}"
        )


@router.post("/{file_id}/process", response_model=APIResponse)
async def process_file(
    file_id: str,
    model_name: str = Form(..., description="Model to use for processing"),
    method: str = Form(default="predict", description="Method to call"),
    parameters: Optional[str] = Form(None, description="Additional parameters as JSON"),
    priority: str = Form(default="normal", regex="^(low|normal|high|critical)$"),
    current_user: dict = Depends(verify_permissions(["write"])),
    scheduler = Depends(get_scheduler)
):
    """
    Process an uploaded file with a specified model
    
    Args:
        file_id: File identifier to process
        model_name: Model to use for processing
        method: Method to call on the model
        parameters: Additional parameters as JSON string
        priority: Processing priority
        current_user: Current authenticated user
        scheduler: Task scheduler dependency
        
    Returns:
        Processing task information
    """
    try:
        # Get file metadata
        file_metadata = await _get_file_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found"
            )
        
        # Check access permissions
        if (file_metadata.get("uploaded_by") != current_user.get("user_id") and 
            "admin" not in current_user.get("permissions", [])):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this file"
            )
        
        # Parse additional parameters
        extra_params = {}
        if parameters:
            try:
                import json
                extra_params = json.loads(parameters)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON in parameters"
                )
        
        # Prepare task inputs
        file_path = file_metadata["file_path"]
        task_inputs = {
            "file_path": file_path,
            **extra_params
        }
        
        # Convert priority
        priority_map = {
            'low': 'LOW',
            'normal': 'NORMAL',
            'high': 'HIGH',
            'critical': 'CRITICAL'
        }
        
        # Submit processing task
        task_id = scheduler.submit_task(
            model_name=model_name,
            method=method,
            kwargs=task_inputs,
            priority=getattr(__import__('src.cv_platform.core', fromlist=['TaskPriority']).TaskPriority, priority_map[priority]),
            metadata={
                "file_id": file_id,
                "user_id": current_user.get("user_id"),
                "processing_requested": True,
                "original_filename": file_metadata.get("original_filename")
            }
        )
        
        # Update file metadata with processing task
        file_metadata["processing_task_id"] = task_id
        file_metadata["status"] = "processing"
        await _store_file_metadata(file_id, file_metadata)
        
        return APIResponse(
            success=True,
            message=f"Processing task submitted for file {file_id}",
            data={
                "file_id": file_id,
                "task_id": task_id,
                "model_name": model_name,
                "method": method,
                "priority": priority,
                "status_url": f"/api/v1/tasks/{task_id}",
                "estimated_completion": time.time() + 30  # Rough estimate
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process file {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {e}"
        )


@router.post("/upload/archive", response_model=APIResponse)
async def upload_and_extract_archive(
    archive: UploadFile = File(..., description="Archive file (ZIP)"),
    auto_process: bool = Form(False, description="Auto-process extracted files"),
    model_name: Optional[str] = Form(None, description="Model for processing"),
    current_user: dict = Depends(verify_permissions(["write"])),
    scheduler = Depends(get_scheduler)
):
    """
    Upload and extract an archive file, optionally processing contents
    
    Args:
        archive: Archive file to upload and extract
        auto_process: Whether to auto-process extracted files
        model_name: Model to use for processing
        current_user: Current authenticated user
        scheduler: Task scheduler dependency
        
    Returns:
        Extraction and processing results
    """
    try:
        # Validate file type
        if archive.content_type not in ALLOWED_ARCHIVE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported archive type: {archive.content_type}"
            )
        
        # Read archive content
        content = await archive.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Archive too large"
            )
        
        # Create temporary extraction directory
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / archive.filename
            
            # Save archive temporarily
            with open(archive_path, 'wb') as f:
                f.write(content)
            
            # Extract archive
            extracted_files = []
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Validate archive contents
                file_list = zip_ref.namelist()
                if len(file_list) > 100:  # Limit extracted files
                    raise HTTPException(
                        status_code=400,
                        detail="Archive contains too many files (max 100)"
                    )
                
                # Extract and process each file
                for file_name in file_list:
                    if file_name.endswith('/'):  # Skip directories
                        continue
                    
                    try:
                        # Extract file
                        extracted_data = zip_ref.read(file_name)
                        
                        # Create UploadFile-like object
                        file_obj = _create_upload_file_from_data(
                            extracted_data, 
                            file_name
                        )
                        
                        # Upload extracted file
                        upload_result = await upload_file(
                            file=file_obj,
                            description=f"Extracted from {archive.filename}",
                            tags=f"archive_extract,{archive.filename}",
                            auto_process=auto_process,
                            model_name=model_name,
                            current_user=current_user,
                            background_tasks=None,
                            scheduler=scheduler
                        )
                        
                        extracted_files.append(upload_result.data)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {file_name} from archive: {e}")
                        extracted_files.append({
                            "filename": file_name,
                            "error": str(e),
                            "status": "failed"
                        })
        
        successful_extractions = [f for f in extracted_files if "error" not in f]
        failed_extractions = [f for f in extracted_files if "error" in f]
        
        return APIResponse(
            success=len(failed_extractions) == 0,
            message=f"Archive processed - {len(successful_extractions)} files extracted successfully",
            data={
                "archive_filename": archive.filename,
                "total_files": len(file_list),
                "successful_extractions": successful_extractions,
                "failed_extractions": failed_extractions,
                "success_count": len(successful_extractions),
                "failure_count": len(failed_extractions)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process archive: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process archive: {e}"
        )


# =============================================================================
# Utility Functions
# =============================================================================

async def _analyze_image(file_path: Path) -> Dict[str, Any]:
    """Analyze uploaded image and extract metadata"""
    try:
        with Image.open(file_path) as img:
            return {
                "image_width": img.width,
                "image_height": img.height,
                "image_mode": img.mode,
                "image_format": img.format,
                "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
    except Exception as e:
        logger.warning(f"Failed to analyze image: {e}")
        return {}


def _infer_model_for_file(content_type: str, image_info: Optional[Dict]) -> Optional[str]:
    """Infer appropriate model for file type"""
    if content_type in ALLOWED_IMAGE_TYPES:
        # For images, default to a detection model
        return "yolov8n"  # Default detection model
    return None


async def _store_file_metadata(file_id: str, metadata: Dict[str, Any]):
    """Store file metadata (placeholder - use database in production)"""
    # In production, store in database
    metadata_file = UPLOAD_DIR / f"{file_id}.metadata.json"
    import json
    async with aiofiles.open(metadata_file, 'w') as f:
        await f.write(json.dumps(metadata, default=str))


async def _get_file_metadata(file_id: str) -> Optional[Dict[str, Any]]:
    """Get file metadata (placeholder - use database in production)"""
    metadata_file = UPLOAD_DIR / f"{file_id}.metadata.json"
    if not metadata_file.exists():
        return None
    
    import json
    async with aiofiles.open(metadata_file, 'r') as f:
        content = await f.read()
        return json.loads(content)


async def _delete_file_metadata(file_id: str):
    """Delete file metadata"""
    metadata_file = UPLOAD_DIR / f"{file_id}.metadata.json"
    if metadata_file.exists():
        metadata_file.unlink()


async def _list_files_from_storage(
    user_id: Optional[str] = None,
    content_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """List files from storage with filtering"""
    # In production, this would be a database query
    all_files = []
    
    for metadata_file in UPLOAD_DIR.glob("*.metadata.json"):
        try:
            file_metadata = await _get_file_metadata(metadata_file.stem)
            if file_metadata:
                # Apply filters
                if user_id and file_metadata.get("uploaded_by") != user_id:
                    continue
                if content_type and file_metadata.get("content_type") != content_type:
                    continue
                if tags:
                    file_tags = file_metadata.get("tags", [])
                    if not any(tag in file_tags for tag in tags):
                        continue
                
                all_files.append(file_metadata)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {metadata_file}: {e}")
    
    # Sort by upload time (newest first)
    all_files.sort(key=lambda x: x.get("upload_timestamp", 0), reverse=True)
    
    # Apply pagination
    paginated_files = all_files[offset:offset + limit]
    
    return {
        "files": paginated_files,
        "total_count": len(all_files),
        "returned_count": len(paginated_files),
        "offset": offset,
        "limit": limit
    }


async def _create_batch_processing_task(task_ids: List[str], user_id: str, scheduler) -> str:
    """Create a batch processing coordination task"""
    # This would coordinate multiple processing tasks
    # For now, just return a placeholder
    batch_id = str(uuid.uuid4())
    logger.info(f"Created batch processing task {batch_id} for {len(task_ids)} tasks")
    return batch_id


def _create_upload_file_from_data(data: bytes, filename: str) -> UploadFile:
    """Create UploadFile object from binary data"""
    from fastapi import UploadFile
    import io
    
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    
    # Create a file-like object
    file_obj = io.BytesIO(data)
    file_obj.name = filename
    
    # Create UploadFile instance
    upload_file = UploadFile(
        filename=filename,
        file=file_obj,
        size=len(data),
        headers={"content-type": content_type}
    )
    
    return upload_file