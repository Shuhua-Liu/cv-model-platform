"""
Request Logging Middleware

Provides comprehensive request/response logging with performance metrics.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses with metrics
    """
    
    def __init__(self, app, log_bodies: bool = False, max_body_size: int = 1024):
        """
        Initialize logging middleware
        
        Args:
            app: FastAPI application
            log_bodies: Whether to log request/response bodies
            max_body_size: Maximum body size to log (in bytes)
        """
        super().__init__(app)
        self.log_bodies = log_bodies
        self.max_body_size = max_body_size
        self.request_count = 0
        self.error_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and response with logging
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response with added headers
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        logger.info(
            f"[{request_id}] {request.method} {request.url} - "
            f"Client: {client_ip} - Agent: {user_agent}"
        )
        
        # Log request body if enabled
        if self.log_bodies and request.method in ["POST", "PUT", "PATCH"]:
            await self._log_request_body(request, request_id)
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            self.request_count += 1
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            response.headers["X-Request-Count"] = str(self.request_count)
            
            # Log response
            logger.info(
                f"[{request_id}] Response: {response.status_code} - "
                f"Time: {processing_time:.3f}s - Size: {self._get_response_size(response)}"
            )
            
            # Log response body if enabled and status indicates error
            if self.log_bodies and response.status_code >= 400:
                await self._log_response_body(response, request_id)
            
            return response
            
        except Exception as e:
            # Handle errors
            processing_time = time.time() - start_time
            self.error_count += 1
            
            logger.error(
                f"[{request_id}] Error: {str(e)} - Time: {processing_time:.3f}s"
            )
            
            # Return structured error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "error": str(e),
                    "request_id": request_id,
                    "timestamp": time.time()
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Processing-Time": f"{processing_time:.3f}"
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check forwarded headers first (for proxy/load balancer setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_response_size(self, response: Response) -> str:
        """Get response size information"""
        content_length = response.headers.get("content-length")
        if content_length:
            size_bytes = int(content_length)
            if size_bytes < 1024:
                return f"{size_bytes}B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f}KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f}MB"
        return "unknown"
    
    async def _log_request_body(self, request: Request, request_id: str):
        """Log request body if enabled"""
        try:
            body = await request.body()
            if len(body) <= self.max_body_size:
                # Try to decode as text
                try:
                    body_text = body.decode('utf-8')
                    logger.debug(f"[{request_id}] Request body: {body_text}")
                except UnicodeDecodeError:
                    logger.debug(f"[{request_id}] Request body: <binary data, {len(body)} bytes>")
            else:
                logger.debug(f"[{request_id}] Request body: <large body, {len(body)} bytes>")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to log request body: {e}")
    
    async def _log_response_body(self, response: Response, request_id: str):
        """Log response body if enabled"""
        try:
            if hasattr(response, 'body') and len(response.body) <= self.max_body_size:
                try:
                    body_text = response.body.decode('utf-8')
                    logger.debug(f"[{request_id}] Response body: {body_text}")
                except (UnicodeDecodeError, AttributeError):
                    logger.debug(f"[{request_id}] Response body: <non-text data>")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to log response body: {e}")