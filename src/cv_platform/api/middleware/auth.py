"""
Authentication Middleware

Provides request-level authentication and security headers.
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authentication and security headers
    """
    
    def __init__(self, app, rate_limit_requests: int = 100, rate_limit_window: int = 60):
        """
        Initialize authentication middleware
        
        Args:
            app: FastAPI application
            rate_limit_requests: Maximum requests per window
            rate_limit_window: Rate limit window in seconds
        """
        super().__init__(app)
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.request_history = {}  # IP -> list of timestamps
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with authentication and rate limiting
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response with security headers
        """
        # Extract client IP
        client_ip = self._get_client_ip(request)
        
        # Apply rate limiting
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "Rate limit exceeded",
                    "error": "Too many requests",
                    "timestamp": time.time()
                },
                headers={
                    "X-Rate-Limit-Limit": str(self.rate_limit_requests),
                    "X-Rate-Limit-Window": str(self.rate_limit_window),
                    "Retry-After": str(self.rate_limit_window)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        # Add rate limit headers
        remaining_requests = self._get_remaining_requests(client_ip)
        response.headers["X-Rate-Limit-Limit"] = str(self.rate_limit_requests)
        response.headers["X-Rate-Limit-Remaining"] = str(remaining_requests)
        response.headers["X-Rate-Limit-Reset"] = str(int(time.time() + self.rate_limit_window))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client is within rate limits
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if within limits, False otherwise
        """
        if client_ip == "unknown":
            return True  # Don't rate limit unknown IPs
        
        current_time = time.time()
        window_start = current_time - self.rate_limit_window
        
        # Initialize or clean up request history
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        
        # Remove old requests outside the window
        self.request_history[client_ip] = [
            timestamp for timestamp in self.request_history[client_ip]
            if timestamp > window_start
        ]
        
        # Check if within limit
        if len(self.request_history[client_ip]) >= self.rate_limit_requests:
            return False
        
        # Add current request
        self.request_history[client_ip].append(current_time)
        return True
    
    def _get_remaining_requests(self, client_ip: str) -> int:
        """
        Get remaining requests for client
        
        Args:
            client_ip: Client IP address
            
        Returns:
            Number of remaining requests
        """
        if client_ip == "unknown" or client_ip not in self.request_history:
            return self.rate_limit_requests
        
        used_requests = len(self.request_history[client_ip])
        return max(0, self.rate_limit_requests - used_requests)
    
    def _add_security_headers(self, response: Response):
        """
        Add security headers to response
        
        Args:
            response: HTTP response to modify
        """
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # API-specific headers
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Powered-By"] = "CV-Model-Platform"
        
        # Cache control for API responses
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        else:
            response.headers["Cache-Control"] = "no-cache"