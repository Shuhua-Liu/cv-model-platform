"""
Authentication Dependencies

Handles user authentication, authorization, and permission checking.
"""

from typing import List, Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Get current authenticated user
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        User information dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    # Allow unauthenticated access for demo purposes
    if not credentials:
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "permissions": ["read"],
            "is_authenticated": False
        }
    
    token = credentials.credentials
    
    try:
        # Simple token validation (replace with your auth system)
        if token == "demo-admin-token":
            return {
                "user_id": "admin",
                "username": "admin",
                "permissions": ["read", "write", "admin"],
                "is_authenticated": True
            }
        elif token == "demo-user-token":
            return {
                "user_id": "demo_user",
                "username": "demo_user", 
                "permissions": ["read", "write"],
                "is_authenticated": True
            }
        elif token.startswith("readonly-"):
            user_id = token[9:]  # Remove "readonly-" prefix
            return {
                "user_id": user_id,
                "username": user_id,
                "permissions": ["read"],
                "is_authenticated": True
            }
        else:
            # Try to validate JWT token here in production
            # For now, reject invalid tokens
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_admin_user(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Ensure current user has admin permissions
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    return current_user


def verify_permissions(required_permissions: List[str]):
    """
    Create a dependency to verify user has required permissions
    
    Args:
        required_permissions: List of required permission strings
        
    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: dict = Depends(get_current_user)
    ) -> dict:
        user_permissions = current_user.get("permissions", [])
        
        missing_permissions = [
            perm for perm in required_permissions 
            if perm not in user_permissions
        ]
        
        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {missing_permissions}"
            )
        
        return current_user
    
    return permission_checker