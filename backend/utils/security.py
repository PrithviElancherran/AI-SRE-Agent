"""
Security utility functions for authentication, authorization, and permission checking.

This module provides authentication and authorization capabilities for the AI SRE Agent
backend with mock implementations suitable for demo purposes.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Any
from functools import wraps
from uuid import uuid4

from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from models.user import User, UserRole, Permission, ROLE_PERMISSIONS

logger = logging.getLogger(__name__)

# Security scheme for bearer token authentication
security_scheme = HTTPBearer(auto_error=False)

class TokenData(BaseModel):
    """Token data model for authentication."""
    user_id: str
    email: str
    role: UserRole
    permissions: List[Permission]
    expires_at: datetime

class SecurityConfig:
    """Security configuration for demo purposes."""
    
    # Demo mode settings
    DEMO_MODE = True
    REQUIRE_AUTHENTICATION = False
    DEFAULT_DEMO_USER_ROLE = UserRole.SRE_ENGINEER
    
    # Token settings
    TOKEN_EXPIRE_HOURS = 24
    SECRET_KEY = "demo-secret-key-for-ai-sre-agent"
    ALGORITHM = "HS256"

# Mock user database for demo
DEMO_USERS = {
    "demo_user": User(
        id=str(uuid4()),
        user_id="demo_user",
        email="demo@aisre.com",
        role=UserRole.SRE_ENGINEER,
        permissions=ROLE_PERMISSIONS[UserRole.SRE_ENGINEER],
        is_active=True,
        full_name="Demo SRE Engineer",
        timezone="UTC",
        notification_preferences={},
        created_at=datetime.utcnow().isoformat()
    ),
    "admin_user": User(
        id=str(uuid4()),
        user_id="admin_user",
        email="admin@aisre.com",
        role=UserRole.SRE_LEAD,
        permissions=ROLE_PERMISSIONS[UserRole.SRE_LEAD],
        is_active=True,
        full_name="Demo SRE Lead",
        timezone="UTC",
        notification_preferences={},
        created_at=datetime.utcnow().isoformat()
    ),
    "readonly_user": User(
        id=str(uuid4()),
        user_id="readonly_user",
        email="readonly@aisre.com",
        role=UserRole.READONLY,
        permissions=ROLE_PERMISSIONS[UserRole.READONLY],
        is_active=True,
        full_name="Demo Read-only User",
        timezone="UTC",
        notification_preferences={},
        created_at=datetime.utcnow().isoformat()
    )
}

def create_demo_token(user: User) -> str:
    """
    Create a demo token for authentication.
    
    Args:
        user: User object
        
    Returns:
        Demo token string
    """
    token_data = TokenData(
        user_id=user.user_id,
        email=user.email,
        role=user.role,
        permissions=user.permissions,
        expires_at=datetime.utcnow() + timedelta(hours=SecurityConfig.TOKEN_EXPIRE_HOURS)
    )
    
    # In demo mode, just return a simple token format
    return f"demo_token_{token_data.user_id}_{token_data.role.value}"

def verify_demo_token(token: str) -> Optional[TokenData]:
    """
    Verify demo token and return token data.
    
    Args:
        token: Token string to verify
        
    Returns:
        TokenData if valid, None if invalid
    """
    if not token or not token.startswith("demo_token_"):
        return None
    
    try:
        # Parse demo token format: demo_token_{user_id}_{role}
        parts = token.split("_")
        if len(parts) != 4:
            return None
        
        user_id = parts[2]
        role_str = parts[3]
        
        # Find user in demo database
        user = DEMO_USERS.get(user_id)
        if not user:
            # Create default demo user
            user = DEMO_USERS["demo_user"]
        
        return TokenData(
            user_id=user_id,
            email=user.email,
            role=user.role,
            permissions=user.permissions,
            expires_at=datetime.utcnow() + timedelta(hours=SecurityConfig.TOKEN_EXPIRE_HOURS)
        )
        
    except Exception as e:
        logger.warning(f"Error verifying demo token: {e}")
        return None

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> User:
    """
    Get current authenticated user.
    
    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If authentication fails
    """
    if SecurityConfig.DEMO_MODE and not SecurityConfig.REQUIRE_AUTHENTICATION:
        # In demo mode without authentication, return default demo user
        return DEMO_USERS["demo_user"]
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Verify token
    token_data = verify_demo_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Check token expiration
    if token_data.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Return user from token data
    user = DEMO_USERS.get(token_data.user_id)
    if not user:
        user = User(
            id=f"user_{str(uuid4())[:8]}",
            user_id=token_data.user_id,
            email=token_data.email,
            role=token_data.role,
            permissions=token_data.permissions,
            is_active=True,
            full_name=f"User {token_data.user_id}",
            timezone="UTC",
            notification_preferences={},
            created_at=datetime.utcnow().isoformat()
        )
    
    # Log successful authentication
    logger.info(f"User authenticated: {user.email} with role {user.role}")
    
    return user

def require_permissions(*required_permissions: Permission):
    """
    Decorator to require specific permissions for endpoint access.
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current user from kwargs or dependency injection
            current_user = None
            
            # Look for current_user in function parameters
            for key, value in kwargs.items():
                if isinstance(value, User):
                    current_user = value
                    break
            
            # If not found in kwargs, try to get from dependencies
            if not current_user:
                # In demo mode, use default user
                if SecurityConfig.DEMO_MODE:
                    current_user = DEMO_USERS["demo_user"]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
            
            # Check permissions
            if not check_permissions(current_user, list(required_permissions)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {[p.value for p in required_permissions]}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def check_permissions(user: User, required_permissions: List[Permission]) -> bool:
    """
    Check if user has required permissions.
    
    Args:
        user: User object to check
        required_permissions: List of required permissions
        
    Returns:
        True if user has all required permissions
    """
    if not user.is_active:
        return False
    
    # In demo mode, SRE_LEAD has all permissions
    if SecurityConfig.DEMO_MODE and user.role == UserRole.SRE_LEAD:
        return True
    
    # Check if user has all required permissions
    user_permissions = set(user.permissions)
    required_permissions_set = set(required_permissions)
    
    return required_permissions_set.issubset(user_permissions)

def get_user_by_email(email: str) -> Optional[User]:
    """
    Get user by email address.
    
    Args:
        email: User email address
        
    Returns:
        User object if found, None otherwise
    """
    for user in DEMO_USERS.values():
        if user.email == email:
            return user
    return None

def get_user_by_id(user_id: str) -> Optional[User]:
    """
    Get user by user ID.
    
    Args:
        user_id: User ID
        
    Returns:
        User object if found, None otherwise
    """
    return DEMO_USERS.get(user_id)

def create_user_session(user: User) -> dict:
    """
    Create user session data.
    
    Args:
        user: User object
        
    Returns:
        Session data dictionary
    """
    token = create_demo_token(user)
    
    return {
        "token": token,
        "user": {
            "id": user.id,
            "user_id": user.user_id,
            "email": user.email,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "full_name": user.full_name,
            "timezone": user.timezone,
            "is_active": user.is_active
        },
        "expires_at": (datetime.utcnow() + timedelta(hours=SecurityConfig.TOKEN_EXPIRE_HOURS)).isoformat()
    }

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key for service-to-service communication.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    # In demo mode, accept demo API keys
    valid_demo_keys = [
        "demo_api_key_monitoring",
        "demo_api_key_logging",
        "demo_api_key_error_reporting",
        "demo_api_key_tracing"
    ]
    
    return api_key in valid_demo_keys

def get_current_user_permissions(user: User) -> List[str]:
    """
    Get current user's permissions as strings.
    
    Args:
        user: User object
        
    Returns:
        List of permission strings
    """
    return [permission.value for permission in user.permissions]

def is_admin_user(user: User) -> bool:
    """
    Check if user has admin privileges.
    
    Args:
        user: User object
        
    Returns:
        True if user is admin
    """
    admin_roles = [UserRole.SRE_LEAD, UserRole.MANAGER]
    return user.role in admin_roles

def log_security_event(event_type: str, user_email: str, details: dict):
    """
    Log security events for audit purposes.
    
    Args:
        event_type: Type of security event
        user_email: User email involved
        details: Additional event details
    """
    logger.info(
        f"Security event: {event_type} | User: {user_email} | Details: {details}",
        extra={
            "event_type": event_type,
            "user_email": user_email,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Dependency functions for FastAPI endpoints
async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to get current active user.
    
    Args:
        current_user: Current user from authentication
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    return current_user

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to require admin user.
    
    Args:
        current_user: Current user from authentication
        
    Returns:
        Admin user object
        
    Raises:
        HTTPException: If user is not admin
    """
    if not is_admin_user(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Initialize demo security setup
def init_demo_security():
    """Initialize demo security configuration."""
    logger.info("Initializing demo security configuration")
    logger.info(f"Demo mode: {SecurityConfig.DEMO_MODE}")
    logger.info(f"Authentication required: {SecurityConfig.REQUIRE_AUTHENTICATION}")
    logger.info(f"Demo users available: {list(DEMO_USERS.keys())}")

# Initialize on module import
init_demo_security()

async def get_current_user_websocket(
    websocket,
    token: Optional[str] = None
) -> Optional[User]:
    """
    Get current authenticated user for WebSocket connections.
    
    Args:
        websocket: WebSocket connection object
        token: Optional authentication token
        
    Returns:
        User object if authenticated, None otherwise
    """
    if SecurityConfig.DEMO_MODE and not SecurityConfig.REQUIRE_AUTHENTICATION:
        # In demo mode without authentication, return default demo user
        return DEMO_USERS["demo_user"]
    
    if not token:
        # Try to get token from query parameters
        try:
            query_params = dict(websocket.query_params)
            token = query_params.get('token')
        except Exception:
            pass
    
    if not token:
        logger.warning("WebSocket connection attempted without authentication token")
        return None
    
    # Verify token
    token_data = verify_demo_token(token)
    if not token_data:
        logger.warning(f"WebSocket connection with invalid token: {token[:10]}...")
        return None
    
    # Check token expiration
    if token_data.expires_at < datetime.utcnow():
        logger.warning(f"WebSocket connection with expired token for user: {token_data.email}")
        return None
    
    # Return user from token data
    user = DEMO_USERS.get(token_data.user_id)
    if not user:
        user = User(
            id=str(uuid4()),
            user_id=token_data.user_id,
            email=token_data.email,
            role=token_data.role,
            permissions=token_data.permissions,
            is_active=True,
            full_name=f"User {token_data.user_id}",
            timezone="UTC",
            notification_preferences={},
            created_at=datetime.utcnow().isoformat()
        )
    
    # Log successful WebSocket authentication
    logger.info(f"WebSocket user authenticated: {user.email} with role {user.role}")
    
    return user
