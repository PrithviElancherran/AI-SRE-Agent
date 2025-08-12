"""
User model with authentication, roles, and permissions.

This module defines the User model and related authentication/authorization
components for the AI SRE Agent system.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Boolean, Column, DateTime, String, Text, JSON
from sqlalchemy import Integer
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.sql import func

from config.database import Base


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    
    SRE_LEAD = "sre_lead"
    SENIOR_SRE = "senior_sre"
    SRE_ENGINEER = "sre_engineer"
    JUNIOR_SRE = "junior_sre"
    MANAGER = "manager"
    READONLY = "readonly"


class Permission(str, Enum):
    """System permissions."""
    
    # Incident permissions
    READ_INCIDENTS = "read_incidents"
    CREATE_INCIDENTS = "create_incidents"
    UPDATE_INCIDENTS = "update_incidents"
    DELETE_INCIDENTS = "delete_incidents"
    
    # Playbook permissions
    READ_PLAYBOOKS = "read_playbooks"
    CREATE_PLAYBOOKS = "create_playbooks"
    UPDATE_PLAYBOOKS = "update_playbooks"
    DELETE_PLAYBOOKS = "delete_playbooks"
    EXECUTE_PLAYBOOKS = "execute_playbooks"
    
    # Analysis permissions
    READ_ANALYSIS = "read_analysis"
    CREATE_ANALYSIS = "create_analysis"
    
    # Action permissions
    APPROVE_ACTIONS = "approve_actions"
    EXECUTE_ACTIONS = "execute_actions"
    
    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_SETTINGS = "manage_settings"
    
    # System permissions
    VIEW_METRICS = "view_metrics"
    ACCESS_LOGS = "access_logs"


# Role-based permission mapping
ROLE_PERMISSIONS = {
    UserRole.SRE_LEAD: [
        Permission.READ_INCIDENTS,
        Permission.CREATE_INCIDENTS,
        Permission.UPDATE_INCIDENTS,
        Permission.READ_PLAYBOOKS,
        Permission.CREATE_PLAYBOOKS,
        Permission.UPDATE_PLAYBOOKS,
        Permission.EXECUTE_PLAYBOOKS,
        Permission.READ_ANALYSIS,
        Permission.CREATE_ANALYSIS,
        Permission.APPROVE_ACTIONS,
        Permission.EXECUTE_ACTIONS,
        Permission.VIEW_METRICS,
        Permission.ACCESS_LOGS,
    ],
    UserRole.SENIOR_SRE: [
        Permission.READ_INCIDENTS,
        Permission.CREATE_INCIDENTS,
        Permission.UPDATE_INCIDENTS,
        Permission.READ_PLAYBOOKS,
        Permission.UPDATE_PLAYBOOKS,
        Permission.EXECUTE_PLAYBOOKS,
        Permission.READ_ANALYSIS,
        Permission.CREATE_ANALYSIS,
        Permission.APPROVE_ACTIONS,
        Permission.EXECUTE_ACTIONS,
        Permission.VIEW_METRICS,
        Permission.ACCESS_LOGS,
    ],
    UserRole.SRE_ENGINEER: [
        Permission.READ_INCIDENTS,
        Permission.CREATE_INCIDENTS,
        Permission.UPDATE_INCIDENTS,
        Permission.READ_PLAYBOOKS,
        Permission.CREATE_PLAYBOOKS,
        Permission.UPDATE_PLAYBOOKS,
        Permission.EXECUTE_PLAYBOOKS,
        Permission.READ_ANALYSIS,
        Permission.CREATE_ANALYSIS,
        Permission.VIEW_METRICS,
        Permission.ACCESS_LOGS,
    ],
    UserRole.JUNIOR_SRE: [
        Permission.READ_INCIDENTS,
        Permission.CREATE_INCIDENTS,
        Permission.READ_PLAYBOOKS,
        Permission.EXECUTE_PLAYBOOKS,
        Permission.READ_ANALYSIS,
        Permission.VIEW_METRICS,
    ],
    UserRole.MANAGER: [
        Permission.READ_INCIDENTS,
        Permission.READ_PLAYBOOKS,
        Permission.READ_ANALYSIS,
        Permission.VIEW_METRICS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SETTINGS,
    ],
    UserRole.READONLY: [
        Permission.READ_INCIDENTS,
        Permission.READ_PLAYBOOKS,
        Permission.READ_ANALYSIS,
        Permission.VIEW_METRICS,
    ],
}


class UserTable(Base):
    """User database table."""
    
    __tablename__ = "users"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(50), nullable=False)
    permissions = Column(JSON, nullable=False, default=list)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Authentication fields
    hashed_password = Column(String(255), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Profile fields
    full_name = Column(String(255), nullable=True)
    timezone = Column(String(50), default="UTC")
    notification_preferences = Column(JSON, default=dict)


class UserBase(BaseModel):
    """Base user model."""
    
    email: EmailStr
    role: UserRole
    permissions: List[Permission] = Field(default_factory=list)
    is_active: bool = True
    full_name: Optional[str] = None
    timezone: str = "UTC"
    notification_preferences: dict = Field(default_factory=dict)


class UserCreate(UserBase):
    """User creation model."""
    
    password: Optional[str] = None
    user_id: Optional[str] = None
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to set default permissions based on role."""
        if not self.permissions:
            self.permissions = ROLE_PERMISSIONS.get(self.role, [])
        
        if not self.user_id:
            self.user_id = str(uuid4())


class UserUpdate(BaseModel):
    """User update model."""
    
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    permissions: Optional[List[Permission]] = None
    is_active: Optional[bool] = None
    full_name: Optional[str] = None
    timezone: Optional[str] = None
    notification_preferences: Optional[dict] = None


class User(UserBase):
    """User model with database fields."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    user_id: str
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(perm in self.permissions for perm in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        return all(perm in self.permissions for perm in permissions)
    
    def can_approve_actions(self) -> bool:
        """Check if user can approve automated actions."""
        return self.has_permission(Permission.APPROVE_ACTIONS)
    
    def can_execute_playbooks(self) -> bool:
        """Check if user can execute playbooks."""
        return self.has_permission(Permission.EXECUTE_PLAYBOOKS)
    
    def can_manage_incidents(self) -> bool:
        """Check if user can manage incidents."""
        return self.has_any_permission([
            Permission.CREATE_INCIDENTS,
            Permission.UPDATE_INCIDENTS,
            Permission.DELETE_INCIDENTS
        ])
    
    def can_create_playbooks(self) -> bool:
        """Check if user can create playbooks."""
        return self.has_permission(Permission.CREATE_PLAYBOOKS)
    
    def can_modify_playbooks(self) -> bool:
        """Check if user can modify playbooks."""
        return self.has_permission(Permission.UPDATE_PLAYBOOKS)
    
    def can_modify_incidents(self) -> bool:
        """Check if user can modify incidents."""
        return self.has_permission(Permission.UPDATE_INCIDENTS)
    
    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return self.role in [UserRole.SRE_LEAD, UserRole.MANAGER]
    
    def is_sre_role(self) -> bool:
        """Check if user has an SRE role."""
        return self.role in [
            UserRole.SRE_LEAD,
            UserRole.SENIOR_SRE,
            UserRole.SRE_ENGINEER,
            UserRole.JUNIOR_SRE
        ]
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    def get_effective_permissions(self) -> List[Permission]:
        """Get effective permissions (role + custom permissions)."""
        role_perms = ROLE_PERMISSIONS.get(self.role, [])
        custom_perms = self.permissions or []
        return list(set(role_perms + custom_perms))


class UserLogin(BaseModel):
    """User login model."""
    
    email: EmailStr
    password: str


class UserProfile(BaseModel):
    """User profile model for responses."""
    
    model_config = {"from_attributes": True}
    
    user_id: str
    email: EmailStr
    role: UserRole
    permissions: List[Permission]
    is_active: bool
    full_name: Optional[str] = None
    timezone: str = "UTC"
    last_login: Optional[datetime] = None
    created_at: datetime


class Token(BaseModel):
    """JWT token model."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserProfile


class TokenData(BaseModel):
    """Token data for JWT payload."""
    
    user_id: str
    email: str
    role: str
    permissions: List[str]
    exp: int


class ApiKey(BaseModel):
    """API key model for service accounts."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    key_id: str
    name: str
    description: Optional[str] = None
    permissions: List[Permission]
    is_active: bool = True
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


class ApiKeyCreate(BaseModel):
    """API key creation model."""
    
    name: str
    description: Optional[str] = None
    permissions: List[Permission]
    expires_at: Optional[datetime] = None


class AuditLog(BaseModel):
    """Audit log model for user actions."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    details: dict = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime


class ApprovalRequest(BaseModel):
    """Approval request model for human-in-the-loop workflows."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    incident_id: str
    requested_by: str
    action_type: str
    action_details: dict
    justification: str
    confidence_score: float
    status: str = "pending"  # pending, approved, rejected, expired
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    expires_at: datetime
    created_at: datetime
    
    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        return datetime.utcnow() > self.expires_at
    
    def can_approve(self, user: User) -> bool:
        """Check if user can approve this request."""
        return user.can_approve_actions() and user.user_id != self.requested_by


class ApprovalRequestCreate(BaseModel):
    """Approval request creation model."""
    
    incident_id: str
    action_type: str
    action_details: dict
    justification: str
    confidence_score: float
    expires_in_minutes: int = 60


class ApprovalResponse(BaseModel):
    """Approval response model."""
    
    status: str  # approved, rejected
    comment: Optional[str] = None


class UserActivity(BaseModel):
    """User activity model for analytics."""
    
    model_config = {"from_attributes": True}
    
    user_id: str
    activity_type: str
    activity_count: int
    last_activity: datetime
    date: datetime
