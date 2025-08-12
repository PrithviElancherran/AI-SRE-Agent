"""
Playbook model with structured troubleshooting workflows, step execution tracking, and effectiveness metrics.

This module defines the Playbook model and related components for managing
systematic troubleshooting workflows in the AI SRE Agent system.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, 
    String, Text, JSON, Index
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from config.database import Base


class PlaybookStatus(str, Enum):
    """Playbook execution status enumeration."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class StepType(str, Enum):
    """Types of playbook steps."""
    
    MONITORING_CHECK = "monitoring_check"
    METRIC_CHECK = "metric_check"
    LOG_ANALYSIS = "log_analysis"
    TRACING_ANALYSIS = "tracing_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    QUERY_ANALYSIS = "query_analysis"
    CHANGE_ANALYSIS = "change_analysis"
    PROCESS_ANALYSIS = "process_analysis"
    MANUAL_ACTION = "manual_action"
    AUTOMATED_ACTION = "automated_action"


class ExecutionStatus(str, Enum):
    """Step execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"


class PlaybookTable(Base):
    """Playbook database table."""
    
    __tablename__ = "playbooks"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    playbook_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default=PlaybookStatus.ACTIVE, index=True)
    effectiveness_score = Column(Float, nullable=False, default=0.0)
    
    # Metadata
    created_by = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    applicable_services = Column(JSON, nullable=False, default=list)
    trigger_conditions = Column(JSON, nullable=False, default=list)
    
    # Statistics
    execution_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    avg_execution_time_minutes = Column(Float, nullable=True)
    last_executed = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    steps = relationship("PlaybookStepTable", back_populates="playbook", cascade="all, delete-orphan", order_by="PlaybookStepTable.order")
    executions = relationship("PlaybookExecutionTable", back_populates="playbook", cascade="all, delete-orphan")
    effectiveness_metrics = relationship("PlaybookEffectivenessTable", back_populates="playbook", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_playbook_status_effectiveness', 'status', 'effectiveness_score'),
        Index('idx_playbook_name_version', 'name', 'version'),
    )


class PlaybookStepTable(Base):
    """Playbook step database table."""
    
    __tablename__ = "playbook_steps"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    playbook_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("playbooks.id"), nullable=False, index=True)
    step_id = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    step_type = Column(String(50), nullable=False)
    order = Column(Integer, nullable=False)
    
    # Step configuration
    query = Column(String(500), nullable=True)
    expected_result = Column(JSON, nullable=True)
    escalation_condition = Column(String(500), nullable=True)
    gcp_integration = Column(JSON, nullable=True)
    timeout_minutes = Column(Integer, nullable=False, default=5)
    retry_count = Column(Integer, nullable=False, default=3)
    
    # Optional fields
    prerequisites = Column(JSON, nullable=True)
    dependencies = Column(JSON, nullable=True)
    approval_required = Column(Boolean, nullable=False, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    playbook = relationship("PlaybookTable", back_populates="steps")
    step_results = relationship("PlaybookStepResultTable", back_populates="step", cascade="all, delete-orphan")
    
    # Unique constraint for step_id within playbook
    __table_args__ = (
        Index('idx_unique_step_per_playbook', 'playbook_id', 'step_id', unique=True),
        Index('idx_step_order', 'playbook_id', 'order'),
    )


class PlaybookExecutionTable(Base):
    """Playbook execution database table."""
    
    __tablename__ = "playbook_executions"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    execution_id = Column(String(50), unique=True, nullable=False, index=True)
    playbook_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("playbooks.id"), nullable=False, index=True)
    incident_id = Column(String(50), nullable=False, index=True)
    
    # Execution details
    status = Column(String(20), nullable=False, default=ExecutionStatus.PENDING)
    started_by = Column(String(50), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_minutes = Column(Float, nullable=True)
    
    # Results
    success = Column(Boolean, nullable=True)
    root_cause_found = Column(Boolean, nullable=False, default=False)
    actions_recommended = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Context
    execution_context = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    playbook = relationship("PlaybookTable", back_populates="executions")
    step_results = relationship("PlaybookStepResultTable", back_populates="execution", cascade="all, delete-orphan")


class PlaybookStepResultTable(Base):
    """Playbook step result database table."""
    
    __tablename__ = "playbook_step_results"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    execution_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("playbook_executions.id"), nullable=False, index=True)
    step_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("playbook_steps.id"), nullable=False, index=True)
    
    # Execution details
    status = Column(String(20), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Results
    success = Column(Boolean, nullable=True)
    result_data = Column(JSON, nullable=True)
    evidence = Column(JSON, nullable=True)
    escalation_triggered = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text, nullable=True)
    
    # Analysis
    threshold_met = Column(Boolean, nullable=True)
    actual_value = Column(Float, nullable=True)
    expected_value = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    execution = relationship("PlaybookExecutionTable", back_populates="step_results")
    step = relationship("PlaybookStepTable", back_populates="step_results")


class PlaybookEffectivenessTable(Base):
    """Playbook effectiveness tracking table."""
    
    __tablename__ = "playbook_effectiveness"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    playbook_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("playbooks.id"), nullable=False, index=True)
    
    # Time period
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Metrics
    execution_count = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=False, default=0.0)
    avg_execution_time = Column(Float, nullable=False, default=0.0)
    root_cause_identification_rate = Column(Float, nullable=False, default=0.0)
    false_positive_rate = Column(Float, nullable=False, default=0.0)
    user_satisfaction_score = Column(Float, nullable=True)
    
    # Calculated effectiveness score
    effectiveness_score = Column(Float, nullable=False, default=0.0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    playbook = relationship("PlaybookTable", back_populates="effectiveness_metrics")


class GCPIntegration(BaseModel):
    """GCP integration configuration for playbook steps."""
    
    service: str  # monitoring, logging, tracing, error_reporting
    metric: Optional[str] = None
    filter: Optional[str] = None
    operation: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    time_range: Optional[str] = None
    aggregation: Optional[str] = None


class ExpectedResult(BaseModel):
    """Expected result configuration for playbook steps."""
    
    threshold: str
    healthy_range: Optional[str] = None
    pattern: Optional[str] = None
    correlation: Optional[str] = None
    action: Optional[str] = None


class PlaybookStepBase(BaseModel):
    """Base playbook step model."""
    
    step_id: str
    description: str
    step_type: StepType
    order: int
    query: Optional[str] = None
    expected_result: Optional[ExpectedResult] = None
    escalation_condition: Optional[str] = None
    gcp_integration: Optional[GCPIntegration] = None
    timeout_minutes: int = 5
    retry_count: int = 3
    prerequisites: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    approval_required: bool = False


class PlaybookStepCreate(PlaybookStepBase):
    """Playbook step creation model."""
    pass


class PlaybookStepUpdate(BaseModel):
    """Playbook step update model."""
    
    description: Optional[str] = None
    step_type: Optional[StepType] = None
    order: Optional[int] = None
    query: Optional[str] = None
    expected_result: Optional[ExpectedResult] = None
    escalation_condition: Optional[str] = None
    gcp_integration: Optional[GCPIntegration] = None
    timeout_minutes: Optional[int] = None
    retry_count: Optional[int] = None
    prerequisites: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    approval_required: Optional[bool] = None


class PlaybookStep(PlaybookStepBase):
    """Full playbook step model."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    created_at: datetime


class PlaybookStepResult(BaseModel):
    """Playbook step execution result model."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    step_id: UUID
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: Optional[bool] = None
    result_data: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    escalation_triggered: bool = False
    error_message: Optional[str] = None
    threshold_met: Optional[bool] = None
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    
    def is_completed(self) -> bool:
        """Check if step execution is completed."""
        return self.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
    
    def is_successful(self) -> bool:
        """Check if step execution was successful."""
        return self.status == ExecutionStatus.COMPLETED and self.success is True
    
    def needs_escalation(self) -> bool:
        """Check if step needs escalation."""
        return self.escalation_triggered or (self.threshold_met is False and self.success is False)


class PlaybookMetadata(BaseModel):
    """Playbook metadata model."""
    
    created_by: str
    created_at: datetime
    last_updated: datetime
    applicable_services: List[str] = Field(default_factory=list)
    trigger_conditions: List[str] = Field(default_factory=list)


class PlaybookBase(BaseModel):
    """Base playbook model."""
    
    name: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=20)
    description: Optional[str] = None
    status: PlaybookStatus = PlaybookStatus.ACTIVE
    applicable_services: List[str] = Field(default_factory=list)
    trigger_conditions: List[str] = Field(default_factory=list)


class PlaybookCreate(PlaybookBase):
    """Playbook creation model."""
    
    playbook_id: Optional[str] = None
    steps: List[PlaybookStepCreate] = Field(default_factory=list)
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to set defaults."""
        if not self.playbook_id:
            # Generate playbook ID based on name
            safe_name = "".join(c.upper() if c.isalnum() else "_" for c in self.name)
            self.playbook_id = f"PB-{safe_name[:20]}-{str(uuid4())[:8].upper()}"


class PlaybookUpdate(BaseModel):
    """Playbook update model."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    version: Optional[str] = Field(None, min_length=1, max_length=20)
    description: Optional[str] = None
    status: Optional[PlaybookStatus] = None
    effectiveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    applicable_services: Optional[List[str]] = None
    trigger_conditions: Optional[List[str]] = None


class Playbook(PlaybookBase):
    """Full playbook model."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    playbook_id: str
    effectiveness_score: float = 0.0
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time_minutes: Optional[float] = None
    last_executed: Optional[datetime] = None
    created_by: str
    created_at: datetime
    last_updated: Optional[datetime] = None
    
    # Related data
    steps: List[PlaybookStep] = Field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.execution_count == 0:
            return 0.0
        return (self.failure_count / self.execution_count) * 100
    
    def is_active(self) -> bool:
        """Check if playbook is active."""
        return self.status == PlaybookStatus.ACTIVE
    
    def is_applicable_to_service(self, service_name: str) -> bool:
        """Check if playbook is applicable to a service."""
        return not self.applicable_services or service_name in self.applicable_services or "all" in self.applicable_services
    
    def matches_symptoms(self, symptoms: List[str]) -> bool:
        """Check if playbook matches incident symptoms."""
        if not self.trigger_conditions:
            return False
        
        # Simple keyword matching for demo
        symptoms_text = " ".join(symptoms).lower()
        return any(
            condition.lower() in symptoms_text 
            for condition in self.trigger_conditions
        )
    
    def get_step_by_id(self, step_id: str) -> Optional[PlaybookStep]:
        """Get step by step_id."""
        return next((step for step in self.steps if step.step_id == step_id), None)
    
    def get_steps_by_type(self, step_type: StepType) -> List[PlaybookStep]:
        """Get steps filtered by type."""
        return [step for step in self.steps if step.step_type == step_type]
    
    def get_next_step(self, current_step_order: int) -> Optional[PlaybookStep]:
        """Get the next step in execution order."""
        next_steps = [step for step in self.steps if step.order > current_step_order]
        return min(next_steps, key=lambda s: s.order) if next_steps else None
    
    def validate_step_dependencies(self, step: PlaybookStep, completed_steps: List[str]) -> bool:
        """Validate that step dependencies are satisfied."""
        if not step.dependencies:
            return True
        return all(dep in completed_steps for dep in step.dependencies)


class PlaybookExecution(BaseModel):
    """Playbook execution model."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    execution_id: str
    playbook_id: str  # Changed from UUID to str to match our playbook ID format
    incident_id: str
    status: ExecutionStatus
    started_by: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    success: Optional[bool] = None
    root_cause_found: bool = False
    actions_recommended: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    execution_context: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Related data
    step_results: List[PlaybookStepResult] = Field(default_factory=list)
    
    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.status == ExecutionStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
    
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED and self.success is True
    
    def get_completed_steps(self) -> List[str]:
        """Get list of completed step IDs."""
        return [
            result.step_id for result in self.step_results 
            if result.is_completed()
        ]
    
    def get_successful_steps(self) -> List[PlaybookStepResult]:
        """Get list of successful step results."""
        return [result for result in self.step_results if result.is_successful()]
    
    def get_failed_steps(self) -> List[PlaybookStepResult]:
        """Get list of failed step results."""
        return [
            result for result in self.step_results 
            if result.status == ExecutionStatus.FAILED
        ]
    
    def get_step_result(self, step_id: UUID) -> Optional[PlaybookStepResult]:
        """Get result for specific step."""
        return next((result for result in self.step_results if result.step_id == step_id), None)
    
    def calculate_progress(self, total_steps: int) -> float:
        """Calculate execution progress percentage."""
        if total_steps == 0:
            return 100.0
        completed = len([r for r in self.step_results if r.is_completed()])
        return (completed / total_steps) * 100
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary for reporting."""
        total_steps = len(self.step_results)
        successful_steps = len(self.get_successful_steps())
        failed_steps = len(self.get_failed_steps())
        
        return {
            "execution_id": self.execution_id,
            "incident_id": self.incident_id,
            "status": self.status,
            "duration_minutes": self.duration_minutes,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "success_rate": (successful_steps / total_steps * 100) if total_steps > 0 else 0,
            "root_cause_found": self.root_cause_found,
            "confidence_score": self.confidence_score,
            "actions_recommended": self.actions_recommended or []
        }


class PlaybookExecutionCreate(BaseModel):
    """Playbook execution creation model."""
    
    playbook_id: str
    incident_id: str
    execution_context: Optional[Dict[str, Any]] = None
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to set defaults."""
        if not hasattr(self, 'execution_id'):
            self.execution_id = f"EXEC-{str(uuid4())[:8].upper()}"


class PlaybookEffectiveness(BaseModel):
    """Playbook effectiveness metrics model."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    playbook_id: UUID
    period_start: datetime
    period_end: datetime
    execution_count: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    root_cause_identification_rate: float = 0.0
    false_positive_rate: float = 0.0
    user_satisfaction_score: Optional[float] = None
    effectiveness_score: float = 0.0
    
    def calculate_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score."""
        # Weighted average of different metrics
        weights = {
            'success_rate': 0.3,
            'root_cause_rate': 0.25,
            'execution_time': 0.2,  # Lower is better
            'false_positive_rate': 0.15,  # Lower is better
            'user_satisfaction': 0.1
        }
        
        score = 0.0
        
        # Success rate (0-100, higher is better)
        score += (self.success_rate / 100) * weights['success_rate']
        
        # Root cause identification rate (0-100, higher is better)
        score += (self.root_cause_identification_rate / 100) * weights['root_cause_rate']
        
        # Execution time (normalized, lower is better)
        if self.avg_execution_time > 0:
            # Assume optimal execution time is 10 minutes, penalize longer times
            time_score = max(0, 1 - (self.avg_execution_time - 10) / 50)
            score += time_score * weights['execution_time']
        
        # False positive rate (0-100, lower is better)
        score += (1 - self.false_positive_rate / 100) * weights['false_positive_rate']
        
        # User satisfaction (0-5 scale, higher is better)
        if self.user_satisfaction_score is not None:
            score += (self.user_satisfaction_score / 5) * weights['user_satisfaction']
        
        return min(1.0, max(0.0, score))


class PlaybookSummary(BaseModel):
    """Playbook summary for list views."""
    
    model_config = {"from_attributes": True}
    
    playbook_id: str
    name: str
    version: str
    status: PlaybookStatus
    effectiveness_score: float
    execution_count: int
    success_rate: float
    applicable_services: List[str]
    last_executed: Optional[datetime] = None
    step_count: int = 0


class PlaybookSearch(BaseModel):
    """Playbook search criteria."""
    
    query: Optional[str] = None
    status: Optional[List[PlaybookStatus]] = None
    applicable_services: Optional[List[str]] = None
    min_effectiveness: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_by: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="effectiveness_score")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class PlaybookRecommendation(BaseModel):
    """Playbook recommendation for incidents."""
    
    playbook_id: str
    name: str
    match_score: float
    applicable_services: List[str]
    trigger_conditions: List[str]
    effectiveness_score: float
    avg_execution_time_minutes: Optional[float] = None
    last_success_rate: float
    recommended_reason: str


class PlaybookStats(BaseModel):
    """Playbook statistics model."""
    
    total_playbooks: int
    active_playbooks: int
    draft_playbooks: int
    deprecated_playbooks: int
    avg_effectiveness_score: float
    total_executions: int
    avg_success_rate: float
    most_used_playbooks: List[Dict[str, Any]] = Field(default_factory=list)
    top_performing_playbooks: List[Dict[str, Any]] = Field(default_factory=list)