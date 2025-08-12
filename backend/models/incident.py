"""
Incident model with symptoms, timeline, resolution tracking, and correlation capabilities.

This module defines the Incident model and related components for managing
production incidents in the AI SRE Agent system.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
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


class IncidentStatus(str, Enum):
    """Incident status enumeration."""
    
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SymptomType(str, Enum):
    """Types of incident symptoms."""
    
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    CACHE_PERFORMANCE = "cache_performance"
    CONNECTION_TIMEOUT = "connection_timeout"
    QUERY_PERFORMANCE = "query_performance"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_ISSUES = "network_issues"
    SERVICE_UNAVAILABLE = "service_unavailable"


class IncidentTable(Base):
    """Incident database table."""
    
    __tablename__ = "incidents"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    status = Column(String(20), nullable=False, default=IncidentStatus.OPEN, index=True)
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    symptoms = Column(JSON, nullable=False, default=list)
    region = Column(String(50), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    confidence_score = Column(Float, nullable=True)
    root_cause = Column(Text, nullable=True)
    resolution = Column(Text, nullable=True)
    resolution_time = Column(DateTime(timezone=True), nullable=True)
    mttr_minutes = Column(Integer, nullable=True)
    affected_users = Column(Integer, nullable=True)
    tags = Column(JSON, nullable=False, default=list)
    
    # Tracking fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(50), nullable=True)
    updated_by = Column(String(50), nullable=True)
    assigned_to = Column(String(50), nullable=True)
    
    # Vector embedding for similarity search
    embedding = Column(JSON, nullable=True)
    
    # Relationships
    incident_symptoms = relationship("IncidentSymptomTable", back_populates="incident", cascade="all, delete-orphan")
    correlations = relationship("IncidentCorrelationTable", foreign_keys="IncidentCorrelationTable.source_incident_id", back_populates="source_incident")
    resolutions = relationship("IncidentResolutionTable", back_populates="incident", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_incident_status_severity', 'status', 'severity'),
        Index('idx_incident_service_region', 'service_name', 'region'),
        Index('idx_incident_timestamp_status', 'timestamp', 'status'),
    )


class IncidentSymptomTable(Base):
    """Incident symptom database table."""
    
    __tablename__ = "incident_symptoms"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    symptom_id = Column(String(50), nullable=False)
    symptom_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    metric_data = Column(JSON, nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    severity_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    incident = relationship("IncidentTable", back_populates="incident_symptoms")


class IncidentResolutionTable(Base):
    """Incident resolution database table."""
    
    __tablename__ = "incident_resolutions"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    resolution_step = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    executed_by = Column(String(50), nullable=False)
    executed_at = Column(DateTime(timezone=True), nullable=False)
    success = Column(Boolean, nullable=False, default=True)
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    incident = relationship("IncidentTable", back_populates="resolutions")


class IncidentCorrelationTable(Base):
    """Incident correlation database table."""
    
    __tablename__ = "incident_correlations"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_incident_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    target_incident_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    similarity_score = Column(Float, nullable=False)
    correlation_type = Column(String(50), nullable=False)  # symptom, root_cause, temporal
    correlation_details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    source_incident = relationship("IncidentTable", foreign_keys=[source_incident_id])
    target_incident = relationship("IncidentTable", foreign_keys=[target_incident_id])
    
    # Unique constraint to prevent duplicate correlations
    __table_args__ = (
        Index('idx_unique_correlation', 'source_incident_id', 'target_incident_id', unique=True),
    )


class MetricData(BaseModel):
    """Metric data structure for symptoms."""
    
    metric_name: str
    threshold: float
    actual_value: float
    unit: str
    timestamp: Optional[datetime] = None


class IncidentSymptom(BaseModel):
    """Incident symptom model."""
    
    model_config = {"from_attributes": True}
    
    symptom_id: str = Field(default_factory=lambda: str(uuid4()))
    symptom_type: SymptomType
    description: str
    metric_data: MetricData
    detected_at: datetime
    severity_score: float = Field(ge=0.0, le=1.0)
    
    @validator('severity_score')
    def validate_severity_score(cls, v):
        """Validate severity score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Severity score must be between 0.0 and 1.0')
        return v


class IncidentResolution(BaseModel):
    """Incident resolution model."""
    
    model_config = {"from_attributes": True}
    
    id: Optional[UUID] = None
    resolution_step: str
    description: str
    executed_by: str
    executed_at: datetime
    success: bool = True
    evidence: Optional[Dict[str, Any]] = None


class IncidentCorrelation(BaseModel):
    """Incident correlation model."""
    
    model_config = {"from_attributes": True}
    
    id: Optional[UUID] = None
    source_incident_id: str
    target_incident_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    correlation_type: str
    correlation_details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    @validator('similarity_score')
    def validate_similarity_score(cls, v):
        """Validate similarity score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity score must be between 0.0 and 1.0')
        return v


class IncidentBase(BaseModel):
    """Base incident model."""
    
    title: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    region: str
    service_name: str
    severity: IncidentSeverity
    tags: List[str] = Field(default_factory=list)
    assigned_to: Optional[str] = None


class IncidentCreate(IncidentBase):
    """Incident creation model."""
    
    incident_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    incident_symptoms: List[IncidentSymptom] = Field(default_factory=list)
    auto_analyze: bool = Field(default=False, description="Whether to trigger automatic analysis")
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to set defaults."""
        if not self.incident_id:
            self.incident_id = f"INC-{str(uuid4())[:8].upper()}"
        
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


class IncidentUpdate(BaseModel):
    """Incident update model."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[IncidentStatus] = None
    severity: Optional[IncidentSeverity] = None
    symptoms: Optional[List[str]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    resolution_time: Optional[datetime] = None
    mttr_minutes: Optional[int] = Field(None, ge=0)
    affected_users: Optional[int] = Field(None, ge=0)
    tags: Optional[List[str]] = None
    assigned_to: Optional[str] = None
    updated_by: Optional[str] = None


class Incident(IncidentBase):
    """Full incident model."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    incident_id: str
    timestamp: datetime
    status: IncidentStatus = IncidentStatus.OPEN
    confidence_score: Optional[float] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    resolution_time: Optional[datetime] = None
    mttr_minutes: Optional[int] = None
    affected_users: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    # Related data
    incident_symptoms: List[IncidentSymptom] = Field(default_factory=list)
    correlations: List[IncidentCorrelation] = Field(default_factory=list)
    resolutions: List[IncidentResolution] = Field(default_factory=list)
    
    def is_open(self) -> bool:
        """Check if incident is still open."""
        return self.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING, IncidentStatus.IDENTIFIED]
    
    def is_resolved(self) -> bool:
        """Check if incident is resolved."""
        return self.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]
    
    def calculate_mttr(self) -> Optional[int]:
        """Calculate Mean Time To Resolution in minutes."""
        if self.resolution_time and self.timestamp:
            delta = self.resolution_time - self.timestamp
            return int(delta.total_seconds() / 60)
        return None
    
    def get_primary_symptom(self) -> Optional[IncidentSymptom]:
        """Get the primary (highest severity) symptom."""
        if not self.incident_symptoms:
            return None
        return max(self.incident_symptoms, key=lambda s: s.severity_score)
    
    def get_symptoms_by_type(self, symptom_type: SymptomType) -> List[IncidentSymptom]:
        """Get symptoms filtered by type."""
        return [s for s in self.incident_symptoms if s.symptom_type == symptom_type]
    
    def has_symptom_type(self, symptom_type: SymptomType) -> bool:
        """Check if incident has specific symptom type."""
        return any(s.symptom_type == symptom_type for s in self.incident_symptoms)
    
    def get_correlation_score(self, other_incident_id: str) -> Optional[float]:
        """Get correlation score with another incident."""
        for correlation in self.correlations:
            if correlation.target_incident_id == other_incident_id:
                return correlation.similarity_score
        return None
    
    def add_symptom(self, symptom: IncidentSymptom) -> None:
        """Add a symptom to the incident."""
        self.incident_symptoms.append(symptom)
    
    def add_resolution_step(self, resolution: IncidentResolution) -> None:
        """Add a resolution step to the incident."""
        self.resolutions.append(resolution)
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get incident timeline with all events."""
        timeline = []
        
        # Incident creation
        timeline.append({
            "timestamp": self.timestamp,
            "event_type": "incident_created",
            "description": f"Incident {self.incident_id} created",
            "severity": self.severity,
            "details": {"title": self.title}
        })
        
        # Symptoms detection
        for symptom in sorted(self.incident_symptoms, key=lambda s: s.detected_at):
            timeline.append({
                "timestamp": symptom.detected_at,
                "event_type": "symptom_detected",
                "description": symptom.description,
                "severity": symptom.severity_score,
                "details": {
                    "symptom_type": symptom.symptom_type,
                    "metric_data": symptom.metric_data.model_dump()
                }
            })
        
        # Resolution steps
        for resolution in sorted(self.resolutions, key=lambda r: r.executed_at):
            timeline.append({
                "timestamp": resolution.executed_at,
                "event_type": "resolution_step",
                "description": resolution.description,
                "success": resolution.success,
                "details": {
                    "executed_by": resolution.executed_by,
                    "evidence": resolution.evidence
                }
            })
        
        # Incident resolution
        if self.resolution_time:
            timeline.append({
                "timestamp": self.resolution_time,
                "event_type": "incident_resolved",
                "description": f"Incident {self.incident_id} resolved",
                "details": {
                    "root_cause": self.root_cause,
                    "resolution": self.resolution,
                    "mttr_minutes": self.mttr_minutes
                }
            })
        
        return sorted(timeline, key=lambda x: x["timestamp"])


class IncidentSummary(BaseModel):
    """Incident summary for list views."""
    
    model_config = {"from_attributes": True}
    
    incident_id: str
    title: str
    status: IncidentStatus
    severity: IncidentSeverity
    service_name: str
    region: str
    timestamp: datetime
    mttr_minutes: Optional[int] = None
    affected_users: Optional[int] = None
    confidence_score: Optional[float] = None
    symptom_count: int = 0
    correlation_count: int = 0


class IncidentStats(BaseModel):
    """Incident statistics model."""
    
    total_incidents: int
    open_incidents: int
    resolved_incidents: int
    critical_incidents: int
    high_incidents: int
    medium_incidents: int
    low_incidents: int
    avg_mttr_minutes: Optional[float] = None
    avg_confidence_score: Optional[float] = None
    top_services: List[Dict[str, Any]] = Field(default_factory=list)
    top_regions: List[Dict[str, Any]] = Field(default_factory=list)


class IncidentSearch(BaseModel):
    """Incident search criteria."""
    
    query: Optional[str] = None
    status: Optional[List[IncidentStatus]] = None
    severity: Optional[List[IncidentSeverity]] = None
    service_name: Optional[List[str]] = None
    region: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="timestamp")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class SimilarIncident(BaseModel):
    """Similar incident model for correlation results."""
    
    incident_id: str
    title: str
    service_name: str
    region: str
    similarity_score: float
    correlation_type: str
    timestamp: datetime
    status: IncidentStatus
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    mttr_minutes: Optional[int] = None


class IncidentCorrelationResult(BaseModel):
    """Incident correlation analysis result."""
    
    incident_id: str
    similar_incidents: List[SimilarIncident] = Field(default_factory=list)
    correlation_analysis: Dict[str, Any] = Field(default_factory=dict)
    recommended_actions: List[str] = Field(default_factory=list)
    confidence_score: float
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)