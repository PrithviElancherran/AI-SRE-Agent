"""
Analysis model with AI-powered analysis results, confidence scoring, reasoning trails, and evidence collection.

This module defines the Analysis model and related components for managing
AI-powered incident analysis in the AI SRE Agent system.
"""

from datetime import datetime
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


class AnalysisType(str, Enum):
    """Types of analysis performed by the AI SRE Agent."""
    
    CORRELATION = "correlation"
    PLAYBOOK = "playbook"
    ML_PREDICTION = "ml_prediction"
    PATTERN_ANALYSIS = "pattern_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    IMPACT_ASSESSMENT = "impact_assessment"


class AnalysisStatus(str, Enum):
    """Analysis execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvidenceType(str, Enum):
    """Types of evidence collected during analysis."""
    
    GCP_MONITORING = "gcp_monitoring"
    GCP_LOGGING = "gcp_logging"
    GCP_TRACING = "gcp_tracing"
    GCP_ERROR_REPORTING = "gcp_error_reporting"
    HISTORICAL_CORRELATION = "historical_correlation"
    METRIC_ANOMALY = "metric_anomaly"
    LOG_PATTERN = "log_pattern"
    PERFORMANCE_DATA = "performance_data"
    USER_FEEDBACK = "user_feedback"


class ReasoningStepType(str, Enum):
    """Types of reasoning steps in analysis trail."""
    
    DATA_COLLECTION = "data_collection"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    CORRELATION_SEARCH = "correlation_search"
    PATTERN_MATCHING = "pattern_matching"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    ROOT_CAUSE_IDENTIFICATION = "root_cause_identification"
    CONFIDENCE_CALCULATION = "confidence_calculation"
    RECOMMENDATION_GENERATION = "recommendation_generation"


class AnalysisResultTable(Base):
    """Analysis result database table."""
    
    __tablename__ = "analysis_results"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(String(50), unique=True, nullable=False, index=True)
    incident_id = Column(String(50), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default=AnalysisStatus.PENDING)
    
    # Analysis results
    findings = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=False, default=0.0)
    recommendation = Column(Text, nullable=True)
    root_cause = Column(Text, nullable=True)
    impact_assessment = Column(JSON, nullable=True)
    
    # Execution details
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    triggered_by = Column(String(50), nullable=True)
    
    # AI model information
    model_version = Column(String(50), nullable=True)
    model_confidence = Column(Float, nullable=True)
    training_data_version = Column(String(50), nullable=True)
    
    # Context and metadata
    analysis_context = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    evidence_items = relationship("EvidenceItemTable", back_populates="analysis", cascade="all, delete-orphan")
    reasoning_trail = relationship("ReasoningTrailTable", back_populates="analysis", cascade="all, delete-orphan")
    confidence_factors = relationship("ConfidenceFactorTable", back_populates="analysis", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_analysis_incident_type', 'incident_id', 'analysis_type'),
        Index('idx_analysis_status_confidence', 'status', 'confidence_score'),
        Index('idx_analysis_created_confidence', 'created_at', 'confidence_score'),
    )


class EvidenceItemTable(Base):
    """Evidence item database table."""
    
    __tablename__ = "evidence_items"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("analysis_results.id"), nullable=False, index=True)
    evidence_type = Column(String(50), nullable=False)
    source = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    data = Column(JSON, nullable=True)
    relevance_score = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    analysis = relationship("AnalysisResultTable", back_populates="evidence_items")


class ReasoningTrailTable(Base):
    """Reasoning trail database table."""
    
    __tablename__ = "reasoning_trails"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("analysis_results.id"), nullable=False, index=True)
    step_number = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    reasoning = Column(Text, nullable=False)
    confidence_impact = Column(Float, nullable=False, default=0.0)
    duration_seconds = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    analysis = relationship("AnalysisResultTable", back_populates="reasoning_trail")
    
    # Unique constraint for step number within analysis
    __table_args__ = (
        Index('idx_unique_step_per_analysis', 'analysis_id', 'step_number', unique=True),
    )


class ConfidenceFactorTable(Base):
    """Confidence factor database table."""
    
    __tablename__ = "confidence_factors"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("analysis_results.id"), nullable=False, index=True)
    factor_name = Column(String(100), nullable=False)
    factor_type = Column(String(50), nullable=False)
    weight = Column(Float, nullable=False)
    score = Column(Float, nullable=False)
    contribution = Column(Float, nullable=False)
    explanation = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    analysis = relationship("AnalysisResultTable", back_populates="confidence_factors")


class AnalysisFindings(BaseModel):
    """Analysis findings structure."""
    
    primary_cause: str
    contributing_factors: List[str] = Field(default_factory=list)
    affected_components: List[str] = Field(default_factory=list)
    impact_scope: Optional[Dict[str, Any]] = None
    timeline: Optional[Dict[str, Any]] = None
    related_incidents: List[str] = Field(default_factory=list)
    risk_assessment: Optional[Dict[str, Any]] = None


class EvidenceItem(BaseModel):
    """Evidence item model."""
    
    model_config = {"from_attributes": True}
    
    id: Optional[UUID] = None
    evidence_type: EvidenceType
    source: str
    description: str
    data: Optional[Dict[str, Any]] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    url: Optional[str] = None
    
    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        """Validate relevance score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Relevance score must be between 0.0 and 1.0')
        return v


class ReasoningStep(BaseModel):
    """Reasoning step model."""
    
    model_config = {"from_attributes": True}
    
    id: Optional[UUID] = None
    step_number: int
    step_type: ReasoningStepType
    description: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    reasoning: str
    confidence_impact: float = Field(default=0.0, ge=-1.0, le=1.0)
    duration_seconds: Optional[float] = None
    
    @validator('confidence_impact')
    def validate_confidence_impact(cls, v):
        """Validate confidence impact is between -1 and 1."""
        if not -1.0 <= v <= 1.0:
            raise ValueError('Confidence impact must be between -1.0 and 1.0')
        return v


class ConfidenceFactor(BaseModel):
    """Confidence factor model."""
    
    model_config = {"from_attributes": True}
    
    id: Optional[UUID] = None
    factor_name: str
    factor_type: str
    weight: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    contribution: float = Field(ge=0.0, le=1.0)
    explanation: Optional[str] = None
    
    @validator('weight', 'score', 'contribution')
    def validate_scores(cls, v):
        """Validate scores are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return v


class ReasoningTrail(BaseModel):
    """Reasoning trail model."""
    
    model_config = {"from_attributes": True}
    
    analysis_id: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    total_confidence_impact: float = 0.0
    reasoning_quality_score: float = 0.0
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trail."""
        step.step_number = len(self.steps) + 1
        self.steps.append(step)
        self.total_confidence_impact += step.confidence_impact
    
    def get_step_by_type(self, step_type: ReasoningStepType) -> Optional[ReasoningStep]:
        """Get first step of specific type."""
        return next((step for step in self.steps if step.step_type == step_type), None)
    
    def get_steps_by_type(self, step_type: ReasoningStepType) -> List[ReasoningStep]:
        """Get all steps of specific type."""
        return [step for step in self.steps if step.step_type == step_type]
    
    def calculate_reasoning_quality(self) -> float:
        """Calculate overall reasoning quality score."""
        if not self.steps:
            return 0.0
        
        # Quality factors
        completeness = len(self.steps) / 9  # Assume 9 ideal steps
        consistency = 1.0 - abs(self.total_confidence_impact) / len(self.steps)
        depth = sum(1 for step in self.steps if step.reasoning and len(step.reasoning) > 50) / len(self.steps)
        
        return min(1.0, (completeness * 0.4 + consistency * 0.3 + depth * 0.3))


class ConfidenceScore(BaseModel):
    """Confidence score model."""
    
    model_config = {"from_attributes": True}
    
    overall_score: float = Field(ge=0.0, le=1.0)
    factors: List[ConfidenceFactor] = Field(default_factory=list)
    calculation_method: str = "weighted_average"
    quality_indicators: Dict[str, float] = Field(default_factory=dict)
    reliability_assessment: Optional[str] = None
    
    @validator('overall_score')
    def validate_overall_score(cls, v):
        """Validate overall score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Overall score must be between 0.0 and 1.0')
        return v
    
    def add_factor(self, factor: ConfidenceFactor) -> None:
        """Add a confidence factor."""
        self.factors.append(factor)
        self.recalculate_score()
    
    def recalculate_score(self) -> None:
        """Recalculate overall confidence score."""
        if not self.factors:
            self.overall_score = 0.0
            return
        
        total_weighted_score = sum(factor.score * factor.weight for factor in self.factors)
        total_weight = sum(factor.weight for factor in self.factors)
        
        if total_weight > 0:
            self.overall_score = total_weighted_score / total_weight
        else:
            self.overall_score = 0.0
        
        # Ensure score is within bounds
        self.overall_score = max(0.0, min(1.0, self.overall_score))
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.overall_score >= 0.9:
            return "Very High"
        elif self.overall_score >= 0.8:
            return "High"
        elif self.overall_score >= 0.7:
            return "Medium"
        elif self.overall_score >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def is_actionable(self, threshold: float = 0.8) -> bool:
        """Check if confidence is high enough for automated actions."""
        return self.overall_score >= threshold


class AnalysisBase(BaseModel):
    """Base analysis model."""
    
    incident_id: str
    analysis_type: AnalysisType
    triggered_by: Optional[str] = None
    analysis_context: Optional[Dict[str, Any]] = None


class AnalysisCreate(AnalysisBase):
    """Analysis creation model."""
    
    analysis_id: Optional[str] = None
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to set defaults."""
        if not self.analysis_id:
            self.analysis_id = f"ANALYSIS-{str(uuid4())[:8].upper()}"


class AnalysisUpdate(BaseModel):
    """Analysis update model."""
    
    status: Optional[AnalysisStatus] = None
    findings: Optional[AnalysisFindings] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recommendation: Optional[str] = None
    root_cause: Optional[str] = None
    impact_assessment: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AnalysisResult(AnalysisBase):
    """Full analysis result model."""
    
    model_config = {"from_attributes": True}
    
    id: UUID
    analysis_id: str
    status: AnalysisStatus = AnalysisStatus.PENDING
    findings: Optional[AnalysisFindings] = None
    confidence_score: float = 0.0
    recommendation: Optional[str] = None
    root_cause: Optional[str] = None
    impact_assessment: Optional[Dict[str, Any]] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    model_version: Optional[str] = None
    model_confidence: Optional[float] = None
    training_data_version: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Related data
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    reasoning_trail: Optional[ReasoningTrail] = None
    confidence_factors: List[ConfidenceFactor] = Field(default_factory=list)
    
    def is_completed(self) -> bool:
        """Check if analysis is completed."""
        return self.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED]
    
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return self.status == AnalysisStatus.COMPLETED and self.confidence_score > 0.0
    
    def has_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if analysis has high confidence."""
        return self.confidence_score >= threshold
    
    def get_evidence_by_type(self, evidence_type: EvidenceType) -> List[EvidenceItem]:
        """Get evidence items by type."""
        return [item for item in self.evidence_items if item.evidence_type == evidence_type]
    
    def get_primary_evidence(self) -> Optional[EvidenceItem]:
        """Get highest relevance evidence item."""
        if not self.evidence_items:
            return None
        return max(self.evidence_items, key=lambda e: e.relevance_score)
    
    def add_evidence(self, evidence: EvidenceItem) -> None:
        """Add evidence item to analysis."""
        self.evidence_items.append(evidence)
    
    def calculate_duration(self) -> Optional[float]:
        """Calculate analysis duration in seconds."""
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds()
        return None
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence_score >= 0.9:
            return "Very High"
        elif self.confidence_score >= 0.8:
            return "High"
        elif self.confidence_score >= 0.7:
            return "Medium"
        elif self.confidence_score >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary for reporting."""
        return {
            "analysis_id": self.analysis_id,
            "incident_id": self.incident_id,
            "analysis_type": self.analysis_type,
            "status": self.status,
            "confidence_score": self.confidence_score,
            "confidence_level": self.get_confidence_level(),
            "duration_seconds": self.duration_seconds,
            "evidence_count": len(self.evidence_items),
            "reasoning_steps": len(self.reasoning_trail.steps) if self.reasoning_trail else 0,
            "root_cause": self.root_cause,
            "recommendation": self.recommendation,
            "primary_evidence": self.get_primary_evidence().description if self.get_primary_evidence() else None
        }


class AnalysisSummary(BaseModel):
    """Analysis summary for list views."""
    
    model_config = {"from_attributes": True}
    
    analysis_id: str
    incident_id: str
    analysis_type: AnalysisType
    status: AnalysisStatus
    confidence_score: float
    confidence_level: str
    duration_seconds: Optional[float] = None
    evidence_count: int = 0
    reasoning_steps: int = 0
    started_at: datetime
    completed_at: Optional[datetime] = None


class AnalysisSearch(BaseModel):
    """Analysis search criteria."""
    
    query: Optional[str] = None
    incident_id: Optional[str] = None
    analysis_type: Optional[List[AnalysisType]] = None
    status: Optional[List[AnalysisStatus]] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    triggered_by: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="started_at")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class AnalysisStats(BaseModel):
    """Analysis statistics model."""
    
    total_analyses: int
    completed_analyses: int
    failed_analyses: int
    avg_confidence_score: float
    avg_duration_seconds: float
    analysis_by_type: Dict[str, int] = Field(default_factory=dict)
    confidence_distribution: Dict[str, int] = Field(default_factory=dict)
    top_root_causes: List[Dict[str, Any]] = Field(default_factory=list)
    performance_trends: List[Dict[str, Any]] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    """Analysis request model for triggering new analysis."""
    
    incident_id: str
    analysis_type: AnalysisType
    priority: Union[int, str] = Field(default=5, description="Priority level (1-10) or string")
    timeout_minutes: int = Field(default=10, ge=1, le=60)
    context: Optional[Dict[str, Any]] = None
    requested_by: str = Field(default="demo_user", description="User requesting analysis")
    request_id: Optional[str] = None
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to set defaults."""
        if not self.request_id:
            self.request_id = f"REQ-{str(uuid4())[:8].upper()}"


class AnalysisResponse(BaseModel):
    """Analysis response model for real-time updates."""
    
    analysis_id: str
    incident_id: str
    status: AnalysisStatus
    progress_percentage: float = Field(ge=0.0, le=100.0)
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    preliminary_findings: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MLModelMetrics(BaseModel):
    """ML model performance metrics."""
    
    model_name: str
    model_version: str
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    training_date: datetime
    evaluation_date: datetime
    sample_size: int
    feature_importance: Dict[str, float] = Field(default_factory=dict)


class AnalysisQualityMetrics(BaseModel):
    """Analysis quality assessment metrics."""
    
    analysis_id: str
    accuracy_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    timeliness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    human_validation: Optional[bool] = None
    validation_notes: Optional[str] = None
    validated_by: Optional[str] = None
    validated_at: Optional[datetime] = None
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall quality score."""
        scores = [
            self.accuracy_score,
            self.completeness_score,
            self.timeliness_score,
            self.relevance_score
        ]
        return sum(scores) / len(scores)


class AnalysisTemplate(BaseModel):
    """Analysis template for consistent analysis structure."""
    
    template_id: str
    name: str
    description: str
    analysis_type: AnalysisType
    required_evidence_types: List[EvidenceType] = Field(default_factory=list)
    reasoning_steps_template: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_factors_template: List[Dict[str, Any]] = Field(default_factory=list)
    output_format: Dict[str, Any] = Field(default_factory=dict)
    validation_criteria: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)