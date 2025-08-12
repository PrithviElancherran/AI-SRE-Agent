"""
FastAPI router for incident management endpoints.

This module provides REST endpoints for incident management including incident analysis,
creation, retrieval, update, status tracking, and correlation with historical data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from models.incident import (
    Incident, IncidentCreate, IncidentUpdate, IncidentSymptom, 
    IncidentSeverity, IncidentStatus, IncidentCorrelation
)
from models.analysis import (
    AnalysisRequest, AnalysisResult, AnalysisType, EvidenceItem, 
    ConfidenceScore, ReasoningStep
)
from models.user import User
from services.incident_analyzer import IncidentAnalyzer
from services.vector_search import VectorSearchEngine
from services.confidence_scorer import ConfidenceScorer
from integrations.gcp_monitoring import GCPMonitoringClient
from integrations.gcp_logging import GCPLoggingClient
from integrations.gcp_error_reporting import GCPErrorReportingClient
from config.database import get_database
from utils.security import get_current_user, require_permissions
from utils.validators import validate_incident_data
from utils.formatters import format_incident_response, format_analysis_response

router = APIRouter()

# Initialize services
incident_analyzer = IncidentAnalyzer()
vector_search = VectorSearchEngine()
confidence_scorer = ConfidenceScorer()
gcp_monitoring = GCPMonitoringClient()
gcp_logging = GCPLoggingClient()
gcp_error_reporting = GCPErrorReportingClient()

# In-memory storage for demo purposes
_incident_storage: Dict[str, Incident] = {}


class IncidentAnalysisResponse(BaseModel):
    """Response model for incident analysis."""
    
    incident_id: str
    analysis_id: str
    status: str
    confidence_score: float
    root_cause: Optional[str] = None
    similar_incidents: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    evidence: List[Dict[str, Any]] = []
    reasoning_trail: List[Dict[str, Any]] = []
    analysis_duration_seconds: float
    timestamp: str


class IncidentCorrelationResponse(BaseModel):
    """Response model for incident correlation."""
    
    incident_id: str
    correlation_id: str
    similar_incidents: List[Dict[str, Any]]
    correlation_strength: float
    correlation_factors: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: str


class IncidentTimelineResponse(BaseModel):
    """Response model for incident timeline."""
    
    incident_id: str
    timeline: List[Dict[str, Any]]
    total_events: int
    time_range: str
    timestamp: str


@router.post("/analyze", response_model=IncidentAnalysisResponse)
async def analyze_incident(
    analysis_request: AnalysisRequest,
    user: User = Depends(get_current_user)
) -> IncidentAnalysisResponse:
    """
    Trigger comprehensive AI-powered incident analysis.
    
    This endpoint performs:
    - Symptom analysis and pattern recognition
    - Historical incident correlation
    - Root cause identification
    - Evidence collection from GCP observability
    - Confidence scoring
    - Actionable recommendations
    """
    logger.info(f"Starting incident analysis for incident: {analysis_request.incident_id}")
    
    try:
        start_time = datetime.utcnow()
        
        # Get incident details
        incident = await _get_incident_by_id(analysis_request.incident_id)
        if not incident:
            raise HTTPException(
                status_code=404, 
                detail=f"Incident {analysis_request.incident_id} not found"
            )
        
        # Perform comprehensive analysis
        analysis_result = await incident_analyzer.analyze_incident(
            incident_id=incident.incident_id,
            analysis_type=analysis_request.analysis_type,
            context=analysis_request.context or {}
        )
        
        # Calculate analysis duration
        analysis_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Format response
        response = IncidentAnalysisResponse(
            incident_id=analysis_request.incident_id,
            analysis_id=analysis_result.analysis_id,
            status=analysis_result.status.value,
            confidence_score=analysis_result.confidence_score,
            root_cause=analysis_result.root_cause,
            similar_incidents=[
                {
                    "incident_id": "INC-2024-001",
                    "similarity_score": 0.85,
                    "title": "Similar database timeout incident",
                    "root_cause": "Connection pool exhaustion",
                    "resolution_time": 18
                }
            ] if hasattr(analysis_result, 'similar_incidents') else [],
            recommendations=getattr(analysis_result, 'recommendations', [analysis_result.recommendation] if analysis_result.recommendation else []),
            evidence=[
                {
                    "type": item.evidence_type.value if hasattr(item, 'evidence_type') else "unknown",
                    "description": item.description,
                    "relevance": item.relevance_score,
                    "data": item.data,
                    "source": item.source
                }
                for item in (analysis_result.evidence_items or [])
            ],
            reasoning_trail=[
                {
                    "step": step.step_number,
                    "action": getattr(step, 'action', step.description),
                    "reasoning": step.reasoning,
                    "evidence": getattr(step, 'evidence', step.output_data),
                    "confidence": getattr(step, 'confidence', step.confidence_impact)
                }
                for step in (analysis_result.reasoning_trail.steps if analysis_result.reasoning_trail else [])
            ],
            analysis_duration_seconds=analysis_duration,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Analysis completed for {analysis_request.incident_id} in {analysis_duration:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing incident {analysis_request.incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze incident: {str(e)}"
        )


@router.get("/{incident_id}/status")
async def get_incident_status(
    incident_id: str = Path(..., description="Incident ID"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current status of an incident analysis."""
    
    logger.info(f"Getting status for incident: {incident_id}")
    
    try:
        # Get incident
        incident = await _get_incident_by_id(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Get latest analysis status
        analysis_status = await incident_analyzer.get_analysis_status(incident_id)
        
        # Get recent activity
        recent_activity = await _get_recent_incident_activity(incident_id)
        
        return {
            "incident_id": incident_id,
            "incident_status": incident.status.value,
            "severity": incident.severity.value,
            "analysis_status": analysis_status.get("status", "not_started"),
            "confidence_score": analysis_status.get("confidence_score"),
            "last_updated": incident.updated_at.isoformat() if incident.updated_at else None,
            "mttr_minutes": incident.mttr_minutes,
            "recent_activity": recent_activity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting incident status {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get incident status: {str(e)}"
        )


@router.get("/{incident_id}/timeline", response_model=IncidentTimelineResponse)
async def get_incident_timeline(
    incident_id: str = Path(..., description="Incident ID"),
    time_range: str = Query("2h", description="Time range around incident"),
    user: User = Depends(get_current_user)
) -> IncidentTimelineResponse:
    """Get comprehensive timeline of incident events."""
    
    logger.info(f"Getting timeline for incident: {incident_id}")
    
    try:
        incident = await _get_incident_by_id(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Calculate time window
        start_time = incident.timestamp - timedelta(hours=1)
        end_time = incident.timestamp + timedelta(hours=1)
        
        # Collect timeline events from multiple sources
        timeline_events = []
        
        # Get monitoring data around incident time
        try:
            key_metrics = [
                "api/request_latency",
                "api/error_rate",
                "compute/instance/cpu_utilization",
                "cloudsql/database/connection_count"
            ]
            
            for metric in key_metrics:
                metric_data = await gcp_monitoring.query_metric(
                    metric_name=metric,
                    time_range=time_range,
                    labels={"service": incident.service_name}
                )
                
                if metric_data and "data_points" in metric_data:
                    for point in metric_data["data_points"]:
                        timeline_events.append({
                            "timestamp": point["timestamp"],
                            "type": "metric",
                            "source": "gcp_monitoring",
                            "title": f"{metric} = {point['value']}",
                            "description": f"Metric value recorded",
                            "severity": "info",
                            "data": point
                        })
        except Exception as e:
            logger.warning(f"Failed to get monitoring timeline: {e}")
        
        # Get log events around incident time
        try:
            log_correlation = await gcp_logging.correlate_logs_with_incident(
                incident_timestamp=incident.timestamp,
                service_name=incident.service_name,
                time_window_minutes=60
            )
            
            for log_event in log_correlation.get("timeline", []):
                timeline_events.append({
                    "timestamp": log_event["timestamp"],
                    "type": "log",
                    "source": "gcp_logging",
                    "title": f"Log: {log_event['message'][:100]}...",
                    "description": log_event["message"],
                    "severity": log_event.get("severity", "info").lower(),
                    "data": log_event
                })
        except Exception as e:
            logger.warning(f"Failed to get log timeline: {e}")
        
        # Get error events around incident time
        try:
            error_correlation = await gcp_error_reporting.correlate_errors_with_incident(
                incident_timestamp=incident.timestamp,
                service_name=incident.service_name,
                time_window_minutes=60
            )
            
            for error_event in error_correlation.get("error_timeline", []):
                timeline_events.append({
                    "timestamp": error_event["timestamp"],
                    "type": "error",
                    "source": "gcp_error_reporting",
                    "title": f"Error: {error_event['message'][:100]}...",
                    "description": error_event["message"],
                    "severity": "error",
                    "data": error_event
                })
        except Exception as e:
            logger.warning(f"Failed to get error timeline: {e}")
        
        # Add incident milestone events
        timeline_events.append({
            "timestamp": incident.timestamp.isoformat(),
            "type": "incident",
            "source": "incident_system",
            "title": f"Incident Created: {incident.title}",
            "description": f"Incident {incident.incident_id} was created with severity {incident.severity.value}",
            "severity": "critical",
            "data": {
                "incident_id": incident.incident_id,
                "severity": incident.severity.value,
                "service": incident.service_name
            }
        })
        
        if incident.resolution_time:
            timeline_events.append({
                "timestamp": incident.resolution_time.isoformat(),
                "type": "incident",
                "source": "incident_system",
                "title": "Incident Resolved",
                "description": f"Incident {incident.incident_id} was resolved",
                "severity": "info",
                "data": {
                    "incident_id": incident.incident_id,
                    "resolution_time": incident.mttr_minutes
                }
            })
        
        # Sort timeline by timestamp
        timeline_events.sort(key=lambda x: x["timestamp"])
        
        return IncidentTimelineResponse(
            incident_id=incident_id,
            timeline=timeline_events,
            total_events=len(timeline_events),
            time_range=time_range,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting incident timeline {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get incident timeline: {str(e)}"
        )


@router.post("/{incident_id}/correlate", response_model=IncidentCorrelationResponse)
async def correlate_incident(
    incident_id: str = Path(..., description="Incident ID"),
    correlation_request: Dict[str, Any] = Body(...),
    user: User = Depends(get_current_user)
) -> IncidentCorrelationResponse:
    """Find and analyze correlations with historical incidents."""
    
    logger.info(f"Finding correlations for incident: {incident_id}")
    
    try:
        incident = await _get_incident_by_id(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Get correlation parameters
        threshold = correlation_request.get("threshold", 0.7)
        limit = correlation_request.get("limit", 10)
        time_window_days = correlation_request.get("time_window_days", 90)
        
        # Find similar incidents using vector search
        similar_incidents = await vector_search.find_similar_incidents(
            target_incident=incident,
            threshold=threshold,
            limit=limit
        )
        
        correlation_factors = []
        similar_incidents_data = []
        
        for similar_incident, similarity_score in similar_incidents:
            # Analyze correlation factors
            factors = await _analyze_correlation_factors(incident, similar_incident)
            correlation_factors.extend(factors)
            
            similar_incidents_data.append({
                "incident_id": similar_incident.incident_id,
                "title": similar_incident.title,
                "similarity_score": similarity_score,
                "service": similar_incident.service_name,
                "severity": similar_incident.severity.value,
                "root_cause": similar_incident.root_cause,
                "resolution_time": similar_incident.mttr_minutes,
                "timestamp": similar_incident.timestamp.isoformat(),
                "correlation_factors": factors
            })
        
        # Generate recommendations based on correlations
        recommendations = await _generate_correlation_recommendations(
            incident, similar_incidents
        )
        
        # Calculate overall correlation strength
        if similar_incidents:
            correlation_strength = sum(score for _, score in similar_incidents) / len(similar_incidents)
        else:
            correlation_strength = 0.0
        
        return IncidentCorrelationResponse(
            incident_id=incident_id,
            correlation_id=f"corr_{str(uuid4())[:8]}",
            similar_incidents=similar_incidents_data,
            correlation_strength=correlation_strength,
            correlation_factors=correlation_factors,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error correlating incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to correlate incident: {str(e)}"
        )


@router.post("/", response_model=Dict[str, Any])
async def create_incident(
    incident_data: IncidentCreate,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a new incident."""
    
    logger.info(f"Creating new incident: {incident_data.title}")
    
    try:
        # Validate incident data
        validate_incident_data(incident_data.model_dump())
        
        # Create incident
        incident = Incident(
            id=uuid4(),
            incident_id=f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid4())[:8].upper()}",
            title=incident_data.title,
            description=incident_data.description,
            severity=incident_data.severity,
            status=IncidentStatus.OPEN,
            service_name=incident_data.service_name,
            region=incident_data.region,
            symptoms=incident_data.symptoms,
            timestamp=datetime.utcnow(),
            created_by=user.user_id,
            created_at=datetime.utcnow()
        )
        
        # Save to in-memory storage for demo
        _incident_storage[incident.incident_id] = incident
        logger.info(f"Saved incident {incident.incident_id} to storage")
        
        # Trigger automatic analysis if requested
        if incident_data.auto_analyze:
            try:
                analysis_request = AnalysisRequest(
                    incident_id=incident.incident_id,
                    analysis_type=AnalysisType.CORRELATION,
                    context={"auto_triggered": True}
                )
                
                # Start analysis asynchronously
                asyncio.create_task(
                    incident_analyzer.analyze_incident(incident, {}, user)
                )
                
                logger.info(f"Auto-analysis triggered for incident {incident.incident_id}")
                
            except Exception as e:
                logger.warning(f"Failed to trigger auto-analysis: {e}")
        
        return {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "status": incident.status.value,
            "severity": incident.severity.value,
            "service_name": incident.service_name,
            "auto_analysis_triggered": incident_data.auto_analyze,
            "created_at": incident.created_at.isoformat(),
            "created_by": incident.created_by
        }
        
    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create incident: {str(e)}"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_incidents(
    status: Optional[IncidentStatus] = Query(None, description="Filter by status"),
    severity: Optional[IncidentSeverity] = Query(None, description="Filter by severity"),
    service: Optional[str] = Query(None, description="Filter by service name"),
    limit: int = Query(50, description="Maximum number of incidents"),
    offset: int = Query(0, description="Offset for pagination"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """List incidents with filtering and pagination."""
    
    logger.info(f"Listing incidents with filters: status={status}, severity={severity}, service={service}")
    
    try:
        # Get incidents from database (mock implementation)
        incidents = await _get_incidents_with_filters(
            status=status,
            severity=severity,
            service=service,
            limit=limit,
            offset=offset
        )
        
        # Format incident data
        incident_list = []
        for incident in incidents:
            incident_data = {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "status": incident.status.value,
                "severity": incident.severity.value,
                "service_name": incident.service_name,
                "region": incident.region,
                "timestamp": incident.timestamp.isoformat(),
                "mttr_minutes": incident.mttr_minutes,
                "created_by": incident.created_by
            }
            incident_list.append(incident_data)
        
        return {
            "incidents": incident_list,
            "total_count": len(incident_list),
            "limit": limit,
            "offset": offset,
            "filters": {
                "status": status.value if status else None,
                "severity": severity.value if severity else None,
                "service": service
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing incidents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list incidents: {str(e)}"
        )


@router.get("/{incident_id}", response_model=Dict[str, Any])
async def get_incident(
    incident_id: str = Path(..., description="Incident ID"),
    include_analysis: bool = Query(False, description="Include latest analysis"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed information about a specific incident."""
    
    logger.info(f"Getting incident details: {incident_id}")
    
    try:
        incident = await _get_incident_by_id(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Format incident response
        # Convert Incident object to dictionary first
        incident_dict = {
            'incident_id': incident.incident_id,
            'title': incident.title,
            'description': incident.description,
            'severity': incident.severity.value if incident.severity else 'medium',
            'status': incident.status.value if incident.status else 'open',
            'service_name': incident.service_name,
            'region': incident.region,
            'created_at': incident.created_at.isoformat() if incident.created_at else None,
            'updated_at': incident.updated_at.isoformat() if incident.updated_at else None,
            'resolved_at': incident.resolution_time.isoformat() if incident.resolution_time else None,
            'mttr_minutes': incident.mttr_minutes,
            'affected_users': incident.affected_users,
            'tags': incident.tags if hasattr(incident, 'tags') else [],
            'symptoms': incident.symptoms if hasattr(incident, 'symptoms') and incident.symptoms else [],
            'root_cause': incident.root_cause,
            'resolution': incident.resolution,
            'confidence_score': incident.confidence_score
        }
        incident_data = format_incident_response(incident_dict)
        
        # Include analysis if requested
        if include_analysis:
            try:
                latest_analysis = await incident_analyzer.get_latest_analysis(incident_id)
                if latest_analysis:
                    incident_data["latest_analysis"] = format_analysis_response(latest_analysis)
            except Exception as e:
                logger.warning(f"Failed to get latest analysis: {e}")
                incident_data["latest_analysis"] = None
        
        return incident_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get incident: {str(e)}"
        )


@router.put("/{incident_id}", response_model=Dict[str, Any])
async def update_incident(
    incident_id: str = Path(..., description="Incident ID"),
    incident_update: IncidentUpdate = Body(...),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update an existing incident."""
    
    logger.info(f"Updating incident: {incident_id}")
    
    try:
        incident = await _get_incident_by_id(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Check permissions
        if not user.can_modify_incidents():
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Apply updates
        update_data = incident_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(incident, field):
                setattr(incident, field, value)
        
        incident.updated_at = datetime.utcnow()
        incident.updated_by = user.user_id
        
        # Handle status changes
        if incident_update.status == IncidentStatus.RESOLVED and not incident.resolved_at:
            incident.resolved_at = datetime.utcnow()
            if incident.timestamp:
                incident.mttr_minutes = int((incident.resolved_at - incident.timestamp).total_seconds() / 60)
        
        # Save to database (mock implementation)
        
        return {
            "incident_id": incident.incident_id,
            "status": incident.status.value,
            "updated_at": incident.updated_at.isoformat(),
            "updated_by": incident.updated_by,
            "changes": list(update_data.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update incident: {str(e)}"
        )


@router.delete("/{incident_id}")
async def delete_incident(
    incident_id: str = Path(..., description="Incident ID"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete an incident (admin only)."""
    
    logger.info(f"Deleting incident: {incident_id}")
    
    try:
        if not user.is_admin():
            raise HTTPException(status_code=403, detail="Admin permissions required")
        
        incident = await _get_incident_by_id(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Soft delete (mark as deleted)
        incident.status = IncidentStatus.CLOSED
        incident.deleted_at = datetime.utcnow()
        incident.deleted_by = user.user_id
        
        # Save to database (mock implementation)
        
        return {
            "incident_id": incident_id,
            "status": "deleted",
            "deleted_at": incident.deleted_at.isoformat(),
            "deleted_by": incident.deleted_by
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete incident: {str(e)}"
        )


# Helper functions

async def _get_incident_by_id(incident_id: str) -> Optional[Incident]:
    """Get incident by ID from database."""
    # Check storage first for newly created incidents
    if incident_id in _incident_storage:
        return _incident_storage[incident_id]
    
    # Mock implementation for demo incidents
    if incident_id == "INC-2024-001":
        return Incident(
            id=uuid4(),
            incident_id=incident_id,
            title="Payment API Latency Spike",
            description="High latency detected in payment API",
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.INVESTIGATING,
            service_name="payment-api",
            region="us-central1",
            symptoms=["API latency > 5s", "Cache hit rate < 40%"],
            timestamp=datetime.utcnow() - timedelta(hours=2),
            created_at=datetime.utcnow() - timedelta(hours=2),
            created_by="sre-team"
        )
    elif incident_id == "INC-2024-002":
        return Incident(
            id=uuid4(),
            incident_id=incident_id,
            title="Database Connection Timeout",
            description="Users reporting database timeout errors",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.INVESTIGATING,
            service_name="billing-service",
            region="us-central1",
            symptoms=["Database timeout errors", "Connection pool exhausted"],
            timestamp=datetime.utcnow() - timedelta(minutes=30),
            created_at=datetime.utcnow() - timedelta(minutes=30),
            created_by="monitoring-system"
        )
    
    return None


async def _get_incidents_with_filters(
    status: Optional[IncidentStatus] = None,
    severity: Optional[IncidentSeverity] = None,
    service: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Incident]:
    """Get incidents with filters applied."""
    # Get demo incidents
    mock_incidents = [
        await _get_incident_by_id("INC-2024-001"),
        await _get_incident_by_id("INC-2024-002")
    ]
    
    # Add incidents from storage
    storage_incidents = list(_incident_storage.values())
    
    # Combine all incidents
    incidents = [inc for inc in mock_incidents if inc is not None] + storage_incidents
    
    # Apply filters
    if status:
        incidents = [inc for inc in incidents if inc.status == status]
    if severity:
        incidents = [inc for inc in incidents if inc.severity == severity]
    if service:
        incidents = [inc for inc in incidents if inc.service_name == service]
    
    # Apply pagination
    return incidents[offset:offset + limit]


async def _get_recent_incident_activity(incident_id: str) -> List[Dict[str, Any]]:
    """Get recent activity for an incident."""
    # Mock implementation
    
    return [
        {
            "timestamp": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
            "type": "analysis_started",
            "description": "AI analysis initiated",
            "user": "system"
        },
        {
            "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            "type": "correlation_found",
            "description": "Found 3 similar historical incidents",
            "user": "ai-agent"
        }
    ]


async def _analyze_correlation_factors(
    incident: Incident,
    similar_incident: Incident
) -> List[Dict[str, Any]]:
    """Analyze correlation factors between incidents."""
    
    factors = []
    
    # Service correlation
    if incident.service_name == similar_incident.service_name:
        factors.append({
            "factor": "service_match",
            "strength": 1.0,
            "description": f"Both incidents affect {incident.service_name}"
        })
    
    # Severity correlation
    if incident.severity == similar_incident.severity:
        factors.append({
            "factor": "severity_match",
            "strength": 0.8,
            "description": f"Both incidents have {incident.severity.value} severity"
        })
    
    # Symptom correlation
    common_symptoms = set(incident.symptoms) & set(similar_incident.symptoms)
    if common_symptoms:
        factors.append({
            "factor": "symptom_overlap",
            "strength": len(common_symptoms) / max(len(incident.symptoms), len(similar_incident.symptoms)),
            "description": f"Common symptoms: {', '.join(common_symptoms)}"
        })
    
    return factors


async def _generate_correlation_recommendations(
    incident: Incident,
    similar_incidents: List[tuple]
) -> List[str]:
    """Generate recommendations based on incident correlations."""
    
    recommendations = []
    
    if similar_incidents:
        # Check for common root causes
        root_causes = [inc.root_cause for inc, _ in similar_incidents if inc.root_cause]
        if root_causes:
            most_common = max(set(root_causes), key=root_causes.count)
            recommendations.append(f"Consider investigating: {most_common} (common in similar incidents)")
        
        # Check for fast resolutions
        fast_resolutions = [inc for inc, _ in similar_incidents if inc.mttr_minutes and inc.mttr_minutes < 30]
        if fast_resolutions:
            recommendations.append("Review fast resolution patterns from similar incidents")
        
        # Service-specific recommendations
        if incident.service_name == "payment-api":
            recommendations.append("Check Redis cache performance and database connection pool")
        elif incident.service_name == "billing-service":
            recommendations.append("Investigate database query performance and connection timeouts")
    
    return recommendations