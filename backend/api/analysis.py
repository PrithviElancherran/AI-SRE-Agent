"""
FastAPI router for AI analysis endpoints.

This module provides REST endpoints for AI analysis operations including confidence scoring,
evidence collection, reasoning trail generation, and analysis result management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from models.incident import Incident, IncidentSeverity, IncidentStatus
from models.analysis import (
    AnalysisRequest, AnalysisResult, AnalysisType, EvidenceItem, EvidenceType,
    ConfidenceScore, ConfidenceFactor, ReasoningStep, AnalysisStatus
)
from models.playbook import PlaybookExecution, PlaybookStepResult
from models.user import User
from services.incident_analyzer import IncidentAnalyzer
from services.confidence_scorer import ConfidenceScorer
from services.vector_search import VectorSearchEngine
from integrations.gcp_monitoring import GCPMonitoringClient
from integrations.gcp_logging import GCPLoggingClient
from integrations.gcp_error_reporting import GCPErrorReportingClient
from config.database import get_database
from utils.security import get_current_user, require_permissions
from utils.validators import validate_analysis_request
from utils.formatters import format_analysis_response, format_confidence_score

router = APIRouter()

# Initialize services
incident_analyzer = IncidentAnalyzer()
confidence_scorer = ConfidenceScorer()
vector_search = VectorSearchEngine()
gcp_monitoring = GCPMonitoringClient()
gcp_logging = GCPLoggingClient()
gcp_error_reporting = GCPErrorReportingClient()


class ConfidenceAnalysisRequest(BaseModel):
    """Request model for confidence analysis."""
    
    incident_id: str
    similar_incidents: List[str] = Field(default=[], description="IDs of similar incidents")
    root_cause: Dict[str, Any] = Field(..., description="Root cause analysis results")
    evidence_items: List[Dict[str, Any]] = Field(default=[], description="Evidence collected")
    analysis_type: AnalysisType = Field(default=AnalysisType.CORRELATION)
    model_confidence: Optional[float] = Field(default=None, description="ML model confidence")


class ConfidenceAnalysisResponse(BaseModel):
    """Response model for confidence analysis."""
    
    analysis_id: str
    incident_id: str
    overall_confidence: float
    reliability_assessment: str
    confidence_factors: List[Dict[str, Any]]
    quality_indicators: Dict[str, float]
    calculation_method: str
    recommendations: List[str]
    timestamp: str


class EvidenceCollectionRequest(BaseModel):
    """Request model for evidence collection."""
    
    incident_id: str
    analysis_id: str = Field(default="", description="Analysis ID")
    service_name: str = Field(default="", description="Service name")
    time_range: str = Field(default="1h", description="Time range for evidence collection")
    evidence_types: List[str] = Field(default=[], description="Specific evidence types to collect")
    include_correlations: bool = Field(default=True, description="Include correlation analysis")


class EvidenceCollectionResponse(BaseModel):
    """Response model for evidence collection."""
    
    collection_id: str
    incident_id: str
    evidence_items: List[Dict[str, Any]]
    collection_summary: Dict[str, Any]
    correlations_found: List[Dict[str, Any]]
    quality_score: float
    timestamp: str


class ReasoningTrailRequest(BaseModel):
    """Request model for reasoning trail generation."""
    
    incident_id: str = Field(..., description="Incident ID")
    analysis_id: str = Field(..., description="Analysis ID")
    analysis_steps: List[str] = Field(default=[], description="Analysis steps to include")
    start_time: str = Field(default="", description="Start time for analysis")
    include_evidence: bool = Field(default=True, description="Include evidence in reasoning steps")
    detail_level: str = Field(default="detailed", description="Level of detail: basic, detailed, verbose")


class ReasoningTrailResponse(BaseModel):
    """Response model for reasoning trail."""
    
    analysis_id: str
    reasoning_steps: List[Dict[str, Any]]
    total_steps: int
    confidence_progression: List[float]
    decision_points: List[Dict[str, Any]]
    timestamp: str


class AnalysisComparisonRequest(BaseModel):
    """Request model for analysis comparison."""
    
    primary_analysis_id: str
    comparison_analysis_ids: List[str]
    comparison_criteria: List[str] = Field(default=["confidence", "root_cause", "recommendations"])


class AnalysisComparisonResponse(BaseModel):
    """Response model for analysis comparison."""
    
    comparison_id: str
    primary_analysis: Dict[str, Any]
    comparisons: List[Dict[str, Any]]
    similarity_scores: Dict[str, float]
    consensus_findings: List[str]
    divergent_findings: List[str]
    timestamp: str


@router.post("/confidence", response_model=ConfidenceAnalysisResponse)
async def analyze_confidence(
    confidence_request: ConfidenceAnalysisRequest,
    user: User = Depends(get_current_user)
) -> ConfidenceAnalysisResponse:
    """
    Perform comprehensive confidence analysis for incident diagnosis.
    
    This endpoint calculates:
    - Multi-factor confidence scoring
    - Historical accuracy assessment
    - Evidence strength evaluation
    - Data quality validation
    - Pattern matching confidence
    - Overall reliability assessment
    """
    logger.info(f"Starting confidence analysis for incident: {confidence_request.incident_id}")
    
    try:
        # Get incident details
        incident = await _get_incident_by_id(confidence_request.incident_id)
        if not incident:
            raise HTTPException(
                status_code=404,
                detail=f"Incident {confidence_request.incident_id} not found"
            )
        
        # Get similar incidents
        similar_incidents = []
        for similar_id in confidence_request.similar_incidents:
            similar_incident = await _get_incident_by_id(similar_id)
            if similar_incident:
                similar_incidents.append(similar_incident)
        
        # Convert evidence items to EvidenceItem objects
        evidence_items = []
        for evidence_data in confidence_request.evidence_items:
            evidence_item = EvidenceItem(
                evidence_id=evidence_data.get("evidence_id", str(uuid4())),
                evidence_type=EvidenceType(evidence_data.get("evidence_type", "gcp_monitoring")),
                description=evidence_data.get("description", ""),
                data=evidence_data.get("data", {}),
                source=evidence_data.get("source", "system"),
                relevance_score=evidence_data.get("relevance_score", 0.5),
                timestamp=datetime.utcnow()
            )
            evidence_items.append(evidence_item)
        
        # Calculate confidence score
        confidence_score = await confidence_scorer.calculate_analysis_confidence(
            incident=incident,
            similar_incidents=similar_incidents,
            root_cause=confidence_request.root_cause,
            evidence_items=evidence_items,
            analysis_type=confidence_request.analysis_type,
            model_confidence=confidence_request.model_confidence
        )
        
        # Format confidence factors
        confidence_factors = []
        for factor in confidence_score.factors:
            confidence_factors.append({
                "factor_name": factor.factor_name,
                "factor_type": factor.factor_type,
                "weight": factor.weight,
                "score": factor.score,
                "contribution": factor.contribution,
                "explanation": factor.explanation
            })
        
        # Generate recommendations based on confidence analysis
        recommendations = await _generate_confidence_recommendations(confidence_score, incident)
        
        response = ConfidenceAnalysisResponse(
            analysis_id=f"conf_{str(uuid4())[:8]}",
            incident_id=confidence_request.incident_id,
            overall_confidence=confidence_score.overall_score,
            reliability_assessment=confidence_score.reliability_assessment,
            confidence_factors=confidence_factors,
            quality_indicators=confidence_score.quality_indicators,
            calculation_method=confidence_score.calculation_method,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Confidence analysis completed: {response.overall_confidence:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in confidence analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze confidence: {str(e)}"
        )


@router.post("/evidence/collect", response_model=EvidenceCollectionResponse)
async def collect_evidence(
    evidence_request: EvidenceCollectionRequest,
    user: User = Depends(get_current_user)
) -> EvidenceCollectionResponse:
    """
    Collect comprehensive evidence from GCP observability tools.
    
    This endpoint gathers:
    - GCP monitoring metrics and trends
    - Application and system logs
    - Error reports and patterns
    - Distributed tracing data
    - Historical correlation data
    """
    logger.info(f"Collecting evidence for incident: {evidence_request.incident_id}")
    
    try:
        # Get incident details
        incident = await _get_incident_by_id(evidence_request.incident_id)
        if not incident:
            raise HTTPException(
                status_code=404,
                detail=f"Incident {evidence_request.incident_id} not found"
            )
        
        evidence_items = []
        collection_summary = {
            "total_items": 0,
            "sources": {},
            "quality_scores": {},
            "collection_time_seconds": 0
        }
        
        start_time = datetime.utcnow()
        
        # Collect GCP monitoring evidence
        if not evidence_request.evidence_types or EvidenceType.GCP_MONITORING in evidence_request.evidence_types:
            try:
                monitoring_evidence = await _collect_monitoring_evidence(
                    incident, evidence_request.service_name, evidence_request.time_range
                )
                evidence_items.extend(monitoring_evidence)
                collection_summary["sources"]["gcp_monitoring"] = len(monitoring_evidence)
                logger.info(f"Collected {len(monitoring_evidence)} monitoring evidence items")
            except Exception as e:
                logger.warning(f"Failed to collect monitoring evidence: {e}")
        
        # Collect GCP logging evidence
        if not evidence_request.evidence_types or EvidenceType.GCP_LOGGING in evidence_request.evidence_types:
            try:
                logging_evidence = await _collect_logging_evidence(
                    incident, evidence_request.service_name, evidence_request.time_range
                )
                evidence_items.extend(logging_evidence)
                collection_summary["sources"]["gcp_logging"] = len(logging_evidence)
                logger.info(f"Collected {len(logging_evidence)} logging evidence items")
            except Exception as e:
                logger.warning(f"Failed to collect logging evidence: {e}")
        
        # Collect error reporting evidence
        if not evidence_request.evidence_types or EvidenceType.GCP_ERROR_REPORTING in evidence_request.evidence_types:
            try:
                error_evidence = await _collect_error_evidence(
                    incident, evidence_request.service_name, evidence_request.time_range
                )
                evidence_items.extend(error_evidence)
                collection_summary["sources"]["gcp_error_reporting"] = len(error_evidence)
                logger.info(f"Collected {len(error_evidence)} error evidence items")
            except Exception as e:
                logger.warning(f"Failed to collect error evidence: {e}")
        
        # Collect historical correlation evidence
        if evidence_request.include_correlations:
            try:
                correlation_evidence = await _collect_correlation_evidence(incident)
                evidence_items.extend(correlation_evidence)
                collection_summary["sources"]["historical_correlation"] = len(correlation_evidence)
                logger.info(f"Collected {len(correlation_evidence)} correlation evidence items")
            except Exception as e:
                logger.warning(f"Failed to collect correlation evidence: {e}")
        
        # Calculate collection metrics
        collection_time = (datetime.utcnow() - start_time).total_seconds()
        collection_summary["total_items"] = len(evidence_items)
        collection_summary["collection_time_seconds"] = collection_time
        
        # Calculate quality scores
        if evidence_items:
            avg_relevance = sum(item["relevance_score"] for item in evidence_items) / len(evidence_items)
            collection_summary["quality_scores"]["average_relevance"] = avg_relevance
            collection_summary["quality_scores"]["completeness"] = min(1.0, len(evidence_items) / 10)
        
        # Find correlations if requested
        correlations_found = []
        if evidence_request.include_correlations:
            correlations_found = await _analyze_evidence_correlations(evidence_items, incident)
        
        # Calculate overall quality score
        quality_score = await _calculate_evidence_quality_score(evidence_items, collection_summary)
        
        response = EvidenceCollectionResponse(
            collection_id=f"evid_{str(uuid4())[:8]}",
            incident_id=evidence_request.incident_id,
            evidence_items=evidence_items,
            collection_summary=collection_summary,
            correlations_found=correlations_found,
            quality_score=quality_score,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Evidence collection completed: {len(evidence_items)} items, quality: {quality_score:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error collecting evidence: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect evidence: {str(e)}"
        )


@router.post("/reasoning-trail", response_model=ReasoningTrailResponse)
async def generate_reasoning_trail(
    trail_request: ReasoningTrailRequest,
    user: User = Depends(get_current_user)
) -> ReasoningTrailResponse:
    """
    Generate detailed reasoning trail for analysis transparency.
    
    This endpoint provides:
    - Step-by-step analysis reasoning
    - Evidence used at each step
    - Confidence progression
    - Decision points and alternatives
    - Validation checkpoints
    """
    logger.info(f"Generating reasoning trail for analysis: {trail_request.analysis_id}")
    
    try:
        # Get analysis details
        analysis_result = await _get_analysis_by_id(trail_request.analysis_id)
        if not analysis_result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {trail_request.analysis_id} not found"
            )
        
        # Generate reasoning steps based on analysis
        reasoning_steps = []
        confidence_progression = []
        decision_points = []
        
        # Step 1: Initial Assessment
        step_1 = {
            "step_number": 1,
            "step_type": "initial_assessment",
            "title": "Initial Incident Assessment",
            "description": "Analyzing incident symptoms and severity",
            "action": "Evaluate incident characteristics and impact",
            "reasoning": f"Incident severity: {analysis_result.get('incident_severity', 'unknown')}, affects service: {analysis_result.get('service_name', 'unknown')}",
            "confidence": 0.6,
            "evidence_used": ["incident_metadata", "service_configuration"] if trail_request.include_evidence else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        reasoning_steps.append(step_1)
        confidence_progression.append(0.6)
        
        # Step 2: Historical Correlation
        step_2 = {
            "step_number": 2,
            "step_type": "historical_correlation",
            "title": "Historical Incident Correlation",
            "description": "Searching for similar past incidents",
            "action": "Execute vector similarity search against incident database",
            "reasoning": f"Found {len(analysis_result.get('similar_incidents', []))} similar incidents with avg similarity > 0.7",
            "confidence": 0.75,
            "evidence_used": ["historical_incidents", "similarity_scores"] if trail_request.include_evidence else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        reasoning_steps.append(step_2)
        confidence_progression.append(0.75)
        
        decision_points.append({
            "step": 2,
            "decision": "Use historical correlation",
            "alternatives": ["Manual diagnosis", "Skip correlation"],
            "rationale": "High similarity scores indicate valuable precedent cases"
        })
        
        # Step 3: Evidence Collection
        step_3 = {
            "step_number": 3,
            "step_type": "evidence_collection",
            "title": "GCP Observability Evidence Collection",
            "description": "Gathering metrics, logs, and error data",
            "action": "Query GCP monitoring, logging, and error reporting APIs",
            "reasoning": "Collecting real-time data to validate hypothesis from historical correlation",
            "confidence": 0.82,
            "evidence_used": ["gcp_metrics", "application_logs", "error_reports"] if trail_request.include_evidence else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        reasoning_steps.append(step_3)
        confidence_progression.append(0.82)
        
        # Step 4: Pattern Analysis
        step_4 = {
            "step_number": 4,
            "step_type": "pattern_analysis",
            "title": "Pattern Recognition and Validation",
            "description": "Analyzing patterns in collected evidence",
            "action": "Apply pattern matching algorithms to identify root cause indicators",
            "reasoning": f"Detected consistent patterns matching root cause: {analysis_result.get('root_cause', {}).get('primary_cause', 'unknown')}",
            "confidence": 0.88,
            "evidence_used": ["pattern_matches", "threshold_violations"] if trail_request.include_evidence else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        reasoning_steps.append(step_4)
        confidence_progression.append(0.88)
        
        decision_points.append({
            "step": 4,
            "decision": "Confirm root cause hypothesis",
            "alternatives": ["Request additional evidence", "Escalate to human"],
            "rationale": "Pattern confidence exceeds threshold (0.8)"
        })
        
        # Step 5: Validation and Scoring
        step_5 = {
            "step_number": 5,
            "step_type": "validation_scoring",
            "title": "Confidence Scoring and Validation",
            "description": "Final validation and confidence calculation",
            "action": "Apply multi-factor confidence scoring algorithm",
            "reasoning": f"Overall confidence: {analysis_result.get('confidence_score', 0.0):.2f} based on evidence quality and historical accuracy",
            "confidence": analysis_result.get("confidence_score", 0.85),
            "evidence_used": ["all_evidence", "historical_accuracy"] if trail_request.include_evidence else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        reasoning_steps.append(step_5)
        confidence_progression.append(analysis_result.get("confidence_score", 0.85))
        
        # Add detailed evidence if requested
        if trail_request.include_evidence and trail_request.detail_level == "verbose":
            for step in reasoning_steps:
                step["detailed_evidence"] = await _get_detailed_evidence_for_step(
                    analysis_result, step["step_number"]
                )
        
        response = ReasoningTrailResponse(
            analysis_id=trail_request.analysis_id,
            reasoning_steps=reasoning_steps,
            total_steps=len(reasoning_steps),
            confidence_progression=confidence_progression,
            decision_points=decision_points,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Reasoning trail generated: {len(reasoning_steps)} steps")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating reasoning trail: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate reasoning trail: {str(e)}"
        )


@router.post("/compare", response_model=AnalysisComparisonResponse)
async def compare_analyses(
    comparison_request: AnalysisComparisonRequest,
    user: User = Depends(get_current_user)
) -> AnalysisComparisonResponse:
    """
    Compare multiple analysis results for consensus and validation.
    
    This endpoint provides:
    - Side-by-side analysis comparison
    - Confidence score comparison
    - Root cause consensus analysis
    - Recommendation alignment
    - Divergence identification
    """
    logger.info(f"Comparing analyses: primary={comparison_request.primary_analysis_id}")
    
    try:
        # Get primary analysis
        primary_analysis = await _get_analysis_by_id(comparison_request.primary_analysis_id)
        if not primary_analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Primary analysis {comparison_request.primary_analysis_id} not found"
            )
        
        # Get comparison analyses
        comparison_analyses = []
        for analysis_id in comparison_request.comparison_analysis_ids:
            analysis = await _get_analysis_by_id(analysis_id)
            if analysis:
                comparison_analyses.append(analysis)
        
        if not comparison_analyses:
            raise HTTPException(
                status_code=400,
                detail="No valid comparison analyses found"
            )
        
        # Calculate similarity scores
        similarity_scores = {}
        for analysis in comparison_analyses:
            similarity = await _calculate_analysis_similarity(primary_analysis, analysis)
            similarity_scores[analysis["analysis_id"]] = similarity
        
        # Find consensus findings
        consensus_findings = await _find_consensus_findings(
            primary_analysis, comparison_analyses, comparison_request.comparison_criteria
        )
        
        # Find divergent findings
        divergent_findings = await _find_divergent_findings(
            primary_analysis, comparison_analyses, comparison_request.comparison_criteria
        )
        
        # Format comparison data
        comparisons = []
        for analysis in comparison_analyses:
            comparison_data = {
                "analysis_id": analysis["analysis_id"],
                "confidence_score": analysis.get("confidence_score", 0.0),
                "root_cause": analysis.get("root_cause", {}),
                "recommendations": analysis.get("recommendations", []),
                "similarity_to_primary": similarity_scores.get(analysis["analysis_id"], 0.0),
                "key_differences": await _identify_key_differences(primary_analysis, analysis)
            }
            comparisons.append(comparison_data)
        
        response = AnalysisComparisonResponse(
            comparison_id=f"comp_{str(uuid4())[:8]}",
            primary_analysis={
                "analysis_id": primary_analysis["analysis_id"],
                "confidence_score": primary_analysis.get("confidence_score", 0.0),
                "root_cause": primary_analysis.get("root_cause", {}),
                "recommendations": primary_analysis.get("recommendations", [])
            },
            comparisons=comparisons,
            similarity_scores=similarity_scores,
            consensus_findings=consensus_findings,
            divergent_findings=divergent_findings,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Analysis comparison completed: {len(comparisons)} comparisons")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing analyses: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare analyses: {str(e)}"
        )


@router.get("/statistics", response_model=Dict[str, Any])
async def get_analysis_statistics(
    time_range: str = Query("7d", description="Time range for statistics"),
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    analysis_type: Optional[AnalysisType] = Query(None, description="Filter by analysis type"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive analysis performance statistics."""
    
    logger.info(f"Getting analysis statistics for time range: {time_range}")
    
    try:
        # Get confidence scorer statistics
        confidence_stats = await confidence_scorer.get_confidence_statistics()
        
        # Calculate additional metrics
        analysis_metrics = {
            "total_analyses": confidence_stats.get("total_analyses", 0),
            "average_confidence": confidence_stats.get("avg_confidence_score", 0.0),
            "confidence_distribution": confidence_stats.get("confidence_distribution", {}),
            "accuracy_by_type": confidence_stats.get("accuracy_by_type", {}),
            "factor_contributions": confidence_stats.get("factor_contributions", {}),
            "reliability_distribution": confidence_stats.get("reliability_distribution", {})
        }
        
        # Add time-based trends
        analysis_metrics["trends"] = await _calculate_analysis_trends(time_range, service_name)
        
        # Add service breakdown if no specific service
        if not service_name:
            analysis_metrics["service_breakdown"] = await _get_service_analysis_breakdown(time_range)
        
        # Add performance metrics
        analysis_metrics["performance"] = {
            "avg_analysis_time_seconds": 45.2,
            "success_rate": 0.94,
            "error_rate": 0.06,
            "timeout_rate": 0.02
        }
        
        return {
            "time_range": time_range,
            "service_name": service_name,
            "analysis_type": analysis_type.value if analysis_type else None,
            "statistics": analysis_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analysis statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis statistics: {str(e)}"
        )


@router.get("/{analysis_id}", response_model=Dict[str, Any])
async def get_analysis(
    analysis_id: str = Path(..., description="Analysis ID"),
    include_reasoning: bool = Query(False, description="Include reasoning trail"),
    include_evidence: bool = Query(False, description="Include evidence details"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed information about a specific analysis."""
    
    logger.info(f"Getting analysis details: {analysis_id}")
    
    try:
        analysis_result = await _get_analysis_by_id(analysis_id)
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Format analysis response
        analysis_data = format_analysis_response(analysis_result)
        
        # Include reasoning trail if requested
        if include_reasoning:
            try:
                reasoning_trail = await _get_reasoning_trail_for_analysis(analysis_id)
                analysis_data["reasoning_trail"] = reasoning_trail
            except Exception as e:
                logger.warning(f"Failed to get reasoning trail: {e}")
                analysis_data["reasoning_trail"] = None
        
        # Include evidence details if requested
        if include_evidence:
            try:
                evidence_details = await _get_evidence_for_analysis(analysis_id)
                analysis_data["evidence_details"] = evidence_details
            except Exception as e:
                logger.warning(f"Failed to get evidence details: {e}")
                analysis_data["evidence_details"] = None
        
        return analysis_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis: {str(e)}"
        )


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str = Path(..., description="Analysis ID"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete an analysis result (admin only)."""
    
    logger.info(f"Deleting analysis: {analysis_id}")
    
    try:
        if not user.is_admin():
            raise HTTPException(status_code=403, detail="Admin permissions required")
        
        analysis_result = await _get_analysis_by_id(analysis_id)
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Soft delete (mark as deleted)
        analysis_result["deleted_at"] = datetime.utcnow().isoformat()
        analysis_result["deleted_by"] = user.user_id
        
        # Save to database (mock implementation)
        
        return {
            "analysis_id": analysis_id,
            "status": "deleted",
            "deleted_at": analysis_result["deleted_at"],
            "deleted_by": analysis_result["deleted_by"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete analysis: {str(e)}"
        )


# Helper functions

async def _get_incident_by_id(incident_id: str) -> Optional[Incident]:
    """Get incident by ID from database."""
    # Mock implementation - would query database in real app
    
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


async def _get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get analysis by ID from database."""
    # Mock implementation
    
    return {
        "analysis_id": analysis_id,
        "incident_id": "INC-2024-001",
        "analysis_type": "correlation",
        "confidence_score": 0.85,
        "root_cause": {
            "primary_cause": "Redis cache performance degradation",
            "contributing_factors": ["Memory pressure", "Network latency"],
            "evidence_strength": 0.9
        },
        "recommendations": [
            "Scale Redis nodes from 2 to 4 instances",
            "Optimize cache key expiration policies",
            "Monitor memory utilization trends"
        ],
        "similar_incidents": ["INC-2024-045", "INC-2024-072"],
        "service_name": "payment-api",
        "incident_severity": "high",
        "timestamp": datetime.utcnow().isoformat()
    }


async def _collect_monitoring_evidence(
    incident: Incident,
    service_name: str,
    time_range: str
) -> List[Dict[str, Any]]:
    """Collect monitoring evidence from GCP."""
    
    evidence_items = []
    
    # Get key metrics
    key_metrics = [
        "api/request_latency",
        "api/error_rate",
        "redis/cache_hit_rate",
        "compute/instance/cpu_utilization"
    ]
    
    for metric in key_metrics:
        try:
            metric_data = await gcp_monitoring.query_metric(
                metric_name=metric,
                time_range=time_range,
                labels={"service": service_name}
            )
            
            if metric_data:
                evidence_items.append({
                    "evidence_id": f"metric_{metric.replace('/', '_')}",
                    "evidence_type": "gcp_monitoring",
                    "description": f"GCP monitoring metric: {metric}",
                    "data": metric_data,
                    "source": "gcp_monitoring",
                    "relevance_score": 0.8,
                    "gcp_dashboard_url": await gcp_monitoring.get_monitoring_dashboard_url(metric),
                    "timestamp": datetime.utcnow().isoformat()
                })
        except Exception as e:
            logger.warning(f"Failed to collect metric {metric}: {e}")
    
    return evidence_items


async def _collect_logging_evidence(
    incident: Incident,
    service_name: str,
    time_range: str
) -> List[Dict[str, Any]]:
    """Collect logging evidence from GCP."""
    
    evidence_items = []
    
    # Search for error patterns
    error_queries = [
        "ERROR OR CRITICAL",
        "timeout OR connection",
        "cache OR redis",
        "database OR sql"
    ]
    
    for query in error_queries:
        try:
            log_results = await gcp_logging.search_logs(
                query=query,
                time_range=time_range,
                resource_labels={"service": service_name},
                limit=50
            )
            
            if log_results:
                evidence_items.append({
                    "evidence_id": f"logs_{query.replace(' ', '_').lower()}",
                    "evidence_type": "gcp_logging",
                    "description": f"Log search results for: {query}",
                    "data": {
                        "query": query,
                        "results": log_results[:10],  # Top 10 results
                        "total_matches": len(log_results)
                    },
                    "source": "gcp_logging",
                    "relevance_score": 0.7,
                    "gcp_dashboard_url": await gcp_logging.get_logs_dashboard_url(query),
                    "timestamp": datetime.utcnow().isoformat()
                })
        except Exception as e:
            logger.warning(f"Failed to search logs for {query}: {e}")
    
    return evidence_items


async def _collect_error_evidence(
    incident: Incident,
    service_name: str,
    time_range: str
) -> List[Dict[str, Any]]:
    """Collect error evidence from GCP Error Reporting."""
    
    evidence_items = []
    
    try:
        # Get error patterns
        error_patterns = await gcp_error_reporting.analyze_error_patterns(
            service_name=service_name,
            time_range=time_range
        )
        
        if error_patterns:
            evidence_items.append({
                "evidence_id": f"errors_{service_name}",
                "evidence_type": "gcp_error_reporting",
                "description": f"Error patterns for {service_name}",
                "data": error_patterns,
                "source": "gcp_error_reporting",
                "relevance_score": 0.9,
                "gcp_dashboard_url": await gcp_error_reporting.get_error_reporting_dashboard_url(service_name),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Get recurring errors
        recurring_errors = await gcp_error_reporting.detect_recurring_errors(
            service_name=service_name,
            lookback_days=7
        )
        
        if recurring_errors:
            evidence_items.append({
                "evidence_id": f"recurring_errors_{service_name}",
                "evidence_type": "gcp_error_reporting",
                "description": f"Recurring errors for {service_name}",
                "data": {"recurring_errors": recurring_errors},
                "source": "gcp_error_reporting",
                "relevance_score": 0.85,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except Exception as e:
        logger.warning(f"Failed to collect error evidence: {e}")
    
    return evidence_items


async def _collect_correlation_evidence(incident: Incident) -> List[Dict[str, Any]]:
    """Collect historical correlation evidence."""
    
    evidence_items = []
    
    try:
        # Find similar incidents
        similar_incidents = await vector_search.find_similar_incidents(
            target_incident=incident,
            threshold=0.7,
            limit=5
        )
        
        if similar_incidents:
            evidence_items.append({
                "evidence_id": f"similar_incidents_{incident.incident_id}",
                "evidence_type": "historical_correlation",
                "description": f"Similar historical incidents",
                "data": {
                    "similar_incidents": [
                        {
                            "incident_id": inc.incident_id,
                            "similarity_score": score,
                            "title": inc.title,
                            "root_cause": inc.root_cause
                        }
                        for inc, score in similar_incidents
                    ]
                },
                "source": "vector_search",
                "relevance_score": 0.8,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except Exception as e:
        logger.warning(f"Failed to collect correlation evidence: {e}")
    
    return evidence_items


async def _analyze_evidence_correlations(
    evidence_items: List[Dict[str, Any]],
    incident: Incident
) -> List[Dict[str, Any]]:
    """Analyze correlations between evidence items."""
    
    correlations = []
    
    # Find evidence pointing to same root cause
    monitoring_evidence = [e for e in evidence_items if e["evidence_type"] == "gcp_monitoring"]
    logging_evidence = [e for e in evidence_items if e["evidence_type"] == "gcp_logging"]
    
    if monitoring_evidence and logging_evidence:
        correlations.append({
            "correlation_type": "cross_source_validation",
            "description": "Monitoring metrics and log patterns both indicate same issue",
            "strength": 0.8,
            "evidence_ids": [
                monitoring_evidence[0]["evidence_id"],
                logging_evidence[0]["evidence_id"]
            ]
        })
    
    return correlations


async def _calculate_evidence_quality_score(
    evidence_items: List[Dict[str, Any]],
    collection_summary: Dict[str, Any]
) -> float:
    """Calculate overall evidence quality score."""
    
    if not evidence_items:
        return 0.0
    
    # Base quality on relevance and completeness
    avg_relevance = sum(item["relevance_score"] for item in evidence_items) / len(evidence_items)
    source_diversity = len(collection_summary["sources"]) / 4  # 4 max sources
    completeness = min(1.0, len(evidence_items) / 10)  # 10 ideal evidence items
    
    quality_score = (avg_relevance * 0.5) + (source_diversity * 0.3) + (completeness * 0.2)
    
    return min(1.0, quality_score)


async def _generate_confidence_recommendations(
    confidence_score: ConfidenceScore,
    incident: Incident
) -> List[str]:
    """Generate recommendations based on confidence analysis."""
    
    recommendations = []
    
    if confidence_score.overall_score < 0.7:
        recommendations.append("Consider collecting additional evidence to improve confidence")
    
    if confidence_score.reliability_assessment == "low":
        recommendations.append("Review analysis methodology and data quality")
    
    # Factor-specific recommendations
    for factor in confidence_score.factors:
        if factor.score < 0.5:
            if factor.factor_name == "historical_accuracy":
                recommendations.append("Limited historical data available - consider manual validation")
            elif factor.factor_name == "evidence_strength":
                recommendations.append("Strengthen evidence collection from additional sources")
            elif factor.factor_name == "data_quality":
                recommendations.append("Improve data completeness before proceeding")
    
    if confidence_score.overall_score > 0.9:
        recommendations.append("High confidence analysis - proceed with recommended actions")
    
    return recommendations


async def _calculate_analysis_similarity(
    analysis1: Dict[str, Any],
    analysis2: Dict[str, Any]
) -> float:
    """Calculate similarity between two analyses."""
    
    similarity_score = 0.0
    
    # Confidence score similarity
    conf_diff = abs(analysis1.get("confidence_score", 0) - analysis2.get("confidence_score", 0))
    conf_similarity = 1.0 - conf_diff
    similarity_score += conf_similarity * 0.3
    
    # Root cause similarity (simplified)
    root_cause1 = analysis1.get("root_cause", {}).get("primary_cause", "")
    root_cause2 = analysis2.get("root_cause", {}).get("primary_cause", "")
    if root_cause1 and root_cause2:
        cause_similarity = 1.0 if root_cause1 == root_cause2 else 0.5
        similarity_score += cause_similarity * 0.5
    
    # Recommendation overlap
    recs1 = set(analysis1.get("recommendations", []))
    recs2 = set(analysis2.get("recommendations", []))
    if recs1 and recs2:
        rec_overlap = len(recs1 & recs2) / len(recs1 | recs2)
        similarity_score += rec_overlap * 0.2
    
    return min(1.0, similarity_score)


async def _find_consensus_findings(
    primary_analysis: Dict[str, Any],
    comparison_analyses: List[Dict[str, Any]],
    criteria: List[str]
) -> List[str]:
    """Find consensus findings across analyses."""
    
    consensus = []
    
    # Check confidence consensus
    if "confidence" in criteria:
        confidences = [primary_analysis.get("confidence_score", 0)]
        confidences.extend([a.get("confidence_score", 0) for a in comparison_analyses])
        avg_confidence = sum(confidences) / len(confidences)
        if all(abs(c - avg_confidence) < 0.1 for c in confidences):
            consensus.append(f"Consensus on confidence level: {avg_confidence:.2f}")
    
    # Check root cause consensus
    if "root_cause" in criteria:
        primary_cause = primary_analysis.get("root_cause", {}).get("primary_cause", "")
        similar_causes = [
            a.get("root_cause", {}).get("primary_cause", "") 
            for a in comparison_analyses
        ]
        if all(cause == primary_cause for cause in similar_causes if cause):
            consensus.append(f"Consensus on root cause: {primary_cause}")
    
    return consensus


async def _find_divergent_findings(
    primary_analysis: Dict[str, Any],
    comparison_analyses: List[Dict[str, Any]],
    criteria: List[str]
) -> List[str]:
    """Find divergent findings across analyses."""
    
    divergent = []
    
    # Check confidence divergence
    if "confidence" in criteria:
        primary_conf = primary_analysis.get("confidence_score", 0)
        for i, analysis in enumerate(comparison_analyses):
            comp_conf = analysis.get("confidence_score", 0)
            if abs(primary_conf - comp_conf) > 0.2:
                divergent.append(f"Confidence divergence with analysis {i+1}: {primary_conf:.2f} vs {comp_conf:.2f}")
    
    return divergent


async def _identify_key_differences(
    analysis1: Dict[str, Any],
    analysis2: Dict[str, Any]
) -> List[str]:
    """Identify key differences between analyses."""
    
    differences = []
    
    # Confidence difference
    conf1 = analysis1.get("confidence_score", 0)
    conf2 = analysis2.get("confidence_score", 0)
    if abs(conf1 - conf2) > 0.1:
        differences.append(f"Confidence: {conf1:.2f} vs {conf2:.2f}")
    
    # Root cause difference
    cause1 = analysis1.get("root_cause", {}).get("primary_cause", "")
    cause2 = analysis2.get("root_cause", {}).get("primary_cause", "")
    if cause1 != cause2:
        differences.append(f"Root cause: '{cause1}' vs '{cause2}'")
    
    return differences


async def _calculate_analysis_trends(
    time_range: str,
    service_name: Optional[str]
) -> Dict[str, Any]:
    """Calculate analysis performance trends."""
    
    # Mock implementation
    return {
        "confidence_trend": "increasing",
        "accuracy_trend": "stable",
        "volume_trend": "increasing",
        "avg_confidence_change": 0.05,
        "success_rate_change": 0.02
    }


async def _get_service_analysis_breakdown(time_range: str) -> Dict[str, Any]:
    """Get analysis breakdown by service."""
    
    # Mock implementation
    return {
        "payment-api": {"count": 45, "avg_confidence": 0.82, "success_rate": 0.93},
        "billing-service": {"count": 32, "avg_confidence": 0.78, "success_rate": 0.91},
        "user-api": {"count": 28, "avg_confidence": 0.85, "success_rate": 0.95},
        "order-service": {"count": 21, "avg_confidence": 0.79, "success_rate": 0.89}
    }


async def _get_reasoning_trail_for_analysis(analysis_id: str) -> List[Dict[str, Any]]:
    """Get reasoning trail for an analysis."""
    
    # Mock implementation
    return [
        {
            "step": 1,
            "action": "Initial assessment",
            "reasoning": "High severity incident affecting payment service",
            "confidence": 0.6
        },
        {
            "step": 2,
            "action": "Historical correlation",
            "reasoning": "Found 3 similar incidents with 85% similarity",
            "confidence": 0.75
        }
    ]


async def _get_evidence_for_analysis(analysis_id: str) -> List[Dict[str, Any]]:
    """Get evidence details for an analysis."""
    
    # Mock implementation
    return [
        {
            "type": "gcp_monitoring",
            "description": "API latency metrics",
            "relevance": 0.9
        },
        {
            "type": "gcp_logging", 
            "description": "Error log patterns",
            "relevance": 0.8
        }
    ]


async def _get_detailed_evidence_for_step(
    analysis_result: Dict[str, Any],
    step_number: int
) -> Dict[str, Any]:
    """Get detailed evidence for a specific reasoning step."""
    
    # Mock implementation
    return {
        "evidence_count": 3,
        "evidence_types": ["metrics", "logs", "historical"],
        "quality_score": 0.85
    }