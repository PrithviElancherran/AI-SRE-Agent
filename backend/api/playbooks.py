"""
FastAPI router for playbook management endpoints.

This module provides REST endpoints for playbook management including playbook execution,
step processing, effectiveness tracking, and configuration management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from models.playbook import (
    Playbook, PlaybookCreate, PlaybookUpdate, PlaybookStep, PlaybookExecution,
    PlaybookStepResult, ExecutionStatus, StepType, PlaybookEffectiveness
)
from models.incident import Incident, IncidentSeverity
from api.incidents import _get_incident_by_id
from models.user import User
from services.playbook_executor import PlaybookExecutor
from services.incident_analyzer import IncidentAnalyzer
from integrations.gcp_monitoring import GCPMonitoringClient
from integrations.gcp_logging import GCPLoggingClient
from integrations.gcp_error_reporting import GCPErrorReportingClient
from config.database import get_database
from utils.security import get_current_user, require_permissions
from utils.validators import validate_playbook_data
from utils.formatters import format_playbook_response, format_execution_response

router = APIRouter()

# Initialize services
playbook_executor = PlaybookExecutor()
incident_analyzer = IncidentAnalyzer()
gcp_monitoring = GCPMonitoringClient()
gcp_logging = GCPLoggingClient()
gcp_error_reporting = GCPErrorReportingClient()

# In-memory storage for demo purposes
_playbook_storage: Dict[str, Dict[str, Any]] = {}
_execution_storage: Dict[str, Dict[str, Any]] = {}
_step_result_storage: Dict[str, List[Dict[str, Any]]] = {}


class PlaybookExecutionRequest(BaseModel):
    """Request model for playbook execution."""
    
    playbook_id: str
    incident_id: str
    execution_mode: str = Field(default="automatic", description="Execution mode: automatic, manual, or step-by-step")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Execution parameters")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User context for execution")


class PlaybookExecutionResponse(BaseModel):
    """Response model for playbook execution."""
    
    execution_id: str
    playbook_id: str
    incident_id: str
    status: str
    progress: float
    current_step: Optional[int] = None
    steps_completed: int
    total_steps: int
    execution_time_seconds: float
    root_cause_found: bool
    confidence_score: Optional[float] = None
    results: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: str


class PlaybookStepExecutionResponse(BaseModel):
    """Response model for individual step execution."""
    
    step_id: str
    step_number: int
    step_type: str
    status: str
    success: Optional[bool] = None
    duration_seconds: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    threshold_met: Optional[bool] = None
    escalation_triggered: bool = False
    error_message: Optional[str] = None
    timestamp: str


class PlaybookEffectivenessResponse(BaseModel):
    """Response model for playbook effectiveness metrics."""
    
    playbook_id: str
    effectiveness_score: float
    total_executions: int
    successful_executions: int
    average_execution_time: float
    success_rate: float
    common_failures: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    last_updated: str


@router.post("/execute", response_model=PlaybookExecutionResponse)
async def execute_playbook(
    execution_request: PlaybookExecutionRequest,
    user: User = Depends(get_current_user)
) -> PlaybookExecutionResponse:
    """
    Execute a playbook for incident diagnosis and resolution.
    
    This endpoint performs:
    - Systematic step-by-step playbook execution
    - Real-time progress tracking
    - Evidence collection and validation
    - Threshold checking and escalation
    - Root cause identification
    - Confidence scoring
    """
    logger.info(f"Starting playbook execution: {execution_request.playbook_id} for incident: {execution_request.incident_id}")
    
    try:
        start_time = datetime.utcnow()
        
        # Get playbook details
        playbook = await _get_playbook_by_id(execution_request.playbook_id)
        if not playbook:
            raise HTTPException(
                status_code=404,
                detail=f"Playbook {execution_request.playbook_id} not found"
            )
        
        # Get incident details
        incident = await _get_incident_by_id(execution_request.incident_id)
        if not incident:
            raise HTTPException(
                status_code=404,
                detail=f"Incident {execution_request.incident_id} not found"
            )
        
        # Execute playbook
        execution_result = await playbook_executor.execute_playbook(
            playbook_id=playbook.playbook_id,
            incident_id=incident.incident_id,
            user=user,
            context={
                "execution_mode": execution_request.execution_mode,
                "parameters": execution_request.parameters or {}
            }
        )
        
        # Calculate execution duration
        execution_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Format step results
        step_results = []
        for step_result in execution_result.step_results:
            step_results.append({
                "step_id": getattr(step_result, 'step_id', 'unknown'),
                "step_number": getattr(step_result, 'step_number', 1),
                "step_type": getattr(step_result, 'step_type', 'unknown'),
                "status": getattr(step_result.status, 'value', str(step_result.status)) if hasattr(step_result, 'status') else 'unknown',
                "success": step_result.success,
                "duration_seconds": step_result.duration_seconds,
                "result_data": step_result.result_data,
                "evidence": step_result.evidence,
                "threshold_met": step_result.threshold_met,
                "escalation_triggered": step_result.escalation_triggered,
                "error_message": step_result.error_message
            })
        
        # Calculate progress and current step
        completed_steps = len([r for r in execution_result.step_results if r.is_completed()])
        progress = (completed_steps / max(1, len(playbook.steps))) * 100
        current_step = completed_steps + 1 if completed_steps < len(playbook.steps) else len(playbook.steps)
        
        # Get recommendations (fallback to empty list if not available)
        recommendations = getattr(execution_result, 'actions_recommended', []) or []
        
        # Create execution record for in-memory storage (matching PlaybookExecution model)
        execution_record = {
            "id": str(getattr(execution_result, 'id', uuid4())),
            "execution_id": execution_result.execution_id,
            "playbook_id": execution_request.playbook_id,
            "incident_id": execution_request.incident_id,
            "status": execution_result.status.value if hasattr(execution_result.status, 'value') else str(execution_result.status),
            "started_by": getattr(execution_result, 'started_by', user.username if hasattr(user, 'username') else 'demo_user'),
            "started_at": getattr(execution_result, 'started_at', datetime.utcnow()).isoformat(),
            "completed_at": getattr(execution_result, 'completed_at', datetime.utcnow()).isoformat(),
            "duration_minutes": execution_duration / 60.0,
            "success": getattr(execution_result, 'success', True),
            "root_cause_found": execution_result.root_cause_found,
            "actions_recommended": recommendations,
            "confidence_score": execution_result.confidence_score,
            "execution_context": {
                "execution_mode": execution_request.execution_mode,
                "parameters": execution_request.parameters or {}
            },
            "step_results": []  # Simplified for now to avoid serialization issues
        }
        
        # Save to in-memory storage
        _execution_storage[execution_result.execution_id] = execution_record
        logger.info(f"Saved execution {execution_result.execution_id} to in-memory storage")
        
        response = PlaybookExecutionResponse(
            execution_id=execution_result.execution_id,
            playbook_id=execution_request.playbook_id,
            incident_id=execution_request.incident_id,
            status=execution_result.status.value,
            progress=progress,
            current_step=current_step,
            steps_completed=completed_steps,
            total_steps=len(playbook.steps),
            execution_time_seconds=execution_duration,
            root_cause_found=execution_result.root_cause_found,
            confidence_score=execution_result.confidence_score,
            results=step_results,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Playbook execution completed: {execution_result.execution_id} in {execution_duration:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing playbook {execution_request.playbook_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute playbook: {str(e)}"
        )


@router.get("/execute/{execution_id}/status")
async def get_execution_status(
    execution_id: str = Path(..., description="Execution ID"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current status of a playbook execution."""
    
    logger.info(f"Getting execution status: {execution_id}")
    
    try:
        # Get execution from storage first, then check active executions
        execution = await _get_execution_by_id(execution_id)
        if not execution:
            # Check active executions as fallback
            execution = await playbook_executor.get_execution_status(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Calculate progress
        total_steps = len(execution.step_results) if execution.step_results else 1
        completed_steps = len([r for r in execution.step_results if getattr(r, 'success', False)]) if execution.step_results else 0
        progress = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
        
        return {
            "execution_id": execution_id,
            "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
            "progress": progress,
            "current_step": completed_steps + 1 if completed_steps < total_steps else total_steps,
            "steps_completed": completed_steps,
            "total_steps": total_steps,
            "root_cause_found": execution.root_cause_found,
            "confidence_score": execution.confidence_score,
            "last_updated": execution.completed_at.isoformat() if execution.completed_at else execution.started_at.isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution status {execution_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution status: {str(e)}"
        )


@router.post("/execute/{execution_id}/step", response_model=PlaybookStepExecutionResponse)
async def execute_next_step(
    execution_id: str = Path(..., description="Execution ID"),
    step_parameters: Optional[Dict[str, Any]] = Body(default=None),
    user: User = Depends(get_current_user)
) -> PlaybookStepExecutionResponse:
    """Execute the next step in a playbook execution."""
    
    logger.info(f"Executing next step for execution: {execution_id}")
    
    try:
        # Execute next step
        step_result = await playbook_executor.execute_next_step(
            execution_id=execution_id,
            step_id=step_parameters.get("step_id", "next") if step_parameters else "next",
            manual_input=step_parameters.get("manual_input") if step_parameters else None,
            context=step_parameters or {}
        )
        
        if not step_result:
            raise HTTPException(status_code=404, detail="No more steps to execute")
        
        return PlaybookStepExecutionResponse(
            step_id=str(step_result.step_id),
            step_number=getattr(step_result, 'step_number', 1),  # Use getattr for missing field
            step_type=getattr(step_result, 'step_type', 'manual_action'),  # Default step type
            status=step_result.status.value if hasattr(step_result.status, 'value') else str(step_result.status),
            success=step_result.success,
            duration_seconds=step_result.duration_seconds,
            result_data=step_result.result_data,
            evidence=step_result.evidence,
            threshold_met=step_result.threshold_met,
            escalation_triggered=step_result.escalation_triggered,
            error_message=step_result.error_message,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing next step for {execution_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute next step: {str(e)}"
        )


@router.post("/execute/{execution_id}/approve")
async def approve_execution_action(
    execution_id: str = Path(..., description="Execution ID"),
    approval_data: Dict[str, Any] = Body(...),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Approve or reject a playbook execution action."""
    
    logger.info(f"Processing approval for execution: {execution_id}")
    
    try:
        action = approval_data.get("action")  # "approve" or "reject"
        step_id = approval_data.get("step_id")
        reason = approval_data.get("reason", "")
        
        if action not in ["approve", "reject"]:
            raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")
        
        # Process approval
        result = await playbook_executor.process_approval(
            execution_id=execution_id,
            step_id=step_id,
            action=action,
            reason=reason,
            user=user
        )
        
        return {
            "execution_id": execution_id,
            "step_id": step_id,
            "action": action,
            "status": "processed",
            "reason": reason,
            "approved_by": user.user_id,
            "approved_at": datetime.utcnow().isoformat(),
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing approval for {execution_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process approval: {str(e)}"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_playbooks(
    category: Optional[str] = Query(None, description="Filter by category"),
    severity: Optional[IncidentSeverity] = Query(None, description="Filter by target severity"),
    active_only: bool = Query(True, description="Show only active playbooks"),
    limit: int = Query(50, description="Maximum number of playbooks"),
    offset: int = Query(0, description="Offset for pagination"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """List available playbooks with filtering and pagination."""
    
    logger.info(f"Listing playbooks with filters: category={category}, severity={severity}")
    
    try:
        # Get playbooks from database
        playbooks = await _get_playbooks_with_filters(
            category=category,
            severity=severity,
            active_only=active_only,
            limit=limit,
            offset=offset
        )
        
        # Format playbook data
        playbook_list = []
        for playbook in playbooks:
            # Get effectiveness metrics
            effectiveness = await _get_playbook_effectiveness(playbook.playbook_id)
            
            playbook_data = {
                "playbook_id": playbook.playbook_id,
                "name": playbook.name,
                "description": playbook.description,
                "version": playbook.version,
                "status": playbook.status,
                "applicable_services": playbook.applicable_services,
                "trigger_conditions": playbook.trigger_conditions,
                "step_count": len(playbook.steps),
                "effectiveness_score": playbook.effectiveness_score,
                "execution_count": playbook.execution_count,
                "success_rate": (playbook.success_count / max(1, playbook.execution_count)) * 100,
                "avg_execution_time": playbook.avg_execution_time_minutes,
                "created_at": playbook.created_at.isoformat() if playbook.created_at else None,
                "last_updated": playbook.last_updated.isoformat() if playbook.last_updated else None
            }
            playbook_list.append(playbook_data)
        
        return {
            "playbooks": playbook_list,
            "total_count": len(playbook_list),
            "limit": limit,
            "offset": offset,
            "filters": {
                "category": category,
                "severity": severity.value if severity else None,
                "active_only": active_only
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing playbooks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list playbooks: {str(e)}"
        )


@router.get("/{playbook_id}", response_model=Dict[str, Any])
async def get_playbook(
    playbook_id: str = Path(..., description="Playbook ID"),
    include_steps: bool = Query(True, description="Include detailed step information"),
    include_effectiveness: bool = Query(True, description="Include effectiveness metrics"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed information about a specific playbook."""
    
    logger.info(f"Getting playbook details: {playbook_id}")
    
    try:
        playbook = await _get_playbook_by_id(playbook_id)
        if not playbook:
            raise HTTPException(status_code=404, detail="Playbook not found")
        
        # Format playbook response
        # Convert Incident object to dictionary first
        playbook_dict = {
            'playbook_id': playbook.playbook_id,
            'name': playbook.name,
            'description': playbook.description,
            'version': playbook.version,
            'status': playbook.status,
            'created_at': playbook.created_at,
            'last_updated': playbook.last_updated,
            'created_by': playbook.created_by,
            'effectiveness_score': playbook.effectiveness_score,
            'success_rate': (playbook.success_count / max(1, playbook.execution_count)) if hasattr(playbook, 'success_count') else 0,
            'usage_count': playbook.execution_count if hasattr(playbook, 'execution_count') else 0,
            'applicable_services': playbook.applicable_services if hasattr(playbook, 'applicable_services') else [],
            'trigger_conditions': playbook.trigger_conditions if hasattr(playbook, 'trigger_conditions') else []
        }
        playbook_data = format_playbook_response(playbook_dict)
        
        # Include detailed steps if requested
        if include_steps:
            steps = []
            for step in playbook.steps:
                step_data = {
                    "step_id": step.step_id,
                    "step_number": getattr(step, 'step_number', getattr(step, 'order', 1)),
                    "step_type": step.step_type.value,
                    "title": getattr(step, 'title', step.description),
                    "description": step.description,
                    "command": getattr(step, 'command', getattr(step, 'query', '')),
                    "expected_output": getattr(step, 'expected_output', ''),
                    "success_criteria": getattr(step, 'success_criteria', {}),
                    "escalation_condition": getattr(step, 'escalation_condition', ''),
                    "timeout_seconds": getattr(step, 'timeout_seconds', getattr(step, 'timeout_minutes', 5) * 60),
                    "requires_approval": getattr(step, 'requires_approval', getattr(step, 'approval_required', False)),
                    "is_critical": getattr(step, 'is_critical', False)
                }
                steps.append(step_data)
            
            playbook_data["steps"] = steps
        
        # Include effectiveness metrics if requested
        if include_effectiveness:
            effectiveness = await _get_playbook_effectiveness(playbook_id)
            playbook_data["effectiveness"] = effectiveness
        
        return playbook_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playbook {playbook_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get playbook: {str(e)}"
        )


@router.get("/{playbook_id}/effectiveness", response_model=PlaybookEffectivenessResponse)
async def get_playbook_effectiveness(
    playbook_id: str = Path(..., description="Playbook ID"),
    time_range_days: int = Query(30, description="Time range for effectiveness analysis"),
    user: User = Depends(get_current_user)
) -> PlaybookEffectivenessResponse:
    """Get effectiveness metrics for a playbook."""
    
    logger.info(f"Getting effectiveness metrics for playbook: {playbook_id}")
    
    try:
        playbook = await _get_playbook_by_id(playbook_id)
        if not playbook:
            raise HTTPException(status_code=404, detail="Playbook not found")
        
        # Get effectiveness data
        effectiveness_data = await _calculate_playbook_effectiveness(
            playbook_id=playbook_id,
            time_range_days=time_range_days
        )
        
        return PlaybookEffectivenessResponse(
            playbook_id=playbook_id,
            effectiveness_score=effectiveness_data["effectiveness_score"],
            total_executions=effectiveness_data["total_executions"],
            successful_executions=effectiveness_data["successful_executions"],
            average_execution_time=effectiveness_data["average_execution_time"],
            success_rate=effectiveness_data["success_rate"],
            common_failures=effectiveness_data["common_failures"],
            improvement_suggestions=effectiveness_data["improvement_suggestions"],
            last_updated=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting effectiveness for {playbook_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get effectiveness metrics: {str(e)}"
        )


@router.post("/", response_model=Dict[str, Any])
async def create_playbook(
    playbook_data: PlaybookCreate,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a new playbook."""
    
    logger.info(f"Creating new playbook: {playbook_data.name}")
    
    try:
        # Check permissions
        if not user.can_create_playbooks():
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Validate playbook data
        validate_playbook_data(playbook_data.model_dump())
        
        # Create playbook ID  
        playbook_id = f"PB-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid4())[:8].upper()}"
        
        # Create basic playbook data for in-memory storage
        playbook_data_dict = {
            "id": str(uuid4()),
            "playbook_id": playbook_id,
            "name": playbook_data.name,
            "version": playbook_data.version,
            "description": playbook_data.description,
            "status": playbook_data.status.value if hasattr(playbook_data.status, 'value') else playbook_data.status,
            "applicable_services": playbook_data.applicable_services,
            "trigger_conditions": playbook_data.trigger_conditions,
            "steps": playbook_data.steps,
            "created_by": user.user_id,
            "created_at": datetime.utcnow().isoformat(),
            "effectiveness_score": 0.0,
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0
        }
        
        # Save to in-memory storage for demo
        _playbook_storage[playbook_id] = playbook_data_dict
        logger.info(f"Saved playbook {playbook_id} to storage")
        
        return {
            "playbook_id": playbook_id,
            "name": playbook_data.name,
            "version": playbook_data.version,
            "step_count": len(playbook_data.steps),
            "created_at": datetime.utcnow().isoformat(),
            "created_by": user.user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating playbook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create playbook: {str(e)}"
        )


@router.put("/{playbook_id}", response_model=Dict[str, Any])
async def update_playbook(
    playbook_id: str = Path(..., description="Playbook ID"),
    playbook_update: PlaybookUpdate = Body(...),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update an existing playbook."""
    
    logger.info(f"Updating playbook: {playbook_id}")
    
    try:
        playbook = await _get_playbook_by_id(playbook_id)
        if not playbook:
            raise HTTPException(status_code=404, detail="Playbook not found")
        
        # Check permissions
        if not user.can_modify_playbooks():
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Apply updates
        update_data = playbook_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(playbook, field):
                setattr(playbook, field, value)
        
        playbook.last_updated = datetime.utcnow()
        # Note: Playbook model doesn't have updated_by field
        # Version is a string, so increment properly
        try:
            current_version = float(playbook.version)
            playbook.version = str(current_version + 0.1)
        except ValueError:
            # If version isn't numeric, append a suffix
            playbook.version = f"{playbook.version}.1"
        
        # Save to database (mock implementation)
        
        return {
            "playbook_id": playbook.playbook_id,
            "version": playbook.version,
            "last_updated": playbook.last_updated.isoformat(),
            "changes": list(update_data.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating playbook {playbook_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update playbook: {str(e)}"
        )


@router.delete("/{playbook_id}")
async def delete_playbook(
    playbook_id: str = Path(..., description="Playbook ID"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete a playbook (admin only)."""
    
    logger.info(f"Deleting playbook: {playbook_id}")
    
    try:
        if not user.is_admin():
            raise HTTPException(status_code=403, detail="Admin permissions required")
        
        playbook = await _get_playbook_by_id(playbook_id)
        if not playbook:
            raise HTTPException(status_code=404, detail="Playbook not found")
        
        # Soft delete (mark as inactive)
        playbook.is_active = False
        playbook.deleted_at = datetime.utcnow()
        playbook.deleted_by = user.user_id
        
        # Save to database (mock implementation)
        
        return {
            "playbook_id": playbook_id,
            "status": "deleted",
            "deleted_at": playbook.deleted_at.isoformat(),
            "deleted_by": playbook.deleted_by
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting playbook {playbook_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete playbook: {str(e)}"
        )


@router.get("/executions/", response_model=Dict[str, Any])
async def list_executions(
    playbook_id: Optional[str] = Query(None, description="Filter by playbook ID"),
    incident_id: Optional[str] = Query(None, description="Filter by incident ID"),
    status: Optional[ExecutionStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of executions"),
    offset: int = Query(0, description="Offset for pagination"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """List playbook executions with filtering."""
    
    logger.info(f"Listing executions with filters: playbook={playbook_id}, incident={incident_id}")
    
    try:
        # Get executions from database
        executions = await _get_executions_with_filters(
            playbook_id=playbook_id,
            incident_id=incident_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        # Format execution data
        execution_list = []
        for execution in executions:
            execution_data = {
                "execution_id": execution.execution_id,
                "playbook_id": str(execution.playbook_id),
                "incident_id": execution.incident_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_minutes": execution.duration_minutes,
                "success": execution.success,
                "root_cause_found": execution.root_cause_found,
                "confidence_score": execution.confidence_score,
                "started_by": execution.started_by,
                "actions_recommended": execution.actions_recommended
            }
            execution_list.append(execution_data)
        
        return {
            "executions": execution_list,
            "total_count": len(execution_list),
            "limit": limit,
            "offset": offset,
            "filters": {
                "playbook_id": playbook_id,
                "incident_id": incident_id,
                "status": status.value if status else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list executions: {str(e)}"
        )


@router.get("/executions/{execution_id}", response_model=Dict[str, Any])
async def get_execution(
    execution_id: str = Path(..., description="Execution ID"),
    include_steps: bool = Query(True, description="Include step results"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed information about a playbook execution."""
    
    logger.info(f"Getting execution details: {execution_id}")
    
    try:
        execution = await _get_execution_by_id(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Format execution response
        execution_data = {
            "execution_id": execution.execution_id,
            "playbook_id": execution.playbook_id,
            "incident_id": execution.incident_id,
            "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
            "started_by": execution.started_by,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration_minutes": execution.duration_minutes,
            "success": execution.success,
            "root_cause_found": execution.root_cause_found,
            "confidence_score": execution.confidence_score,
            "actions_recommended": execution.actions_recommended or [],
            "execution_context": execution.execution_context or {}
        }
        
        # Include step results if requested
        if include_steps:
            step_results = []
            for step_result in execution.step_results:
                step_data = {
                    "step_id": getattr(step_result, 'step_id', 'unknown'),
                    "step_number": getattr(step_result, 'step_number', 1),
                    "step_type": getattr(step_result, 'step_type', 'unknown'),
                    "status": getattr(step_result.status, 'value', str(step_result.status)) if hasattr(step_result, 'status') else 'unknown',
                    "success": step_result.success,
                    "started_at": step_result.started_at.isoformat() if step_result.started_at else None,
                    "completed_at": step_result.completed_at.isoformat() if step_result.completed_at else None,
                    "duration_seconds": step_result.duration_seconds,
                    "result_data": step_result.result_data,
                    "evidence": step_result.evidence,
                    "threshold_met": step_result.threshold_met,
                    "escalation_triggered": step_result.escalation_triggered,
                    "error_message": step_result.error_message
                }
                step_results.append(step_data)
            
            execution_data["step_results"] = step_results
        
        return execution_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution {execution_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution: {str(e)}"
        )


# Helper functions

async def _get_playbook_by_id(playbook_id: str) -> Optional[Playbook]:
    """Get playbook by ID from database."""
    # Check storage first for newly created playbooks
    if playbook_id in _playbook_storage:
        playbook_data = _playbook_storage[playbook_id]
        return Playbook(
            id=playbook_data["id"],
            playbook_id=playbook_data["playbook_id"],
            name=playbook_data["name"],
            version=playbook_data["version"],
            description=playbook_data["description"],
            status=playbook_data["status"],
            applicable_services=playbook_data["applicable_services"],
            trigger_conditions=playbook_data["trigger_conditions"],
            effectiveness_score=playbook_data["effectiveness_score"],
            execution_count=playbook_data["execution_count"],
            success_count=playbook_data["success_count"],
            failure_count=playbook_data["failure_count"],
            created_by=playbook_data["created_by"],
            created_at=datetime.fromisoformat(playbook_data["created_at"]),
            steps=[
                PlaybookStep(
                    id=uuid4(),
                    step_id=step.step_id if hasattr(step, 'step_id') else getattr(step, 'step_id', f"step_{i}"),
                    step_type=step.step_type if hasattr(step, 'step_type') else getattr(step, 'step_type', StepType.MANUAL_ACTION),
                    description=step.description if hasattr(step, 'description') else getattr(step, 'description', 'Unknown step'),
                    order=step.order if hasattr(step, 'order') else getattr(step, 'order', i + 1),
                    approval_required=step.approval_required if hasattr(step, 'approval_required') else getattr(step, 'approval_required', False),
                    created_at=datetime.utcnow()
                ) for i, step in enumerate(playbook_data.get("steps", []))
            ]
        )
    
    # Mock implementation for demo playbooks
    if playbook_id == "PB-DB-TIMEOUT-001":
        from models.playbook import ExpectedResult
        
        return Playbook(
            id=uuid4(),
            playbook_id=playbook_id,
            name="Database Connection Timeout Troubleshooting",
            description="Systematic troubleshooting for database connection timeouts",
            version="1.0.0",
            status="active",
            applicable_services=["billing-service", "payment-api"],
            trigger_conditions=["Database timeout errors", "Connection pool exhausted"],
            effectiveness_score=0.88,
            execution_count=25,
            success_count=23,
            failure_count=2,
            avg_execution_time_minutes=12.5,
            last_executed=datetime.utcnow() - timedelta(days=2),
            created_by="sre_team",
            created_at=datetime.utcnow() - timedelta(days=30),
            last_updated=datetime.utcnow() - timedelta(days=5),
            steps=[
                PlaybookStep(
                    id=uuid4(),
                    step_id="step_1",
                    description="Verify connection pool utilization",
                    step_type=StepType.METRIC_CHECK,
                    order=1,
                    query="gcp_monitoring.query('cloudsql_database_connection_count')",
                    expected_result=ExpectedResult(
                        threshold="< 180",
                        healthy_range="100-150"
                    ),
                    escalation_condition="greater_than 190",
                    timeout_minutes=1,
                    retry_count=3,
                    prerequisites=[],
                    dependencies=[],
                    approval_required=False,
                    created_at=datetime.utcnow()
                ),
                PlaybookStep(
                    id=uuid4(),
                    step_id="step_2",
                    description="Search for connection timeout patterns",
                    step_type=StepType.LOG_ANALYSIS,
                    order=2,
                    query="gcp_logging.search('connection timeout OR connection refused')",
                    expected_result=ExpectedResult(
                        threshold="< 5 errors",
                        pattern="timeout"
                    ),
                    escalation_condition="errors > 10",
                    timeout_minutes=2,
                    retry_count=2,
                    prerequisites=[],
                    dependencies=[],
                    approval_required=False,
                    created_at=datetime.utcnow()
                ),
                PlaybookStep(
                    id=uuid4(),
                    step_id="step_3",
                    description="Analyze query performance and execution plans",
                    step_type=StepType.QUERY_ANALYSIS,
                    order=3,
                    query="gcp_monitoring.analyze_query_performance('database_query_execution_time')",
                    expected_result=ExpectedResult(
                        threshold="< 5s",
                        healthy_range="1-3s"
                    ),
                    escalation_condition="avg_time > 5000ms",
                    timeout_minutes=3,
                    retry_count=2,
                    prerequisites=[],
                    dependencies=[],
                    approval_required=False,
                    created_at=datetime.utcnow()
                )
            ]
        )
    
    return None


# _get_incident_by_id is now imported from api.incidents


async def _get_playbooks_with_filters(
    category: Optional[str] = None,
    severity: Optional[IncidentSeverity] = None,
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0
) -> List[Playbook]:
    """Get playbooks with filters applied."""
    # Get mock playbooks
    mock_playbooks = [
        await _get_playbook_by_id("PB-DB-TIMEOUT-001")
    ]
    
    # Get playbooks from storage
    storage_playbooks = []
    for playbook_id in _playbook_storage.keys():
        playbook = await _get_playbook_by_id(playbook_id)
        if playbook:
            storage_playbooks.append(playbook)
    
    # Combine all playbooks
    playbooks = [pb for pb in mock_playbooks if pb is not None] + storage_playbooks
    
    # Apply filters
    if category:
        # Filter by applicable_services
        playbooks = [pb for pb in playbooks if category.lower() in [service.lower() for service in pb.applicable_services]]
    if severity:
        # Since target_severity isn't used in our model, skip this filter for now
        pass
    if active_only:
        # Filter by status since is_active isn't a direct attribute
        playbooks = [pb for pb in playbooks if pb.status == "active"]
    
    # Apply pagination
    return playbooks[offset:offset + limit]


async def _get_playbook_effectiveness(playbook_id: str) -> Dict[str, Any]:
    """Get effectiveness metrics for a playbook."""
    # Mock implementation
    
    return {
        "effectiveness_score": 0.85,
        "success_rate": 0.92,
        "avg_execution_time": 12.5,
        "total_executions": 24,
        "successful_executions": 22
    }


async def _calculate_playbook_effectiveness(
    playbook_id: str,
    time_range_days: int
) -> Dict[str, Any]:
    """Calculate detailed effectiveness metrics."""
    # Mock implementation
    
    return {
        "effectiveness_score": 0.85,
        "total_executions": 24,
        "successful_executions": 22,
        "average_execution_time": 12.5,
        "success_rate": 0.92,
        "common_failures": [
            {
                "failure_type": "timeout",
                "count": 2,
                "percentage": 8.3,
                "description": "Step timeout due to slow API response"
            }
        ],
        "improvement_suggestions": [
            "Increase timeout for slow API calls",
            "Add retry logic for transient failures",
            "Consider parallel execution for independent steps"
        ]
    }


async def _get_executions_with_filters(
    playbook_id: Optional[str] = None,
    incident_id: Optional[str] = None,
    status: Optional[ExecutionStatus] = None,
    limit: int = 50,
    offset: int = 0
) -> List[PlaybookExecution]:
    """Get executions with filters applied."""
    
    # Get stored executions from in-memory storage
    stored_executions = []
    for exec_id, exec_data in _execution_storage.items():
        # Apply filters
        if playbook_id and exec_data["playbook_id"] != playbook_id:
            continue
        if incident_id and exec_data["incident_id"] != incident_id:
            continue
        if status and exec_data["status"] != status.value:
            continue
            
        # Convert stored dict to PlaybookExecution model
        try:
            execution = PlaybookExecution(**exec_data)
            stored_executions.append(execution)
        except Exception as e:
            logger.warning(f"Error converting execution {exec_id} to model: {e}")
            continue
    
    # Add a mock execution if no stored ones exist
    if not stored_executions:
        mock_execution = PlaybookExecution(
            id=uuid4(),
            execution_id="EXEC-MOCK-001",
            playbook_id="PB-DB-TIMEOUT-001",
            incident_id="INC-2024-002",
            status=ExecutionStatus.COMPLETED,
            started_by="sre-team",
            started_at=datetime.utcnow() - timedelta(minutes=15),
            completed_at=datetime.utcnow() - timedelta(minutes=3),
            duration_minutes=12.0,
            success=True,
            root_cause_found=True,
            actions_recommended=["Check connection pool settings", "Restart database connections"],
            confidence_score=0.88,
            execution_context={"mode": "automatic", "triggered_by": "alert"},
            step_results=[]
        )
        stored_executions.append(mock_execution)
    
    # Apply pagination
    start_idx = offset
    end_idx = offset + limit
    
    return stored_executions[start_idx:end_idx]


async def _get_execution_by_id(execution_id: str) -> Optional[PlaybookExecution]:
    """Get execution by ID from in-memory storage."""
    
    # Check in-memory storage first
    if execution_id in _execution_storage:
        execution_data = _execution_storage[execution_id].copy()
        
        # Convert datetime strings back to datetime objects if needed
        if isinstance(execution_data.get('started_at'), str):
            execution_data['started_at'] = datetime.fromisoformat(execution_data['started_at'].replace('Z', '+00:00'))
        if isinstance(execution_data.get('completed_at'), str):
            execution_data['completed_at'] = datetime.fromisoformat(execution_data['completed_at'].replace('Z', '+00:00'))
        
        # Convert id string back to UUID if needed
        if isinstance(execution_data.get('id'), str):
            try:
                execution_data['id'] = UUID(execution_data['id'])
            except:
                execution_data['id'] = uuid4()
        
        return PlaybookExecution(**execution_data)
    
    # If not found in storage, return None (or create a mock for demo)
    return None