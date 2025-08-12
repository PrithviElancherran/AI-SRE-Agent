"""
Demo Scenarios API Endpoints

This module provides FastAPI endpoints for the two debugging scenarios:
1. Payment API Latency Spike (Scenario 1)
2. Database Connection Pool Exhaustion (Scenario 2)

Integrates with synthetic data to provide realistic demo experiences.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta
import json
from uuid import uuid4

from backend.models.incident import Incident, IncidentCreate, IncidentSeverity, IncidentStatus
from backend.models.playbook import PlaybookExecution, ExecutionStatus, PlaybookExecutionRequest
from backend.models.analysis import AnalysisResult, EvidenceItem, EvidenceType
from backend.services.synthetic_data_loader import synthetic_data_loader
from backend.services.incident_analyzer import incident_analyzer
from backend.services.playbook_executor import playbook_executor
from backend.integrations.gcp_monitoring import gcp_monitoring_client
from backend.integrations.gcp_logging import gcp_logging_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scenarios", tags=["Demo Scenarios"])

# Demo state management
demo_state = {
    "scenario_1": {
        "active": False,
        "incident_id": None,
        "analysis_id": None,
        "execution_id": None,
        "start_time": None,
        "current_step": 1,
        "total_steps": 8
    },
    "scenario_2": {
        "active": False,
        "incident_id": None,
        "analysis_id": None,
        "execution_id": None,
        "start_time": None,
        "current_step": 1,
        "total_steps": 6
    }
}

@router.post("/initialize")
async def initialize_demo_data():
    """Initialize synthetic data for demo scenarios"""
    try:
        success = await synthetic_data_loader.load_all_synthetic_data()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load synthetic data")
        
        return {
            "success": True,
            "message": "Demo data initialized successfully",
            "data": {
                "incidents_loaded": len(synthetic_data_loader.get_incidents()),
                "playbooks_loaded": len(synthetic_data_loader.get_playbooks()),
                "analyses_loaded": len(synthetic_data_loader.get_analyses()),
                "executions_loaded": len(synthetic_data_loader.get_executions())
            }
        }
    except Exception as e:
        logger.error(f"Error initializing demo data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_demo_status():
    """Get current status of all demo scenarios"""
    return {
        "success": True,
        "data": {
            "scenarios": demo_state,
            "data_loaded": len(synthetic_data_loader.get_incidents()) > 0,
            "available_scenarios": [
                {
                    "id": "scenario_1",
                    "name": "Payment API Latency Spike",
                    "description": "Redis cache performance degradation causing API latency",
                    "severity": "high",
                    "estimated_duration": "10-15 minutes"
                },
                {
                    "id": "scenario_2", 
                    "name": "Database Connection Pool Exhaustion",
                    "description": "Database connection pool exhausted causing timeouts",
                    "severity": "critical",
                    "estimated_duration": "8-12 minutes"
                }
            ]
        }
    }

@router.post("/scenario1/start")
async def start_scenario_1(background_tasks: BackgroundTasks):
    """Start Scenario 1: Payment API Latency Spike"""
    try:
        if demo_state["scenario_1"]["active"]:
            raise HTTPException(status_code=400, detail="Scenario 1 is already active")
        
        # Get or create the scenario incident
        incident = await synthetic_data_loader.get_scenario_1_incident()
        
        # Initialize scenario state
        demo_state["scenario_1"] = {
            "active": True,
            "incident_id": incident.incident_id,
            "analysis_id": None,
            "execution_id": None,
            "start_time": datetime.utcnow().isoformat(),
            "current_step": 1,
            "total_steps": 8
        }
        
        # Start background scenario progression
        background_tasks.add_task(execute_scenario_1_progression, incident)
        
        return {
            "success": True,
            "message": "Scenario 1 started: Payment API Latency Spike",
            "data": {
                "scenario_id": "scenario_1",
                "incident": {
                    "incident_id": incident.incident_id,
                    "title": incident.title,
                    "description": incident.description,
                    "severity": incident.severity,
                    "service_name": incident.service_name,
                    "region": incident.region
                },
                "expected_progression": [
                    "Alert received: Payment API latency spike",
                    "Analyzing GCP monitoring metrics", 
                    "Searching historical incidents",
                    "Pattern correlation analysis",
                    "Root cause identification",
                    "Playbook selection and execution",
                    "Evidence collection and validation",
                    "Resolution recommendation"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error starting scenario 1: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenario2/start")
async def start_scenario_2(background_tasks: BackgroundTasks):
    """Start Scenario 2: Database Connection Pool Exhaustion"""
    try:
        if demo_state["scenario_2"]["active"]:
            raise HTTPException(status_code=400, detail="Scenario 2 is already active")
        
        # Get or create the scenario incident
        incident = await synthetic_data_loader.get_scenario_2_incident()
        
        # Initialize scenario state
        demo_state["scenario_2"] = {
            "active": True,
            "incident_id": incident.incident_id,
            "analysis_id": None,
            "execution_id": None,
            "start_time": datetime.utcnow().isoformat(),
            "current_step": 1,
            "total_steps": 6
        }
        
        # Start background scenario progression
        background_tasks.add_task(execute_scenario_2_progression, incident)
        
        return {
            "success": True,
            "message": "Scenario 2 started: Database Connection Pool Exhaustion",
            "data": {
                "scenario_id": "scenario_2",
                "incident": {
                    "incident_id": incident.incident_id,
                    "title": incident.title,
                    "description": incident.description,
                    "severity": incident.severity,
                    "service_name": incident.service_name,
                    "region": incident.region
                },
                "expected_progression": [
                    "User report: Database timeout errors",
                    "Playbook selection: Database troubleshooting",
                    "Step 1: Check connection pool utilization",
                    "Step 2: Analyze timeout errors in logs",
                    "Step 3: Identify long-running queries",
                    "Root cause and resolution recommendation"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error starting scenario 2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scenario1/data")
async def get_scenario_1_data():
    """Get GCP observability data for Scenario 1"""
    try:
        redis_metrics = synthetic_data_loader.get_gcp_metrics("1", "redis")
        api_metrics = synthetic_data_loader.get_gcp_metrics("1", "api")
        logs = synthetic_data_loader.get_gcp_logs("1")
        errors = synthetic_data_loader.get_gcp_errors("1")
        traces = synthetic_data_loader.get_gcp_traces("1")
        
        return {
            "success": True,
            "data": {
                "scenario": "Payment API Latency Spike",
                "observability_data": {
                    "metrics": {
                        "redis_cache_hit_rate": redis_metrics[:50],  # Last 50 data points
                        "api_latency": api_metrics[:50]
                    },
                    "logs": logs[:100],  # Last 100 log entries
                    "errors": errors[:50],  # Last 50 errors
                    "traces": traces[:20],  # Last 20 traces
                    "summary": {
                        "redis_metrics_count": len(redis_metrics),
                        "api_metrics_count": len(api_metrics),
                        "log_entries_count": len(logs),
                        "error_count": len(errors),
                        "trace_count": len(traces)
                    }
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting scenario 1 data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scenario2/data")
async def get_scenario_2_data():
    """Get GCP observability data for Scenario 2"""
    try:
        db_metrics = synthetic_data_loader.get_gcp_metrics("2", "db")
        conn_metrics = synthetic_data_loader.get_gcp_metrics("2", "connections")
        logs = synthetic_data_loader.get_gcp_logs("2")
        errors = synthetic_data_loader.get_gcp_errors("2")
        traces = synthetic_data_loader.get_gcp_traces("2")
        
        return {
            "success": True,
            "data": {
                "scenario": "Database Connection Pool Exhaustion",
                "observability_data": {
                    "metrics": {
                        "database_connections": db_metrics[:50],
                        "connection_pool": conn_metrics[:50]
                    },
                    "logs": logs[:100],
                    "errors": errors[:50],
                    "traces": traces[:20],
                    "summary": {
                        "db_metrics_count": len(db_metrics),
                        "connection_metrics_count": len(conn_metrics),
                        "log_entries_count": len(logs),
                        "error_count": len(errors),
                        "trace_count": len(traces)
                    }
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting scenario 2 data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scenario1/progress")
async def get_scenario_1_progress():
    """Get current progress of Scenario 1"""
    scenario = demo_state["scenario_1"]
    
    if not scenario["active"]:
        raise HTTPException(status_code=404, detail="Scenario 1 is not active")
    
    # Calculate progress percentage
    progress = (scenario["current_step"] / scenario["total_steps"]) * 100
    
    return {
        "success": True,
        "data": {
            "scenario_id": "scenario_1",
            "active": scenario["active"],
            "incident_id": scenario["incident_id"],
            "current_step": scenario["current_step"],
            "total_steps": scenario["total_steps"],
            "progress_percentage": round(progress, 2),
            "start_time": scenario["start_time"],
            "elapsed_time_minutes": _calculate_elapsed_time(scenario["start_time"]),
            "analysis_id": scenario["analysis_id"],
            "execution_id": scenario["execution_id"]
        }
    }

@router.get("/scenario2/progress")
async def get_scenario_2_progress():
    """Get current progress of Scenario 2"""
    scenario = demo_state["scenario_2"]
    
    if not scenario["active"]:
        raise HTTPException(status_code=404, detail="Scenario 2 is not active")
    
    # Calculate progress percentage
    progress = (scenario["current_step"] / scenario["total_steps"]) * 100
    
    return {
        "success": True,
        "data": {
            "scenario_id": "scenario_2",
            "active": scenario["active"],
            "incident_id": scenario["incident_id"],
            "current_step": scenario["current_step"],
            "total_steps": scenario["total_steps"],
            "progress_percentage": round(progress, 2),
            "start_time": scenario["start_time"],
            "elapsed_time_minutes": _calculate_elapsed_time(scenario["start_time"]),
            "analysis_id": scenario["analysis_id"],
            "execution_id": scenario["execution_id"]
        }
    }

@router.post("/scenario1/stop")
async def stop_scenario_1():
    """Stop Scenario 1"""
    try:
        demo_state["scenario_1"] = {
            "active": False,
            "incident_id": None,
            "analysis_id": None,
            "execution_id": None,
            "start_time": None,
            "current_step": 1,
            "total_steps": 8
        }
        
        return {
            "success": True,
            "message": "Scenario 1 stopped successfully"
        }
    except Exception as e:
        logger.error(f"Error stopping scenario 1: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenario2/stop")
async def stop_scenario_2():
    """Stop Scenario 2"""
    try:
        demo_state["scenario_2"] = {
            "active": False,
            "incident_id": None,
            "analysis_id": None,
            "execution_id": None,
            "start_time": None,
            "current_step": 1,
            "total_steps": 6
        }
        
        return {
            "success": True,
            "message": "Scenario 2 stopped successfully"
        }
    except Exception as e:
        logger.error(f"Error stopping scenario 2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical-correlation/{incident_id}")
async def get_historical_correlation(incident_id: str):
    """Get historical incident correlation for demo"""
    try:
        # Get historical incidents from synthetic data
        historical_incidents = synthetic_data_loader.get_incidents()
        
        # Find similar incidents based on service and symptoms
        similar_incidents = []
        target_incident = None
        
        # Find target incident
        for incident in historical_incidents:
            if incident.incident_id == incident_id:
                target_incident = incident
                break
        
        if not target_incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Find similar incidents
        for incident in historical_incidents:
            if incident.incident_id != incident_id:
                similarity_score = _calculate_similarity(target_incident, incident)
                if similarity_score > 0.7:  # 70% similarity threshold
                    similar_incidents.append({
                        "incident_id": incident.incident_id,
                        "service_name": incident.service_name,
                        "severity": incident.severity,
                        "root_cause": incident.root_cause,
                        "resolution": incident.resolution,
                        "mttr_minutes": incident.mttr_minutes,
                        "similarity_score": similarity_score,
                        "matching_symptoms": _get_matching_symptoms(target_incident, incident)
                    })
        
        # Sort by similarity score
        similar_incidents.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "success": True,
            "data": {
                "target_incident_id": incident_id,
                "similar_incidents": similar_incidents[:5],  # Top 5 matches
                "correlation_summary": {
                    "total_matches": len(similar_incidents),
                    "high_confidence_matches": len([s for s in similar_incidents if s["similarity_score"] > 0.9]),
                    "avg_similarity": sum(s["similarity_score"] for s in similar_incidents) / len(similar_incidents) if similar_incidents else 0
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting historical correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks for scenario progression

async def execute_scenario_1_progression(incident: Incident):
    """Execute the progression of Scenario 1"""
    try:
        scenario = demo_state["scenario_1"]
        
        # Step 1: Alert received (already done)
        await asyncio.sleep(2)
        scenario["current_step"] = 2
        
        # Step 2: Start analysis
        await asyncio.sleep(3)
        analysis_id = str(uuid4())
        scenario["analysis_id"] = analysis_id
        scenario["current_step"] = 3
        
        # Step 3: Historical search
        await asyncio.sleep(4)
        scenario["current_step"] = 4
        
        # Step 4: Pattern correlation
        await asyncio.sleep(3)
        scenario["current_step"] = 5
        
        # Step 5: Root cause identification
        await asyncio.sleep(5)
        scenario["current_step"] = 6
        
        # Step 6: Playbook execution
        await asyncio.sleep(4)
        execution_id = str(uuid4())
        scenario["execution_id"] = execution_id
        scenario["current_step"] = 7
        
        # Step 7: Evidence collection
        await asyncio.sleep(3)
        scenario["current_step"] = 8
        
        # Step 8: Resolution recommendation
        await asyncio.sleep(2)
        scenario["active"] = False
        
        logger.info(f"Scenario 1 completed for incident {incident.incident_id}")
        
    except Exception as e:
        logger.error(f"Error in scenario 1 progression: {e}")
        scenario["active"] = False

async def execute_scenario_2_progression(incident: Incident):
    """Execute the progression of Scenario 2"""
    try:
        scenario = demo_state["scenario_2"]
        
        # Step 1: User report received (already done)
        await asyncio.sleep(2)
        scenario["current_step"] = 2
        
        # Step 2: Playbook selection
        await asyncio.sleep(3)
        execution_id = str(uuid4())
        scenario["execution_id"] = execution_id
        scenario["current_step"] = 3
        
        # Step 3: Check connection pool utilization
        await asyncio.sleep(4)
        scenario["current_step"] = 4
        
        # Step 4: Analyze timeout errors
        await asyncio.sleep(5)
        scenario["current_step"] = 5
        
        # Step 5: Identify long-running queries
        await asyncio.sleep(4)
        scenario["current_step"] = 6
        
        # Step 6: Root cause and resolution
        await asyncio.sleep(3)
        scenario["active"] = False
        
        logger.info(f"Scenario 2 completed for incident {incident.incident_id}")
        
    except Exception as e:
        logger.error(f"Error in scenario 2 progression: {e}")
        scenario["active"] = False

# Helper functions

def _calculate_elapsed_time(start_time: str) -> float:
    """Calculate elapsed time in minutes"""
    if not start_time:
        return 0
    
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        now = datetime.utcnow().replace(tzinfo=start.tzinfo)
        elapsed = now - start
        return round(elapsed.total_seconds() / 60, 2)
    except Exception:
        return 0

def _calculate_similarity(incident1: Incident, incident2: Incident) -> float:
    """Calculate similarity score between two incidents"""
    score = 0.0
    
    # Service name match (40% weight)
    if incident1.service_name == incident2.service_name:
        score += 0.4
    
    # Severity match (20% weight)
    if incident1.severity == incident2.severity:
        score += 0.2
    
    # Symptoms similarity (40% weight)
    if incident1.symptoms and incident2.symptoms:
        symptom_overlap = _calculate_symptom_overlap(incident1.symptoms, incident2.symptoms)
        score += 0.4 * symptom_overlap
    
    return min(score, 1.0)

def _calculate_symptom_overlap(symptoms1, symptoms2) -> float:
    """Calculate overlap between symptom lists"""
    if not symptoms1 or not symptoms2:
        return 0.0
    
    # Extract symptom descriptions
    desc1 = {s.description.lower() for s in symptoms1}
    desc2 = {s.description.lower() for s in symptoms2}
    
    # Calculate Jaccard similarity
    intersection = len(desc1.intersection(desc2))
    union = len(desc1.union(desc2))
    
    return intersection / union if union > 0 else 0.0

def _get_matching_symptoms(incident1: Incident, incident2: Incident) -> List[str]:
    """Get list of matching symptoms between incidents"""
    if not incident1.symptoms or not incident2.symptoms:
        return []
    
    desc1 = {s.description.lower() for s in incident1.symptoms}
    desc2 = {s.description.lower() for s in incident2.symptoms}
    
    return list(desc1.intersection(desc2))