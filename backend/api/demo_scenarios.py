"""
Demo Scenarios API for AI SRE Agent.

This module provides REST endpoints for demonstrating the two main debugging scenarios:
1. Past incident correlation and diagnosis
2. Playbook-driven debugging

Integrates with synthetic data to provide realistic demo experiences.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta
import json
from uuid import uuid4

from services.synthetic_data_loader import synthetic_data_loader
from integrations.gcp_monitoring import gcp_monitoring_client
from integrations.gcp_logging import gcp_logging_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scenarios", tags=["Demo Scenarios"])

# Demo state management
demo_state = {
    "scenario_1": {
        "active_sessions": {},
        "last_execution": None
    },
    "scenario_2": {
        "active_sessions": {},
        "last_execution": None
    }
}

@router.get("/", summary="List available demo scenarios")
async def list_scenarios():
    """Get list of available demo scenarios."""
    
    scenarios = [
        {
            "scenario_id": "scenario_1",
            "name": "Past Incident Correlation & Diagnosis",
            "description": "Payment API latency spike detection and correlation with historical incidents",
            "type": "correlation",
            "estimated_duration_minutes": 3,
            "difficulty": "intermediate",
            "technologies": ["GCP Monitoring", "Vector Search", "Historical Analysis"],
            "synthetic_data": {
                "historical_incidents": 2,
                "current_metrics": "payment_api_latency_spike",
                "infrastructure": "e-commerce_microservices"
            }
        },
        {
            "scenario_id": "scenario_2", 
            "name": "Playbook-Driven Debugging",
            "description": "Database connection timeout troubleshooting using structured playbook",
            "type": "playbook",
            "estimated_duration_minutes": 4,
            "difficulty": "beginner",
            "technologies": ["GCP Logging", "Database Monitoring", "Playbook Execution"],
            "synthetic_data": {
                "playbook_steps": 3,
                "current_metrics": "database_connection_timeout",
                "infrastructure": "saas_application"
            }
        }
    ]
    
    return {
        "scenarios": scenarios,
        "total_count": len(scenarios),
        "demo_mode": True,
        "synthetic_data_loaded": synthetic_data_loader.is_loaded(),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/scenario_1/start", summary="Start Scenario 1: Past Incident Correlation")
async def start_scenario_1(background_tasks: BackgroundTasks):
    """
    Start the first demo scenario: Payment API latency spike with historical correlation.
    
    This scenario demonstrates:
    - Automated incident detection
    - Historical incident correlation using vector search
    - Root cause analysis with confidence scoring
    - Evidence collection from GCP observability tools
    """
    
    session_id = str(uuid4())
    incident_id = f"INC-2024-{str(uuid4())[:6].upper()}"
    
    # Initialize session state
    demo_state["scenario_1"]["active_sessions"][session_id] = {
        "incident_id": incident_id,
        "started_at": datetime.utcnow().isoformat(),
        "status": "starting",
        "progress": 0,
        "steps_completed": 0,
        "total_steps": 6
    }
    
    # Load synthetic data for this scenario
    synthetic_incident = synthetic_data_loader.get_current_incident_scenario_1()
    historical_incidents = synthetic_data_loader.get_historical_incidents()
    
    # Execute scenario in background
    background_tasks.add_task(execute_scenario_1, session_id, incident_id, synthetic_incident, historical_incidents)
    
    return {
        "session_id": session_id,
        "incident_id": incident_id,
        "scenario": "past_incident_correlation",
        "status": "started",
        "message": "Scenario 1 initiated: Payment API latency spike detected",
        "estimated_completion": (datetime.utcnow() + timedelta(minutes=3)).isoformat(),
        "synthetic_data": {
            "service_name": synthetic_incident.get("service_name", "payment-api"),
            "region": synthetic_incident.get("region", "us-central1"),
            "severity": synthetic_incident.get("severity", "high")
        }
    }

@router.post("/scenario_2/start", summary="Start Scenario 2: Playbook-Driven Debugging")
async def start_scenario_2(background_tasks: BackgroundTasks):
    """
    Start the second demo scenario: Database timeout troubleshooting with playbook.
    
    This scenario demonstrates:
    - Manual incident reporting via chat
    - Systematic playbook execution
    - Step-by-step debugging process
    - Root cause identification through guided analysis
    """
    
    session_id = str(uuid4())
    incident_id = f"INC-2024-{str(uuid4())[:6].upper()}"
    playbook_id = "PB-DB-TIMEOUT-001"
    execution_id = str(uuid4())
    
    # Initialize session state
    demo_state["scenario_2"]["active_sessions"][session_id] = {
        "incident_id": incident_id,
        "playbook_id": playbook_id,
        "execution_id": execution_id,
        "started_at": datetime.utcnow().isoformat(),
        "status": "starting",
        "progress": 0,
        "current_step": 0,
        "total_steps": 3
    }
    
    # Load synthetic data for this scenario
    synthetic_incident = synthetic_data_loader.get_current_incident_scenario_2()
    playbook_data = synthetic_data_loader.get_database_timeout_playbook()
    
    # Execute scenario in background
    background_tasks.add_task(execute_scenario_2, session_id, incident_id, playbook_id, execution_id, synthetic_incident, playbook_data)
    
    return {
        "session_id": session_id,
        "incident_id": incident_id,
        "playbook_id": playbook_id,
        "execution_id": execution_id,
        "scenario": "playbook_driven_debugging",
        "status": "started",
        "message": "Scenario 2 initiated: Database timeout troubleshooting",
        "estimated_completion": (datetime.utcnow() + timedelta(minutes=4)).isoformat(),
        "synthetic_data": {
            "playbook_name": playbook_data.get("name", "Database Connection Timeout Troubleshooting"),
            "service_name": synthetic_incident.get("service_name", "billing-service"),
            "total_steps": len(playbook_data.get("steps", []))
        }
    }

@router.get("/scenario_1/{session_id}/status", summary="Get Scenario 1 Status")
async def get_scenario_1_status(session_id: str):
    """Get the current status of a running Scenario 1 session."""
    
    if session_id not in demo_state["scenario_1"]["active_sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = demo_state["scenario_1"]["active_sessions"][session_id]
    
    return {
        "session_id": session_id,
        "scenario": "past_incident_correlation",
        "status": session["status"],
        "progress": session["progress"],
        "steps_completed": session["steps_completed"],
        "total_steps": session["total_steps"],
        "incident_id": session["incident_id"],
        "started_at": session["started_at"],
        "current_step": session.get("current_step", ""),
        "last_update": datetime.utcnow().isoformat()
    }

@router.get("/scenario_2/{session_id}/status", summary="Get Scenario 2 Status")
async def get_scenario_2_status(session_id: str):
    """Get the current status of a running Scenario 2 session."""
    
    if session_id not in demo_state["scenario_2"]["active_sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = demo_state["scenario_2"]["active_sessions"][session_id]
    
    return {
        "session_id": session_id,
        "scenario": "playbook_driven_debugging", 
        "status": session["status"],
        "progress": session["progress"],
        "current_step": session["current_step"],
        "total_steps": session["total_steps"],
        "incident_id": session["incident_id"],
        "playbook_id": session["playbook_id"],
        "execution_id": session["execution_id"],
        "started_at": session["started_at"],
        "last_update": datetime.utcnow().isoformat()
    }

@router.get("/synthetic-data/status", summary="Get Synthetic Data Status")
async def get_synthetic_data_status():
    """Get the status of loaded synthetic data."""
    
    return {
        "loaded": synthetic_data_loader.is_loaded(),
        "data_sources": {
            "historical_incidents": len(synthetic_data_loader.get_historical_incidents()) if synthetic_data_loader.is_loaded() else 0,
            "infrastructure_blueprints": len(synthetic_data_loader.get_infrastructure_blueprints()) if synthetic_data_loader.is_loaded() else 0,
            "playbooks": len(synthetic_data_loader.get_playbooks()) if synthetic_data_loader.is_loaded() else 0,
            "gcp_metrics": len(synthetic_data_loader.get_gcp_observability_data()) if synthetic_data_loader.is_loaded() else 0
        },
        "last_loaded": synthetic_data_loader.get_last_loaded_timestamp() if synthetic_data_loader.is_loaded() else None,
        "demo_scenarios_ready": synthetic_data_loader.is_loaded(),
        "timestamp": datetime.utcnow().isoformat()
    }

# Background execution functions

async def execute_scenario_1(session_id: str, incident_id: str, synthetic_incident: Dict[str, Any], historical_incidents: List[Dict[str, Any]]):
    """Execute Scenario 1: Past Incident Correlation in background."""
    
    try:
        session = demo_state["scenario_1"]["active_sessions"][session_id]
        
        # Step 1: Alert Reception (0-15%)
        session["status"] = "alert_received"
        session["progress"] = 10
        session["current_step"] = "Alert received: Payment API latency spike detected"
        await asyncio.sleep(2)
        
        # Step 2: Historical Search (15-35%)
        session["status"] = "searching_history"
        session["progress"] = 25
        session["current_step"] = "Searching incident history for similar patterns"
        session["steps_completed"] = 1
        await asyncio.sleep(3)
        
        # Step 3: Correlation Analysis (35-55%)
        session["status"] = "correlating"
        session["progress"] = 45
        session["current_step"] = "Found 2 similar incidents, analyzing correlation"
        session["steps_completed"] = 2
        await asyncio.sleep(2)
        
        # Step 4: Hypothesis Testing (55-75%)
        session["status"] = "testing_hypothesis"
        session["progress"] = 65
        session["current_step"] = "Testing hypothesis: Redis cache performance degradation"
        session["steps_completed"] = 3
        await asyncio.sleep(3)
        
        # Step 5: Verification (75-90%)
        session["status"] = "verifying"
        session["progress"] = 80
        session["current_step"] = "Verifying cache metrics: Hit rate 38% (normal: >90%)"
        session["steps_completed"] = 4
        await asyncio.sleep(2)
        
        # Step 6: Root Cause & Resolution (90-100%)
        session["status"] = "completed"
        session["progress"] = 100
        session["current_step"] = "Root cause identified: Redis memory eviction. Recommended: Scale Redis nodes"
        session["steps_completed"] = 6
        session["root_cause"] = "Redis memory eviction causing cache misses"
        session["confidence_score"] = 0.92
        session["resolution"] = "Scale Redis nodes from 2 to 4 instances"
        session["completed_at"] = datetime.utcnow().isoformat()
        
        # Update last execution
        demo_state["scenario_1"]["last_execution"] = {
            "session_id": session_id,
            "incident_id": incident_id,
            "completed_at": session["completed_at"],
            "result": "success"
        }
        
    except Exception as e:
        logger.error(f"Error executing scenario 1: {e}")
        session["status"] = "error"
        session["error"] = str(e)

async def execute_scenario_2(session_id: str, incident_id: str, playbook_id: str, execution_id: str, synthetic_incident: Dict[str, Any], playbook_data: Dict[str, Any]):
    """Execute Scenario 2: Playbook-Driven Debugging in background."""
    
    try:
        session = demo_state["scenario_2"]["active_sessions"][session_id]
        
        # Step 1: Playbook Selection & Step 1 (0-30%)
        session["status"] = "executing_step_1"
        session["progress"] = 15
        session["current_step"] = "Step 1: Check database connection pool utilization"
        await asyncio.sleep(3)
        
        session["progress"] = 30
        session["current_step"] = "Step 1 Result: Pool utilization at 95% (190/200 connections)"
        session["current_step_number"] = 1
        await asyncio.sleep(2)
        
        # Step 2: Connection Timeout Analysis (30-60%)
        session["status"] = "executing_step_2"
        session["progress"] = 45
        session["current_step"] = "Step 2: Analyze connection timeout errors in logs"
        await asyncio.sleep(3)
        
        session["progress"] = 60
        session["current_step"] = "Step 2 Result: Found 15 timeout errors in last 30 minutes"
        session["current_step_number"] = 2
        await asyncio.sleep(2)
        
        # Step 3: Long-Running Queries (60-100%)
        session["status"] = "executing_step_3"
        session["progress"] = 75
        session["current_step"] = "Step 3: Check for long-running queries"
        await asyncio.sleep(3)
        
        session["progress"] = 90
        session["current_step"] = "Step 3 Result: Average query time 8.5s (threshold: < 5s)"
        session["current_step_number"] = 3
        await asyncio.sleep(2)
        
        # Final Analysis
        session["status"] = "completed"
        session["progress"] = 100
        session["current_step"] = "Playbook execution complete: Root cause identified"
        session["root_cause"] = "Long-running bulk update query monopolizing connections"
        session["confidence_score"] = 0.88
        session["recommendations"] = [
            "Kill long-running query (PID: 12345)",
            "Optimize bulk update to use batching (50k → 5k rows)",
            "Increase connection pool size (200 → 300)"
        ]
        session["completed_at"] = datetime.utcnow().isoformat()
        
        # Update last execution
        demo_state["scenario_2"]["last_execution"] = {
            "session_id": session_id,
            "incident_id": incident_id,
            "playbook_id": playbook_id,
            "execution_id": execution_id,
            "completed_at": session["completed_at"],
            "result": "success"
        }
        
    except Exception as e:
        logger.error(f"Error executing scenario 2: {e}")
        session["status"] = "error"
        session["error"] = str(e)

@router.delete("/scenario_1/{session_id}", summary="Clean up Scenario 1 session")
async def cleanup_scenario_1(session_id: str):
    """Clean up a completed or failed Scenario 1 session."""
    
    if session_id in demo_state["scenario_1"]["active_sessions"]:
        del demo_state["scenario_1"]["active_sessions"][session_id]
        return {"message": f"Session {session_id} cleaned up successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.delete("/scenario_2/{session_id}", summary="Clean up Scenario 2 session") 
async def cleanup_scenario_2(session_id: str):
    """Clean up a completed or failed Scenario 2 session."""
    
    if session_id in demo_state["scenario_2"]["active_sessions"]:
        del demo_state["scenario_2"]["active_sessions"][session_id]
        return {"message": f"Session {session_id} cleaned up successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/health", summary="Demo scenarios health check")
async def health_check():
    """Health check endpoint for demo scenarios service."""
    
    active_scenario_1 = len(demo_state["scenario_1"]["active_sessions"])
    active_scenario_2 = len(demo_state["scenario_2"]["active_sessions"])
    
    return {
        "status": "healthy",
        "service": "demo_scenarios",
        "synthetic_data_loaded": synthetic_data_loader.is_loaded(),
        "active_sessions": {
            "scenario_1": active_scenario_1,
            "scenario_2": active_scenario_2,
            "total": active_scenario_1 + active_scenario_2
        },
        "gcp_integrations": {
            "monitoring": gcp_monitoring_client.is_mock(),
            "logging": gcp_logging_client.is_mock()
        },
        "timestamp": datetime.utcnow().isoformat()
    }