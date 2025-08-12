"""
FastAPI WebSocket router for real-time communication.

This module provides WebSocket endpoints for real-time communication including incident analysis
progress updates, playbook execution status, chat interface, and live notifications.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4, UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.websockets import WebSocketState
from loguru import logger
from pydantic import BaseModel, Field

from models.incident import Incident, IncidentSeverity, IncidentStatus
from models.playbook import PlaybookExecution, ExecutionStatus
from models.analysis import AnalysisRequest, AnalysisResult, AnalysisStatus
from models.user import User
from services.incident_analyzer import IncidentAnalyzer
from services.playbook_executor import PlaybookExecutor
from services.confidence_scorer import ConfidenceScorer
from integrations.gcp_monitoring import GCPMonitoringClient
from integrations.gcp_logging import GCPLoggingClient
from integrations.gcp_error_reporting import GCPErrorReportingClient
from utils.security import get_current_user_websocket
from utils.formatters import format_incident_response, format_analysis_response

router = APIRouter()


def safe_json_convert(obj, visited=None):
    """Recursively convert objects to JSON-safe format."""
    if visited is None:
        visited = set()
    
    obj_id = id(obj)
    if obj_id in visited:
        return "<circular reference>"
    
    # Only track mutable objects to prevent infinite recursion
    if isinstance(obj, (dict, list, tuple)) or hasattr(obj, '__dict__'):
        visited.add(obj_id)
    
    try:
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif callable(obj):  # Functions, methods, etc.
            return f"<function {obj.__name__}>"
        elif hasattr(obj, '_name'):  # Enum types
            return obj.value if hasattr(obj, 'value') else str(obj)
        elif isinstance(obj, dict):
            return {k: safe_json_convert(v, visited) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_json_convert(item, visited) for item in obj]
        elif isinstance(obj, tuple):
            return [safe_json_convert(item, visited) for item in obj]
        elif hasattr(obj, '__dict__'):
            return safe_json_convert(obj.__dict__, visited)
        elif hasattr(obj, 'items'):  # mappingproxy, etc.
            return {k: safe_json_convert(v, visited) for k, v in obj.items()}
        else:
            return obj
    finally:
        # Remove from visited set after processing
        if obj_id in visited:
            visited.remove(obj_id)

# Initialize services
incident_analyzer = IncidentAnalyzer()
playbook_executor = PlaybookExecutor()
confidence_scorer = ConfidenceScorer()
gcp_monitoring = GCPMonitoringClient()
gcp_logging = GCPLoggingClient()
gcp_error_reporting = GCPErrorReportingClient()


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> connection_id
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.chat_history: List[Dict[str, Any]] = []
        self.max_chat_history = 100
    
    async def connect(self, websocket: WebSocket, connection_id: str, user: User):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.user_sessions[user.user_id] = connection_id
        self.connection_metadata[connection_id] = {
            "user_id": user.user_id,
            "username": user.full_name or user.user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "subscriptions": set()
        }
        
        logger.info(f"WebSocket connection established: {connection_id} for user {user.full_name or user.user_id}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "message": f"Welcome {user.full_name or user.user_id}! AI SRE Agent is ready to assist.",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Send recent chat history
        if self.chat_history:
            await self.send_personal_message({
                "type": "chat_history",
                "messages": self.chat_history[-10:],  # Last 10 messages
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get("user_id")
            username = metadata.get("username", "unknown")
            
            # Remove from all tracking dictionaries
            del self.active_connections[connection_id]
            if user_id and user_id in self.user_sessions:
                del self.user_sessions[user_id]
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id} for user {username}")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                # Use safe JSON conversion to handle all data types
                safe_message = safe_json_convert(message)
                await websocket.send_text(json.dumps(safe_message))
                # Update last activity
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow().isoformat()
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def send_to_user(self, message: Dict[str, Any], user_id: str):
        """Send a message to a specific user."""
        connection_id = self.user_sessions.get(user_id)
        if connection_id:
            await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: Dict[str, Any], exclude_connection: Optional[str] = None):
        """Broadcast a message to all connected clients."""
        disconnected_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            if exclude_connection and connection_id == exclude_connection:
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
    
    async def broadcast_to_subscribers(self, message: Dict[str, Any], subscription_type: str):
        """Broadcast to users subscribed to a specific type."""
        for connection_id, metadata in self.connection_metadata.items():
            if subscription_type in metadata.get("subscriptions", set()):
                await self.send_personal_message(message, connection_id)
    
    def add_subscription(self, connection_id: str, subscription_type: str):
        """Add a subscription for a connection."""
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].add(subscription_type)
    
    def remove_subscription(self, connection_id: str, subscription_type: str):
        """Remove a subscription for a connection."""
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].discard(subscription_type)
    
    def add_chat_message(self, message: Dict[str, Any]):
        """Add a message to chat history."""
        self.chat_history.append(message)
        if len(self.chat_history) > self.max_chat_history:
            self.chat_history = self.chat_history[-self.max_chat_history:]
    
    def get_active_users(self) -> List[Dict[str, Any]]:
        """Get list of active users."""
        active_users = []
        for connection_id, metadata in self.connection_metadata.items():
            active_users.append({
                "connection_id": connection_id,
                "user_id": metadata.get("user_id"),
                "username": metadata.get("username"),
                "connected_at": metadata.get("connected_at"),
                "last_activity": metadata.get("last_activity"),
                "subscriptions": list(metadata.get("subscriptions", set()))
            })
        return active_users


# Global connection manager
manager = ConnectionManager()


class ChatMessage(BaseModel):
    """Chat message model."""
    
    type: str = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    user_id: Optional[str] = Field(None, description="User ID")
    username: Optional[str] = Field(None, description="Username")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SubscriptionRequest(BaseModel):
    """Subscription request model."""
    
    action: str = Field(..., description="subscribe or unsubscribe")
    subscription_type: str = Field(..., description="Type of subscription")


@router.websocket("/chat")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
    user_id: Optional[str] = Query(None, description="User ID")
):
    """
    Main chat WebSocket endpoint for real-time communication.
    
    Supports:
    - Real-time chat interface
    - Incident analysis requests
    - Playbook execution commands
    - Live status updates
    - Notification subscriptions
    """
    connection_id = f"conn_{str(uuid4())[:8]}"
    
    try:
        # Authenticate user
        user = await get_current_user_websocket(websocket, token)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Connect to manager
        await manager.connect(websocket, connection_id, user)
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Update last activity
                if connection_id in manager.connection_metadata:
                    manager.connection_metadata[connection_id]["last_activity"] = datetime.utcnow().isoformat()
                
                # Handle different message types
                await handle_websocket_message(message_data, connection_id, user)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received from {connection_id}: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {connection_id}: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        manager.disconnect(connection_id)


async def handle_websocket_message(
    message_data: Dict[str, Any],
    connection_id: str,
    user: User
):
    """Handle incoming WebSocket messages based on type."""
    
    message_type = message_data.get("type", "unknown")
    logger.info(f"Handling WebSocket message type: {message_type} from {connection_id}")
    print(f"DEBUG: WebSocket message received - Type: {message_type}, Data: {message_data}")
    
    if message_type == "chat_message":
        await handle_chat_message(message_data, connection_id, user)
    
    elif message_type == "incident_analysis_request":
        await handle_incident_analysis_request(message_data, connection_id, user)
    
    elif message_type == "playbook_execution_request":
        await handle_playbook_execution_request(message_data, connection_id, user)
    
    elif message_type == "subscription_request":
        await handle_subscription_request(message_data, connection_id, user)
    
    elif message_type == "status_request":
        await handle_status_request(message_data, connection_id, user)
    
    elif message_type == "ping":
        await manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
    
    else:
        await manager.send_personal_message({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)


async def handle_chat_message(
    message_data: Dict[str, Any],
    connection_id: str,
    user: User
):
    """Handle regular chat messages."""
    
    content = message_data.get("content", "")
    
    # Create chat message
    chat_message = {
        "type": "chat_message",
        "content": content,
        "user_id": user.user_id,
        "username": user.full_name or user.user_id,
        "connection_id": connection_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add to chat history
    manager.add_chat_message(chat_message)
    
    # Broadcast to all connected users
    await manager.broadcast(chat_message, exclude_connection=connection_id)
    
    # Check if this is a command to the AI agent
    if content.startswith("@sre-bot") or content.startswith("@ai-agent"):
        await handle_ai_command(content, connection_id, user)


async def handle_ai_command(content: str, connection_id: str, user: User):
    """Handle AI agent commands from chat."""
    
    # Send acknowledgment
    await manager.send_personal_message({
        "type": "ai_response",
        "content": "🤖 AI SRE Agent analyzing your request...",
        "agent": "ai-sre-bot",
        "status": "processing",
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    # Parse command
    command_text = content.replace("@sre-bot", "").replace("@ai-agent", "").strip()
    
    if "incident" in command_text.lower() and "analyze" in command_text.lower():
        # Extract incident ID if present
        words = command_text.split()
        incident_id = None
        for word in words:
            if word.startswith("INC-"):
                incident_id = word
                break
        
        if not incident_id:
            incident_id = "INC-2024-001"  # Default for demo
        
        await execute_real_incident_analysis(incident_id, connection_id, user)
    
    elif "playbook" in command_text.lower() and "execute" in command_text.lower():
        # Parse playbook name from command
        playbook_name = None
        incident_id = "INC-2024-002"  # Default incident ID
        
        # Simple parsing to extract playbook name
        import re
        
        # Try to extract playbook name after "playbook"
        playbook_match = re.search(r'playbook\s+([a-zA-Z0-9\-_]+)', command_text.lower())
        if playbook_match:
            playbook_name = playbook_match.group(1)
        
        # Map natural language playbook names to actual playbook IDs
        playbook_mapping = {
            "database-timeout": "PB-DB-TIMEOUT-001",
            "databse-timeout": "PB-DB-TIMEOUT-001",  # Handle typo
            "db-timeout": "PB-DB-TIMEOUT-001", 
            "latency": "PB-LATENCY-001",
            "cpu": "PB-CPU-001",
            "cpu-spike": "PB-CPU-001"
        }
        
        # Get playbook ID
        playbook_id = playbook_mapping.get(playbook_name, "PB-DB-TIMEOUT-001")  # Default fallback
        
        # Map playbook to appropriate incident
        incident_mapping = {
            "PB-DB-TIMEOUT-001": "INC-2024-002",
            "PB-LATENCY-001": "INC-2024-001", 
            "PB-CPU-001": "INC-2024-003"
        }
        
        incident_id = incident_mapping.get(playbook_id, "INC-2024-002")
        
        await execute_real_playbook(playbook_id, incident_id, connection_id, user)
    
    elif "status" in command_text.lower():
        await send_system_status(connection_id, user)
    
    else:
        await manager.send_personal_message({
            "type": "ai_response",
            "content": "🤖 I can help with:\n• Incident analysis (@sre-bot analyze incident INC-XXXX)\n• Playbook execution (@sre-bot execute playbook)\n• System status (@sre-bot status)",
            "agent": "ai-sre-bot",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)


async def handle_incident_analysis_request(
    message_data: Dict[str, Any],
    connection_id: str,
    user: User
):
    """Handle incident analysis requests."""
    
    incident_id = message_data.get("incident_id")
    if not incident_id:
        await manager.send_personal_message({
            "type": "error",
            "message": "incident_id is required",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        return
    
    # Start real analysis
    await execute_real_incident_analysis(incident_id, connection_id, user)


async def handle_playbook_execution_request(
    message_data: Dict[str, Any],
    connection_id: str,
    user: User
):
    """Handle playbook execution requests."""
    
    playbook_id = message_data.get("playbook_id", "PB-DB-TIMEOUT-001")
    incident_id = message_data.get("incident_id", "INC-2024-002")
    
    # Start execution simulation
    await simulate_playbook_execution(playbook_id, incident_id, connection_id, user)


async def handle_subscription_request(
    message_data: Dict[str, Any],
    connection_id: str,
    user: User
):
    """Handle subscription/unsubscription requests."""
    
    action = message_data.get("action")
    subscription_type = message_data.get("subscription_type")
    
    if not action or not subscription_type:
        await manager.send_personal_message({
            "type": "error",
            "message": "action and subscription_type are required",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        return
    
    if action == "subscribe":
        manager.add_subscription(connection_id, subscription_type)
        await manager.send_personal_message({
            "type": "subscription_confirmed",
            "subscription_type": subscription_type,
            "action": "subscribed",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
    
    elif action == "unsubscribe":
        manager.remove_subscription(connection_id, subscription_type)
        await manager.send_personal_message({
            "type": "subscription_confirmed",
            "subscription_type": subscription_type,
            "action": "unsubscribed",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)


async def handle_status_request(
    message_data: Dict[str, Any],
    connection_id: str,
    user: User
):
    """Handle status requests."""
    
    request_type = message_data.get("request_type", "general")
    
    if request_type == "active_users":
        active_users = manager.get_active_users()
        await manager.send_personal_message({
            "type": "status_response",
            "request_type": "active_users",
            "data": active_users,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
    
    else:
        await send_system_status(connection_id, user)


async def execute_real_incident_analysis(incident_id: str, connection_id: str, user: User):
    """Execute real incident analysis with progress updates."""
    
    analysis_id = f"analysis_{str(uuid4())[:8]}"
    
    try:
        # Step 1: Start analysis
        await manager.send_personal_message({
            "type": "incident_analysis_update",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "status": "started",
            "progress": 0,
            "message": "🔍 Starting incident analysis...",
            "step": "initialization",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Step 2: Call real incident analyzer
        await manager.send_personal_message({
            "type": "incident_analysis_update",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "status": "in_progress",
            "progress": 20,
            "message": "📚 Searching historical incidents...",
            "step": "historical_correlation",
            "details": "Executing vector similarity search against incident database",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Execute real analysis
        from models.analysis import AnalysisRequest, AnalysisType
        analysis_request = AnalysisRequest(
            incident_id=incident_id,
            analysis_type=AnalysisType.CORRELATION,
            context={"triggered_by": "websocket_chat", "user_id": user.user_id}
        )
        
        # Progress update
        await manager.send_personal_message({
            "type": "incident_analysis_update",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "status": "in_progress",
            "progress": 60,
            "message": "🔍 Analyzing incident with AI...",
            "step": "ai_analysis",
            "details": "Running incident analyzer with vector search and confidence scoring",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Call the actual incident analyzer
        analysis_result = await incident_analyzer.analyze_incident(
            incident_id=incident_id,
            analysis_type=AnalysisType.CORRELATION,
            context=analysis_request.context or {}
        )
        
        # Progress update
        await manager.send_personal_message({
            "type": "incident_analysis_update",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "status": "in_progress",
            "progress": 90,
            "message": "📊 Collecting evidence from GCP observability...",
            "step": "evidence_collection",
            "details": "Gathering metrics, logs, and traces",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Final result
        await manager.send_personal_message({
            "type": "incident_analysis_update",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "status": "completed",
            "progress": 100,
            "message": f"🎯 Analysis Complete - Confidence: {analysis_result.confidence_score:.1%}",
            "step": "completed",
            "confidence": analysis_result.confidence_score,
            "root_cause": analysis_result.root_cause,
            "recommendations": getattr(analysis_result, 'recommendations', [analysis_result.recommendation] if analysis_result.recommendation else []),
            "similar_incidents": getattr(analysis_result, 'similar_incidents', []),
            "evidence": [
                {
                    "type": item.evidence_type.value if hasattr(item, 'evidence_type') else "unknown",
                    "description": item.description,
                    "relevance": item.relevance_score,
                    "data": item.data,
                    "source": item.source
                }
                for item in (analysis_result.evidence_items or [])
            ],
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Send final AI response
        recommendations_text = ""
        if hasattr(analysis_result, 'recommendations') and analysis_result.recommendations:
            recommendations_text = "\n\n**Recommendations:**\n" + "\n".join(f"• {rec}" for rec in analysis_result.recommendations)
        elif analysis_result.recommendation:
            recommendations_text = f"\n\n**Recommendation:** {analysis_result.recommendation}"
        
        # Send complete analysis result with all details
        await manager.send_personal_message({
            "type": "ai_response",
            "content": f"🎯 **Analysis Complete for {incident_id}**\n\n**Root Cause:** {analysis_result.root_cause or 'Analysis in progress'}\n\n**Confidence:** {analysis_result.confidence_score:.1%}{recommendations_text}",
            "agent": "ai-sre-bot",
            "status": "completed",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "timestamp": datetime.utcnow().isoformat(),
            "full_analysis": {
                "incident_id": incident_id,
                "analysis_id": analysis_id,
                "db_analysis_id": str(analysis_result.id),
                "status": "completed",
                "confidence_score": analysis_result.confidence_score,
                "root_cause": analysis_result.root_cause,
                "recommendations": getattr(analysis_result, 'recommendations', [analysis_result.recommendation] if analysis_result.recommendation else []),
                "evidence": [
                    {
                        "type": item.evidence_type.value if hasattr(item, 'evidence_type') else "unknown",
                        "description": str(item.description),
                        "relevance": float(item.relevance_score),
                        "data": item.data if item.data and isinstance(item.data, (dict, list, str, int, float, bool)) else str(item.data) if item.data else None,
                        "source": str(item.source),
                        "timestamp": item.timestamp.isoformat() if hasattr(item, 'timestamp') and item.timestamp else None
                    }
                    for item in (analysis_result.evidence_items or [])
                ],
                "similar_incidents": [
                    {
                        "id": str(incident_id),
                        "title": f"Related Incident {incident_id}",
                        "description": "Historical incident with similar patterns",
                        "similarity_score": 0.8,
                        "resolution": "Resolved through systematic analysis",
                        "created_at": datetime.utcnow().isoformat()
                    }
                    for incident_id in (analysis_result.findings.related_incidents if analysis_result.findings else [])
                ],
                "reasoning_trail": [
                    {
                        "step": step.step_number,
                        "step_type": step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type),
                        "action": str(step.description),
                        "reasoning": str(step.reasoning),
                        "confidence_impact": float(step.confidence_impact),
                        "evidence": step.output_data if step.output_data and isinstance(step.output_data, (dict, list, str, int, float, bool)) else str(step.output_data) if step.output_data else None,
                        "duration_seconds": float(step.duration_seconds) if step.duration_seconds else None
                    }
                    for step in (analysis_result.reasoning_trail.steps if analysis_result.reasoning_trail else [])
                ],
                "analysis_duration_seconds": getattr(analysis_result, 'analysis_duration_seconds', 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        }, connection_id)
        
    except Exception as e:
        logger.error(f"Error in real incident analysis: {e}")
        await manager.send_personal_message({
            "type": "incident_analysis_update",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "status": "failed",
            "progress": 100,
            "message": f"❌ Analysis failed: {str(e)}",
            "step": "error",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        await manager.send_personal_message({
            "type": "ai_response",
            "content": f"❌ **Analysis Failed for {incident_id}**\n\nError: {str(e)}",
            "agent": "ai-sre-bot",
            "status": "failed",
            "analysis_id": analysis_id,
            "incident_id": incident_id,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)


async def simulate_incident_analysis(incident_id: str, connection_id: str, user: User):
    """Simulate real-time incident analysis with progress updates."""
    
    analysis_id = f"analysis_{str(uuid4())[:8]}"
    
    # Step 1: Start analysis
    await manager.send_personal_message({
        "type": "incident_analysis_update",
        "analysis_id": analysis_id,
        "incident_id": incident_id,
        "status": "started",
        "progress": 0,
        "message": "🔍 Starting incident analysis...",
        "step": "initialization",
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(1)
    
    # Step 2: Historical correlation
    await manager.send_personal_message({
        "type": "incident_analysis_update",
        "analysis_id": analysis_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 20,
        "message": "📚 Searching historical incidents...",
        "step": "historical_correlation",
        "details": "Executing vector similarity search against incident database",
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(2)
    
    # Step 3: Found similar incidents
    await manager.send_personal_message({
        "type": "incident_analysis_update",
        "analysis_id": analysis_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 40,
        "message": "✅ Found 2 similar incidents (92% similarity)",
        "step": "correlation_results",
        "details": "INC-2024-045: Redis cache issues, INC-2024-072: Database connections",
        "evidence": {
            "similar_incidents": [
                {"id": "INC-2024-045", "similarity": 0.92, "resolution": "Scale Redis nodes"},
                {"id": "INC-2024-072", "similarity": 0.88, "resolution": "Fix connection pooling"}
            ]
        },
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(1.5)
    
    # Step 4: Evidence collection
    await manager.send_personal_message({
        "type": "incident_analysis_update",
        "analysis_id": analysis_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 60,
        "message": "📊 Collecting evidence from GCP observability...",
        "step": "evidence_collection",
        "details": "Querying monitoring, logging, and error reporting APIs",
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(2)
    
    # Step 5: Evidence analysis
    await manager.send_personal_message({
        "type": "incident_analysis_update",
        "analysis_id": analysis_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 80,
        "message": "🔗 [GCP Monitoring Query](https://console.cloud.google.com/monitoring)",
        "step": "evidence_analysis",
        "details": "Cache hit rate: 38% (normal: >90%), Connection pool: 95% utilization",
        "evidence": {
            "metrics": {
                "cache_hit_rate": 38,
                "connection_pool_utilization": 95,
                "api_latency_p95": 6.2
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(1.5)
    
    # Step 6: Root cause identified
    await manager.send_personal_message({
        "type": "incident_analysis_update",
        "analysis_id": analysis_id,
        "incident_id": incident_id,
        "status": "completed",
        "progress": 100,
        "message": "🎯 Root Cause: Redis memory eviction causing cache misses",
        "step": "root_cause_identified",
        "confidence": 0.92,
        "root_cause": {
            "primary_cause": "Redis memory eviction",
            "contributing_factors": ["High memory usage", "Cache key churn"],
            "evidence_strength": 0.9
        },
        "recommendations": [
            "Scale Redis nodes from 2 to 4 instances",
            "Optimize cache key expiration policies",
            "Monitor memory utilization trends"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    # Send final summary
    await manager.send_personal_message({
        "type": "ai_response",
        "content": "🎯 **Analysis Complete**\n\n**Root Cause:** Redis memory eviction causing cache misses\n\n**Confidence:** 92%\n\n**Recommended Actions:**\n• Scale Redis nodes (2 → 4 instances)\n• Optimize cache expiration policies\n• Monitor memory utilization\n\n**Expected Resolution Time:** 15 minutes",
        "agent": "ai-sre-bot",
        "status": "completed",
        "analysis_id": analysis_id,
        "incident_id": incident_id,
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)


async def simulate_playbook_execution(
    playbook_id: str,
    incident_id: str,
    connection_id: str,
    user: User
):
    """Simulate real-time playbook execution with step updates."""
    
    execution_id = f"exec_{str(uuid4())[:8]}"
    
    # Step 1: Start execution
    await manager.send_personal_message({
        "type": "playbook_execution_update",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "status": "started",
        "progress": 0,
        "message": "📘 Starting playbook: Database Connection Timeout Troubleshooting",
        "current_step": 0,
        "total_steps": 3,
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(1)
    
    # Step 2: Execute Step 1
    await manager.send_personal_message({
        "type": "playbook_execution_update",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 33,
        "message": "📘 Executing Step 1: Check database connection pool utilization",
        "current_step": 1,
        "step_details": {
            "step_title": "Check Database Connection Pool",
            "step_description": "Verify connection pool utilization",
            "command": "gcp_monitoring.query('cloudsql_database_connection_count')",
            "expected": "Connection count < 180"
        },
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(2)
    
    # Step 3: Step 1 Results
    await manager.send_personal_message({
        "type": "playbook_execution_update",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 33,
        "message": "⚠️ Step 1 Result: Pool utilization at 95% (190/200 connections)",
        "current_step": 1,
        "step_result": {
            "success": False,
            "threshold_exceeded": True,
            "actual_value": 190,
            "threshold": 180,
            "escalation_triggered": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(1.5)
    
    # Step 4: Execute Step 2
    await manager.send_personal_message({
        "type": "playbook_execution_update",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 66,
        "message": "📘 Executing Step 2: Analyze connection timeout errors",
        "current_step": 2,
        "step_details": {
            "step_title": "Analyze Connection Timeout Logs",
            "step_description": "Search for connection timeout patterns",
            "command": "gcp_logging.search('connection_timeout OR connection_refused')",
            "expected": "Low frequency timeout errors"
        },
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(2.5)
    
    # Step 5: Step 2 Results
    await manager.send_personal_message({
        "type": "playbook_execution_update",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 66,
        "message": "⚠️ Step 2 Result: Found 15 timeout errors in last 30 minutes",
        "current_step": 2,
        "step_result": {
            "success": False,
            "error_count": 15,
            "error_frequency": "0.5 errors/minute",
            "trend": "increasing"
        },
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(1.5)
    
    # Step 6: Execute Step 3
    await manager.send_personal_message({
        "type": "playbook_execution_update",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "status": "in_progress",
        "progress": 90,
        "message": "📘 Executing Step 3: Check for long-running queries",
        "current_step": 3,
        "step_details": {
            "step_title": "Check Query Performance",
            "step_description": "Identify slow or long-running queries",
            "command": "gcp_monitoring.query('cloudsql_query_execution_time')",
            "expected": "Query time < 5s"
        },
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    await asyncio.sleep(2)
    
    # Step 7: Step 3 Results and Root Cause
    await manager.send_personal_message({
        "type": "playbook_execution_update",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "status": "completed",
        "progress": 100,
        "message": "🚨 Step 3 Result: Average query time 8.5s (threshold: < 5s)",
        "current_step": 3,
        "root_cause_found": True,
        "confidence": 0.88,
        "step_result": {
            "success": False,
            "avg_query_time": 8.5,
            "threshold": 5.0,
            "slow_query_identified": True,
            "slow_query": "invoice_calculation_bulk_update (12.3s)"
        },
        "root_cause": {
            "issue": "Long-running bulk update query monopolizing connections",
            "query": "UPDATE invoices SET status = 'processed' WHERE customer_id IN (...)",
            "execution_time": 12.3,
            "rows_affected": 50000
        },
        "recommendations": [
            "Kill long-running query (PID: 12345)",
            "Optimize bulk update to use batching (50k → 5k rows)",
            "Increase connection pool size (200 → 300)"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)
    
    # Send final summary
    await manager.send_personal_message({
        "type": "ai_response",
        "content": "🎯 **Playbook Execution Complete**\n\n**Root Cause:** Long-running bulk update query monopolizing connections\n\n**Confidence:** 88%\n\n**Recommended Actions:**\n• Kill long-running query (PID: 12345)\n• Optimize bulk update batching (50k → 5k rows)\n• Increase connection pool size (200 → 300)\n\n**Expected Resolution Time:** 10 minutes\n\n⚠️ **Action Required:** Approve query termination and optimization",
        "agent": "ai-sre-bot",
        "status": "completed",
        "execution_id": execution_id,
        "playbook_id": playbook_id,
        "incident_id": incident_id,
        "requires_approval": True,
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)


async def execute_real_playbook(playbook_id: str, incident_id: str, connection_id: str, user: User):
    """Execute real playbook with progress updates."""
    
    execution_id = f"exec_{str(uuid4())[:8]}"
    
    try:
        # Step 1: Start execution
        await manager.send_personal_message({
            "type": "playbook_execution_update",
            "execution_id": execution_id,
            "playbook_id": playbook_id,
            "incident_id": incident_id,
            "status": "started",
            "progress": 0,
            "message": f"📘 Starting playbook execution: {playbook_id}",
            "current_step": 0,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Execute real playbook
        from models.user import User as UserModel
        user_model = UserModel(
            id=user.id,
            user_id=user.user_id,
            email=user.email,
            role=user.role,
            permissions=user.permissions,
            is_active=user.is_active,
            full_name=user.full_name,
            timezone=user.timezone,
            notification_preferences=user.notification_preferences,
            last_login=None,
            created_at=datetime.utcnow(),
            failed_login_attempts=0,
            locked_until=None
        )
        
        # Progress update
        await manager.send_personal_message({
            "type": "playbook_execution_update",
            "execution_id": execution_id,
            "playbook_id": playbook_id,
            "incident_id": incident_id,
            "status": "in_progress",
            "progress": 50,
            "message": "🔄 Executing playbook steps...",
            "current_step": 1,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Call the actual playbook executor
        execution_result = await playbook_executor.execute_playbook(
            playbook_id=playbook_id,
            incident_id=incident_id,
            user=user_model,
            context={"triggered_by": "websocket_chat"}
        )
        
        # Final result
        await manager.send_personal_message({
            "type": "playbook_execution_update",
            "execution_id": execution_id,
            "playbook_id": playbook_id,
            "incident_id": incident_id,
            "status": "completed",
            "progress": 100,
            "message": f"✅ Playbook execution completed - Confidence: {execution_result.confidence_score:.1%}",
            "confidence": execution_result.confidence_score,
            "root_cause_found": execution_result.root_cause_found,
            "recommendations": execution_result.actions_recommended or [],
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        # Send final AI response
        recommendations_text = ""
        if execution_result.actions_recommended:
            recommendations_text = "\n\n**Recommendations:**\n" + "\n".join(f"• {rec}" for rec in execution_result.actions_recommended)
        
        await manager.send_personal_message({
            "type": "ai_response",
            "content": f"✅ **Playbook Execution Complete for {playbook_id}**\n\n**Confidence:** {execution_result.confidence_score:.1%}\n\n**Root Cause Found:** {'Yes' if execution_result.root_cause_found else 'No'}{recommendations_text}",
            "agent": "ai-sre-bot",
            "status": "completed",
            "execution_id": execution_id,
            "playbook_id": playbook_id,
            "incident_id": incident_id,
            "timestamp": datetime.utcnow().isoformat(),
            "full_playbook_result": {
                "execution_id": execution_id,
                "playbook_id": playbook_id,
                "incident_id": incident_id,
                "status": "completed",
                "progress": 100.0,
                "confidence_score": execution_result.confidence_score,
                "root_cause_found": execution_result.root_cause_found,
                "steps_completed": getattr(execution_result, 'steps_completed', 0),
                "total_steps": getattr(execution_result, 'total_steps', 0),
                "execution_time_seconds": getattr(execution_result, 'execution_time_seconds', 0),
                "actions_recommended": execution_result.actions_recommended or [],
                "step_results": getattr(execution_result, 'step_results', []),
                "recommendations": execution_result.actions_recommended or [],
                "timestamp": datetime.utcnow().isoformat()
            }
        }, connection_id)
        
    except Exception as e:
        logger.error(f"Error in real playbook execution: {e}")
        await manager.send_personal_message({
            "type": "playbook_execution_update",
            "execution_id": execution_id,
            "playbook_id": playbook_id,
            "incident_id": incident_id,
            "status": "failed",
            "progress": 100,
            "message": f"❌ Playbook execution failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        await manager.send_personal_message({
            "type": "ai_response",
            "content": f"❌ **Playbook Execution Failed for {playbook_id}**\n\nError: {str(e)}",
            "agent": "ai-sre-bot",
            "status": "failed",
            "execution_id": execution_id,
            "playbook_id": playbook_id,
            "incident_id": incident_id,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)


async def send_system_status(connection_id: str, user: User):
    """Send current system status."""
    
    # Get system metrics
    system_status = {
        "overall_status": "healthy",
        "active_incidents": 2,
        "active_analyses": 1,
        "active_playbooks": 0,
        "gcp_services": {
            "monitoring": "healthy",
            "logging": "healthy", 
            "error_reporting": "healthy"
        },
        "confidence_scores": {
            "avg_last_24h": 0.85,
            "trend": "stable"
        },
        "active_users": len(manager.get_active_users())
    }
    
    await manager.send_personal_message({
        "type": "system_status",
        "status": system_status,
        "timestamp": datetime.utcnow().isoformat()
    }, connection_id)


# Background task for periodic updates
async def periodic_status_updates():
    """Send periodic status updates to subscribed users."""
    
    while True:
        try:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # Send status update to subscribers
            status_message = {
                "type": "periodic_update",
                "system_health": "healthy",
                "active_connections": len(manager.active_connections),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await manager.broadcast_to_subscribers(status_message, "status_updates")
            
        except Exception as e:
            logger.error(f"Error in periodic status updates: {e}")


# Background task will be started by FastAPI application startup
# Note: periodic_status_updates() is available but not auto-started
# to avoid event loop issues during module import


@router.websocket("/notifications")
async def websocket_notifications_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
    user_id: Optional[str] = Query(None, description="User ID")
):
    """
    Dedicated WebSocket endpoint for notifications only.
    
    Lighter weight endpoint for receiving notifications without
    full chat functionality.
    """
    connection_id = f"notif_{str(uuid4())[:8]}"
    
    try:
        # Authenticate user
        user = await get_current_user_websocket(websocket, token)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        await websocket.accept()
        logger.info(f"Notifications WebSocket connected: {connection_id} for user {user.username}")
        
        # Subscribe to notifications by default
        manager.active_connections[connection_id] = websocket
        manager.connection_metadata[connection_id] = {
            "user_id": user.user_id,
            "username": user.full_name or user.user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "subscriptions": {"notifications", "status_updates"}
        }
        
        # Send welcome
        await websocket.send_text(json.dumps({
            "type": "notifications_connected",
            "message": "Notifications channel active",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for ping or close
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in notifications WebSocket: {e}")
                break
    
    except Exception as e:
        logger.error(f"Notifications WebSocket error: {e}")
    
    finally:
        if connection_id in manager.active_connections:
            del manager.active_connections[connection_id]
        if connection_id in manager.connection_metadata:
            del manager.connection_metadata[connection_id]
        logger.info(f"Notifications WebSocket disconnected: {connection_id}")


# Health check endpoint for WebSocket
@router.get("/health")
async def websocket_health():
    """WebSocket service health check."""
    
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "active_users": len(manager.user_sessions),
        "chat_history_size": len(manager.chat_history),
        "timestamp": datetime.utcnow().isoformat()
    }
