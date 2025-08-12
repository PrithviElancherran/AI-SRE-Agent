"""
Synthetic Data Loader Service for AI SRE Agent Demo

This service loads synthetic data for demo scenarios including GCP metrics,
historical incidents, playbooks, and analysis results.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from uuid import uuid4

from models.incident import (
    Incident, IncidentCreate, IncidentSeverity, IncidentStatus,
    IncidentSymptom, SymptomType, MetricData
)
from models.playbook import (
    Playbook, PlaybookStep, PlaybookExecution,
    ExecutionStatus, StepType
)
from models.analysis import (
    AnalysisResult, EvidenceItem, EvidenceType
)
from models.user import User, UserRole

logger = logging.getLogger(__name__)

class SyntheticDataLoader:
    """Service for loading synthetic data for demo scenarios"""
    
    def __init__(self, data_directory: str = None):
        # Use environment variable or default based on context
        if data_directory:
            self.data_directory = Path(data_directory)
        else:
            # Check if running in Docker (by checking if /app exists)
            if os.path.exists("/app") and os.getcwd().startswith("/app"):
                self.data_directory = Path("/app/data/synthetic")
            else:
                # Local development path
                self.data_directory = Path("/Users/prith/Hazel_Projects/4.0-ai-sre-agent/data/synthetic")
        self.scenario_1_data = None
        self.scenario_2_data = None
        self.historical_incidents = None
        self.playbook_configurations = None
        self.analysis_results = None
        self.infrastructure_blueprint = None
        self.users = None
        self._loaded = False
        self._last_loaded = None
        
        # In-memory storage for demo
        self.loaded_incidents: List[Incident] = []
        self.loaded_playbooks: List[Playbook] = []
        self.loaded_executions: List[PlaybookExecution] = []
        self.loaded_analyses: List[AnalysisResult] = []
        self.loaded_users: List[User] = []
        self.gcp_metrics_cache: Dict[str, List[Dict]] = {}
        self.gcp_logs_cache: Dict[str, List[Dict]] = {}
        self.gcp_errors_cache: Dict[str, List[Dict]] = {}
        self.gcp_traces_cache: Dict[str, List[Dict]] = {}
    
    async def load_all_synthetic_data(self) -> bool:
        """Load all synthetic data files"""
        try:
            logger.info("Loading synthetic data for AI SRE Agent demo...")
            
            # Load all data files
            await self._load_scenario_data()
            await self._load_historical_incidents()
            await self._load_playbook_configurations()
            await self._load_analysis_results()
            await self._load_infrastructure_blueprint()
            await self._load_users()
            
            # Process and transform data
            await self._process_incidents()
            await self._process_playbooks()
            await self._process_analyses()
            await self._process_gcp_data()
            await self._create_demo_executions()
            
            self._loaded = True
            self._last_loaded = datetime.utcnow().isoformat()
            
            logger.info(f"Loaded {len(self.loaded_incidents)} incidents, "
                       f"{len(self.loaded_playbooks)} playbooks, "
                       f"{len(self.loaded_analyses)} analyses, "
                       f"{len(self.loaded_executions)} executions")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading synthetic data: {e}")
            # Create fallback data if files don't exist
            await self._create_fallback_data()
            self._loaded = True
            self._last_loaded = datetime.utcnow().isoformat()
            return True
    
    async def _create_fallback_data(self):
        """Create minimal fallback data for demo when files don't exist"""
        logger.info("Creating fallback synthetic data for demo")
        
        # Create fallback historical incidents
        fallback_incidents = [
            {
                "incident_id": "INC-2024-045",
                "timestamp": (datetime.utcnow() - timedelta(hours=24)).isoformat(),
                "service_name": "payment-api",
                "region": "us-central1",
                "severity": "high",
                "status": "resolved",
                "symptoms": ["Payment API latency > 5s", "Redis cache hit rate 35%"],
                "root_cause": "Redis memory eviction due to insufficient capacity",
                "resolution": "Scaled Redis nodes from 2 to 4, cache hit rate restored to 95%",
                "resolution_time": (datetime.utcnow() - timedelta(hours=23, minutes=35)).isoformat(),
                "mttr_minutes": 25,
                "affected_users": 1500
            },
            {
                "incident_id": "INC-2024-072",
                "timestamp": (datetime.utcnow() - timedelta(hours=48)).isoformat(),
                "service_name": "billing-service",
                "region": "us-central1",
                "severity": "medium",
                "status": "resolved",
                "symptoms": ["Database connection pool exhausted", "Connection timeout errors"],
                "root_cause": "Database connection leak in payment processing service",
                "resolution": "Restarted service instances, fixed connection pooling bug",
                "resolution_time": (datetime.utcnow() - timedelta(hours=47, minutes=42)).isoformat(),
                "mttr_minutes": 18,
                "affected_users": 800
            }
        ]
        
        self.historical_incidents = fallback_incidents
        
        # Create fallback playbook
        fallback_playbooks = [
            {
                "playbook_id": "PB-DB-TIMEOUT-001",
                "name": "Database Connection Timeout Troubleshooting",
                "description": "Systematic approach to diagnose and resolve database connection timeout issues",
                "version": "1.2",
                "category": "database",
                "effectiveness_score": 0.88,
                "steps": [
                    {
                        "step_id": "1",
                        "description": "Check database connection pool utilization",
                        "type": "query",
                        "query": "gcp_monitoring.query('cloudsql_database_connection_count')",
                        "expected_result": {"threshold": "< 80%"},
                        "escalation_condition": "> 90%"
                    },
                    {
                        "step_id": "2", 
                        "description": "Analyze connection timeout errors in logs",
                        "type": "search",
                        "query": "gcp_logging.search('connection_timeout OR connection_refused')",
                        "time_range": "last_30_minutes"
                    },
                    {
                        "step_id": "3",
                        "description": "Check for long-running queries",
                        "type": "query",
                        "query": "gcp_monitoring.query('cloudsql_query_execution_time')",
                        "expected_result": {"threshold": "< 5s"}
                    }
                ]
            }
        ]
        
        self.playbook_configurations = fallback_playbooks
        
        # Create scenario data
        self.scenario_1_data = {
            "incident": {
                "service_name": "payment-api",
                "region": "us-central1",
                "severity": "high",
                "symptoms": ["Payment API latency > 5s", "Redis cache hit rate 35%"]
            },
            "metrics": [
                {
                    "metric_name": "payment_api_latency_p95",
                    "value": 6.2,
                    "timestamp": datetime.utcnow().isoformat(),
                    "labels": {"service": "payment-api", "region": "us-central1"}
                },
                {
                    "metric_name": "redis_cache_hit_rate",
                    "value": 38,
                    "timestamp": datetime.utcnow().isoformat(),
                    "labels": {"service": "redis-cache", "region": "us-central1"}
                }
            ]
        }
        
        self.scenario_2_data = {
            "incident": {
                "service_name": "billing-service",
                "region": "us-central1",
                "severity": "medium",
                "symptoms": ["Database timeout errors", "Connection pool exhausted"]
            },
            "metrics": [
                {
                    "metric_name": "database_connection_pool_utilization",
                    "value": 95,
                    "timestamp": datetime.utcnow().isoformat(),
                    "labels": {"service": "billing-db", "region": "us-central1"}
                },
                {
                    "metric_name": "average_query_time",
                    "value": 8.5,
                    "timestamp": datetime.utcnow().isoformat(),
                    "labels": {"service": "billing-db", "region": "us-central1"}
                }
            ]
        }
    
    async def _load_scenario_data(self):
        """Load scenario-specific GCP data"""
        scenario_1_path = self.data_directory / "scenario_1_gcp_data.json"
        scenario_2_path = self.data_directory / "scenario_2_gcp_data.json"
        
        if scenario_1_path.exists():
            with open(scenario_1_path, 'r') as f:
                self.scenario_1_data = json.load(f)
                logger.info(f"Loaded scenario 1 data with {len(self.scenario_1_data.get('metrics', []))} metrics")
        
        if scenario_2_path.exists():
            with open(scenario_2_path, 'r') as f:
                self.scenario_2_data = json.load(f)
                logger.info(f"Loaded scenario 2 data with {len(self.scenario_2_data.get('metrics', []))} metrics")
    
    async def _load_historical_incidents(self):
        """Load historical incidents data"""
        incidents_path = self.data_directory / "historical_incidents.json"
        
        if incidents_path.exists():
            with open(incidents_path, 'r') as f:
                self.historical_incidents = json.load(f)
                logger.info(f"Loaded {len(self.historical_incidents)} historical incidents")
    
    async def _load_playbook_configurations(self):
        """Load playbook configurations"""
        playbooks_path = self.data_directory / "playbook_configurations.json"
        
        if playbooks_path.exists():
            with open(playbooks_path, 'r') as f:
                self.playbook_configurations = json.load(f)
                logger.info(f"Loaded {len(self.playbook_configurations)} playbook configurations")
    
    async def _load_analysis_results(self):
        """Load analysis results data"""
        analysis_path = self.data_directory / "analysis_results.json"
        
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                self.analysis_results = json.load(f)
                logger.info(f"Loaded {len(self.analysis_results)} analysis results")
    
    async def _load_infrastructure_blueprint(self):
        """Load infrastructure blueprint"""
        blueprint_path = self.data_directory / "infrastructure_blueprint.json"
        
        if blueprint_path.exists():
            with open(blueprint_path, 'r') as f:
                self.infrastructure_blueprint = json.load(f)
                logger.info("Loaded infrastructure blueprint")
    
    async def _load_users(self):
        """Load users data"""
        users_path = self.data_directory / "users.json"
        
        if users_path.exists():
            with open(users_path, 'r') as f:
                self.users = json.load(f)
                logger.info(f"Loaded {len(self.users)} users")
    
    async def _process_incidents(self):
        """Process historical incidents into Incident models"""
        if not self.historical_incidents:
            logger.warning("No historical incidents data loaded")
            return
        
        logger.info(f"Processing {len(self.historical_incidents)} historical incidents")
        
        for incident_data in self.historical_incidents:
            try:
                # Convert to proper enums
                severity = self._map_severity(incident_data.get('severity', 'medium'))
                status = self._map_status(incident_data.get('status', 'resolved'))
                
                # Create symptoms
                symptoms = []
                for idx, symptom_text in enumerate(incident_data.get('symptoms', [])):
                    # Map symptom text to appropriate symptom type
                    symptom_type = self._map_symptom_type(symptom_text)
                    
                    # Create proper metric data based on symptom type
                    metric_data = self._create_metric_data(symptom_type, symptom_text)
                    
                    symptoms.append(IncidentSymptom(
                        symptom_id=str(uuid4()),
                        symptom_type=symptom_type,
                        description=symptom_text,
                        metric_data=metric_data,
                        detected_at=incident_data['timestamp'],
                        severity_score=self._calculate_symptom_severity(symptom_text)
                    ))
                
                # Create incident
                incident = Incident(
                    id=uuid4(),
                    incident_id=incident_data['incident_id'],
                    title=f"Incident {incident_data['incident_id']}: {incident_data.get('service_name', 'Unknown')} Issue",
                    description=incident_data.get('root_cause', 'Production incident'),
                    severity=severity,
                    status=status,
                    service_name=incident_data.get('service_name', 'unknown-service'),
                    region=incident_data.get('region', 'us-central1'),
                    created_at=incident_data['timestamp'],
                    timestamp=incident_data['timestamp'],
                    incident_symptoms=symptoms,
                    root_cause=incident_data.get('root_cause'),
                    resolution=incident_data.get('resolution'),
                    mttr_minutes=incident_data.get('mttr_minutes'),
                    affected_users=incident_data.get('affected_users'),
                    created_by="system",
                    assigned_to="sre_team",
                    acknowledged_at=incident_data['timestamp'],
                    resolved_at=incident_data.get('resolution_time'),
                    tags=["demo", "synthetic", incident_data.get('service_name', 'unknown')]
                )
                
                self.loaded_incidents.append(incident)
                
            except Exception as e:
                logger.error(f"Error processing incident {incident_data.get('incident_id')}: {e}")
        
        logger.info(f"Successfully processed {len(self.loaded_incidents)} incidents")
    
    async def _process_playbooks(self):
        """Process playbook configurations into Playbook models"""
        if not self.playbook_configurations:
            return
        
        for playbook_data in self.playbook_configurations:
            try:
                # Process steps
                steps = []
                for step_data in playbook_data.get('steps', []):
                    # Convert escalation_condition to string if it's a dict
                    escalation_condition = step_data.get('escalation_condition', '')
                    if isinstance(escalation_condition, dict):
                        threshold = escalation_condition.get('threshold', '')
                        operator = escalation_condition.get('operator', '')
                        escalation_condition = f"{operator} {threshold}" if operator and threshold else str(escalation_condition)
                    
                    step = PlaybookStep(
                        id=uuid4(),
                        step_id=step_data['step_id'],
                        description=step_data['description'],
                        step_type=self._map_step_type(step_data.get('step_type', 'manual_action')),
                        order=step_data.get('order', len(steps) + 1),
                        query=step_data.get('query', ''),
                        expected_result=step_data.get('expected_result'),
                        escalation_condition=escalation_condition,
                        gcp_integration=step_data.get('gcp_integration'),
                        timeout_minutes=step_data.get('timeout_minutes', 5),
                        retry_count=step_data.get('retry_count', 3),
                        prerequisites=step_data.get('prerequisites', []),
                        dependencies=step_data.get('dependencies', []),
                        approval_required=step_data.get('approval_required', False),
                        created_at=datetime.utcnow()
                    )
                    steps.append(step)
                
                # Create playbook
                playbook = Playbook(
                    playbook_id=playbook_data['playbook_id'],
                    name=playbook_data['name'],
                    description=playbook_data['description'],
                    version=playbook_data.get('version', '1.0'),
                    category=self._categorize_playbook(playbook_data['name']),
                    target_severity=[IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
                    estimated_duration_minutes=self._estimate_duration(steps),
                    steps=steps,
                    prerequisites=[],
                    success_criteria=[],
                    rollback_steps=[],
                    tags=["demo", "synthetic", playbook_data.get('category', 'general')],
                    is_active=True,
                    created_by="sre_team",
                    created_at=datetime.utcnow().isoformat(),
                    author="SRE Team",
                    effectiveness_score=playbook_data.get('effectiveness_score', 0.85),
                    success_rate=0.92,
                    usage_count=25,
                    last_executed_at=(datetime.utcnow() - timedelta(days=2)).isoformat()
                )
                
                self.loaded_playbooks.append(playbook)
                
            except Exception as e:
                logger.error(f"Error processing playbook {playbook_data.get('playbook_id')}: {e}")
    
    async def _process_analyses(self):
        """Process analysis results into AnalysisResult models"""
        if not self.analysis_results:
            return
        
        for analysis_data in self.analysis_results:
            try:
                # Create evidence items
                evidence_items = []
                for evidence_text in analysis_data.get('evidence', []):
                    evidence_item = EvidenceItem(
                        evidence_id=str(uuid4()),
                        evidence_type=self._map_evidence_type(evidence_text),
                        description=evidence_text,
                        data={},
                        source="GCP Monitoring",
                        relevance_score=0.9,
                        quality_score=0.85,
                        timestamp=analysis_data['created_at'],
                        collected_at=datetime.utcnow().isoformat(),
                        validation_status="validated"
                    )
                    evidence_items.append(evidence_item)
                
                # Create analysis result
                analysis = AnalysisResult(
                    analysis_id=analysis_data['analysis_id'],
                    incident_id=analysis_data['incident_id'],
                    analysis_type=analysis_data.get('analysis_type', 'incident_analysis'),
                    confidence_score=analysis_data.get('confidence_score', 0.8),
                    reasoning_steps=[
                        "Analyzed GCP monitoring metrics",
                        "Correlated with historical incidents",
                        "Identified pattern matches",
                        "Generated root cause hypothesis"
                    ],
                    evidence_items=evidence_items,
                    recommendations=[analysis_data.get('recommendation', 'Follow standard procedures')],
                    created_at=analysis_data['created_at'],
                    updated_at=analysis_data['created_at'],
                    analyzed_by="ai_sre_agent",
                    status="completed"
                )
                
                self.loaded_analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Error processing analysis {analysis_data.get('analysis_id')}: {e}")
    
    async def _process_gcp_data(self):
        """Process GCP observability data for caching"""
        try:
            # Process scenario 1 data
            if self.scenario_1_data:
                self.gcp_metrics_cache['scenario_1'] = self.scenario_1_data.get('metrics', [])
                self.gcp_logs_cache['scenario_1'] = self.scenario_1_data.get('logs', [])
                self.gcp_errors_cache['scenario_1'] = self.scenario_1_data.get('errors', [])
                
            # Process scenario 2 data
            if self.scenario_2_data:
                self.gcp_metrics_cache['scenario_2'] = self.scenario_2_data.get('metrics', [])
                self.gcp_logs_cache['scenario_2'] = self.scenario_2_data.get('logs', [])
                self.gcp_errors_cache['scenario_2'] = self.scenario_2_data.get('errors', [])
                
            logger.info("Processed GCP observability data for caching")
            
        except Exception as e:
            logger.error(f"Error processing GCP data: {e}")
    
    async def _create_demo_executions(self):
        """Create demo playbook executions"""
        try:
            if self.loaded_playbooks:
                for playbook in self.loaded_playbooks[:2]:  # Create executions for first 2 playbooks
                    execution = PlaybookExecution(
                        execution_id=str(uuid4()),
                        playbook_id=playbook.playbook_id,
                        incident_id=f"INC-DEMO-{str(uuid4())[:6].upper()}",
                        status=ExecutionStatus.COMPLETED,
                        execution_mode="manual",
                        started_at=(datetime.utcnow() - timedelta(hours=1)).isoformat(),
                        completed_at=datetime.utcnow().isoformat(),
                        executed_by="demo_user",
                        step_results=[],
                        root_cause_found=True,
                        confidence_score=0.88
                    )
                    self.loaded_executions.append(execution)
                    
        except Exception as e:
            logger.error(f"Error creating demo executions: {e}")
    
    # Helper methods
    def _map_severity(self, severity_str: str) -> IncidentSeverity:
        """Map string severity to enum"""
        mapping = {
            'low': IncidentSeverity.LOW,
            'medium': IncidentSeverity.MEDIUM,
            'high': IncidentSeverity.HIGH,
            'critical': IncidentSeverity.CRITICAL
        }
        return mapping.get(severity_str.lower(), IncidentSeverity.MEDIUM)
    
    def _map_status(self, status_str: str) -> IncidentStatus:
        """Map string status to enum"""
        mapping = {
            'open': IncidentStatus.OPEN,
            'investigating': IncidentStatus.INVESTIGATING,
            'in_progress': IncidentStatus.INVESTIGATING,
            'identified': IncidentStatus.IDENTIFIED,
            'monitoring': IncidentStatus.MONITORING,
            'resolved': IncidentStatus.RESOLVED,
            'closed': IncidentStatus.CLOSED,
            'reopened': IncidentStatus.REOPENED
        }
        return mapping.get(status_str.lower(), IncidentStatus.OPEN)
    
    def _map_symptom_type(self, symptom_text: str) -> SymptomType:
        """Map symptom text to appropriate symptom type enum"""
        symptom_lower = symptom_text.lower()
        
        if 'latency' in symptom_lower or 'response time' in symptom_lower or 'slow' in symptom_lower:
            return SymptomType.LATENCY
        elif 'error rate' in symptom_lower or 'errors' in symptom_lower or '500' in symptom_lower:
            return SymptomType.ERROR_RATE
        elif 'cache' in symptom_lower or 'hit rate' in symptom_lower:
            return SymptomType.CACHE_PERFORMANCE
        elif 'timeout' in symptom_lower or 'connection' in symptom_lower:
            return SymptomType.CONNECTION_TIMEOUT
        elif 'query' in symptom_lower or 'database' in symptom_lower or 'db' in symptom_lower:
            return SymptomType.QUERY_PERFORMANCE
        elif 'cpu' in symptom_lower or 'processor' in symptom_lower:
            return SymptomType.CPU_USAGE
        elif 'memory' in symptom_lower or 'ram' in symptom_lower:
            return SymptomType.MEMORY_USAGE
        elif 'disk' in symptom_lower or 'storage' in symptom_lower:
            return SymptomType.DISK_USAGE
        elif 'network' in symptom_lower or 'connectivity' in symptom_lower:
            return SymptomType.NETWORK_ISSUES
        elif 'service' in symptom_lower or 'unavailable' in symptom_lower:
            return SymptomType.SERVICE_UNAVAILABLE
        else:
            return SymptomType.LATENCY  # Default fallback
    
    def _create_metric_data(self, symptom_type: SymptomType, symptom_text: str) -> MetricData:
        """Create appropriate metric data based on symptom type"""
        
        # Extract numeric values from symptom text if possible
        import re
        numbers = re.findall(r'\d+\.?\d*', symptom_text)
        
        if symptom_type == SymptomType.LATENCY:
            return MetricData(
                metric_name="response_time_ms",
                threshold=200.0,
                actual_value=float(numbers[0]) * 1000 if numbers else 2500.0,
                unit="milliseconds",
                timestamp=datetime.utcnow()
            )
        elif symptom_type == SymptomType.ERROR_RATE:
            return MetricData(
                metric_name="error_rate_percent",
                threshold=1.0,
                actual_value=float(numbers[0]) if numbers else 15.0,
                unit="percentage",
                timestamp=datetime.utcnow()
            )
        elif symptom_type == SymptomType.CACHE_PERFORMANCE:
            return MetricData(
                metric_name="cache_hit_rate",
                threshold=85.0,
                actual_value=float(numbers[0]) if numbers else 40.0,
                unit="percentage",
                timestamp=datetime.utcnow()
            )
        elif symptom_type == SymptomType.CONNECTION_TIMEOUT:
            return MetricData(
                metric_name="connection_timeouts",
                threshold=5.0,
                actual_value=float(numbers[0]) if numbers else 45.0,
                unit="errors_per_minute",
                timestamp=datetime.utcnow()
            )
        elif symptom_type == SymptomType.QUERY_PERFORMANCE:
            return MetricData(
                metric_name="query_execution_time",
                threshold=2.0,
                actual_value=float(numbers[0]) if numbers else 8.2,
                unit="seconds",
                timestamp=datetime.utcnow()
            )
        elif symptom_type == SymptomType.CPU_USAGE:
            return MetricData(
                metric_name="cpu_utilization",
                threshold=80.0,
                actual_value=float(numbers[0]) if numbers else 95.0,
                unit="percentage",
                timestamp=datetime.utcnow()
            )
        elif symptom_type == SymptomType.MEMORY_USAGE:
            return MetricData(
                metric_name="memory_utilization",
                threshold=85.0,
                actual_value=float(numbers[0]) if numbers else 92.0,
                unit="percentage",
                timestamp=datetime.utcnow()
            )
        else:
            # Default metric data
            return MetricData(
                metric_name="generic_metric",
                threshold=100.0,
                actual_value=float(numbers[0]) if numbers else 150.0,
                unit="units",
                timestamp=datetime.utcnow()
            )
    
    def _calculate_symptom_severity(self, symptom_text: str) -> float:
        """Calculate symptom severity score based on text"""
        symptom_lower = symptom_text.lower()
        
        # High severity indicators
        if any(word in symptom_lower for word in ['critical', 'severe', 'timeout', 'unavailable', 'down']):
            return 0.9
        elif any(word in symptom_lower for word in ['high', 'increased', 'spike', 'dropped']):
            return 0.8
        elif any(word in symptom_lower for word in ['degraded', 'slow', 'delayed']):
            return 0.7
        elif any(word in symptom_lower for word in ['warning', 'elevated']):
            return 0.6
        else:
            return 0.7  # Default moderate severity
    
    def _map_step_type(self, type_str: str) -> StepType:
        """Map string step type to enum"""
        mapping = {
            'query': StepType.QUERY,
            'search': StepType.COMMAND,
            'manual': StepType.MANUAL,
            'api': StepType.API_CALL,
            'script': StepType.SCRIPT
        }
        return mapping.get(type_str.lower(), StepType.MANUAL)
    
    def _map_evidence_type(self, evidence_text: str) -> EvidenceType:
        """Map evidence text to evidence type"""
        if 'metric' in evidence_text.lower() or 'cpu' in evidence_text.lower():
            return EvidenceType.METRIC
        elif 'log' in evidence_text.lower() or 'error' in evidence_text.lower():
            return EvidenceType.LOG
        elif 'trace' in evidence_text.lower():
            return EvidenceType.TRACE
        else:
            return EvidenceType.CONFIGURATION
    
    def _categorize_playbook(self, name: str) -> str:
        """Categorize playbook based on name"""
        name_lower = name.lower()
        if 'database' in name_lower or 'db' in name_lower:
            return 'database'
        elif 'network' in name_lower:
            return 'network'
        elif 'cache' in name_lower or 'redis' in name_lower:
            return 'cache'
        else:
            return 'general'
    
    def _estimate_duration(self, steps: List[PlaybookStep]) -> int:
        """Estimate playbook duration in minutes"""
        return max(5, len(steps) * 2)  # Minimum 5 minutes, 2 minutes per step
    
    # Public API methods
    def is_loaded(self) -> bool:
        """Check if synthetic data is loaded"""
        return self._loaded
    
    def get_last_loaded_timestamp(self) -> Optional[str]:
        """Get timestamp of last successful load"""
        return self._last_loaded
    
    def get_historical_incidents(self) -> List[Dict[str, Any]]:
        """Get historical incidents as dictionaries"""
        return self.historical_incidents or []
    
    def get_infrastructure_blueprints(self) -> Dict[str, Any]:
        """Get infrastructure blueprints"""
        return self.infrastructure_blueprint or {}
    
    def get_playbooks(self) -> List[Dict[str, Any]]:
        """Get playbook configurations"""
        return self.playbook_configurations or []
    
    def get_gcp_observability_data(self) -> Dict[str, Any]:
        """Get cached GCP observability data"""
        return {
            'metrics': self.gcp_metrics_cache,
            'logs': self.gcp_logs_cache,
            'errors': self.gcp_errors_cache,
            'traces': self.gcp_traces_cache
        }
    
    def get_current_incident_scenario_1(self) -> Dict[str, Any]:
        """Get current incident data for scenario 1"""
        if self.scenario_1_data:
            return self.scenario_1_data.get('incident', {})
        return {
            "service_name": "payment-api",
            "region": "us-central1",
            "severity": "high",
            "symptoms": ["Payment API latency > 5s", "Redis cache hit rate 35%"]
        }
    
    def get_current_incident_scenario_2(self) -> Dict[str, Any]:
        """Get current incident data for scenario 2"""
        if self.scenario_2_data:
            return self.scenario_2_data.get('incident', {})
        return {
            "service_name": "billing-service", 
            "region": "us-central1",
            "severity": "medium",
            "symptoms": ["Database timeout errors", "Connection pool exhausted"]
        }
    
    def get_database_timeout_playbook(self) -> Dict[str, Any]:
        """Get database timeout playbook for scenario 2"""
        if self.playbook_configurations:
            for playbook in self.playbook_configurations:
                if 'database' in playbook['name'].lower() or 'timeout' in playbook['name'].lower():
                    return playbook
        
        # Return fallback playbook
        return {
            "playbook_id": "PB-DB-TIMEOUT-001",
            "name": "Database Connection Timeout Troubleshooting",
            "description": "Systematic approach to diagnose and resolve database connection timeout issues",
            "version": "1.2",
            "steps": [
                {
                    "step_id": "1",
                    "description": "Check database connection pool utilization",
                    "type": "query"
                },
                {
                    "step_id": "2",
                    "description": "Analyze connection timeout errors in logs", 
                    "type": "search"
                },
                {
                    "step_id": "3",
                    "description": "Check for long-running queries",
                    "type": "query"
                }
            ]
        }
    
    def get_loaded_incidents_models(self) -> List[Incident]:
        """Get loaded incident models"""
        return self.loaded_incidents
    
    def get_loaded_playbooks_models(self) -> List[Playbook]:
        """Get loaded playbook models"""
        return self.loaded_playbooks
    
    def get_loaded_analyses_models(self) -> List[AnalysisResult]:
        """Get loaded analysis models"""
        return self.loaded_analyses

# Global instance
synthetic_data_loader = SyntheticDataLoader()