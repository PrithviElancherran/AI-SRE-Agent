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

from backend.models.incident import (
    Incident, IncidentCreate, IncidentSeverity, IncidentStatus,
    IncidentSymptom, IncidentTimelineEvent
)
from backend.models.playbook import (
    Playbook, PlaybookCreate, PlaybookStep, PlaybookExecution,
    ExecutionStatus, StepStatus, StepType, PlaybookStepResult
)
from backend.models.analysis import (
    AnalysisResult, EvidenceItem, EvidenceType, ConfidenceLevel,
    RootCauseAnalysis, PatternDetectionResult
)
from backend.models.user import User, UserCreate, UserRole

logger = logging.getLogger(__name__)

class SyntheticDataLoader:
    """Service for loading synthetic data for demo scenarios"""
    
    def __init__(self, data_directory: str = "/home/brama/hazel/prod/hazel-core/workspace/GenAI_CE_AI_SRE1/data/synthetic"):
        self.data_directory = Path(data_directory)
        self.scenario_1_data = None
        self.scenario_2_data = None
        self.historical_incidents = None
        self.playbook_configurations = None
        self.analysis_results = None
        self.infrastructure_blueprint = None
        self.users = None
        
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
            
            logger.info(f"Loaded {len(self.loaded_incidents)} incidents, "
                       f"{len(self.loaded_playbooks)} playbooks, "
                       f"{len(self.loaded_analyses)} analyses, "
                       f"{len(self.loaded_executions)} executions")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading synthetic data: {e}")
            return False
    
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
            return
        
        for incident_data in self.historical_incidents:
            try:
                # Convert to proper enums
                severity = self._map_severity(incident_data.get('severity', 'medium'))
                status = self._map_status(incident_data.get('status', 'resolved'))
                
                # Create symptoms
                symptoms = []
                for symptom_text in incident_data.get('symptoms', []):
                    symptoms.append(IncidentSymptom(
                        symptom_id=str(uuid4()),
                        symptom_type="performance",  # Default type
                        description=symptom_text,
                        severity=severity,
                        first_observed=incident_data['timestamp'],
                        metric_data={}
                    ))
                
                # Create incident
                incident = Incident(
                    incident_id=incident_data['incident_id'],
                    title=f"Incident {incident_data['incident_id']}: {incident_data.get('service_name', 'Unknown')} Issue",
                    description=incident_data.get('root_cause', 'Production incident'),
                    severity=severity,
                    status=status,
                    service_name=incident_data.get('service_name', 'unknown-service'),
                    region=incident_data.get('region', 'us-central1'),
                    created_at=incident_data['timestamp'],
                    timestamp=incident_data['timestamp'],
                    symptoms=symptoms,
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
    
    async def _process_playbooks(self):
        """Process playbook configurations into Playbook models"""
        if not self.playbook_configurations:
            return
        
        for playbook_data in self.playbook_configurations:
            try:
                # Process steps
                steps = []
                for step_data in playbook_data.get('steps', []):
                    step = PlaybookStep(
                        step_id=step_data['step_id'],
                        step_number=len(steps) + 1,
                        title=step_data['description'],
                        description=step_data['description'],
                        step_type=self._map_step_type(step_data.get('type', 'manual')),
                        command=step_data.get('query', ''),
                        expected_output=step_data.get('expected_result', {}).get('threshold', ''),
                        timeout_seconds=300,
                        is_critical=step_data.get('critical', False),
                        requires_approval=step_data.get('requires_approval', False),
                        dependencies=[],
                        retry_policy=None
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
                
                # Create root cause analysis
                root_cause = None
                if analysis_data.get('findings', {}).get('primary_cause'):
                    root_cause = RootCauseAnalysis(
                        primary_cause=analysis_data['findings']['primary_cause'],
                        contributing_factors=analysis_data['findings'].get('contributing_factors', []),
                        confidence_score=analysis_data.get('confidence_score', 0.8),
                        evidence_chain=evidence_items[:3],  # Top 3 evidence items
                        affected_components=analysis_data['findings'].get('affected_components', []),
                        timeline_correlation=[]
                    )
                
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
                    root_cause=root_cause,
                    recommendations=[analysis_data.get('recommendation', 'Follow standard procedures')],
                    pattern_detection=None,
                    correlation_analysis=None,
                    created_at=analysis_data['created_at'],
                    updated_at=analysis_data['created_at'],
                    analyzed_by="ai_sre_agent",
                    status="completed"
                )
                
                self.loaded_analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Error processing analysis {analysis_data.get('analysis_id')}: {e}")
    
    async def _process_gcp_data(self):
        """Process GCP observability data for both scenarios"""
        # Process Scenario 1 (Redis cache data)
        if self.scenario_1_data:
            self._process_scenario_1_data()
        
        # Process Scenario 2 (Database connection data)
        if self.scenario_2_data:
            self._process_scenario_2_data()
    
    def _process_scenario_1_data(self):
        """Process Scenario 1: Payment API Latency Spike"""
        metrics = self.scenario_1_data.get('metrics', [])
        logs = self.scenario_1_data.get('logs', [])
        errors = self.scenario_1_data.get('errors', [])
        traces = self.scenario_1_data.get('traces', [])
        
        # Cache Redis metrics
        redis_metrics = [m for m in metrics if 'redis' in m.get('metric_name', '').lower()]
        self.gcp_metrics_cache['scenario_1_redis'] = redis_metrics
        
        # Cache API latency metrics
        api_metrics = [m for m in metrics if 'latency' in m.get('metric_name', '').lower()]
        self.gcp_metrics_cache['scenario_1_api'] = api_metrics
        
        # Cache logs and errors
        self.gcp_logs_cache['scenario_1'] = logs
        self.gcp_errors_cache['scenario_1'] = errors
        self.gcp_traces_cache['scenario_1'] = traces
        
        logger.info(f"Processed Scenario 1: {len(redis_metrics)} Redis metrics, {len(api_metrics)} API metrics")
    
    def _process_scenario_2_data(self):
        """Process Scenario 2: Database Connection Pool Exhaustion"""
        metrics = self.scenario_2_data.get('metrics', [])
        logs = self.scenario_2_data.get('logs', [])
        errors = self.scenario_2_data.get('errors', [])
        traces = self.scenario_2_data.get('traces', [])
        
        # Cache database metrics
        db_metrics = [m for m in metrics if 'database' in m.get('metric_name', '').lower() or 'cloudsql' in m.get('metric_name', '').lower()]
        self.gcp_metrics_cache['scenario_2_db'] = db_metrics
        
        # Cache connection pool metrics
        conn_metrics = [m for m in metrics if 'connection' in m.get('metric_name', '').lower()]
        self.gcp_metrics_cache['scenario_2_connections'] = conn_metrics
        
        # Cache logs and errors
        self.gcp_logs_cache['scenario_2'] = logs
        self.gcp_errors_cache['scenario_2'] = errors
        self.gcp_traces_cache['scenario_2'] = traces
        
        logger.info(f"Processed Scenario 2: {len(db_metrics)} DB metrics, {len(conn_metrics)} connection metrics")
    
    async def _create_demo_executions(self):
        """Create sample playbook executions for demo"""
        if not self.loaded_playbooks:
            return
        
        # Create execution for first playbook
        for i, playbook in enumerate(self.loaded_playbooks[:2]):
            execution = PlaybookExecution(
                execution_id=f"exec_{str(uuid4())[:8]}",
                playbook_id=playbook.playbook_id,
                incident_id=f"INC-2024-{str(i+1).zfill(3)}",
                status=ExecutionStatus.COMPLETED if i == 0 else ExecutionStatus.RUNNING,
                execution_mode="automated",
                current_step_number=len(playbook.steps) if i == 0 else 2,
                progress=100 if i == 0 else 40,
                started_at=(datetime.utcnow() - timedelta(minutes=10)).isoformat(),
                completed_at=datetime.utcnow().isoformat() if i == 0 else None,
                executed_by="ai_sre_agent",
                step_results=self._create_step_results(playbook, completed=(i == 0)),
                root_cause_found=i == 0,
                confidence_score=0.92 if i == 0 else 0.75,
                recommendations=[
                    "Scale Redis cluster to handle increased load",
                    "Implement cache warming strategy",
                    "Add monitoring for cache hit rates"
                ]
            )
            
            self.loaded_executions.append(execution)
    
    def _create_step_results(self, playbook: Playbook, completed: bool = False) -> List[PlaybookStepResult]:
        """Create step results for playbook execution"""
        results = []
        
        for i, step in enumerate(playbook.steps):
            if not completed and i >= 2:  # Only first 2 steps for running execution
                break
                
            result = PlaybookStepResult(
                step_id=step.step_id,
                step_number=step.step_number,
                step_type=step.step_type,
                status=StepStatus.COMPLETED if completed or i < 2 else StepStatus.RUNNING,
                success=True,
                duration_seconds=30 + (i * 15),
                result_data={
                    "metrics_checked": True,
                    "threshold_exceeded": True if i == 0 else False,
                    "action_taken": f"Executed step {i+1}"
                },
                evidence=[f"Evidence from step {i+1}"],
                escalation_triggered=False
            )
            results.append(result)
        
        return results
    
    # Helper methods for data mapping
    
    def _map_severity(self, severity: str) -> IncidentSeverity:
        """Map string severity to enum"""
        mapping = {
            'critical': IncidentSeverity.CRITICAL,
            'high': IncidentSeverity.HIGH,
            'medium': IncidentSeverity.MEDIUM,
            'low': IncidentSeverity.LOW
        }
        return mapping.get(severity.lower(), IncidentSeverity.MEDIUM)
    
    def _map_status(self, status: str) -> IncidentStatus:
        """Map string status to enum"""
        mapping = {
            'open': IncidentStatus.OPEN,
            'investigating': IncidentStatus.INVESTIGATING,
            'identified': IncidentStatus.IDENTIFIED,
            'monitoring': IncidentStatus.MONITORING,
            'resolved': IncidentStatus.RESOLVED,
            'closed': IncidentStatus.CLOSED
        }
        return mapping.get(status.lower(), IncidentStatus.OPEN)
    
    def _map_step_type(self, step_type: str) -> StepType:
        """Map string step type to enum"""
        mapping = {
            'check_metrics': StepType.CHECK_METRICS,
            'query_logs': StepType.QUERY_LOGS,
            'run_command': StepType.RUN_COMMAND,
            'manual': StepType.MANUAL_INTERVENTION,
            'api_call': StepType.API_CALL,
            'notification': StepType.SEND_NOTIFICATION
        }
        return mapping.get(step_type.lower(), StepType.MANUAL_INTERVENTION)
    
    def _map_evidence_type(self, evidence_text: str) -> EvidenceType:
        """Map evidence text to evidence type"""
        text_lower = evidence_text.lower()
        
        if 'monitoring' in text_lower or 'metric' in text_lower:
            return EvidenceType.GCP_MONITORING
        elif 'log' in text_lower:
            return EvidenceType.GCP_LOGGING
        elif 'error' in text_lower:
            return EvidenceType.GCP_ERROR_REPORTING
        elif 'tracing' in text_lower or 'trace' in text_lower:
            return EvidenceType.GCP_TRACING
        elif 'historical' in text_lower or 'correlation' in text_lower:
            return EvidenceType.HISTORICAL_CORRELATION
        else:
            return EvidenceType.INFRASTRUCTURE_DATA
    
    def _categorize_playbook(self, name: str) -> str:
        """Categorize playbook based on name"""
        name_lower = name.lower()
        
        if 'latency' in name_lower or 'performance' in name_lower:
            return 'performance'
        elif 'database' in name_lower or 'db' in name_lower:
            return 'database'
        elif 'security' in name_lower:
            return 'security'
        elif 'network' in name_lower:
            return 'network'
        else:
            return 'general'
    
    def _estimate_duration(self, steps: List[PlaybookStep]) -> int:
        """Estimate playbook duration in minutes"""
        total_seconds = sum(step.timeout_seconds for step in steps)
        return max(5, total_seconds // 60)  # Minimum 5 minutes
    
    # Public methods for accessing loaded data
    
    def get_incidents(self, limit: int = 50) -> List[Incident]:
        """Get loaded incidents"""
        return self.loaded_incidents[:limit]
    
    def get_playbooks(self, limit: int = 20) -> List[Playbook]:
        """Get loaded playbooks"""
        return self.loaded_playbooks[:limit]
    
    def get_executions(self, limit: int = 10) -> List[PlaybookExecution]:
        """Get loaded executions"""
        return self.loaded_executions[:limit]
    
    def get_analyses(self, limit: int = 20) -> List[AnalysisResult]:
        """Get loaded analyses"""
        return self.loaded_analyses[:limit]
    
    def get_gcp_metrics(self, scenario: str, metric_type: str = None) -> List[Dict]:
        """Get GCP metrics for scenario"""
        key = f"scenario_{scenario}_{metric_type}" if metric_type else f"scenario_{scenario}"
        return self.gcp_metrics_cache.get(key, [])
    
    def get_gcp_logs(self, scenario: str) -> List[Dict]:
        """Get GCP logs for scenario"""
        return self.gcp_logs_cache.get(f"scenario_{scenario}", [])
    
    def get_gcp_errors(self, scenario: str) -> List[Dict]:
        """Get GCP errors for scenario"""
        return self.gcp_errors_cache.get(f"scenario_{scenario}", [])
    
    def get_gcp_traces(self, scenario: str) -> List[Dict]:
        """Get GCP traces for scenario"""
        return self.gcp_traces_cache.get(f"scenario_{scenario}", [])
    
    async def get_scenario_1_incident(self) -> Optional[Incident]:
        """Get the main incident for Scenario 1: Payment API Latency Spike"""
        for incident in self.loaded_incidents:
            if 'payment' in incident.service_name.lower() or 'redis' in str(incident.symptoms).lower():
                return incident
        
        # Create demo incident if not found
        return Incident(
            incident_id="INC-2024-001",
            title="Payment API Latency Spike",
            description="Payment API experiencing high latency due to Redis cache performance issues",
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.INVESTIGATING,
            service_name="payment-api",
            region="us-central1",
            created_at=datetime.utcnow().isoformat(),
            timestamp=datetime.utcnow().isoformat(),
            symptoms=[
                IncidentSymptom(
                    symptom_id=str(uuid4()),
                    symptom_type="latency",
                    description="Payment API P95 latency > 5s",
                    severity=IncidentSeverity.HIGH,
                    first_observed=datetime.utcnow().isoformat(),
                    metric_data={"threshold": 2.0, "current": 6.2}
                )
            ],
            created_by="monitoring_system",
            assigned_to="sre_team",
            tags=["demo", "scenario1", "latency", "redis"]
        )
    
    async def get_scenario_2_incident(self) -> Optional[Incident]:
        """Get the main incident for Scenario 2: Database Connection Pool Exhaustion"""
        for incident in self.loaded_incidents:
            if 'database' in incident.service_name.lower() or 'connection' in str(incident.symptoms).lower():
                return incident
        
        # Create demo incident if not found
        return Incident(
            incident_id="INC-2024-002",
            title="Database Connection Pool Exhaustion",
            description="Database connection pool exhausted causing timeout errors in billing service",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.INVESTIGATING,
            service_name="billing-service",
            region="us-central1",
            created_at=datetime.utcnow().isoformat(),
            timestamp=datetime.utcnow().isoformat(),
            symptoms=[
                IncidentSymptom(
                    symptom_id=str(uuid4()),
                    symptom_type="database",
                    description="Database connection pool utilization > 95%",
                    severity=IncidentSeverity.CRITICAL,
                    first_observed=datetime.utcnow().isoformat(),
                    metric_data={"threshold": 80, "current": 95}
                )
            ],
            created_by="monitoring_system",
            assigned_to="sre_team",
            tags=["demo", "scenario2", "database", "connections"]
        )

# Global instance for the application
synthetic_data_loader = SyntheticDataLoader()