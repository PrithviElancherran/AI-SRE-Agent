"""
Systematic playbook execution engine with step-by-step workflow processing, dynamic adaptation based on findings, and human-in-the-loop approval workflow integration.

This service provides the core playbook execution capabilities for the SRE Agent, including:
- Step-by-step playbook execution
- Dynamic adaptation based on findings
- GCP observability integration for data collection
- Human-in-the-loop approval workflows
- Real-time progress tracking and reporting
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from loguru import logger
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func

from models.playbook import (
    Playbook, PlaybookStep, PlaybookExecution, PlaybookStepResult,
    PlaybookStatus, StepType, ExecutionStatus, PlaybookTable,
    PlaybookExecutionTable, PlaybookStepResultTable
)
from models.incident import Incident, IncidentSymptom
from models.user import User, ApprovalRequest, ApprovalRequestCreate
from models.analysis import EvidenceItem, EvidenceType
from config.database import get_database
from config.settings import get_settings
from utils.formatters import format_duration, format_confidence_score
from integrations.gcp_monitoring import GCPMonitoringClient
from integrations.gcp_logging import GCPLoggingClient
from integrations.gcp_error_reporting import GCPErrorReportingClient
from integrations.gcp_tracing import GCPTracingClient

settings = get_settings()


class StepExecutionResult:
    """Result of step execution with detailed information."""
    
    def __init__(
        self,
        success: bool,
        result_data: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
        escalation_triggered: bool = False,
        error_message: Optional[str] = None,
        threshold_met: Optional[bool] = None,
        actual_value: Optional[float] = None,
        expected_value: Optional[float] = None
    ):
        self.success = success
        self.result_data = result_data
        self.evidence = evidence or {}
        self.escalation_triggered = escalation_triggered
        self.error_message = error_message
        self.threshold_met = threshold_met
        self.actual_value = actual_value
        self.expected_value = expected_value


class PlaybookExecutor:
    """Systematic playbook execution engine."""
    
    def __init__(self):
        """Initialize the playbook executor."""
        self.gcp_monitoring = GCPMonitoringClient()
        self.gcp_logging = GCPLoggingClient()
        self.gcp_error_reporting = GCPErrorReportingClient()
        self.gcp_tracing = GCPTracingClient()
        
        # Execution cache and state management
        self._active_executions = {}
        self._execution_locks = {}
        
        # Step processors for different step types
        self._step_processors = {
            StepType.MONITORING_CHECK: self._execute_monitoring_check,
            StepType.LOG_ANALYSIS: self._execute_log_analysis,
            StepType.TRACING_ANALYSIS: self._execute_tracing_analysis,
            StepType.PERFORMANCE_ANALYSIS: self._execute_performance_analysis,
            StepType.QUERY_ANALYSIS: self._execute_query_analysis,
            StepType.CHANGE_ANALYSIS: self._execute_change_analysis,
            StepType.PROCESS_ANALYSIS: self._execute_process_analysis,
            StepType.MANUAL_ACTION: self._execute_manual_action,
            StepType.AUTOMATED_ACTION: self._execute_automated_action
        }
        
        logger.info("PlaybookExecutor initialized with GCP integrations")
    
    async def execute_playbook(
        self,
        playbook_id: str,
        incident_id: str,
        user: User,
        context: Optional[Dict[str, Any]] = None
    ) -> PlaybookExecution:
        """
        Execute a playbook for an incident with step-by-step processing.
        
        Args:
            playbook_id: ID of the playbook to execute
            incident_id: ID of the incident being investigated
            user: User executing the playbook
            context: Additional execution context
            
        Returns:
            PlaybookExecution with results and progress
        """
        logger.info(f"Starting playbook execution: {playbook_id} for incident {incident_id}")
        
        # Create execution record
        execution = PlaybookExecution(
            id=uuid4(),
            execution_id=f"EXEC-{str(uuid4())[:8].upper()}",
            playbook_id=playbook_id,
            incident_id=incident_id,
            status=ExecutionStatus.RUNNING,
            started_by=user.user_id,
            started_at=datetime.utcnow(),
            execution_context=context or {}
        )
        
        # Create execution lock to prevent concurrent executions
        execution_key = f"{playbook_id}:{incident_id}"
        if execution_key in self._execution_locks:
            raise ValueError(f"Playbook {playbook_id} is already being executed for incident {incident_id}")
        
        self._execution_locks[execution_key] = execution.execution_id
        self._active_executions[execution.execution_id] = execution
        
        try:
            # Get playbook details
            playbook = await self._get_playbook(playbook_id)
            if not playbook:
                raise ValueError(f"Playbook {playbook_id} not found")
            
            if not playbook.is_active():
                raise ValueError(f"Playbook {playbook_id} is not active")
            
            # Get incident details
            incident = await self._get_incident(incident_id)
            if not incident:
                raise ValueError(f"Incident {incident_id} not found")
            
            logger.info(f"Executing playbook '{playbook.name}' v{playbook.version} with {len(playbook.steps)} steps")
            
            # Execute steps in order
            completed_steps = []
            step_results = []
            
            for step in sorted(playbook.steps, key=lambda s: s.order):
                logger.info(f"Executing step {step.order}: {step.description}")
                
                # Check dependencies
                if not playbook.validate_step_dependencies(step, completed_steps):
                    logger.warning(f"Step {step.step_id} dependencies not satisfied, skipping")
                    step_result = await self._create_step_result(
                        execution, step, ExecutionStatus.SKIPPED,
                        success=None, error_message="Dependencies not satisfied"
                    )
                    step_results.append(step_result)
                    continue
                
                # Execute the step
                step_result = await self._execute_step(execution, playbook, step, incident, user)
                step_results.append(step_result)
                
                # Update execution progress
                execution.step_results = step_results
                
                # Check if step was successful
                if step_result.is_successful():
                    completed_steps.append(step.step_id)
                    logger.info(f"Step {step.step_id} completed successfully")
                elif step_result.status == ExecutionStatus.FAILED:
                    logger.error(f"Step {step.step_id} failed: {step_result.error_message}")
                    
                    # Check if this is a critical failure
                    if step.step_type in [StepType.MONITORING_CHECK, StepType.LOG_ANALYSIS]:
                        logger.info("Critical step failed, continuing with remaining steps")
                    else:
                        logger.warning("Non-critical step failed, continuing execution")
                
                # Check for escalation conditions
                if step_result.escalation_triggered:
                    logger.warning(f"Escalation triggered for step {step.step_id}")
                    await self._handle_escalation(execution, step, step_result, user)
                
                # Dynamic adaptation based on findings
                adaptation_result = await self._adapt_execution(
                    execution, playbook, step, step_result, incident
                )
                if adaptation_result:
                    logger.info(f"Execution adapted: {adaptation_result}")
            
            # Analyze overall execution results
            execution_analysis = await self._analyze_execution_results(
                execution, playbook, step_results, incident
            )
            
            # Finalize execution
            execution.completed_at = datetime.utcnow()
            execution.duration_minutes = (execution.completed_at - execution.started_at).total_seconds() / 60
            execution.step_results = step_results
            execution.success = len([r for r in step_results if r.is_successful()]) > len(step_results) * 0.5
            execution.root_cause_found = execution_analysis.get('root_cause_found', False)
            execution.actions_recommended = execution_analysis.get('recommendations', [])
            execution.confidence_score = execution_analysis.get('confidence_score', 0.0)
            execution.status = ExecutionStatus.COMPLETED
            
            logger.info(f"Playbook execution completed: {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Playbook execution failed: {e}")
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            execution.duration_minutes = (execution.completed_at - execution.started_at).total_seconds() / 60
        
        finally:
            # Clean up execution state
            if execution_key in self._execution_locks:
                del self._execution_locks[execution_key]
            if execution.execution_id in self._active_executions:
                del self._active_executions[execution.execution_id]
        
        return execution
    
    async def get_execution_status(self, execution_id: str) -> Optional[PlaybookExecution]:
        """Get current status of a playbook execution."""
        return self._active_executions.get(execution_id)
    
    async def cancel_execution(self, execution_id: str, user: User) -> bool:
        """Cancel a running playbook execution."""
        execution = self._active_executions.get(execution_id)
        if not execution:
            return False
        
        if execution.started_by != user.user_id and not user.can_approve_actions():
            raise ValueError("Insufficient permissions to cancel execution")
        
        execution.status = ExecutionStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        execution.duration_minutes = (execution.completed_at - execution.started_at).total_seconds() / 60
        execution.error_message = f"Cancelled by {user.user_id}"
        
        logger.info(f"Execution {execution_id} cancelled by {user.user_id}")
        return True
    
    async def _execute_step(
        self,
        execution: PlaybookExecution,
        playbook: Playbook,
        step: PlaybookStep,
        incident: Incident,
        user: User
    ) -> PlaybookStepResult:
        """Execute a single playbook step."""
        step_start_time = datetime.utcnow()
        
        try:
            # Check if step requires approval
            if step.approval_required and not user.can_approve_actions():
                return await self._request_step_approval(execution, step, user)
            
            # Execute step based on type
            processor = self._step_processors.get(step.step_type)
            if not processor:
                raise ValueError(f"No processor found for step type: {step.step_type}")
            
            # Execute with timeout
            execution_result = await asyncio.wait_for(
                processor(step, incident, execution.execution_context),
                timeout=step.timeout_minutes * 60
            )
            
            # Create step result
            step_result = await self._create_step_result(
                execution=execution,
                step=step,
                status=ExecutionStatus.COMPLETED if execution_result.success else ExecutionStatus.FAILED,
                success=execution_result.success,
                result_data=execution_result.result_data,
                evidence=execution_result.evidence,
                escalation_triggered=execution_result.escalation_triggered,
                error_message=execution_result.error_message,
                threshold_met=execution_result.threshold_met,
                actual_value=execution_result.actual_value,
                expected_value=execution_result.expected_value,
                started_at=step_start_time
            )
            
            return step_result
            
        except asyncio.TimeoutError:
            logger.error(f"Step {step.step_id} timed out after {step.timeout_minutes} minutes")
            return await self._create_step_result(
                execution, step, ExecutionStatus.FAILED,
                success=False, error_message=f"Step timed out after {step.timeout_minutes} minutes",
                started_at=step_start_time
            )
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            return await self._create_step_result(
                execution, step, ExecutionStatus.FAILED,
                success=False, error_message=str(e),
                started_at=step_start_time
            )
    
    async def _execute_monitoring_check(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a monitoring check step."""
        try:
            # Check if step has GCP integration configuration
            if not step.gcp_integration or not step.gcp_integration.metric:
                # Default to a direct metric query if no GCP integration is configured
                metric_name = "cloudsql/database/connection_count"
                time_range = "1h"
                
                logger.info(f"No GCP integration configured, using default metric: {metric_name}")
                
                metric_data = await self.gcp_monitoring.query_metric(
                    metric_name=metric_name,
                    time_range=time_range,
                    labels={"service": incident.service_name}
                )
            else:
                # Query GCP monitoring with configured integration
                metric_name = step.gcp_integration.metric
                time_range = step.gcp_integration.time_range or "1h"
                
                metric_data = await self.gcp_monitoring.query_metric(
                    metric_name=metric_name,
                    time_range=time_range,
                    labels=step.gcp_integration.labels,
                    aggregation=step.gcp_integration.aggregation
                )
            
            if not metric_data:
                return StepExecutionResult(
                    success=False,
                    result_data={"error": "No metric data found"},
                    error_message=f"No data found for metric {metric_name}"
                )
            
            # Get latest value
            latest_value = metric_data.get('latest_value', 0)
            
            # Parse expected result threshold
            threshold_result = self._parse_threshold(step.expected_result.threshold if step.expected_result else "< 100")
            expected_value = threshold_result['value']
            operator = threshold_result['operator']
            
            # Check threshold
            threshold_met = self._evaluate_threshold(latest_value, operator, expected_value)
            
            # Check escalation condition
            escalation_triggered = False
            if step.escalation_condition:
                escalation_result = self._parse_threshold(step.escalation_condition)
                escalation_triggered = self._evaluate_threshold(
                    latest_value, escalation_result['operator'], escalation_result['value']
                )
            
            # Collect evidence
            evidence = {
                "metric_name": metric_name,
                "latest_value": latest_value,
                "threshold": step.expected_result.threshold if step.expected_result else "< 100",
                "threshold_met": threshold_met,
                "gcp_dashboard_url": f"https://console.cloud.google.com/monitoring/metrics-explorer?project={settings.GCP_PROJECT_ID}",
                "data_points": metric_data.get('data_points', [])
            }
            
            return StepExecutionResult(
                success=threshold_met,
                result_data={
                    "metric_value": latest_value,
                    "threshold_met": threshold_met,
                    "metric_trend": metric_data.get('trend', 'stable')
                },
                evidence=evidence,
                escalation_triggered=escalation_triggered,
                threshold_met=threshold_met,
                actual_value=latest_value,
                expected_value=expected_value
            )
            
        except Exception as e:
            logger.error(f"Error in monitoring check: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _execute_log_analysis(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a log analysis step."""
        try:
            if not step.query:
                raise ValueError("Query is required for log analysis step")
            
            # Search logs using GCP Logging
            log_entries = await self.gcp_logging.search_logs(
                query=step.query,
                time_range="1h",
                resource_labels={"service": incident.service_name}
            )
            
            # Analyze log entries
            error_count = len([entry for entry in log_entries if entry.get('severity') in ['ERROR', 'CRITICAL']])
            warning_count = len([entry for entry in log_entries if entry.get('severity') == 'WARNING'])
            total_count = len(log_entries)
            
            # Determine threshold
            threshold_str = step.expected_result.threshold if step.expected_result else "< 5 errors/hour"
            threshold_parts = threshold_str.split()
            
            # Extract expected errors count with proper parsing
            expected_errors = 5  # Default
            if len(threshold_parts) > 1:
                last_part = threshold_parts[-1]
                if '/' in last_part:
                    # Handle "5 errors/hour" format
                    error_part = last_part.split('/')[0]
                    # Remove non-numeric characters like "errors"
                    import re
                    numbers = re.findall(r'\d+', error_part)
                    expected_errors = int(numbers[0]) if numbers else 5
                else:
                    # Handle simple numeric format
                    try:
                        expected_errors = int(last_part)
                    except ValueError:
                        # If it's not a number, try to extract numbers from the string
                        import re
                        numbers = re.findall(r'\d+', threshold_str)
                        expected_errors = int(numbers[0]) if numbers else 5
            
            threshold_met = error_count <= expected_errors
            
            # Check escalation condition
            escalation_triggered = False
            if step.escalation_condition:
                escalation_parts = step.escalation_condition.split()
                escalation_limit = 10  # Default
                if len(escalation_parts) > 1:
                    last_part = escalation_parts[-1]
                    if '/' in last_part:
                        # Handle "10 errors/hour" format
                        error_part = last_part.split('/')[0]
                        # Remove non-numeric characters like "errors"
                        import re
                        numbers = re.findall(r'\d+', error_part)
                        escalation_limit = int(numbers[0]) if numbers else 10
                    else:
                        # Handle simple numeric format
                        try:
                            escalation_limit = int(last_part)
                        except ValueError:
                            # If it's not a number, try to extract numbers from the string
                            import re
                            numbers = re.findall(r'\d+', step.escalation_condition)
                            escalation_limit = int(numbers[0]) if numbers else 10
                escalation_triggered = error_count > escalation_limit
            
            # Collect evidence
            evidence = {
                "query": step.query,
                "total_entries": total_count,
                "error_count": error_count,
                "warning_count": warning_count,
                "sample_errors": log_entries[:5] if log_entries else [],
                "gcp_logs_url": f"https://console.cloud.google.com/logs/query?project={settings.GCP_PROJECT_ID}"
            }
            
            return StepExecutionResult(
                success=threshold_met,
                result_data={
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "total_entries": total_count,
                    "error_rate": error_count / max(1, total_count) * 100
                },
                evidence=evidence,
                escalation_triggered=escalation_triggered,
                threshold_met=threshold_met,
                actual_value=float(error_count),
                expected_value=float(expected_errors)
            )
            
        except Exception as e:
            logger.error(f"Error in log analysis: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _execute_tracing_analysis(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a tracing analysis step."""
        try:
            if not step.gcp_integration or not step.gcp_integration.operation:
                raise ValueError("GCP integration with operation is required for tracing analysis")
            
            operation = step.gcp_integration.operation
            
            # Query GCP tracing
            traces = await self.gcp_tracing.query_traces(
                operation=operation,
                time_range="1h",
                service_name=incident.service_name
            )
            
            if not traces:
                return StepExecutionResult(
                    success=False,
                    result_data={"error": "No trace data found"},
                    error_message=f"No traces found for operation {operation}"
                )
            
            # Analyze trace latencies
            latencies = [trace.get('duration_ms', 0) for trace in traces]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            
            # Parse threshold (e.g., "< 1s" -> 1000ms)
            threshold_str = step.expected_result.threshold if step.expected_result else "< 1s"
            expected_latency = self._parse_latency_threshold(threshold_str)
            
            threshold_met = avg_latency <= expected_latency
            
            # Check escalation condition
            escalation_triggered = False
            if step.escalation_condition:
                escalation_latency = self._parse_latency_threshold(step.escalation_condition)
                escalation_triggered = p95_latency > escalation_latency
            
            # Collect evidence
            evidence = {
                "operation": operation,
                "trace_count": len(traces),
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "slow_traces": [t for t in traces if t.get('duration_ms', 0) > expected_latency],
                "gcp_trace_url": f"https://console.cloud.google.com/traces/list?project={settings.GCP_PROJECT_ID}"
            }
            
            return StepExecutionResult(
                success=threshold_met,
                result_data={
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "trace_count": len(traces),
                    "slow_trace_count": len([t for t in traces if t.get('duration_ms', 0) > expected_latency])
                },
                evidence=evidence,
                escalation_triggered=escalation_triggered,
                threshold_met=threshold_met,
                actual_value=avg_latency,
                expected_value=expected_latency
            )
            
        except Exception as e:
            logger.error(f"Error in tracing analysis: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _execute_performance_analysis(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a performance analysis step."""
        try:
            # Query multiple performance metrics
            metrics_to_check = [
                "cloudsql/database/query_execution_time",
                "cloudsql/database/connection_count",
                "compute/instance/cpu_utilization"
            ]
            
            performance_data = {}
            issues_found = []
            
            for metric in metrics_to_check:
                try:
                    data = await self.gcp_monitoring.query_metric(
                        metric_name=metric,
                        time_range="1h",
                        labels={"service": incident.service_name}
                    )
                    if data:
                        performance_data[metric] = data
                        
                        # Check for performance issues
                        latest_value = data.get('latest_value', 0)
                        if metric == "cloudsql/database/query_execution_time" and latest_value > 5000:  # 5s
                            issues_found.append(f"Slow database queries detected: {latest_value}ms")
                        elif metric == "cloudsql/database/connection_count" and latest_value > 80:  # 80% of pool
                            issues_found.append(f"High database connection usage: {latest_value}%")
                        elif metric == "compute/instance/cpu_utilization" and latest_value > 80:
                            issues_found.append(f"High CPU utilization: {latest_value}%")
                            
                except Exception as e:
                    logger.warning(f"Failed to query metric {metric}: {e}")
            
            # Determine success based on issues found
            success = len(issues_found) == 0
            
            # Check escalation condition based on severity
            escalation_triggered = len(issues_found) > 2
            
            evidence = {
                "performance_metrics": performance_data,
                "issues_found": issues_found,
                "analysis_time": datetime.utcnow().isoformat()
            }
            
            return StepExecutionResult(
                success=success,
                result_data={
                    "issues_count": len(issues_found),
                    "issues": issues_found,
                    "metrics_analyzed": len(performance_data)
                },
                evidence=evidence,
                escalation_triggered=escalation_triggered,
                threshold_met=success
            )
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _execute_query_analysis(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a query analysis step."""
        try:
            # Simulate query analysis using GCP monitoring for query performance
            slow_queries = await self.gcp_monitoring.query_metric(
                metric_name="cloudsql/database/query_execution_time",
                time_range="1h",
                labels={"service": incident.service_name}
            )
            
            if not slow_queries:
                return StepExecutionResult(
                    success=True,
                    result_data={"message": "No slow queries detected"},
                    threshold_met=True
                )
            
            # Parse threshold from expected result
            threshold_ms = 5000  # Default 5 seconds
            if step.expected_result and step.expected_result.threshold:
                threshold_str = step.expected_result.threshold
                logger.info(f"Query analysis threshold string: '{threshold_str}'")
                if "5s" in threshold_str:
                    threshold_ms = 5000
                elif "10s" in threshold_str:
                    threshold_ms = 10000
                elif "30s" in threshold_str:
                    threshold_ms = 30000
                elif "60s" in threshold_str:
                    threshold_ms = 60000
                else:
                    # Try to extract number from threshold string
                    import re
                    numbers = re.findall(r'(\d+)', threshold_str)
                    if numbers:
                        threshold_ms = int(numbers[0]) * 1000  # Convert to milliseconds
                logger.info(f"Parsed threshold_ms: {threshold_ms}")
            else:
                logger.info("No expected_result or threshold found, using default 5000ms")
            
            avg_query_time = slow_queries.get('latest_value', 0)
            threshold_met = avg_query_time < threshold_ms
            
            # Check escalation condition
            escalation_triggered = False
            if step.escalation_condition:
                escalation_limit = 5000  # Default 5 seconds in ms
                if "5000ms" in step.escalation_condition:
                    escalation_limit = 5000
                elif "10000ms" in step.escalation_condition:
                    escalation_limit = 10000
                elif "30000ms" in step.escalation_condition:
                    escalation_limit = 30000
                elif "60000ms" in step.escalation_condition:
                    escalation_limit = 60000
                else:
                    # Try to extract number from escalation condition
                    import re
                    numbers = re.findall(r'(\d+)', step.escalation_condition)
                    if numbers:
                        escalation_limit = int(numbers[0])
                escalation_triggered = avg_query_time > escalation_limit
            
            # Simulate finding specific slow queries
            slow_query_details = []
            if avg_query_time > threshold_ms:
                slow_query_details = [
                    {
                        "query": "UPDATE invoices SET status = 'processed' WHERE customer_id IN (...)",
                        "execution_time_ms": avg_query_time,
                        "rows_affected": 50000
                    }
                ]
            
            evidence = {
                "avg_query_time_ms": avg_query_time,
                "threshold_ms": threshold_ms,
                "slow_queries": slow_query_details,
                "analysis_method": "GCP Cloud SQL Performance Insights"
            }
            
            return StepExecutionResult(
                success=threshold_met,
                result_data={
                    "avg_query_time_ms": avg_query_time,
                    "slow_queries_count": len(slow_query_details),
                    "threshold_exceeded": not threshold_met
                },
                evidence=evidence,
                escalation_triggered=escalation_triggered,
                threshold_met=threshold_met,
                actual_value=avg_query_time,
                expected_value=float(threshold_ms)
            )
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _execute_change_analysis(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a change analysis step."""
        try:
            # Simulate deployment history analysis
            # In a real implementation, this would query deployment systems
            recent_deployments = [
                {
                    "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "service": incident.service_name,
                    "version": "v1.2.3",
                    "deployed_by": "deployment-bot",
                    "status": "success"
                }
            ]
            
            # Check if deployments coincide with incident timeline
            incident_time = incident.timestamp
            correlation_found = False
            
            for deployment in recent_deployments:
                deploy_time = datetime.fromisoformat(deployment["timestamp"].replace('Z', '+00:00'))
                time_diff = abs((incident_time - deploy_time).total_seconds())
                
                # If deployment was within 4 hours of incident
                if time_diff < 4 * 3600:
                    correlation_found = True
                    break
            
            # Check escalation condition
            escalation_triggered = correlation_found and step.escalation_condition == "recent deployment found"
            
            evidence = {
                "recent_deployments": recent_deployments,
                "incident_timestamp": incident_time.isoformat(),
                "correlation_found": correlation_found,
                "analysis_window": "24 hours"
            }
            
            return StepExecutionResult(
                success=not correlation_found,  # Success if no recent deployments
                result_data={
                    "deployments_found": len(recent_deployments),
                    "correlation_with_incident": correlation_found,
                    "potential_cause": correlation_found
                },
                evidence=evidence,
                escalation_triggered=escalation_triggered,
                threshold_met=not correlation_found
            )
            
        except Exception as e:
            logger.error(f"Error in change analysis: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _execute_process_analysis(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a process analysis step."""
        try:
            # Simulate process analysis using CPU metrics as proxy
            cpu_data = await self.gcp_monitoring.query_metric(
                metric_name="compute/instance/cpu_utilization",
                time_range="1h",
                labels={"service": incident.service_name}
            )
            
            if not cpu_data:
                return StepExecutionResult(
                    success=False,
                    result_data={"error": "No CPU data available"},
                    error_message="No process data available for analysis"
                )
            
            cpu_usage = cpu_data.get('latest_value', 0)
            
            # Simulate finding top processes
            top_processes = [
                {"name": "java", "cpu_percent": cpu_usage * 0.6, "memory_mb": 2048},
                {"name": "redis", "cpu_percent": cpu_usage * 0.2, "memory_mb": 512},
                {"name": "postgres", "cpu_percent": cpu_usage * 0.15, "memory_mb": 1024}
            ]
            
            # Check for resource hogs
            resource_hogs = [p for p in top_processes if p["cpu_percent"] > 50]
            threshold_met = len(resource_hogs) == 0
            
            # Check escalation condition
            escalation_triggered = any(p["cpu_percent"] > 80 for p in top_processes)
            
            evidence = {
                "cpu_utilization": cpu_usage,
                "top_processes": top_processes,
                "resource_hogs": resource_hogs,
                "analysis_method": "GCP Compute Engine metrics"
            }
            
            return StepExecutionResult(
                success=threshold_met,
                result_data={
                    "cpu_usage": cpu_usage,
                    "resource_hogs_count": len(resource_hogs),
                    "top_processes": top_processes
                },
                evidence=evidence,
                escalation_triggered=escalation_triggered,
                threshold_met=threshold_met,
                actual_value=cpu_usage,
                expected_value=70.0  # Expected threshold
            )
            
        except Exception as e:
            logger.error(f"Error in process analysis: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _execute_manual_action(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute a manual action step (requires human intervention)."""
        # Manual actions always require approval
        return StepExecutionResult(
            success=False,
            result_data={"requires_manual_intervention": True},
            error_message="Manual action required - human intervention needed"
        )
    
    async def _execute_automated_action(
        self,
        step: PlaybookStep,
        incident: Incident,
        context: Dict[str, Any]
    ) -> StepExecutionResult:
        """Execute an automated action step."""
        try:
            # In demo mode, simulate automated actions
            action_type = step.description.lower()
            
            if "scale" in action_type:
                # Simulate scaling action
                result = {
                    "action": "scale_service",
                    "service": incident.service_name,
                    "before_instances": 3,
                    "after_instances": 6,
                    "status": "success"
                }
            elif "restart" in action_type:
                # Simulate restart action
                result = {
                    "action": "restart_service",
                    "service": incident.service_name,
                    "restart_time": datetime.utcnow().isoformat(),
                    "status": "success"
                }
            else:
                # Generic automated action
                result = {
                    "action": "generic_automation",
                    "description": step.description,
                    "status": "simulated"
                }
            
            evidence = {
                "action_details": result,
                "execution_time": datetime.utcnow().isoformat(),
                "automation_type": "demo_simulation"
            }
            
            return StepExecutionResult(
                success=True,
                result_data=result,
                evidence=evidence,
                threshold_met=True
            )
            
        except Exception as e:
            logger.error(f"Error in automated action: {e}")
            return StepExecutionResult(
                success=False,
                result_data={"error": str(e)},
                error_message=str(e)
            )
    
    async def _request_step_approval(
        self,
        execution: PlaybookExecution,
        step: PlaybookStep,
        user: User
    ) -> PlaybookStepResult:
        """Request approval for a step that requires human intervention."""
        approval_request = ApprovalRequestCreate(
            incident_id=execution.incident_id,
            action_type=f"playbook_step_{step.step_type}",
            action_details={
                "step_id": step.step_id,
                "description": step.description,
                "playbook_execution_id": execution.execution_id
            },
            justification=f"Step '{step.description}' requires approval before execution",
            confidence_score=0.8,
            expires_in_minutes=60
        )
        
        # In a real implementation, this would create an approval request in the database
        # For demo purposes, we'll simulate the approval process
        
        return await self._create_step_result(
            execution=execution,
            step=step,
            status=ExecutionStatus.WAITING_APPROVAL,
            success=None,
            result_data={"approval_request": approval_request.model_dump()},
            error_message="Waiting for human approval"
        )
    
    async def _handle_escalation(
        self,
        execution: PlaybookExecution,
        step: PlaybookStep,
        step_result: PlaybookStepResult,
        user: User
    ) -> None:
        """Handle escalation when a step triggers escalation conditions."""
        logger.warning(f"Escalation triggered for step {step.step_id} in execution {execution.execution_id}")
        
        # In a real implementation, this would:
        # 1. Send notifications to on-call engineers
        # 2. Create high-priority alerts
        # 3. Potentially trigger additional playbooks
        # 4. Update incident severity
        
        escalation_details = {
            "execution_id": execution.execution_id,
            "step_id": step.step_id,
            "escalation_reason": step_result.error_message or "Threshold exceeded",
            "timestamp": datetime.utcnow().isoformat(),
            "escalated_by": user.user_id
        }
        
        # Add escalation details to execution context
        if "escalations" not in execution.execution_context:
            execution.execution_context["escalations"] = []
        execution.execution_context["escalations"].append(escalation_details)
    
    async def _adapt_execution(
        self,
        execution: PlaybookExecution,
        playbook: Playbook,
        step: PlaybookStep,
        step_result: PlaybookStepResult,
        incident: Incident
    ) -> Optional[str]:
        """Dynamically adapt execution based on step results."""
        # Check if we should skip remaining steps based on findings
        if step_result.is_successful() and step.step_type == StepType.MONITORING_CHECK:
            # If monitoring check shows system is healthy, might skip some diagnostic steps
            if step_result.threshold_met and step_result.actual_value and step_result.actual_value < step_result.expected_value * 0.5:
                return "System metrics are well within normal range - skipping additional diagnostic steps"
        
        # If we found a clear root cause, might accelerate to resolution steps
        if step_result.escalation_triggered and step.step_type in [StepType.LOG_ANALYSIS, StepType.QUERY_ANALYSIS]:
            return "Critical issue identified - prioritizing immediate resolution steps"
        
        # Check if we should add additional analysis steps
        if step_result.result_data and "issues_found" in step_result.result_data:
            issues = step_result.result_data["issues_found"]
            if isinstance(issues, list) and len(issues) > 3:
                return "Multiple issues detected - expanding diagnostic scope"
        
        return None
    
    async def _analyze_execution_results(
        self,
        execution: PlaybookExecution,
        playbook: Playbook,
        step_results: List[PlaybookStepResult],
        incident: Incident
    ) -> Dict[str, Any]:
        """Analyze overall execution results and generate conclusions."""
        successful_steps = [r for r in step_results if r.is_successful()]
        failed_steps = [r for r in step_results if r.status == ExecutionStatus.FAILED]
        
        # Determine if root cause was found
        root_cause_found = False
        root_cause_evidence = []
        
        for result in step_results:
            if result.escalation_triggered or (result.threshold_met is False and result.evidence):
                root_cause_found = True
                root_cause_evidence.append(result.evidence)
        
        # Generate recommendations based on findings
        recommendations = []
        
        # Analyze patterns in failed steps
        failed_step_types = [r.step.step_type for r in failed_steps if hasattr(r, 'step')]
        
        if StepType.MONITORING_CHECK in failed_step_types:
            recommendations.append("Scale infrastructure to handle current load")
        if StepType.LOG_ANALYSIS in failed_step_types:
            recommendations.append("Investigate application errors and implement fixes")
        if StepType.QUERY_ANALYSIS in failed_step_types:
            recommendations.append("Optimize slow database queries")
        
        # Calculate confidence score based on successful diagnostics
        confidence_score = len(successful_steps) / max(1, len(step_results))
        
        # Adjust confidence based on evidence quality
        if root_cause_found:
            confidence_score = min(1.0, confidence_score + 0.2)
        
        return {
            "root_cause_found": root_cause_found,
            "confidence_score": confidence_score,
            "recommendations": recommendations,
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "evidence_collected": len(root_cause_evidence),
            "execution_quality": "high" if confidence_score > 0.8 else "medium" if confidence_score > 0.5 else "low"
        }
    
    async def _create_step_result(
        self,
        execution: PlaybookExecution,
        step: PlaybookStep,
        status: ExecutionStatus,
        success: Optional[bool] = None,
        result_data: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        escalation_triggered: bool = False,
        error_message: Optional[str] = None,
        threshold_met: Optional[bool] = None,
        actual_value: Optional[float] = None,
        expected_value: Optional[float] = None,
        started_at: Optional[datetime] = None
    ) -> PlaybookStepResult:
        """Create a step result record."""
        now = datetime.utcnow()
        start_time = started_at or now
        
        return PlaybookStepResult(
            id=uuid4(),
            step_id=step.id,
            status=status,
            started_at=start_time,
            completed_at=now if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED] else None,
            duration_seconds=(now - start_time).total_seconds() if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED] else None,
            success=success,
            result_data=result_data or {},
            evidence=evidence,
            escalation_triggered=escalation_triggered,
            error_message=error_message,
            threshold_met=threshold_met,
            actual_value=actual_value,
            expected_value=expected_value
        )
    
    async def _get_playbook(self, playbook_id: str) -> Optional[Playbook]:
        """Get playbook by ID from database or cache."""
        # In a real implementation, this would query the database
        # For demo purposes, we'll create a mock playbook based on synthetic data
        
        if "PB-LATENCY" in playbook_id:
            return self._create_mock_latency_playbook()
        elif "PB-DB-TIMEOUT" in playbook_id:
            return self._create_mock_db_timeout_playbook()
        elif "PB-CPU" in playbook_id:
            return self._create_mock_cpu_playbook()
        
        return None
    
    async def _get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID from database."""
        # In a real implementation, this would query the database
        # For demo purposes, return a mock incident
        from models.incident import IncidentSeverity, IncidentStatus
        
        return Incident(
            id=uuid4(),
            incident_id=incident_id,
            title=f"Mock Incident {incident_id}",
            timestamp=datetime.utcnow(),
            status=IncidentStatus.INVESTIGATING,
            severity=IncidentSeverity.HIGH,
            symptoms=["API latency > 5s", "Database connection timeouts"],
            region="us-central1",
            service_name="payment-api",
            created_at=datetime.utcnow(),
            incident_symptoms=[]
        )
    
    def _create_mock_latency_playbook(self) -> Playbook:
        """Create mock latency troubleshooting playbook."""
        from models.playbook import GCPIntegration, ExpectedResult
        
        steps = [
            PlaybookStep(
                id=uuid4(),
                step_id="step_1",
                description="Check cache hit rate for potential cache misses",
                step_type=StepType.MONITORING_CHECK,
                order=1,
                gcp_integration=GCPIntegration(
                    service="monitoring",
                    metric="redis/cache_hit_rate"
                ),
                expected_result=ExpectedResult(threshold="> 80%"),
                escalation_condition="< 60%",
                created_at=datetime.utcnow()
            ),
            PlaybookStep(
                id=uuid4(),
                step_id="step_2",
                description="Verify database connection pool utilization",
                step_type=StepType.MONITORING_CHECK,
                order=2,
                gcp_integration=GCPIntegration(
                    service="monitoring",
                    metric="cloudsql/database/connection_count"
                ),
                expected_result=ExpectedResult(threshold="< 80%"),
                escalation_condition="> 90%",
                created_at=datetime.utcnow()
            )
        ]
        
        return Playbook(
            id=uuid4(),
            playbook_id="PB-LATENCY-001",
            name="High Latency Troubleshooting",
            version="1.2.0",
            description="Systematic approach to diagnose and resolve API latency issues",
            status=PlaybookStatus.ACTIVE,
            effectiveness_score=0.87,
            applicable_services=["payment-api", "user-api", "order-api"],
            trigger_conditions=["API latency > 3s", "Response time degradation > 50%"],
            created_by="SRE Team",
            created_at=datetime.utcnow(),
            steps=steps
        )
    
    def _create_mock_db_timeout_playbook(self) -> Playbook:
        """Create mock database timeout troubleshooting playbook."""
        from models.playbook import GCPIntegration, ExpectedResult
        
        steps = [
            PlaybookStep(
                id=uuid4(),
                step_id="step_1",
                description="Check database connection pool utilization",
                step_type=StepType.MONITORING_CHECK,
                order=1,
                gcp_integration=GCPIntegration(
                    service="monitoring",
                    metric="cloudsql/database/connection_count"
                ),
                expected_result=ExpectedResult(threshold="< 80%"),
                escalation_condition="> 90%",
                created_at=datetime.utcnow()
            ),
            PlaybookStep(
                id=uuid4(),
                step_id="step_2",
                description="Search for connection timeout errors in logs",
                step_type=StepType.LOG_ANALYSIS,
                order=2,
                query="connection timeout OR connection refused",
                expected_result=ExpectedResult(threshold="< 5 errors/hour"),
                escalation_condition="> 10 errors/hour",
                created_at=datetime.utcnow()
            ),
            PlaybookStep(
                id=uuid4(),
                step_id="step_3",
                description="Analyze slow query performance",
                step_type=StepType.QUERY_ANALYSIS,
                order=3,
                expected_result=ExpectedResult(threshold="< 5s"),
                escalation_condition="> 60s",
                created_at=datetime.utcnow()
            )
        ]
        
        return Playbook(
            id=uuid4(),
            playbook_id="PB-DB-TIMEOUT-001",
            name="Database Timeout Troubleshooting",
            version="1.1.0",
            description="Systematic approach to diagnose database connection and query timeout issues",
            status=PlaybookStatus.ACTIVE,
            effectiveness_score=0.92,
            applicable_services=["ecommerce-api", "inventory-service", "order-service"],
            trigger_conditions=["Database connection timeout", "Connection pool exhaustion"],
            created_by="Database Team",
            created_at=datetime.utcnow(),
            steps=steps
        )
    
    def _create_mock_cpu_playbook(self) -> Playbook:
        """Create mock CPU troubleshooting playbook."""
        steps = [
            PlaybookStep(
                id=uuid4(),
                step_id="step_1",
                description="Check CPU utilization across all instances",
                step_type=StepType.MONITORING_CHECK,
                order=1,
                gcp_integration=GCPIntegration(
                    service="monitoring",
                    metric="compute/instance/cpu_utilization"
                ),
                expected_result=ExpectedResult(threshold="< 70%"),
                escalation_condition="> 90%",
                created_at=datetime.utcnow()
            ),
            PlaybookStep(
                id=uuid4(),
                step_id="step_2",
                description="Identify top CPU-consuming processes",
                step_type=StepType.PROCESS_ANALYSIS,
                order=2,
                expected_result=ExpectedResult(threshold="no single process > 50%"),
                escalation_condition="process consuming > 80%",
                created_at=datetime.utcnow()
            )
        ]
        
        return Playbook(
            id=uuid4(),
            playbook_id="PB-CPU-001",
            name="High CPU Troubleshooting",
            version="1.0.0",
            description="Systematic approach to diagnose high CPU utilization issues",
            status=PlaybookStatus.ACTIVE,
            effectiveness_score=0.79,
            applicable_services=["all"],
            trigger_conditions=["CPU utilization > 80%"],
            created_by="Infrastructure Team",
            created_at=datetime.utcnow(),
            steps=steps
        )
    
    def _parse_threshold(self, threshold_str: str) -> Dict[str, Any]:
        """Parse threshold string like '< 80%' or '> 90' into operator and value."""
        threshold_str = threshold_str.strip()
        
        if threshold_str.startswith('>'):
            operator = '>'
            value_str = threshold_str[1:].strip()
        elif threshold_str.startswith('<'):
            operator = '<'
            value_str = threshold_str[1:].strip()
        elif threshold_str.startswith('>='):
            operator = '>='
            value_str = threshold_str[2:].strip()
        elif threshold_str.startswith('<='):
            operator = '<='
            value_str = threshold_str[2:].strip()
        else:
            operator = '=='
            value_str = threshold_str
        
        # Extract numeric value
        value = 0.0
        if '%' in value_str:
            value = float(value_str.replace('%', ''))
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = 0.0
        
        return {"operator": operator, "value": value}
    
    def _evaluate_threshold(self, actual: float, operator: str, expected: float) -> bool:
        """Evaluate if actual value meets threshold condition."""
        if operator == '>':
            return actual > expected
        elif operator == '<':
            return actual < expected
        elif operator == '>=':
            return actual >= expected
        elif operator == '<=':
            return actual <= expected
        elif operator == '==':
            return abs(actual - expected) < 0.001
        else:
            return False
    
    def _parse_latency_threshold(self, threshold_str: str) -> float:
        """Parse latency threshold string like '< 1s' or '> 500ms' into milliseconds."""
        threshold_str = threshold_str.strip().lower()
        
        # Extract numeric part
        import re
        numeric_match = re.search(r'[\d.]+', threshold_str)
        if not numeric_match:
            return 1000.0  # Default 1 second
        
        value = float(numeric_match.group())
        
        # Convert to milliseconds
        if 's' in threshold_str and 'ms' not in threshold_str:
            return value * 1000  # seconds to milliseconds
        elif 'ms' in threshold_str:
            return value  # already milliseconds
        else:
            return value  # assume milliseconds
    
    async def execute_next_step(
        self,
        execution_id: str,
        step_id: str,
        manual_input: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PlaybookStepResult:
        """
        Execute the next step in a playbook execution.
        
        Args:
            execution_id: ID of the playbook execution
            step_id: ID of the step to execute
            manual_input: Optional manual input from user
            context: Additional context for step execution
            
        Returns:
            PlaybookStepResult with execution details
        """
        logger.info(f"Executing next step {step_id} for execution {execution_id}")
        
        try:
            # Get execution from active executions or mock one
            if execution_id in self._active_executions:
                execution = self._active_executions[execution_id]
            else:
                # Create a mock execution for demo
                from uuid import uuid4
                execution = PlaybookExecution(
                    id=uuid4(),  # Required field
                    execution_id=execution_id,
                    playbook_id="PB-DEMO-001",
                    incident_id="INC-DEMO-001",
                    status=ExecutionStatus.RUNNING,
                    started_by="demo_user",  # Required field (changed from executed_by)
                    started_at=datetime.utcnow(),
                    execution_context=context or {}  # Changed from context
                )
                self._active_executions[execution_id] = execution
            
            # Create a mock step result with proper UUID fields
            from uuid import uuid4, UUID
            
            # Convert step_id to UUID if it's a string
            if isinstance(step_id, str):
                try:
                    step_uuid = UUID(step_id)
                except ValueError:
                    # If step_id is not a valid UUID string, generate a new UUID
                    step_uuid = uuid4()
            else:
                step_uuid = step_id
            
            step_result = PlaybookStepResult(
                id=uuid4(),  # Required field
                step_id=step_uuid,  # Convert to UUID
                status=ExecutionStatus.COMPLETED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_seconds=0.1,
                result_data={
                    "manual_input": manual_input,
                    "step_completed": True,
                    "user_confirmation": "Step executed successfully"
                },
                evidence={
                    "user_input": manual_input,
                    "timestamp": datetime.utcnow().isoformat()
                },
                success=True,
                error_message=None,
                threshold_met=True,
                escalation_triggered=False
            )
            
            logger.info(f"Step {step_id} executed successfully")
            return step_result
            
        except Exception as e:
            logger.error(f"Error executing step {step_id}: {e}")
            
            # Return failed step result
            return PlaybookStepResult(
                execution_id=execution_id,
                step_id=step_id,
                step_type=StepType.MANUAL_ACTION,
                status=ExecutionStatus.FAILED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                result_data={},
                evidence={},
                success=False,
                error_message=str(e)
            )
    
    async def process_approval(
        self,
        execution_id: str,
        step_id: str,
        action: str,
        reason: str,
        user: Any
    ) -> Dict[str, Any]:
        """
        Process approval/rejection for a playbook step.
        
        Args:
            execution_id: ID of the playbook execution
            step_id: ID of the step to approve/reject
            action: 'approve' or 'reject'
            reason: Reason for the action
            user: User making the decision
            
        Returns:
            Processing result
        """
        logger.info(f"Processing {action} for step {step_id} in execution {execution_id}")
        
        try:
            # Get execution from active executions
            if execution_id in self._active_executions:
                execution = self._active_executions[execution_id]
            else:
                # Create a mock execution for demo
                from uuid import uuid4
                execution = PlaybookExecution(
                    id=uuid4(),  # Required field
                    execution_id=execution_id,
                    playbook_id="PB-DEMO-001",
                    incident_id="INC-DEMO-001",
                    status=ExecutionStatus.RUNNING,
                    started_by=getattr(user, 'username', getattr(user, 'user_id', 'demo_user')),  # Required field
                    started_at=datetime.utcnow(),
                    execution_context={}  # Changed from context
                )
                self._active_executions[execution_id] = execution
            
            if action == "approve":
                # Continue with the execution
                execution.status = ExecutionStatus.RUNNING
                result = {
                    "action_taken": "approved",
                    "next_action": "continue_execution",
                    "message": f"Step {step_id} approved by {getattr(user, 'user_id', 'unknown')}"
                }
            else:  # reject
                # Stop the execution
                execution.status = ExecutionStatus.FAILED
                result = {
                    "action_taken": "rejected",
                    "next_action": "stop_execution", 
                    "message": f"Step {step_id} rejected by {getattr(user, 'user_id', 'unknown')}: {reason}"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing approval for step {step_id}: {e}")
            return {
                "action_taken": "error",
                "next_action": "manual_intervention_required",
                "message": f"Error processing approval: {str(e)}"
            }