"""
GCP Cloud Logging integration with mock implementation for demo.

This module provides integration with GCP Cloud Logging API, simulating real
logging API responses using synthetic log data for error analysis, pattern
detection, and incident correlation.
"""

import asyncio
import json
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from loguru import logger
from collections import Counter, defaultdict

try:
    from google.cloud import logging
    from google.cloud.logging_v2 import entries
    from google.oauth2 import service_account
    GCP_LOGGING_AVAILABLE = True
except ImportError:
    GCP_LOGGING_AVAILABLE = False
    logger.warning("Google Cloud Logging client not available, using mock implementation")

from config.settings import get_settings

settings = get_settings()


class LogEntry:
    """Represents a log entry with all relevant fields."""
    
    def __init__(
        self,
        timestamp: datetime,
        severity: str,
        message: str,
        resource: Dict[str, Any],
        labels: Optional[Dict[str, str]] = None,
        source_location: Optional[Dict[str, str]] = None,
        http_request: Optional[Dict[str, Any]] = None,
        operation: Optional[Dict[str, str]] = None,
        trace: Optional[str] = None,
        span_id: Optional[str] = None
    ):
        self.timestamp = timestamp
        self.severity = severity
        self.message = message
        self.resource = resource
        self.labels = labels or {}
        self.source_location = source_location
        self.http_request = http_request
        self.operation = operation
        self.trace = trace
        self.span_id = span_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "message": self.message,
            "resource": self.resource,
            "labels": self.labels,
            "source_location": self.source_location,
            "http_request": self.http_request,
            "operation": self.operation,
            "trace": self.trace,
            "span_id": self.span_id
        }


class GCPLoggingClient:
    """GCP Cloud Logging integration client."""
    
    def __init__(self):
        """Initialize the GCP Logging client."""
        self.project_id = settings.GCP_PROJECT_ID
        self.use_mock = settings.USE_MOCK_GCP_SERVICES or not GCP_LOGGING_AVAILABLE
        self.client = None
        self.synthetic_data_path = Path(settings.SYNTHETIC_DATA_PATH)
        
        # Mock data cache
        self._mock_logs_cache = []
        self._last_cache_update = None
        self._cache_ttl = timedelta(minutes=5)
        
        # Log patterns for different scenarios
        self.log_patterns = {
            "database_timeout": [
                "Connection timeout: Unable to acquire connection from pool within 5000ms",
                "Database query timeout after 5000ms for user_profile_lookup",
                "Connection pool exhausted: 190/200 connections in use",
                "Slow query detected: SELECT * FROM invoices WHERE customer_id IN (...) took 8.2s",
                "Database connection refused: Too many connections"
            ],
            "cache_miss": [
                "Cache miss for user_profile_12345, falling back to database",
                "Redis connection timeout: Could not connect to Redis cluster",
                "Cache cluster node failure detected, switching to backup",
                "High cache miss rate detected: 62% of requests missed",
                "Redis memory limit exceeded, evicting keys"
            ],
            "api_errors": [
                "HTTP 500: Internal server error in payment processing",
                "API rate limit exceeded: 1000 requests/minute limit hit",
                "Authentication failed: Invalid JWT token",
                "Request validation error: Missing required field 'amount'",
                "Downstream service unavailable: billing-service not responding"
            ],
            "memory_issues": [
                "OutOfMemoryError: Java heap space exceeded",
                "Memory usage warning: 85% of allocated memory in use",
                "Garbage collection took 2.3s, application paused",
                "Memory leak detected in user session management",
                "Container memory limit reached: 2GB"
            ],
            "network_issues": [
                "Network timeout: Request to external API timed out after 30s",
                "DNS resolution failed for api.external-service.com",
                "Connection reset by peer during data transfer",
                "SSL handshake failed: Certificate validation error",
                "Load balancer health check failed for 3 consecutive attempts"
            ]
        }
        
        # Severity mappings
        self.severity_weights = {
            "EMERGENCY": 800,
            "ALERT": 700,
            "CRITICAL": 600,
            "ERROR": 500,
            "WARNING": 400,
            "NOTICE": 300,
            "INFO": 200,
            "DEBUG": 100
        }
        
        if not self.use_mock:
            self._initialize_real_client()
        else:
            logger.info("Using mock GCP Logging implementation for demo")
    
    def _initialize_real_client(self) -> None:
        """Initialize real GCP Logging client."""
        try:
            if settings.GCP_SERVICE_ACCOUNT_KEY_PATH:
                credentials = service_account.Credentials.from_service_account_file(
                    settings.GCP_SERVICE_ACCOUNT_KEY_PATH
                )
                self.client = logging.Client(
                    project=self.project_id,
                    credentials=credentials
                )
            else:
                # Use default credentials
                self.client = logging.Client(project=self.project_id)
            
            logger.info(f"Initialized GCP Logging client for project: {self.project_id}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize real GCP Logging client: {e}")
            logger.info("Falling back to mock implementation")
            self.use_mock = True
            self.client = None
    
    async def search_logs(
        self,
        query: str,
        time_range: str = "1h",
        resource_labels: Optional[Dict[str, str]] = None,
        severity_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search logs using a query string.
        
        Args:
            query: Search query (e.g., "ERROR", "connection timeout")
            time_range: Time range for search (e.g., "1h", "6h", "1d")
            resource_labels: Resource labels to filter by
            severity_filter: List of severities to include
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entry dictionaries
        """
        logger.info(f"Searching logs with query: '{query}' for time range: {time_range}")
        
        try:
            if self.use_mock:
                return await self._search_logs_mock(query, time_range, resource_labels, severity_filter, limit)
            else:
                return await self._search_logs_real(query, time_range, resource_labels, severity_filter, limit)
                
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []
    
    async def get_log_entries_by_resource(
        self,
        resource_type: str,
        resource_name: str,
        time_range: str = "1h",
        severity_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get log entries for a specific resource.
        
        Args:
            resource_type: Type of resource (e.g., "cloud_run_revision", "gce_instance")
            resource_name: Name/ID of the resource
            time_range: Time range for search
            severity_filter: List of severities to include
            
        Returns:
            List of log entry dictionaries
        """
        logger.info(f"Getting logs for resource: {resource_type}/{resource_name}")
        
        try:
            if self.use_mock:
                return await self._get_logs_by_resource_mock(resource_type, resource_name, time_range, severity_filter)
            else:
                return await self._get_logs_by_resource_real(resource_type, resource_name, time_range, severity_filter)
                
        except Exception as e:
            logger.error(f"Error getting logs by resource: {e}")
            return []
    
    async def analyze_error_patterns(
        self,
        service_name: str,
        time_range: str = "1h",
        min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze error patterns for a service.
        
        Args:
            service_name: Name of the service to analyze
            time_range: Time range for analysis
            min_occurrences: Minimum occurrences for pattern recognition
            
        Returns:
            Dictionary containing error pattern analysis
        """
        logger.info(f"Analyzing error patterns for service: {service_name}")
        
        try:
            # Search for error logs
            error_logs = await self.search_logs(
                query="ERROR OR CRITICAL OR ALERT",
                time_range=time_range,
                resource_labels={"service": service_name},
                severity_filter=["ERROR", "CRITICAL", "ALERT"]
            )
            
            if not error_logs:
                return {
                    "service_name": service_name,
                    "time_range": time_range,
                    "total_errors": 0,
                    "error_patterns": [],
                    "top_errors": [],
                    "error_distribution": {},
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            
            # Analyze patterns
            error_messages = [log.get("message", "") for log in error_logs]
            error_patterns = self._extract_error_patterns(error_messages, min_occurrences)
            
            # Group by severity
            severity_distribution = Counter(log.get("severity", "UNKNOWN") for log in error_logs)
            
            # Find most common errors
            error_counter = Counter(error_messages)
            top_errors = [
                {"message": msg, "count": count, "percentage": (count / len(error_logs)) * 100}
                for msg, count in error_counter.most_common(10)
            ]
            
            # Time-based analysis
            time_distribution = self._analyze_error_timeline(error_logs)
            
            # Calculate error rate
            total_logs = await self._get_total_log_count(service_name, time_range)
            error_rate = (len(error_logs) / max(1, total_logs)) * 100
            
            return {
                "service_name": service_name,
                "time_range": time_range,
                "total_errors": len(error_logs),
                "error_rate_percentage": round(error_rate, 2),
                "error_patterns": error_patterns,
                "top_errors": top_errors,
                "severity_distribution": dict(severity_distribution),
                "time_distribution": time_distribution,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing error patterns: {e}")
            return {
                "service_name": service_name,
                "error": str(e),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_log_statistics(
        self,
        service_name: Optional[str] = None,
        time_range: str = "1h"
    ) -> Dict[str, Any]:
        """
        Get comprehensive log statistics.
        
        Args:
            service_name: Optional service name to filter by
            time_range: Time range for statistics
            
        Returns:
            Dictionary containing log statistics
        """
        logger.info(f"Getting log statistics for service: {service_name or 'all'}")
        
        try:
            # Get all logs for the time range
            resource_labels = {"service": service_name} if service_name else None
            all_logs = await self.search_logs(
                query="*",
                time_range=time_range,
                resource_labels=resource_labels,
                limit=1000
            )
            
            if not all_logs:
                return {
                    "service_name": service_name,
                    "time_range": time_range,
                    "total_logs": 0,
                    "statistics": {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Calculate statistics
            severity_counts = Counter(log.get("severity", "UNKNOWN") for log in all_logs)
            resource_counts = Counter(
                log.get("resource", {}).get("labels", {}).get("service_name", "unknown")
                for log in all_logs
            )
            
            # Calculate log volume over time
            time_buckets = self._bucket_logs_by_time(all_logs, time_range)
            
            # Error rate calculation
            error_logs = [log for log in all_logs if log.get("severity") in ["ERROR", "CRITICAL", "ALERT"]]
            error_rate = (len(error_logs) / len(all_logs)) * 100
            
            # Find peak periods
            peak_periods = self._find_peak_periods(time_buckets)
            
            statistics = {
                "total_logs": len(all_logs),
                "error_rate_percentage": round(error_rate, 2),
                "severity_distribution": dict(severity_counts),
                "resource_distribution": dict(resource_counts.most_common(10)),
                "logs_per_minute": self._calculate_logs_per_minute(all_logs, time_range),
                "peak_periods": peak_periods,
                "time_buckets": time_buckets
            }
            
            return {
                "service_name": service_name,
                "time_range": time_range,
                "statistics": statistics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting log statistics: {e}")
            return {
                "service_name": service_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def correlate_logs_with_incident(
        self,
        incident_timestamp: datetime,
        service_name: str,
        time_window_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Correlate logs with an incident timeframe.
        
        Args:
            incident_timestamp: When the incident occurred
            service_name: Service involved in the incident
            time_window_minutes: Time window around incident to analyze
            
        Returns:
            Dictionary containing correlated log analysis
        """
        logger.info(f"Correlating logs with incident for {service_name} at {incident_timestamp}")
        
        try:
            # Define time window around incident
            start_time = incident_timestamp - timedelta(minutes=time_window_minutes)
            end_time = incident_timestamp + timedelta(minutes=time_window_minutes)
            
            # Search for logs in the incident timeframe
            incident_logs = await self._search_logs_timeframe(
                start_time=start_time,
                end_time=end_time,
                resource_labels={"service": service_name}
            )
            
            if not incident_logs:
                return {
                    "incident_timestamp": incident_timestamp.isoformat(),
                    "service_name": service_name,
                    "correlated_logs": 0,
                    "findings": [],
                    "timeline": [],
                    "correlation_strength": 0.0
                }
            
            # Analyze logs around incident time
            pre_incident_logs = [
                log for log in incident_logs
                if datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')) < incident_timestamp
            ]
            
            during_incident_logs = [
                log for log in incident_logs
                if abs((datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')) - incident_timestamp).total_seconds()) <= 300  # 5 minutes
            ]
            
            post_incident_logs = [
                log for log in incident_logs
                if datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')) > incident_timestamp
            ]
            
            # Find correlating patterns
            correlation_findings = []
            correlation_strength = 0.0
            
            # Check for error spikes
            error_logs_during = [log for log in during_incident_logs if log.get("severity") in ["ERROR", "CRITICAL"]]
            if error_logs_during:
                correlation_strength += 0.4
                correlation_findings.append({
                    "type": "error_spike",
                    "description": f"Detected {len(error_logs_during)} error logs during incident timeframe",
                    "confidence": 0.8,
                    "evidence": error_logs_during[:5]  # First 5 errors
                })
            
            # Check for pattern changes
            pre_patterns = self._extract_log_patterns([log.get("message", "") for log in pre_incident_logs])
            during_patterns = self._extract_log_patterns([log.get("message", "") for log in during_incident_logs])
            
            new_patterns = set(during_patterns.keys()) - set(pre_patterns.keys())
            if new_patterns:
                correlation_strength += 0.3
                correlation_findings.append({
                    "type": "new_error_patterns",
                    "description": f"Detected {len(new_patterns)} new error patterns during incident",
                    "confidence": 0.7,
                    "patterns": list(new_patterns)[:3]
                })
            
            # Check for log volume changes
            pre_volume = len(pre_incident_logs) / max(1, time_window_minutes)
            during_volume = len(during_incident_logs) / 10  # 10 minutes window
            
            if during_volume > pre_volume * 1.5:
                correlation_strength += 0.2
                correlation_findings.append({
                    "type": "log_volume_spike",
                    "description": f"Log volume increased from {pre_volume:.1f} to {during_volume:.1f} logs/min",
                    "confidence": 0.6
                })
            
            # Create timeline
            timeline = self._create_incident_timeline(incident_logs, incident_timestamp)
            
            return {
                "incident_timestamp": incident_timestamp.isoformat(),
                "service_name": service_name,
                "time_window_minutes": time_window_minutes,
                "correlated_logs": len(incident_logs),
                "pre_incident_logs": len(pre_incident_logs),
                "during_incident_logs": len(during_incident_logs),
                "post_incident_logs": len(post_incident_logs),
                "correlation_strength": min(1.0, correlation_strength),
                "findings": correlation_findings,
                "timeline": timeline,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error correlating logs with incident: {e}")
            return {
                "incident_timestamp": incident_timestamp.isoformat(),
                "error": str(e),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def _search_logs_mock(
        self,
        query: str,
        time_range: str,
        resource_labels: Optional[Dict[str, str]] = None,
        severity_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Mock implementation of log searching."""
        
        # Load synthetic data
        await self._load_synthetic_logs()
        
        # Parse time range
        end_time = datetime.utcnow()
        start_time = self._parse_time_range(time_range, end_time)
        
        # Generate logs based on query and filters
        matching_logs = []
        
        # Determine log scenario based on query
        scenario = self._determine_log_scenario(query, resource_labels)
        
        # Generate appropriate number of logs
        num_logs = min(limit, random.randint(10, 50))
        
        for i in range(num_logs):
            log_entry = self._generate_mock_log_entry(
                scenario=scenario,
                timestamp=self._random_timestamp_in_range(start_time, end_time),
                resource_labels=resource_labels
            )
            
            # Apply filters
            if severity_filter and log_entry["severity"] not in severity_filter:
                continue
            
            # Apply query filter
            if self._matches_query(log_entry["message"], query):
                matching_logs.append(log_entry)
        
        # Ensure we have timeout errors for database_timeout scenario
        if scenario == "database_timeout" and len(matching_logs) == 0:
            # Force generate some matching logs for demo purposes
            for i in range(random.randint(5, 15)):
                log_entry = self._generate_mock_log_entry(
                    scenario="database_timeout",
                    timestamp=self._random_timestamp_in_range(start_time, end_time),
                    resource_labels=resource_labels
                )
                # Force match by ensuring it has timeout/refused keywords
                if "timeout" in log_entry["message"].lower() or "refused" in log_entry["message"].lower():
                    matching_logs.append(log_entry)
        
        # Sort by timestamp (newest first)
        matching_logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return matching_logs[:limit]
    
    async def _search_logs_real(
        self,
        query: str,
        time_range: str,
        resource_labels: Optional[Dict[str, str]] = None,
        severity_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Real implementation of log searching using GCP Logging API."""
        
        if not self.client:
            raise Exception("GCP Logging client not initialized")
        
        # Parse time range
        end_time = datetime.utcnow()
        start_time = self._parse_time_range(time_range, end_time)
        
        # Build filter
        filter_parts = []
        
        # Time filter
        filter_parts.append(f'timestamp >= "{start_time.isoformat()}Z"')
        filter_parts.append(f'timestamp <= "{end_time.isoformat()}Z"')
        
        # Query filter
        if query and query != "*":
            filter_parts.append(f'textPayload:"{query}" OR jsonPayload.message:"{query}"')
        
        # Resource filters
        if resource_labels:
            for key, value in resource_labels.items():
                filter_parts.append(f'resource.labels.{key}="{value}"')
        
        # Severity filter
        if severity_filter:
            severity_conditions = [f'severity="{sev}"' for sev in severity_filter]
            filter_parts.append(f'({" OR ".join(severity_conditions)})')
        
        filter_str = " AND ".join(filter_parts)
        
        # Execute query
        entries = self.client.list_entries(
            filter_=filter_str,
            order_by=logging.DESCENDING,
            max_results=limit
        )
        
        # Convert to dictionaries
        result_logs = []
        for entry in entries:
            log_dict = {
                "timestamp": entry.timestamp.isoformat(),
                "severity": entry.severity,
                "message": entry.payload.get("message", str(entry.payload)) if hasattr(entry.payload, 'get') else str(entry.payload),
                "resource": {
                    "type": entry.resource.type,
                    "labels": dict(entry.resource.labels)
                },
                "labels": dict(entry.labels) if entry.labels else {},
                "log_name": entry.log_name,
                "insert_id": entry.insert_id
            }
            
            # Add optional fields
            if hasattr(entry, 'source_location') and entry.source_location:
                log_dict["source_location"] = {
                    "file": entry.source_location.file,
                    "line": entry.source_location.line,
                    "function": entry.source_location.function
                }
            
            if hasattr(entry, 'http_request') and entry.http_request:
                log_dict["http_request"] = {
                    "request_method": entry.http_request.request_method,
                    "request_url": entry.http_request.request_url,
                    "status": entry.http_request.status
                }
            
            result_logs.append(log_dict)
        
        return result_logs
    
    async def _get_logs_by_resource_mock(
        self,
        resource_type: str,
        resource_name: str,
        time_range: str,
        severity_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Mock implementation of getting logs by resource."""
        
        return await self._search_logs_mock(
            query="*",
            time_range=time_range,
            resource_labels={"service": resource_name, "resource_type": resource_type},
            severity_filter=severity_filter
        )
    
    async def _get_logs_by_resource_real(
        self,
        resource_type: str,
        resource_name: str,
        time_range: str,
        severity_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Real implementation of getting logs by resource."""
        
        return await self._search_logs_real(
            query="*",
            time_range=time_range,
            resource_labels={"resource_name": resource_name, "resource_type": resource_type},
            severity_filter=severity_filter
        )
    
    async def _search_logs_timeframe(
        self,
        start_time: datetime,
        end_time: datetime,
        resource_labels: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Search logs within a specific timeframe."""
        
        if self.use_mock:
            # Generate logs for the specific timeframe
            logs = []
            current_time = start_time
            
            while current_time < end_time:
                if random.random() < 0.3:  # 30% chance of log per minute
                    log_entry = self._generate_mock_log_entry(
                        scenario="mixed",
                        timestamp=current_time,
                        resource_labels=resource_labels
                    )
                    logs.append(log_entry)
                
                current_time += timedelta(minutes=1)
            
            return logs
        else:
            # Use real search with timeframe
            duration = end_time - start_time
            time_range_str = f"{int(duration.total_seconds()/3600)}h"
            
            return await self._search_logs_real(
                query="*",
                time_range=time_range_str,
                resource_labels=resource_labels
            )
    
    async def _load_synthetic_logs(self) -> None:
        """Load synthetic log data for mock responses."""
        
        # Check if cache needs refresh
        if (self._last_cache_update and 
            datetime.utcnow() - self._last_cache_update < self._cache_ttl and
            self._mock_logs_cache):
            return
        
        try:
            # Load GCP data file
            gcp_data_file = self.synthetic_data_path / "scenario_1_gcp_data.json"
            if gcp_data_file.exists():
                with open(gcp_data_file, 'r') as f:
                    content = f.read(10000)  # Read first 10KB
                    try:
                        gcp_data = json.loads(content)
                        if "logging_data" in gcp_data:
                            self._mock_logs_cache = gcp_data["logging_data"]
                    except json.JSONDecodeError:
                        # Parse line by line for partial data
                        lines = content.split('\n')
                        for line in lines[:50]:  # First 50 lines
                            if line.strip() and '"message"' in line:
                                try:
                                    log_obj = json.loads(line)
                                    self._mock_logs_cache.append(log_obj)
                                except:
                                    continue
            
            self._last_cache_update = datetime.utcnow()
            logger.debug(f"Loaded {len(self._mock_logs_cache)} synthetic log entries")
            
        except Exception as e:
            logger.warning(f"Error loading synthetic logs: {e}")
            self._mock_logs_cache = []
    
    def _determine_log_scenario(
        self,
        query: str,
        resource_labels: Optional[Dict[str, str]] = None
    ) -> str:
        """Determine which log scenario to use based on query."""
        
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["connection", "timeout", "database"]):
            return "database_timeout"
        elif any(term in query_lower for term in ["cache", "redis", "miss"]):
            return "cache_miss"
        elif any(term in query_lower for term in ["error", "500", "api"]):
            return "api_errors"
        elif any(term in query_lower for term in ["memory", "oom", "heap"]):
            return "memory_issues"
        elif any(term in query_lower for term in ["network", "dns", "ssl"]):
            return "network_issues"
        else:
            return "mixed"
    
    def _generate_mock_log_entry(
        self,
        scenario: str,
        timestamp: datetime,
        resource_labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate a mock log entry for the given scenario."""
        
        # Select appropriate message pattern
        if scenario in self.log_patterns:
            message = random.choice(self.log_patterns[scenario])
        else:
            # Mixed scenario - random from all patterns
            all_messages = []
            for patterns in self.log_patterns.values():
                all_messages.extend(patterns)
            message = random.choice(all_messages)
        
        # Determine severity based on message content
        severity = self._determine_severity_from_message(message)
        
        # Create resource info
        service_name = resource_labels.get("service", "payment-api") if resource_labels else "payment-api"
        resource = {
            "type": "cloud_run_revision",
            "labels": {
                "service_name": service_name,
                "revision_name": f"{service_name}-v1-abc123",
                "location": "us-central1",
                "project_id": self.project_id
            }
        }
        
        # Add some contextual labels
        labels = {
            "component": self._extract_component_from_message(message),
            "environment": "production"
        }
        
        # Add source location for errors
        source_location = None
        if severity in ["ERROR", "CRITICAL"]:
            source_location = {
                "file": f"src/main/java/com/{service_name}/Service.java",
                "line": str(random.randint(100, 500)),
                "function": "processRequest"
            }
        
        # Add HTTP request info for API errors
        http_request = None
        if "api" in message.lower() or "http" in message.lower():
            http_request = {
                "request_method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                "request_url": f"https://{service_name}.example.com/api/v1/endpoint",
                "status": 500 if severity == "ERROR" else 200,
                "response_size": str(random.randint(1000, 10000))
            }
        
        return {
            "timestamp": timestamp.isoformat() + "Z",
            "severity": severity,
            "message": message,
            "resource": resource,
            "labels": labels,
            "source_location": source_location,
            "http_request": http_request,
            "log_name": f"projects/{self.project_id}/logs/cloudrun.googleapis.com%2Fstdout",
            "insert_id": f"log_{random.randint(1000000, 9999999)}"
        }
    
    def _determine_severity_from_message(self, message: str) -> str:
        """Determine log severity from message content."""
        
        message_lower = message.lower()
        
        if any(term in message_lower for term in ["critical", "fatal", "emergency"]):
            return "CRITICAL"
        elif any(term in message_lower for term in ["error", "exception", "failed", "failure"]):
            return "ERROR"
        elif any(term in message_lower for term in ["warning", "warn", "timeout", "slow"]):
            return "WARNING"
        elif any(term in message_lower for term in ["info", "started", "completed", "success"]):
            return "INFO"
        else:
            return random.choice(["INFO", "WARNING", "ERROR"])
    
    def _extract_component_from_message(self, message: str) -> str:
        """Extract component name from log message."""
        
        message_lower = message.lower()
        
        if any(term in message_lower for term in ["database", "sql", "connection"]):
            return "database"
        elif any(term in message_lower for term in ["cache", "redis"]):
            return "cache"
        elif any(term in message_lower for term in ["api", "http", "request"]):
            return "api"
        elif any(term in message_lower for term in ["memory", "heap", "gc"]):
            return "memory"
        elif any(term in message_lower for term in ["network", "dns", "ssl"]):
            return "network"
        else:
            return "application"
    
    def _matches_query(self, message: str, query: str) -> bool:
        """Check if log message matches search query."""
        
        if query == "*" or not query:
            return True
        
        query_lower = query.lower()
        message_lower = message.lower()
        
        # Support boolean operators
        if " OR " in query_lower:
            terms = [term.strip() for term in query_lower.split(" or ")]
            return any(term in message_lower for term in terms)
        elif " AND " in query_lower:
            terms = [term.strip() for term in query_lower.split(" and ")]
            return all(term in message_lower for term in terms)
        else:
            return query_lower in message_lower
    
    def _extract_error_patterns(
        self,
        error_messages: List[str],
        min_occurrences: int
    ) -> List[Dict[str, Any]]:
        """Extract common error patterns from messages."""
        
        # Group similar error messages
        pattern_groups = defaultdict(list)
        
        for message in error_messages:
            # Create pattern by replacing numbers and IDs with placeholders
            pattern = re.sub(r'\d+', '{number}', message)
            pattern = re.sub(r'[a-f0-9]{8,}', '{id}', pattern)
            pattern = re.sub(r'\d{4}-\d{2}-\d{2}', '{date}', pattern)
            
            pattern_groups[pattern].append(message)
        
        # Filter by minimum occurrences
        patterns = []
        for pattern, messages in pattern_groups.items():
            if len(messages) >= min_occurrences:
                patterns.append({
                    "pattern": pattern,
                    "occurrences": len(messages),
                    "percentage": (len(messages) / len(error_messages)) * 100,
                    "sample_messages": messages[:3],
                    "severity": "high" if len(messages) > len(error_messages) * 0.3 else "medium"
                })
        
        # Sort by occurrences
        patterns.sort(key=lambda x: x["occurrences"], reverse=True)
        
        return patterns
    
    def _analyze_error_timeline(self, error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error distribution over time."""
        
        if not error_logs:
            return {}
        
        # Group errors by hour
        hourly_counts = defaultdict(int)
        for log in error_logs:
            timestamp = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1
        
        # Find peak hours
        sorted_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)
        peak_hours = sorted_hours[:3]
        
        return {
            "hourly_distribution": dict(hourly_counts),
            "peak_hours": [{"hour": hour, "error_count": count} for hour, count in peak_hours],
            "total_hours_with_errors": len(hourly_counts),
            "max_errors_per_hour": max(hourly_counts.values()) if hourly_counts else 0
        }
    
    def _bucket_logs_by_time(
        self,
        logs: List[Dict[str, Any]],
        time_range: str
    ) -> List[Dict[str, Any]]:
        """Bucket logs by time intervals."""
        
        # Determine bucket size based on time range
        if "h" in time_range and int(time_range.replace("h", "")) <= 6:
            bucket_minutes = 10  # 10-minute buckets for short ranges
        elif "h" in time_range:
            bucket_minutes = 60  # 1-hour buckets for longer ranges
        else:
            bucket_minutes = 30  # 30-minute default
        
        buckets = defaultdict(int)
        
        for log in logs:
            timestamp = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
            # Round down to bucket boundary
            bucket_time = timestamp.replace(
                minute=(timestamp.minute // bucket_minutes) * bucket_minutes,
                second=0,
                microsecond=0
            )
            bucket_key = bucket_time.strftime("%Y-%m-%d %H:%M")
            buckets[bucket_key] += 1
        
        # Convert to list format
        return [
            {"time": time, "count": count}
            for time, count in sorted(buckets.items())
        ]
    
    def _find_peak_periods(self, time_buckets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find peak logging periods."""
        
        if not time_buckets:
            return []
        
        # Calculate average
        counts = [bucket["count"] for bucket in time_buckets]
        avg_count = sum(counts) / len(counts)
        
        # Find buckets significantly above average
        peak_threshold = avg_count * 1.5
        peaks = [
            bucket for bucket in time_buckets
            if bucket["count"] > peak_threshold
        ]
        
        return sorted(peaks, key=lambda x: x["count"], reverse=True)[:5]
    
    def _calculate_logs_per_minute(
        self,
        logs: List[Dict[str, Any]],
        time_range: str
    ) -> float:
        """Calculate average logs per minute."""
        
        # Parse time range to minutes
        if "h" in time_range:
            total_minutes = int(time_range.replace("h", "")) * 60
        elif "m" in time_range:
            total_minutes = int(time_range.replace("m", ""))
        elif "d" in time_range:
            total_minutes = int(time_range.replace("d", "")) * 24 * 60
        else:
            total_minutes = 60  # default
        
        return len(logs) / max(1, total_minutes)
    
    def _extract_log_patterns(self, messages: List[str]) -> Dict[str, int]:
        """Extract patterns from log messages."""
        
        patterns = defaultdict(int)
        
        for message in messages:
            # Simple pattern extraction
            normalized = re.sub(r'\d+', 'N', message)
            normalized = re.sub(r'[a-f0-9]{8,}', 'ID', normalized)
            patterns[normalized] += 1
        
        return dict(patterns)
    
    def _create_incident_timeline(
        self,
        logs: List[Dict[str, Any]],
        incident_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Create timeline of events around incident."""
        
        timeline_events = []
        
        for log in logs:
            log_time = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
            time_delta = (log_time - incident_timestamp).total_seconds()
            
            if abs(time_delta) <= 1800:  # Within 30 minutes
                timeline_events.append({
                    "timestamp": log["timestamp"],
                    "relative_time_seconds": int(time_delta),
                    "severity": log["severity"],
                    "message": log["message"],
                    "component": log.get("labels", {}).get("component", "unknown")
                })
        
        # Sort by timestamp
        timeline_events.sort(key=lambda x: x["timestamp"])
        
        return timeline_events
    
    async def _get_total_log_count(self, service_name: str, time_range: str) -> int:
        """Get total log count for calculating error rates."""
        
        # Mock implementation - estimate based on service activity
        base_logs_per_minute = {
            "payment-api": 50,
            "billing-service": 30,
            "user-api": 40,
            "order-service": 35
        }.get(service_name, 25)
        
        # Parse time range to minutes
        if "h" in time_range:
            total_minutes = int(time_range.replace("h", "")) * 60
        elif "m" in time_range:
            total_minutes = int(time_range.replace("m", ""))
        else:
            total_minutes = 60
        
        return base_logs_per_minute * total_minutes
    
    def _random_timestamp_in_range(self, start_time: datetime, end_time: datetime) -> datetime:
        """Generate random timestamp within range."""
        
        time_diff = (end_time - start_time).total_seconds()
        random_seconds = random.uniform(0, time_diff)
        return start_time + timedelta(seconds=random_seconds)
    
    def _parse_time_range(self, time_range: str, end_time: datetime) -> datetime:
        """Parse time range string to start datetime."""
        
        time_range = time_range.lower().strip()
        
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
            return end_time - timedelta(hours=hours)
        elif time_range.endswith('m'):
            minutes = int(time_range[:-1])
            return end_time - timedelta(minutes=minutes)
        elif time_range.endswith('d'):
            days = int(time_range[:-1])
            return end_time - timedelta(days=days)
        elif time_range.endswith('s'):
            seconds = int(time_range[:-1])
            return end_time - timedelta(seconds=seconds)
        else:
            # Default to 1 hour
            return end_time - timedelta(hours=1)
    
    async def get_logs_dashboard_url(
        self,
        query: str,
        resource_labels: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate URL for GCP Logs Explorer dashboard."""
        
        base_url = "https://console.cloud.google.com/logs/query"
        params = [f"project={self.project_id}"]
        
        if query:
            # URL encode the query
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            params.append(f"query={encoded_query}")
        
        return f"{base_url}?{'&'.join(params)}"