"""
GCP Error Reporting integration with mock implementation for demo.

This module provides integration with GCP Error Reporting API, simulating real
error reporting API responses using synthetic error data for pattern analysis,
frequency tracking, and recurring error detection.
"""

import asyncio
import json
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from loguru import logger
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

try:
    from google.cloud import error_reporting
    from google.cloud.error_reporting_v1beta1 import ErrorEvent, ErrorGroup
    from google.oauth2 import service_account
    GCP_ERROR_REPORTING_AVAILABLE = True
except ImportError:
    GCP_ERROR_REPORTING_AVAILABLE = False
    logger.warning("Google Cloud Error Reporting client not available, using mock implementation")

from config.settings import get_settings

settings = get_settings()


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ErrorGroupInfo:
    """Information about an error group."""
    
    group_id: str
    name: str
    message: str
    service: str
    version: str
    first_seen: datetime
    last_seen: datetime
    count: int
    affected_users: int
    error_rate: float
    resolution_status: str
    tracking_issues: List[str]


@dataclass
class ErrorEventInfo:
    """Information about an error event."""
    
    event_time: datetime
    service_name: str
    service_version: str
    message: str
    user: str
    file_path: str
    line_number: int
    function_name: str
    stack_trace: List[str]
    http_request: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class GCPErrorReportingClient:
    """GCP Error Reporting integration client."""
    
    def __init__(self):
        """Initialize the GCP Error Reporting client."""
        self.project_id = settings.GCP_PROJECT_ID
        self.use_mock = settings.USE_MOCK_GCP_SERVICES or not GCP_ERROR_REPORTING_AVAILABLE
        self.client = None
        self.synthetic_data_path = Path(settings.SYNTHETIC_DATA_PATH)
        
        # Mock data cache
        self._mock_error_cache = []
        self._mock_error_groups = {}
        self._last_cache_update = None
        self._cache_ttl = timedelta(minutes=5)
        
        # Error patterns for different scenarios
        self.error_patterns = {
            "database_errors": [
                {
                    "message": "Connection pool exhausted: Unable to acquire connection within timeout",
                    "exception": "java.sql.SQLException",
                    "file": "src/main/java/com/app/database/ConnectionManager.java",
                    "function": "getConnection",
                    "stack_trace": [
                        "at com.app.database.ConnectionManager.getConnection(ConnectionManager.java:145)",
                        "at com.app.service.PaymentService.processPayment(PaymentService.java:89)",
                        "at com.app.controller.PaymentController.handlePayment(PaymentController.java:67)"
                    ]
                },
                {
                    "message": "Query execution timeout: Statement cancelled due to timeout",
                    "exception": "java.sql.SQLTimeoutException",
                    "file": "src/main/java/com/app/repository/UserRepository.java",
                    "function": "findUsersByCriteria",
                    "stack_trace": [
                        "at com.app.repository.UserRepository.findUsersByCriteria(UserRepository.java:234)",
                        "at com.app.service.UserService.searchUsers(UserService.java:156)",
                        "at com.app.controller.UserController.search(UserController.java:123)"
                    ]
                }
            ],
            "cache_errors": [
                {
                    "message": "Redis connection failed: Connection refused",
                    "exception": "redis.exceptions.ConnectionError",
                    "file": "src/main/python/cache/redis_client.py",
                    "function": "connect",
                    "stack_trace": [
                        "File \"/app/cache/redis_client.py\", line 78, in connect",
                        "File \"/app/service/cache_service.py\", line 45, in get_cached_data",
                        "File \"/app/controller/api_controller.py\", line 234, in handle_request"
                    ]
                },
                {
                    "message": "Cache operation timeout: SET operation timed out after 5000ms",
                    "exception": "redis.exceptions.TimeoutError",
                    "file": "src/main/python/cache/redis_client.py",
                    "function": "set_with_expiry",
                    "stack_trace": [
                        "File \"/app/cache/redis_client.py\", line 156, in set_with_expiry",
                        "File \"/app/service/session_service.py\", line 89, in store_session",
                        "File \"/app/controller/auth_controller.py\", line 167, in login"
                    ]
                }
            ],
            "api_errors": [
                {
                    "message": "HTTP 500: Internal server error in payment processing",
                    "exception": "com.app.exception.PaymentProcessingException",
                    "file": "src/main/java/com/app/service/PaymentService.java",
                    "function": "processPayment",
                    "stack_trace": [
                        "at com.app.service.PaymentService.processPayment(PaymentService.java:189)",
                        "at com.app.service.PaymentService.handlePaymentRequest(PaymentService.java:134)",
                        "at com.app.controller.PaymentController.processPayment(PaymentController.java:78)"
                    ]
                },
                {
                    "message": "Authentication token validation failed: JWT signature verification failed",
                    "exception": "com.auth0.jwt.exceptions.SignatureVerificationException",
                    "file": "src/main/java/com/app/security/JwtValidator.java",
                    "function": "validateToken",
                    "stack_trace": [
                        "at com.app.security.JwtValidator.validateToken(JwtValidator.java:67)",
                        "at com.app.filter.AuthenticationFilter.doFilter(AuthenticationFilter.java:89)",
                        "at com.app.controller.BaseController.authenticate(BaseController.java:45)"
                    ]
                }
            ],
            "memory_errors": [
                {
                    "message": "OutOfMemoryError: Java heap space",
                    "exception": "java.lang.OutOfMemoryError",
                    "file": "src/main/java/com/app/service/DataProcessingService.java",
                    "function": "processLargeDataset",
                    "stack_trace": [
                        "at com.app.service.DataProcessingService.processLargeDataset(DataProcessingService.java:234)",
                        "at com.app.service.DataProcessingService.handleBatchJob(DataProcessingService.java:156)",
                        "at com.app.controller.JobController.executeBatch(JobController.java:89)"
                    ]
                },
                {
                    "message": "Memory allocation failed: Cannot allocate memory for buffer",
                    "exception": "std::bad_alloc",
                    "file": "src/cpp/memory/buffer_manager.cpp",
                    "function": "allocate_buffer",
                    "stack_trace": [
                        "at buffer_manager.cpp:178 in allocate_buffer()",
                        "at image_processor.cpp:234 in process_image()",
                        "at api_handler.cpp:167 in handle_upload()"
                    ]
                }
            ],
            "network_errors": [
                {
                    "message": "Network timeout: Connection to external service timed out",
                    "exception": "java.net.SocketTimeoutException",
                    "file": "src/main/java/com/app/client/ExternalServiceClient.java",
                    "function": "makeRequest",
                    "stack_trace": [
                        "at com.app.client.ExternalServiceClient.makeRequest(ExternalServiceClient.java:123)",
                        "at com.app.service.IntegrationService.callExternalAPI(IntegrationService.java:78)",
                        "at com.app.controller.WebhookController.handleWebhook(WebhookController.java:45)"
                    ]
                }
            ]
        }
        
        # Service configurations
        self.service_configs = {
            "payment-api": {
                "language": "java",
                "version": "1.2.3",
                "error_rate_baseline": 0.2,
                "typical_errors": ["database_errors", "api_errors"]
            },
            "billing-service": {
                "language": "java", 
                "version": "1.1.8",
                "error_rate_baseline": 0.15,
                "typical_errors": ["database_errors", "memory_errors"]
            },
            "user-api": {
                "language": "python",
                "version": "2.4.1",
                "error_rate_baseline": 0.18,
                "typical_errors": ["cache_errors", "api_errors"]
            },
            "cache-service": {
                "language": "python",
                "version": "1.3.2",
                "error_rate_baseline": 0.25,
                "typical_errors": ["cache_errors", "network_errors"]
            }
        }
        
        if not self.use_mock:
            self._initialize_real_client()
        else:
            logger.info("Using mock GCP Error Reporting implementation for demo")
    
    def _initialize_real_client(self) -> None:
        """Initialize real GCP Error Reporting client."""
        try:
            if settings.GCP_SERVICE_ACCOUNT_KEY_PATH:
                credentials = service_account.Credentials.from_service_account_file(
                    settings.GCP_SERVICE_ACCOUNT_KEY_PATH
                )
                self.client = error_reporting.Client(
                    project=self.project_id,
                    credentials=credentials
                )
            else:
                # Use default credentials
                self.client = error_reporting.Client(project=self.project_id)
            
            logger.info(f"Initialized GCP Error Reporting client for project: {self.project_id}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize real GCP Error Reporting client: {e}")
            logger.info("Falling back to mock implementation")
            self.use_mock = True
            self.client = None
    
    async def get_error_groups(
        self,
        service_name: Optional[str] = None,
        time_range: str = "24h",
        order_by: str = "count_desc"
    ) -> List[ErrorGroupInfo]:
        """
        Get error groups with filtering and sorting options.
        
        Args:
            service_name: Filter by service name
            time_range: Time range for analysis
            order_by: Sorting order (count_desc, first_seen_desc, last_seen_desc)
            
        Returns:
            List of ErrorGroupInfo objects
        """
        logger.info(f"Getting error groups for service: {service_name or 'all'}")
        
        try:
            if self.use_mock:
                return await self._get_error_groups_mock(service_name, time_range, order_by)
            else:
                return await self._get_error_groups_real(service_name, time_range, order_by)
                
        except Exception as e:
            logger.error(f"Error getting error groups: {e}")
            return []
    
    async def get_error_events(
        self,
        group_id: Optional[str] = None,
        service_name: Optional[str] = None,
        time_range: str = "1h",
        limit: int = 100
    ) -> List[ErrorEventInfo]:
        """
        Get error events for a specific group or service.
        
        Args:
            group_id: Specific error group ID
            service_name: Filter by service name
            time_range: Time range for events
            limit: Maximum number of events to return
            
        Returns:
            List of ErrorEventInfo objects
        """
        logger.info(f"Getting error events for group: {group_id} service: {service_name}")
        
        try:
            if self.use_mock:
                return await self._get_error_events_mock(group_id, service_name, time_range, limit)
            else:
                return await self._get_error_events_real(group_id, service_name, time_range, limit)
                
        except Exception as e:
            logger.error(f"Error getting error events: {e}")
            return []
    
    async def analyze_error_patterns(
        self,
        service_name: str,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """
        Analyze error patterns for a service.
        
        Args:
            service_name: Service to analyze
            time_range: Time range for analysis
            
        Returns:
            Dictionary containing pattern analysis results
        """
        logger.info(f"Analyzing error patterns for service: {service_name}")
        
        try:
            # Get error groups for the service
            error_groups = await self.get_error_groups(service_name, time_range)
            
            if not error_groups:
                return {
                    "service_name": service_name,
                    "time_range": time_range,
                    "total_error_groups": 0,
                    "total_error_events": 0,
                    "error_patterns": [],
                    "trending_errors": [],
                    "severity_distribution": {},
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            
            # Analyze patterns
            patterns = await self._analyze_error_group_patterns(error_groups)
            
            # Find trending errors
            trending_errors = await self._find_trending_errors(error_groups, time_range)
            
            # Calculate severity distribution
            severity_distribution = self._calculate_severity_distribution(error_groups)
            
            # Calculate impact metrics
            impact_metrics = self._calculate_error_impact(error_groups)
            
            # Generate recommendations
            recommendations = self._generate_error_recommendations(error_groups, patterns)
            
            total_events = sum(group.count for group in error_groups)
            
            return {
                "service_name": service_name,
                "time_range": time_range,
                "total_error_groups": len(error_groups),
                "total_error_events": total_events,
                "error_rate": self._calculate_error_rate(service_name, total_events, time_range),
                "error_patterns": patterns,
                "trending_errors": trending_errors,
                "severity_distribution": severity_distribution,
                "impact_metrics": impact_metrics,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing error patterns: {e}")
            return {
                "service_name": service_name,
                "error": str(e),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_error_statistics(
        self,
        service_name: Optional[str] = None,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get comprehensive error statistics.
        
        Args:
            service_name: Optional service name filter
            time_range: Time range for statistics
            
        Returns:
            Dictionary containing error statistics
        """
        logger.info(f"Getting error statistics for service: {service_name or 'all'}")
        
        try:
            # Get error groups
            error_groups = await self.get_error_groups(service_name, time_range)
            
            if not error_groups:
                return {
                    "service_name": service_name,
                    "time_range": time_range,
                    "statistics": {
                        "total_error_groups": 0,
                        "total_error_events": 0,
                        "error_rate": 0.0,
                        "affected_users": 0,
                        "top_errors": [],
                        "error_frequency": {},
                        "resolution_status": {}
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Calculate statistics
            total_events = sum(group.count for group in error_groups)
            total_affected_users = sum(group.affected_users for group in error_groups)
            
            # Top errors by frequency
            top_errors = sorted(error_groups, key=lambda g: g.count, reverse=True)[:10]
            top_errors_data = [
                {
                    "group_id": group.group_id,
                    "name": group.name,
                    "count": group.count,
                    "affected_users": group.affected_users,
                    "error_rate": group.error_rate
                }
                for group in top_errors
            ]
            
            # Error frequency over time
            error_frequency = await self._calculate_error_frequency_timeline(error_groups, time_range)
            
            # Resolution status distribution
            resolution_status = Counter(group.resolution_status for group in error_groups)
            
            # Service breakdown if no specific service
            service_breakdown = {}
            if not service_name:
                service_errors = defaultdict(int)
                for group in error_groups:
                    service_errors[group.service] += group.count
                service_breakdown = dict(service_errors)
            
            statistics = {
                "total_error_groups": len(error_groups),
                "total_error_events": total_events,
                "error_rate": self._calculate_error_rate(service_name, total_events, time_range),
                "affected_users": total_affected_users,
                "unique_affected_users": len(set(group.affected_users for group in error_groups)),
                "top_errors": top_errors_data,
                "error_frequency": error_frequency,
                "resolution_status": dict(resolution_status),
                "service_breakdown": service_breakdown
            }
            
            return {
                "service_name": service_name,
                "time_range": time_range,
                "statistics": statistics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting error statistics: {e}")
            return {
                "service_name": service_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def detect_recurring_errors(
        self,
        service_name: str,
        lookback_days: int = 7,
        min_occurrences: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Detect recurring error patterns.
        
        Args:
            service_name: Service to analyze
            lookback_days: Number of days to look back
            min_occurrences: Minimum occurrences to consider recurring
            
        Returns:
            List of recurring error patterns
        """
        logger.info(f"Detecting recurring errors for {service_name} over {lookback_days} days")
        
        try:
            # Get error groups for extended time range
            time_range = f"{lookback_days}d"
            error_groups = await self.get_error_groups(service_name, time_range)
            
            # Find recurring patterns
            recurring_errors = []
            
            for group in error_groups:
                if group.count >= min_occurrences:
                    # Calculate recurrence metrics
                    days_active = (group.last_seen - group.first_seen).days + 1
                    avg_occurrences_per_day = group.count / max(1, days_active)
                    
                    # Determine if error is truly recurring (not just a burst)
                    is_recurring = days_active > 1 and avg_occurrences_per_day >= 1
                    
                    if is_recurring:
                        # Calculate trend
                        trend = await self._calculate_error_trend(group, lookback_days)
                        
                        # Determine severity
                        severity = self._determine_error_severity(group, avg_occurrences_per_day)
                        
                        recurring_errors.append({
                            "group_id": group.group_id,
                            "name": group.name,
                            "message": group.message,
                            "total_occurrences": group.count,
                            "days_active": days_active,
                            "avg_occurrences_per_day": round(avg_occurrences_per_day, 2),
                            "affected_users": group.affected_users,
                            "first_seen": group.first_seen.isoformat(),
                            "last_seen": group.last_seen.isoformat(),
                            "trend": trend,
                            "severity": severity,
                            "resolution_status": group.resolution_status,
                            "recurrence_score": self._calculate_recurrence_score(
                                group.count, days_active, avg_occurrences_per_day
                            )
                        })
            
            # Sort by recurrence score
            recurring_errors.sort(key=lambda x: x["recurrence_score"], reverse=True)
            
            return recurring_errors
            
        except Exception as e:
            logger.error(f"Error detecting recurring errors: {e}")
            return []
    
    async def correlate_errors_with_incident(
        self,
        incident_timestamp: datetime,
        service_name: str,
        time_window_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Correlate errors with an incident timeframe.
        
        Args:
            incident_timestamp: When the incident occurred
            service_name: Service involved in the incident
            time_window_minutes: Time window around incident to analyze
            
        Returns:
            Dictionary containing correlated error analysis
        """
        logger.info(f"Correlating errors with incident for {service_name} at {incident_timestamp}")
        
        try:
            # Get error events around incident time
            incident_errors = await self._get_errors_in_timeframe(
                incident_timestamp,
                service_name,
                time_window_minutes
            )
            
            if not incident_errors:
                return {
                    "incident_timestamp": incident_timestamp.isoformat(),
                    "service_name": service_name,
                    "correlated_errors": 0,
                    "error_spike_detected": False,
                    "correlation_strength": 0.0,
                    "findings": [],
                    "error_timeline": []
                }
            
            # Analyze error patterns around incident
            correlation_findings = []
            correlation_strength = 0.0
            
            # Check for error spikes
            pre_incident_count = len([
                e for e in incident_errors 
                if e.event_time < incident_timestamp
            ])
            
            during_incident_count = len([
                e for e in incident_errors
                if abs((e.event_time - incident_timestamp).total_seconds()) <= 300  # 5 minutes
            ])
            
            spike_detected = during_incident_count > pre_incident_count * 2
            
            if spike_detected:
                correlation_strength += 0.4
                correlation_findings.append({
                    "type": "error_spike",
                    "description": f"Error spike detected: {during_incident_count} errors during incident vs {pre_incident_count} before",
                    "confidence": 0.8
                })
            
            # Check for new error types
            pre_incident_messages = set(
                e.message for e in incident_errors 
                if e.event_time < incident_timestamp - timedelta(minutes=10)
            )
            
            during_incident_messages = set(
                e.message for e in incident_errors
                if abs((e.event_time - incident_timestamp).total_seconds()) <= 300
            )
            
            new_error_types = during_incident_messages - pre_incident_messages
            
            if new_error_types:
                correlation_strength += 0.3
                correlation_findings.append({
                    "type": "new_error_types",
                    "description": f"New error types appeared during incident: {len(new_error_types)} unique errors",
                    "confidence": 0.7,
                    "new_errors": list(new_error_types)[:3]
                })
            
            # Check for error clustering
            error_timeline = self._create_error_timeline(incident_errors, incident_timestamp)
            
            # Find error clusters (multiple errors in short time spans)
            clusters = self._find_error_clusters(incident_errors, incident_timestamp)
            
            if clusters:
                correlation_strength += 0.2
                correlation_findings.append({
                    "type": "error_clustering",
                    "description": f"Found {len(clusters)} error clusters around incident time",
                    "confidence": 0.6,
                    "clusters": clusters
                })
            
            return {
                "incident_timestamp": incident_timestamp.isoformat(),
                "service_name": service_name,
                "time_window_minutes": time_window_minutes,
                "correlated_errors": len(incident_errors),
                "error_spike_detected": spike_detected,
                "correlation_strength": min(1.0, correlation_strength),
                "findings": correlation_findings,
                "error_timeline": error_timeline,
                "pre_incident_errors": pre_incident_count,
                "during_incident_errors": during_incident_count,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error correlating errors with incident: {e}")
            return {
                "incident_timestamp": incident_timestamp.isoformat(),
                "error": str(e),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_error_groups_mock(
        self,
        service_name: Optional[str],
        time_range: str,
        order_by: str
    ) -> List[ErrorGroupInfo]:
        """Mock implementation of getting error groups."""
        
        # Load synthetic data
        await self._load_synthetic_errors()
        
        # Generate error groups based on service and scenario
        error_groups = []
        
        services = [service_name] if service_name else list(self.service_configs.keys())
        
        for service in services:
            config = self.service_configs.get(service, self.service_configs["payment-api"])
            typical_errors = config.get("typical_errors", ["api_errors"])
            
            # Generate error groups for this service
            for error_category in typical_errors:
                patterns = self.error_patterns.get(error_category, [])
                
                for i, pattern in enumerate(patterns):
                    # Generate realistic error group data
                    group_id = f"group_{service}_{error_category}_{i}"
                    
                    # Calculate realistic counts and timings
                    base_count = random.randint(5, 50)
                    if service in ["payment-api", "billing-service"]:
                        base_count *= 2  # Higher error rates for incident scenarios
                    
                    last_seen = datetime.utcnow() - timedelta(minutes=random.randint(1, 120))
                    first_seen = last_seen - timedelta(days=random.randint(1, 30))
                    
                    # Calculate affected users (typically less than error count)
                    affected_users = max(1, int(base_count * random.uniform(0.3, 0.8)))
                    
                    # Calculate error rate
                    error_rate = base_count / max(1, (last_seen - first_seen).total_seconds() / 3600)
                    
                    error_group = ErrorGroupInfo(
                        group_id=group_id,
                        name=f"{service}: {pattern['exception']}",
                        message=pattern["message"],
                        service=service,
                        version=config.get("version", "1.0.0"),
                        first_seen=first_seen,
                        last_seen=last_seen,
                        count=base_count,
                        affected_users=affected_users,
                        error_rate=round(error_rate, 2),
                        resolution_status=random.choice(["OPEN", "ACKNOWLEDGED", "RESOLVED", "MUTED"]),
                        tracking_issues=[]
                    )
                    
                    error_groups.append(error_group)
        
        # Apply sorting
        if order_by == "count_desc":
            error_groups.sort(key=lambda g: g.count, reverse=True)
        elif order_by == "first_seen_desc":
            error_groups.sort(key=lambda g: g.first_seen, reverse=True)
        elif order_by == "last_seen_desc":
            error_groups.sort(key=lambda g: g.last_seen, reverse=True)
        
        return error_groups[:20]  # Limit to 20 groups for demo
    
    async def _get_error_groups_real(
        self,
        service_name: Optional[str],
        time_range: str,
        order_by: str
    ) -> List[ErrorGroupInfo]:
        """Real implementation of getting error groups using GCP Error Reporting API."""
        
        if not self.client:
            raise Exception("GCP Error Reporting client not initialized")
        
        # Note: Real implementation would use the GCP Error Reporting API
        # This is a simplified mock for demonstration
        return []
    
    async def _get_error_events_mock(
        self,
        group_id: Optional[str],
        service_name: Optional[str],
        time_range: str,
        limit: int
    ) -> List[ErrorEventInfo]:
        """Mock implementation of getting error events."""
        
        error_events = []
        
        # Determine which error patterns to use
        if group_id:
            # Extract service and category from group_id
            parts = group_id.split("_")
            if len(parts) >= 3:
                service = parts[1]
                category = parts[2]
            else:
                service = service_name or "payment-api"
                category = "api_errors"
        else:
            service = service_name or "payment-api"
            category = random.choice(["database_errors", "api_errors", "cache_errors"])
        
        patterns = self.error_patterns.get(category, self.error_patterns["api_errors"])
        
        # Generate error events
        end_time = datetime.utcnow()
        start_time = self._parse_time_range(time_range, end_time)
        
        num_events = min(limit, random.randint(5, 25))
        
        for i in range(num_events):
            pattern = random.choice(patterns)
            
            # Generate random timestamp in range
            event_time = self._random_timestamp_in_range(start_time, end_time)
            
            # Generate user identifier
            user = f"user_{random.randint(1000, 9999)}"
            
            # Extract file info from pattern
            file_path = pattern.get("file", "unknown.java")
            line_number = random.randint(50, 500)
            function_name = pattern.get("function", "unknown")
            
            # Generate HTTP request data for API errors
            http_request = None
            if "api" in category or "HTTP" in pattern["message"]:
                http_request = {
                    "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                    "url": f"/api/v1/{service.replace('-', '/')}/endpoint",
                    "status": random.choice([500, 503, 400, 401]),
                    "response_size": random.randint(100, 5000),
                    "user_agent": "Mozilla/5.0 (compatible; APIClient/1.0)"
                }
            
            # Generate context
            context = {
                "service_version": self.service_configs.get(service, {}).get("version", "1.0.0"),
                "environment": "production",
                "region": "us-central1",
                "instance_id": f"instance-{random.randint(1, 10)}"
            }
            
            error_event = ErrorEventInfo(
                event_time=event_time,
                service_name=service,
                service_version=context["service_version"],
                message=pattern["message"],
                user=user,
                file_path=file_path,
                line_number=line_number,
                function_name=function_name,
                stack_trace=pattern.get("stack_trace", []),
                http_request=http_request,
                context=context
            )
            
            error_events.append(error_event)
        
        # Sort by event time (newest first)
        error_events.sort(key=lambda e: e.event_time, reverse=True)
        
        return error_events
    
    async def _get_error_events_real(
        self,
        group_id: Optional[str],
        service_name: Optional[str],
        time_range: str,
        limit: int
    ) -> List[ErrorEventInfo]:
        """Real implementation of getting error events using GCP Error Reporting API."""
        
        if not self.client:
            raise Exception("GCP Error Reporting client not initialized")
        
        # Note: Real implementation would use the GCP Error Reporting API
        return []
    
    async def _get_errors_in_timeframe(
        self,
        incident_timestamp: datetime,
        service_name: str,
        time_window_minutes: int
    ) -> List[ErrorEventInfo]:
        """Get errors within a specific timeframe around an incident."""
        
        start_time = incident_timestamp - timedelta(minutes=time_window_minutes)
        end_time = incident_timestamp + timedelta(minutes=time_window_minutes)
        
        # Get error events for the timeframe
        time_range_str = f"{time_window_minutes * 2}m"  # Total window
        all_events = await self._get_error_events_mock(
            group_id=None,
            service_name=service_name,
            time_range=time_range_str,
            limit=100
        )
        
        # Filter to the specific timeframe
        timeframe_events = [
            event for event in all_events
            if start_time <= event.event_time <= end_time
        ]
        
        return timeframe_events
    
    async def _load_synthetic_errors(self) -> None:
        """Load synthetic error data for mock responses."""
        
        # Check if cache needs refresh
        if (self._last_cache_update and 
            datetime.utcnow() - self._last_cache_update < self._cache_ttl and
            self._mock_error_cache):
            return
        
        try:
            # Load GCP data file
            gcp_data_file = self.synthetic_data_path / "scenario_1_gcp_data.json"
            if gcp_data_file.exists():
                with open(gcp_data_file, 'r') as f:
                    content = f.read(15000)  # Read first 15KB
                    try:
                        gcp_data = json.loads(content)
                        if "error_reporting_data" in gcp_data:
                            self._mock_error_cache = gcp_data["error_reporting_data"]
                    except json.JSONDecodeError:
                        # Parse line by line for partial data
                        lines = content.split('\n')
                        for line in lines[:30]:  # First 30 lines
                            if line.strip() and '"exception"' in line:
                                try:
                                    error_obj = json.loads(line)
                                    self._mock_error_cache.append(error_obj)
                                except:
                                    continue
            
            self._last_cache_update = datetime.utcnow()
            logger.debug(f"Loaded {len(self._mock_error_cache)} synthetic error entries")
            
        except Exception as e:
            logger.warning(f"Error loading synthetic error data: {e}")
            self._mock_error_cache = []
    
    async def _analyze_error_group_patterns(
        self,
        error_groups: List[ErrorGroupInfo]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in error groups."""
        
        patterns = []
        
        # Group by exception type
        exception_groups = defaultdict(list)
        for group in error_groups:
            exception_type = self._extract_exception_type(group.name)
            exception_groups[exception_type].append(group)
        
        for exception_type, groups in exception_groups.items():
            if len(groups) >= 2:  # Pattern needs at least 2 occurrences
                total_count = sum(g.count for g in groups)
                total_users = sum(g.affected_users for g in groups)
                
                patterns.append({
                    "pattern_type": "exception_clustering",
                    "exception_type": exception_type,
                    "occurrences": len(groups),
                    "total_errors": total_count,
                    "affected_users": total_users,
                    "services": list(set(g.service for g in groups)),
                    "severity": "high" if total_count > 50 else "medium",
                    "description": f"Multiple error groups with {exception_type} affecting {len(set(g.service for g in groups))} services"
                })
        
        # Look for time-based patterns
        recent_groups = [g for g in error_groups if (datetime.utcnow() - g.last_seen).total_seconds() < 3600]
        if len(recent_groups) > len(error_groups) * 0.5:
            patterns.append({
                "pattern_type": "recent_error_spike",
                "occurrences": len(recent_groups),
                "total_errors": sum(g.count for g in recent_groups),
                "time_window": "1 hour",
                "severity": "high",
                "description": f"High concentration of recent errors: {len(recent_groups)} groups active in last hour"
            })
        
        return patterns
    
    async def _find_trending_errors(
        self,
        error_groups: List[ErrorGroupInfo],
        time_range: str
    ) -> List[Dict[str, Any]]:
        """Find trending errors based on frequency and recency."""
        
        trending = []
        
        for group in error_groups:
            # Calculate trend factors
            recency_score = self._calculate_recency_score(group.last_seen)
            frequency_score = min(1.0, group.count / 100)  # Normalize to max of 100
            user_impact_score = min(1.0, group.affected_users / 50)  # Normalize to max of 50
            
            # Combined trending score
            trending_score = (recency_score * 0.4) + (frequency_score * 0.4) + (user_impact_score * 0.2)
            
            if trending_score > 0.6:  # Threshold for trending
                trending.append({
                    "group_id": group.group_id,
                    "name": group.name,
                    "message": group.message,
                    "trending_score": round(trending_score, 2),
                    "count": group.count,
                    "affected_users": group.affected_users,
                    "last_seen": group.last_seen.isoformat(),
                    "error_rate": group.error_rate,
                    "trend_factors": {
                        "recency": round(recency_score, 2),
                        "frequency": round(frequency_score, 2),
                        "user_impact": round(user_impact_score, 2)
                    }
                })
        
        # Sort by trending score
        trending.sort(key=lambda x: x["trending_score"], reverse=True)
        
        return trending[:10]  # Top 10 trending errors
    
    def _calculate_severity_distribution(
        self,
        error_groups: List[ErrorGroupInfo]
    ) -> Dict[str, int]:
        """Calculate severity distribution of error groups."""
        
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for group in error_groups:
            if group.count > 100 or group.affected_users > 50:
                severity_counts["CRITICAL"] += 1
            elif group.count > 50 or group.affected_users > 20:
                severity_counts["HIGH"] += 1
            elif group.count > 10 or group.affected_users > 5:
                severity_counts["MEDIUM"] += 1
            else:
                severity_counts["LOW"] += 1
        
        return severity_counts
    
    def _calculate_error_impact(
        self,
        error_groups: List[ErrorGroupInfo]
    ) -> Dict[str, Any]:
        """Calculate overall error impact metrics."""
        
        total_errors = sum(g.count for g in error_groups)
        total_users = sum(g.affected_users for g in error_groups)
        unique_users = len(set(g.affected_users for g in error_groups))
        
        # Find most impactful errors
        high_impact_errors = [
            g for g in error_groups 
            if g.count > 20 or g.affected_users > 10
        ]
        
        # Calculate time-based impact
        recent_errors = [
            g for g in error_groups
            if (datetime.utcnow() - g.last_seen).total_seconds() < 3600
        ]
        
        return {
            "total_error_events": total_errors,
            "total_affected_users": total_users,
            "unique_affected_users": unique_users,
            "high_impact_error_groups": len(high_impact_errors),
            "recent_error_groups": len(recent_errors),
            "avg_errors_per_group": round(total_errors / max(1, len(error_groups)), 2),
            "avg_users_per_group": round(total_users / max(1, len(error_groups)), 2)
        }
    
    def _generate_error_recommendations(
        self,
        error_groups: List[ErrorGroupInfo],
        patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on error analysis."""
        
        recommendations = []
        
        # Check for database connection issues
        db_errors = [g for g in error_groups if "connection" in g.message.lower() or "sql" in g.message.lower()]
        if db_errors:
            recommendations.append("Investigate database connection pool configuration and query performance")
        
        # Check for memory issues
        memory_errors = [g for g in error_groups if "memory" in g.message.lower() or "heap" in g.message.lower()]
        if memory_errors:
            recommendations.append("Review memory allocation and garbage collection settings")
        
        # Check for high error rates
        high_rate_errors = [g for g in error_groups if g.error_rate > 10]
        if high_rate_errors:
            recommendations.append("Implement circuit breakers for services with high error rates")
        
        # Check for unresolved errors
        unresolved_errors = [g for g in error_groups if g.resolution_status == "OPEN"]
        if len(unresolved_errors) > len(error_groups) * 0.5:
            recommendations.append("Prioritize resolution of open error groups to improve service reliability")
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern["pattern_type"] == "exception_clustering":
                recommendations.append(f"Address recurring {pattern['exception_type']} exceptions across multiple services")
            elif pattern["pattern_type"] == "recent_error_spike":
                recommendations.append("Investigate recent changes that may have caused error spike")
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def _calculate_error_frequency_timeline(
        self,
        error_groups: List[ErrorGroupInfo],
        time_range: str
    ) -> Dict[str, Any]:
        """Calculate error frequency over time."""
        
        # Parse time range
        if "h" in time_range:
            hours = int(time_range.replace("h", ""))
            bucket_size = max(1, hours // 12)  # 12 buckets max
        elif "d" in time_range:
            days = int(time_range.replace("d", ""))
            bucket_size = max(1, days)  # Daily buckets
            hours = days * 24
        else:
            hours = 24
            bucket_size = 2  # 2-hour buckets
        
        # Create time buckets
        buckets = []
        end_time = datetime.utcnow()
        
        for i in range(12):  # 12 time points
            bucket_time = end_time - timedelta(hours=bucket_size * i)
            bucket_errors = 0
            
            # Count errors in this time bucket
            for group in error_groups:
                if abs((group.last_seen - bucket_time).total_seconds()) < bucket_size * 3600:
                    bucket_errors += group.count // 10  # Distribute errors across time
            
            buckets.append({
                "timestamp": bucket_time.isoformat(),
                "error_count": bucket_errors
            })
        
        buckets.reverse()  # Chronological order
        
        return {
            "time_buckets": buckets,
            "bucket_size_hours": bucket_size,
            "total_buckets": len(buckets)
        }
    
    async def _calculate_error_trend(
        self,
        group: ErrorGroupInfo,
        lookback_days: int
    ) -> str:
        """Calculate error trend for a group."""
        
        # Simulate trend calculation
        days_active = (group.last_seen - group.first_seen).days + 1
        avg_per_day = group.count / max(1, days_active)
        
        # Simple trend logic
        if days_active <= 2:
            return "new"
        elif avg_per_day > 5:
            return "increasing"
        elif avg_per_day < 1:
            return "decreasing"
        else:
            return "stable"
    
    def _determine_error_severity(
        self,
        group: ErrorGroupInfo,
        avg_occurrences_per_day: float
    ) -> ErrorSeverity:
        """Determine error severity based on metrics."""
        
        if group.affected_users > 50 or avg_occurrences_per_day > 20:
            return ErrorSeverity.CRITICAL
        elif group.affected_users > 20 or avg_occurrences_per_day > 10:
            return ErrorSeverity.HIGH
        elif group.affected_users > 5 or avg_occurrences_per_day > 2:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _calculate_recurrence_score(
        self,
        total_count: int,
        days_active: int,
        avg_per_day: float
    ) -> float:
        """Calculate recurrence score for ranking."""
        
        frequency_score = min(1.0, avg_per_day / 10)  # Normalize to max 10/day
        persistence_score = min(1.0, days_active / 30)  # Normalize to max 30 days
        volume_score = min(1.0, total_count / 100)  # Normalize to max 100 total
        
        return (frequency_score * 0.4) + (persistence_score * 0.3) + (volume_score * 0.3)
    
    def _calculate_error_rate(
        self,
        service_name: Optional[str],
        total_errors: int,
        time_range: str
    ) -> float:
        """Calculate error rate percentage."""
        
        # Estimate total requests for error rate calculation
        if service_name and service_name in self.service_configs:
            baseline = self.service_configs[service_name]["error_rate_baseline"]
        else:
            baseline = 0.2  # Default baseline
        
        # Parse time range to get estimated request volume
        if "h" in time_range:
            hours = int(time_range.replace("h", ""))
            estimated_requests = hours * 1000  # 1000 requests/hour estimate
        elif "d" in time_range:
            days = int(time_range.replace("d", ""))
            estimated_requests = days * 24 * 1000
        else:
            estimated_requests = 1000
        
        error_rate = (total_errors / max(1, estimated_requests)) * 100
        return round(error_rate, 3)
    
    def _calculate_recency_score(self, last_seen: datetime) -> float:
        """Calculate recency score (higher for more recent)."""
        
        hours_ago = (datetime.utcnow() - last_seen).total_seconds() / 3600
        
        if hours_ago < 1:
            return 1.0
        elif hours_ago < 6:
            return 0.8
        elif hours_ago < 24:
            return 0.6
        elif hours_ago < 168:  # 1 week
            return 0.4
        else:
            return 0.2
    
    def _extract_exception_type(self, error_name: str) -> str:
        """Extract exception type from error name."""
        
        # Common exception patterns
        if "SQLException" in error_name:
            return "SQL Exception"
        elif "ConnectionError" in error_name:
            return "Connection Error"
        elif "TimeoutException" in error_name or "timeout" in error_name.lower():
            return "Timeout Exception"
        elif "OutOfMemoryError" in error_name:
            return "Memory Error"
        elif "NullPointerException" in error_name:
            return "Null Pointer Exception"
        elif "IllegalArgumentException" in error_name:
            return "Illegal Argument Exception"
        else:
            return "General Exception"
    
    def _create_error_timeline(
        self,
        error_events: List[ErrorEventInfo],
        incident_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Create timeline of error events around incident."""
        
        timeline = []
        
        for event in error_events:
            time_delta = (event.event_time - incident_timestamp).total_seconds()
            
            timeline.append({
                "timestamp": event.event_time.isoformat(),
                "relative_time_seconds": int(time_delta),
                "message": event.message,
                "service": event.service_name,
                "user": event.user,
                "file": event.file_path,
                "line": event.line_number
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline
    
    def _find_error_clusters(
        self,
        error_events: List[ErrorEventInfo],
        incident_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Find clusters of errors in time."""
        
        clusters = []
        
        # Group errors by 5-minute windows
        time_buckets = defaultdict(list)
        
        for event in error_events:
            # Round to 5-minute buckets
            bucket_time = event.event_time.replace(
                minute=(event.event_time.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            time_buckets[bucket_time].append(event)
        
        # Find clusters (buckets with >3 errors)
        for bucket_time, events in time_buckets.items():
            if len(events) > 3:
                time_delta = (bucket_time - incident_timestamp).total_seconds()
                
                clusters.append({
                    "timestamp": bucket_time.isoformat(),
                    "relative_time_seconds": int(time_delta),
                    "error_count": len(events),
                    "unique_errors": len(set(e.message for e in events)),
                    "affected_users": len(set(e.user for e in events))
                })
        
        return clusters
    
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
    
    async def get_error_reporting_dashboard_url(
        self,
        service_name: Optional[str] = None,
        time_range: str = "24h"
    ) -> str:
        """Generate URL for GCP Error Reporting dashboard."""
        
        base_url = "https://console.cloud.google.com/errors"
        params = [f"project={self.project_id}"]
        
        if service_name:
            params.append(f"service={service_name}")
        
        if time_range:
            params.append(f"time={time_range}")
        
        return f"{base_url}?{'&'.join(params)}"
    
    async def create_error_report(
        self,
        message: str,
        service_name: str,
        service_version: str,
        user: str,
        file_path: str,
        line_number: int,
        function_name: str,
        stack_trace: List[str],
        http_request: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new error report.
        
        Args:
            message: Error message
            service_name: Name of the service
            service_version: Version of the service
            user: User identifier
            file_path: Source file path
            line_number: Line number where error occurred
            function_name: Function name where error occurred
            stack_trace: Stack trace lines
            http_request: HTTP request information
            context: Additional context
            
        Returns:
            True if report was created successfully
        """
        logger.info(f"Creating error report for {service_name}: {message}")
        
        try:
            if self.use_mock:
                # Mock implementation - just log the error
                logger.info(f"Mock error report created: {message}")
                return True
            else:
                # Real implementation would use GCP Error Reporting API
                if not self.client:
                    raise Exception("GCP Error Reporting client not initialized")
                
                # Note: Real implementation would create error report via API
                return True
                
        except Exception as e:
            logger.error(f"Error creating error report: {e}")
            return False