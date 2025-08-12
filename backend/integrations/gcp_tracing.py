"""
GCP Cloud Tracing integration client with mock implementation for demo purposes.

This module provides distributed tracing analysis capabilities for the AI SRE Agent,
allowing analysis of request flows and performance bottlenecks across microservices.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Try to import GCP Cloud Trace client
try:
    from google.cloud import trace_v2
    GCP_TRACING_AVAILABLE = True
except ImportError:
    logger.warning("Google Cloud Tracing client not available, using mock implementation")
    GCP_TRACING_AVAILABLE = False


class GCPTracingData:
    """Structured representation of GCP tracing data."""
    
    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str] = None,
        operation_name: str = "",
        service_name: str = "",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_ms: float = 0.0,
        status_code: int = 200,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.service_name = service_name
        self.start_time = start_time or datetime.utcnow()
        self.end_time = end_time or datetime.utcnow()
        self.duration_ms = duration_ms
        self.status_code = status_code
        self.labels = labels or {}
        self.annotations = annotations or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'service_name': self.service_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'status_code': self.status_code,
            'labels': self.labels,
            'annotations': self.annotations
        }


class MockGCPTracingClient:
    """Mock implementation of GCP Cloud Tracing client for demo purposes."""
    
    def __init__(self):
        self.project_id = settings.GCP_PROJECT_ID or "demo-project"
        self.traces = self._generate_mock_traces()
        logger.info("Using mock GCP Tracing implementation for demo")
    
    def _generate_mock_traces(self) -> List[GCPTracingData]:
        """Generate realistic mock tracing data for demo scenarios."""
        traces = []
        
        # Generate traces for payment API scenario
        payment_trace_id = str(uuid4())
        
        # Root span - API Gateway
        traces.append(GCPTracingData(
            trace_id=payment_trace_id,
            span_id=str(uuid4()),
            operation_name="POST /api/v1/payments",
            service_name="api-gateway",
            start_time=datetime.utcnow() - timedelta(minutes=5),
            end_time=datetime.utcnow() - timedelta(minutes=4, seconds=58),
            duration_ms=2000.0,
            status_code=200,
            labels={
                "http.method": "POST",
                "http.url": "/api/v1/payments",
                "user.id": "user_12345"
            }
        ))
        
        # Child span - Payment Service
        payment_span_id = str(uuid4())
        traces.append(GCPTracingData(
            trace_id=payment_trace_id,
            span_id=payment_span_id,
            parent_span_id=traces[-1].span_id,
            operation_name="process_payment",
            service_name="payment-service",
            start_time=datetime.utcnow() - timedelta(minutes=4, seconds=59),
            end_time=datetime.utcnow() - timedelta(minutes=4, seconds=52),
            duration_ms=7000.0,
            status_code=200,
            labels={
                "payment.amount": "99.99",
                "payment.currency": "USD",
                "payment.method": "credit_card"
            }
        ))
        
        # Child span - Redis Cache (slow)
        traces.append(GCPTracingData(
            trace_id=payment_trace_id,
            span_id=str(uuid4()),
            parent_span_id=payment_span_id,
            operation_name="redis.get",
            service_name="redis-cache",
            start_time=datetime.utcnow() - timedelta(minutes=4, seconds=58),
            end_time=datetime.utcnow() - timedelta(minutes=4, seconds=53),
            duration_ms=5000.0,
            status_code=200,
            labels={
                "cache.key": "user_profile_12345",
                "cache.hit": "false",
                "cache.evicted": "true"
            }
        ))
        
        # Child span - Database query (fallback)
        traces.append(GCPTracingData(
            trace_id=payment_trace_id,
            span_id=str(uuid4()),
            parent_span_id=payment_span_id,
            operation_name="sql.query",
            service_name="payment-db",
            start_time=datetime.utcnow() - timedelta(minutes=4, seconds=53),
            end_time=datetime.utcnow() - timedelta(minutes=4, seconds=52),
            duration_ms=1000.0,
            status_code=200,
            labels={
                "db.statement": "SELECT * FROM user_profiles WHERE id = ?",
                "db.rows_affected": "1"
            }
        ))
        
        # Generate traces for database timeout scenario
        billing_trace_id = str(uuid4())
        
        # Root span - Billing API
        traces.append(GCPTracingData(
            trace_id=billing_trace_id,
            span_id=str(uuid4()),
            operation_name="POST /api/v1/billing/process",
            service_name="billing-service",
            start_time=datetime.utcnow() - timedelta(minutes=2),
            end_time=datetime.utcnow() - timedelta(minutes=1, seconds=48),
            duration_ms=12000.0,
            status_code=500,
            labels={
                "http.method": "POST",
                "http.status_code": "500",
                "error.type": "timeout"
            }
        ))
        
        # Child span - Database connection (timeout)
        traces.append(GCPTracingData(
            trace_id=billing_trace_id,
            span_id=str(uuid4()),
            parent_span_id=traces[-1].span_id,
            operation_name="sql.bulk_update",
            service_name="billing-db",
            start_time=datetime.utcnow() - timedelta(minutes=1, seconds=59),
            end_time=datetime.utcnow() - timedelta(minutes=1, seconds=48),
            duration_ms=11000.0,
            status_code=500,
            labels={
                "db.statement": "UPDATE invoices SET status = 'processed' WHERE customer_id IN (...)",
                "db.rows_affected": "50000",
                "error.message": "connection timeout"
            }
        ))
        
        return traces
    
    def get_traces(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filter_expr: Optional[str] = None,
        limit: int = 100
    ) -> List[GCPTracingData]:
        """Get traces with optional filtering."""
        filtered_traces = self.traces.copy()
        
        # Apply time filtering
        if start_time:
            filtered_traces = [
                trace for trace in filtered_traces
                if trace.start_time >= start_time
            ]
        
        if end_time:
            filtered_traces = [
                trace for trace in filtered_traces
                if trace.start_time <= end_time
            ]
        
        # Apply text filtering
        if filter_expr:
            filtered_traces = [
                trace for trace in filtered_traces
                if (filter_expr.lower() in trace.operation_name.lower() or
                    filter_expr.lower() in trace.service_name.lower())
            ]
        
        return filtered_traces[:limit]
    
    def get_trace_by_id(self, trace_id: str) -> List[GCPTracingData]:
        """Get all spans for a specific trace ID."""
        return [trace for trace in self.traces if trace.trace_id == trace_id]
    
    def get_slow_traces(
        self,
        min_duration_ms: float = 1000.0,
        limit: int = 50
    ) -> List[GCPTracingData]:
        """Get traces with duration exceeding threshold."""
        slow_traces = [
            trace for trace in self.traces
            if trace.duration_ms >= min_duration_ms
        ]
        return sorted(slow_traces, key=lambda x: x.duration_ms, reverse=True)[:limit]
    
    def get_error_traces(
        self,
        start_time: Optional[datetime] = None,
        limit: int = 50
    ) -> List[GCPTracingData]:
        """Get traces with error status codes."""
        error_traces = [
            trace for trace in self.traces
            if trace.status_code >= 400
        ]
        
        if start_time:
            error_traces = [
                trace for trace in error_traces
                if trace.start_time >= start_time
            ]
        
        return error_traces[:limit]
    
    def analyze_service_dependencies(self) -> Dict[str, List[str]]:
        """Analyze service dependencies from trace data."""
        dependencies = {}
        
        for trace in self.traces:
            if trace.parent_span_id:
                # Find parent span
                parent_traces = [
                    t for t in self.traces
                    if t.span_id == trace.parent_span_id
                ]
                if parent_traces:
                    parent_service = parent_traces[0].service_name
                    if parent_service not in dependencies:
                        dependencies[parent_service] = []
                    if trace.service_name not in dependencies[parent_service]:
                        dependencies[parent_service].append(trace.service_name)
        
        return dependencies
    
    def get_service_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics by service."""
        service_stats = {}
        
        for trace in self.traces:
            service = trace.service_name
            if service not in service_stats:
                service_stats[service] = {
                    'total_duration': 0.0,
                    'count': 0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0
                }
            
            stats = service_stats[service]
            stats['total_duration'] += trace.duration_ms
            stats['count'] += 1
            stats['min_duration'] = min(stats['min_duration'], trace.duration_ms)
            stats['max_duration'] = max(stats['max_duration'], trace.duration_ms)
        
        # Calculate averages
        for service, stats in service_stats.items():
            if stats['count'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['count']
            else:
                stats['avg_duration'] = 0.0
        
        return service_stats


class RealGCPTracingClient:
    """Real GCP Cloud Tracing client implementation."""
    
    def __init__(self):
        self.project_id = settings.GCP_PROJECT_ID
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID must be set for real GCP Tracing client")
        
        # Initialize the real GCP client
        self.client = trace_v2.TraceServiceClient()
        self.project_name = f"projects/{self.project_id}"
        logger.info(f"Initialized real GCP Tracing client for project: {self.project_id}")
    
    def get_traces(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filter_expr: Optional[str] = None,
        limit: int = 100
    ) -> List[GCPTracingData]:
        """Get traces from real GCP Cloud Trace."""
        try:
            # Build request
            request = trace_v2.ListTracesRequest(
                parent=self.project_name,
                page_size=limit
            )
            
            # Add time filter if provided
            if start_time or end_time:
                request.filter = self._build_time_filter(start_time, end_time)
            
            # Add custom filter
            if filter_expr:
                if request.filter:
                    request.filter += f" AND {filter_expr}"
                else:
                    request.filter = filter_expr
            
            # Execute request
            response = self.client.list_traces(request=request)
            
            # Convert to our data format
            traces = []
            for trace in response:
                traces.extend(self._convert_gcp_trace(trace))
            
            return traces
            
        except Exception as e:
            logger.error(f"Error fetching traces from GCP: {e}")
            return []
    
    def _build_time_filter(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> str:
        """Build time filter for GCP Trace API."""
        filters = []
        
        if start_time:
            filters.append(f"start_time >= \"{start_time.isoformat()}\"")
        
        if end_time:
            filters.append(f"start_time <= \"{end_time.isoformat()}\"")
        
        return " AND ".join(filters)
    
    def _convert_gcp_trace(self, gcp_trace) -> List[GCPTracingData]:
        """Convert GCP trace to our data format."""
        traces = []
        
        for span in gcp_trace.spans:
            trace_data = GCPTracingData(
                trace_id=gcp_trace.trace_id,
                span_id=span.span_id,
                parent_span_id=span.parent_span_id if span.parent_span_id else None,
                operation_name=span.display_name.value if span.display_name else "",
                service_name=self._extract_service_name(span),
                start_time=span.start_time,
                end_time=span.end_time,
                duration_ms=self._calculate_duration_ms(span.start_time, span.end_time),
                status_code=self._extract_status_code(span),
                labels=dict(span.attributes.attribute_map) if span.attributes else {},
                annotations=self._extract_annotations(span)
            )
            traces.append(trace_data)
        
        return traces
    
    def _extract_service_name(self, span) -> str:
        """Extract service name from span."""
        if span.attributes and span.attributes.attribute_map:
            attrs = span.attributes.attribute_map
            return attrs.get('service.name', attrs.get('component', 'unknown'))
        return 'unknown'
    
    def _calculate_duration_ms(self, start_time, end_time) -> float:
        """Calculate duration in milliseconds."""
        if start_time and end_time:
            duration = end_time - start_time
            return duration.total_seconds() * 1000.0
        return 0.0
    
    def _extract_status_code(self, span) -> int:
        """Extract HTTP status code from span."""
        if span.status and span.status.code:
            return span.status.code
        return 200  # Default to success
    
    def _extract_annotations(self, span) -> List[Dict[str, Any]]:
        """Extract annotations from span."""
        annotations = []
        if span.time_events:
            for event in span.time_events.time_event:
                annotation = {
                    'timestamp': event.time.isoformat() if event.time else None,
                    'description': event.annotation.description.value if event.annotation else ""
                }
                annotations.append(annotation)
        return annotations


class GCPTracingClient:
    """Main GCP Cloud Tracing client with automatic mock/real switching."""
    
    def __init__(self):
        """Initialize the appropriate tracing client based on configuration."""
        self.use_mock = settings.USE_MOCK_GCP_SERVICES or not GCP_TRACING_AVAILABLE
        
        if self.use_mock:
            self.client = MockGCPTracingClient()
            logger.info("Using mock GCP Tracing implementation for demo")
        else:
            self.client = RealGCPTracingClient()
            logger.info("Using real GCP Tracing implementation")
    
    def get_traces(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filter_expr: Optional[str] = None,
        limit: int = 100
    ) -> List[GCPTracingData]:
        """Get traces with optional filtering."""
        return self.client.get_traces(start_time, end_time, filter_expr, limit)
    
    def get_trace_by_id(self, trace_id: str) -> List[GCPTracingData]:
        """Get all spans for a specific trace ID."""
        if hasattr(self.client, 'get_trace_by_id'):
            return self.client.get_trace_by_id(trace_id)
        else:
            # For real client, filter by trace ID
            return self.get_traces(filter_expr=f"trace_id=\"{trace_id}\"")
    
    def get_slow_traces(
        self,
        min_duration_ms: float = 1000.0,
        limit: int = 50
    ) -> List[GCPTracingData]:
        """Get traces with duration exceeding threshold."""
        if hasattr(self.client, 'get_slow_traces'):
            return self.client.get_slow_traces(min_duration_ms, limit)
        else:
            # For real client, get all traces and filter
            all_traces = self.get_traces(limit=1000)
            slow_traces = [
                trace for trace in all_traces
                if trace.duration_ms >= min_duration_ms
            ]
            return sorted(slow_traces, key=lambda x: x.duration_ms, reverse=True)[:limit]
    
    def get_error_traces(
        self,
        start_time: Optional[datetime] = None,
        limit: int = 50
    ) -> List[GCPTracingData]:
        """Get traces with error status codes."""
        if hasattr(self.client, 'get_error_traces'):
            return self.client.get_error_traces(start_time, limit)
        else:
            # For real client, use filter
            filter_expr = "status.code >= 400"
            return self.get_traces(start_time=start_time, filter_expr=filter_expr, limit=limit)
    
    def analyze_service_dependencies(self) -> Dict[str, List[str]]:
        """Analyze service dependencies from trace data."""
        if hasattr(self.client, 'analyze_service_dependencies'):
            return self.client.analyze_service_dependencies()
        else:
            # Implement for real client
            traces = self.get_traces(limit=1000)
            dependencies = {}
            
            # Group traces by trace_id
            traces_by_id = {}
            for trace in traces:
                if trace.trace_id not in traces_by_id:
                    traces_by_id[trace.trace_id] = []
                traces_by_id[trace.trace_id].append(trace)
            
            # Analyze dependencies within each trace
            for trace_spans in traces_by_id.values():
                for span in trace_spans:
                    if span.parent_span_id:
                        parent_spans = [
                            s for s in trace_spans
                            if s.span_id == span.parent_span_id
                        ]
                        if parent_spans:
                            parent_service = parent_spans[0].service_name
                            if parent_service not in dependencies:
                                dependencies[parent_service] = []
                            if span.service_name not in dependencies[parent_service]:
                                dependencies[parent_service].append(span.service_name)
            
            return dependencies
    
    def get_service_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics by service."""
        if hasattr(self.client, 'get_service_latency_stats'):
            return self.client.get_service_latency_stats()
        else:
            # Implement for real client
            traces = self.get_traces(limit=1000)
            service_stats = {}
            
            for trace in traces:
                service = trace.service_name
                if service not in service_stats:
                    service_stats[service] = {
                        'total_duration': 0.0,
                        'count': 0,
                        'min_duration': float('inf'),
                        'max_duration': 0.0
                    }
                
                stats = service_stats[service]
                stats['total_duration'] += trace.duration_ms
                stats['count'] += 1
                stats['min_duration'] = min(stats['min_duration'], trace.duration_ms)
                stats['max_duration'] = max(stats['max_duration'], trace.duration_ms)
            
            # Calculate averages
            for service, stats in service_stats.items():
                if stats['count'] > 0:
                    stats['avg_duration'] = stats['total_duration'] / stats['count']
                else:
                    stats['avg_duration'] = 0.0
            
            return service_stats
    
    def is_mock(self) -> bool:
        """Check if using mock implementation."""
        return self.use_mock
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary information for a trace."""
        spans = self.get_trace_by_id(trace_id)
        if not spans:
            return {}
        
        # Find root span (no parent)
        root_spans = [span for span in spans if not span.parent_span_id]
        root_span = root_spans[0] if root_spans else spans[0]
        
        # Calculate total duration
        total_duration = max(span.duration_ms for span in spans)
        
        # Count services
        services = set(span.service_name for span in spans)
        
        # Check for errors
        has_errors = any(span.status_code >= 400 for span in spans)
        
        return {
            'trace_id': trace_id,
            'root_operation': root_span.operation_name,
            'total_duration_ms': total_duration,
            'span_count': len(spans),
            'service_count': len(services),
            'services': list(services),
            'has_errors': has_errors,
            'start_time': root_span.start_time.isoformat() if root_span.start_time else None,
            'end_time': root_span.end_time.isoformat() if root_span.end_time else None
        }