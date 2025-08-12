"""
GCP Cloud Monitoring integration with mock implementation for demo.

This module provides integration with GCP Cloud Monitoring API, simulating real
monitoring API responses using synthetic data for metrics like CPU, memory,
latency, and database performance.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from loguru import logger

try:
    from google.cloud import monitoring_v3
    from google.oauth2 import service_account
    GCP_MONITORING_AVAILABLE = True
except ImportError:
    GCP_MONITORING_AVAILABLE = False
    logger.warning("Google Cloud Monitoring client not available, using mock implementation")

from config.settings import get_settings

settings = get_settings()


class GCPMonitoringClient:
    """GCP Cloud Monitoring integration client."""
    
    def __init__(self):
        """Initialize the GCP Monitoring client."""
        self.project_id = settings.GCP_PROJECT_ID
        self.use_mock = settings.USE_MOCK_GCP_SERVICES or not GCP_MONITORING_AVAILABLE
        self.client = None
        self.synthetic_data_path = Path(settings.SYNTHETIC_DATA_PATH)
        
        # Mock data cache
        self._mock_data_cache = {}
        self._last_cache_update = None
        self._cache_ttl = timedelta(minutes=5)
        
        # Metric configurations
        self.metric_configs = {
            "compute/instance/cpu_utilization": {
                "unit": "percentage",
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE",
                "display_name": "CPU Utilization",
                "description": "Fraction of the allocated CPU that is currently in use on the instance"
            },
            "compute/instance/memory_utilization": {
                "unit": "percentage", 
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE",
                "display_name": "Memory Utilization",
                "description": "Fraction of the allocated memory that is currently in use on the instance"
            },
            "compute/instance/disk_read_bytes_count": {
                "unit": "bytes",
                "value_type": "INT64", 
                "metric_kind": "CUMULATIVE",
                "display_name": "Disk Read Bytes",
                "description": "Count of bytes read from disk"
            },
            "compute/instance/disk_write_bytes_count": {
                "unit": "bytes",
                "value_type": "INT64",
                "metric_kind": "CUMULATIVE", 
                "display_name": "Disk Write Bytes",
                "description": "Count of bytes written to disk"
            },
            "cloudsql/database/cpu_utilization": {
                "unit": "percentage",
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE",
                "display_name": "Database CPU Utilization",
                "description": "Current CPU utilization for the database"
            },
            "cloudsql/database/memory_utilization": {
                "unit": "percentage",
                "value_type": "DOUBLE", 
                "metric_kind": "GAUGE",
                "display_name": "Database Memory Utilization",
                "description": "Current memory utilization for the database"
            },
            "cloudsql/database/connection_count": {
                "unit": "connections",
                "value_type": "INT64",
                "metric_kind": "GAUGE", 
                "display_name": "Database Connections",
                "description": "Number of connections to the database"
            },
            "cloudsql/database/query_execution_time": {
                "unit": "ms",
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE",
                "display_name": "Query Execution Time", 
                "description": "Average query execution time"
            },
            "redis/cache_hit_rate": {
                "unit": "percentage",
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE",
                "display_name": "Cache Hit Rate",
                "description": "Percentage of cache requests that resulted in hits"
            },
            "redis/memory_usage": {
                "unit": "bytes", 
                "value_type": "INT64",
                "metric_kind": "GAUGE",
                "display_name": "Redis Memory Usage",
                "description": "Current memory usage of Redis instance"
            },
            "loadbalancer/request_count": {
                "unit": "requests",
                "value_type": "INT64",
                "metric_kind": "CUMULATIVE",
                "display_name": "Load Balancer Requests",
                "description": "Number of requests processed by load balancer"
            },
            "loadbalancer/latency": {
                "unit": "ms",
                "value_type": "DOUBLE", 
                "metric_kind": "GAUGE",
                "display_name": "Load Balancer Latency",
                "description": "Average request latency through load balancer"
            },
            "api/request_latency": {
                "unit": "ms",
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE", 
                "display_name": "API Request Latency",
                "description": "Average API request latency"
            },
            "api/error_rate": {
                "unit": "percentage",
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE",
                "display_name": "API Error Rate", 
                "description": "Percentage of API requests that resulted in errors"
            },
            "kubernetes/container/cpu_usage": {
                "unit": "cores",
                "value_type": "DOUBLE",
                "metric_kind": "GAUGE",
                "display_name": "Container CPU Usage",
                "description": "CPU usage by container"
            },
            "kubernetes/container/memory_usage": {
                "unit": "bytes",
                "value_type": "INT64", 
                "metric_kind": "GAUGE",
                "display_name": "Container Memory Usage",
                "description": "Memory usage by container"
            }
        }
        
        if not self.use_mock:
            self._initialize_real_client()
        else:
            logger.info("Using mock GCP Monitoring implementation for demo")
    
    def _initialize_real_client(self) -> None:
        """Initialize real GCP Monitoring client."""
        try:
            if settings.GCP_SERVICE_ACCOUNT_KEY_PATH:
                credentials = service_account.Credentials.from_service_account_file(
                    settings.GCP_SERVICE_ACCOUNT_KEY_PATH
                )
                self.client = monitoring_v3.MetricServiceClient(credentials=credentials)
            else:
                # Use default credentials
                self.client = monitoring_v3.MetricServiceClient()
            
            self.project_name = f"projects/{self.project_id}"
            logger.info(f"Initialized GCP Monitoring client for project: {self.project_id}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize real GCP Monitoring client: {e}")
            logger.info("Falling back to mock implementation")
            self.use_mock = True
            self.client = None
    
    async def query_metric(
        self,
        metric_name: str,
        time_range: str = "1h",
        labels: Optional[Dict[str, str]] = None,
        aggregation: Optional[str] = None,
        interval: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query a specific metric from GCP Monitoring.
        
        Args:
            metric_name: Name of the metric to query
            time_range: Time range for the query (e.g., "1h", "6h", "1d")
            labels: Resource labels to filter by
            aggregation: Aggregation method (e.g., "MEAN", "MAX", "SUM")
            interval: Alignment interval for time series
            
        Returns:
            Dictionary containing metric data and metadata
        """
        logger.info(f"Querying metric: {metric_name} for time range: {time_range}")
        
        try:
            if self.use_mock:
                return await self._query_metric_mock(metric_name, time_range, labels, aggregation)
            else:
                return await self._query_metric_real(metric_name, time_range, labels, aggregation, interval)
                
        except Exception as e:
            logger.error(f"Error querying metric {metric_name}: {e}")
            return None
    
    async def query_multiple_metrics(
        self,
        metric_names: List[str],
        time_range: str = "1h",
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Query multiple metrics simultaneously.
        
        Args:
            metric_names: List of metric names to query
            time_range: Time range for the query
            labels: Resource labels to filter by
            
        Returns:
            Dictionary mapping metric names to their data
        """
        logger.info(f"Querying {len(metric_names)} metrics for time range: {time_range}")
        
        results = {}
        
        # Execute queries concurrently
        tasks = [
            self.query_metric(metric_name, time_range, labels)
            for metric_name in metric_names
        ]
        
        metric_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for metric_name, result in zip(metric_names, metric_results):
            if isinstance(result, Exception):
                logger.error(f"Error querying metric {metric_name}: {result}")
                results[metric_name] = None
            else:
                results[metric_name] = result
        
        return results
    
    async def get_metric_descriptors(
        self,
        metric_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available metric descriptors.
        
        Args:
            metric_filter: Filter string for metric types
            
        Returns:
            List of metric descriptor dictionaries
        """
        logger.info("Getting metric descriptors")
        
        try:
            if self.use_mock:
                return await self._get_metric_descriptors_mock(metric_filter)
            else:
                return await self._get_metric_descriptors_real(metric_filter)
                
        except Exception as e:
            logger.error(f"Error getting metric descriptors: {e}")
            return []
    
    async def create_custom_metric(
        self,
        metric_type: str,
        display_name: str,
        description: str,
        value_type: str = "DOUBLE",
        metric_kind: str = "GAUGE",
        unit: str = ""
    ) -> bool:
        """
        Create a custom metric descriptor.
        
        Args:
            metric_type: Type identifier for the metric
            display_name: Human-readable name
            description: Description of the metric
            value_type: Type of values (DOUBLE, INT64, BOOL)
            metric_kind: Kind of metric (GAUGE, CUMULATIVE, DELTA)
            unit: Unit of measurement
            
        Returns:
            True if creation was successful
        """
        logger.info(f"Creating custom metric: {metric_type}")
        
        try:
            if self.use_mock:
                return await self._create_custom_metric_mock(
                    metric_type, display_name, description, value_type, metric_kind, unit
                )
            else:
                return await self._create_custom_metric_real(
                    metric_type, display_name, description, value_type, metric_kind, unit
                )
                
        except Exception as e:
            logger.error(f"Error creating custom metric {metric_type}: {e}")
            return False
    
    async def write_time_series(
        self,
        metric_type: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Write a time series data point.
        
        Args:
            metric_type: Type of the metric
            value: Value to write
            labels: Resource and metric labels
            timestamp: Timestamp for the data point
            
        Returns:
            True if write was successful
        """
        logger.info(f"Writing time series data for metric: {metric_type}")
        
        try:
            if self.use_mock:
                return await self._write_time_series_mock(metric_type, value, labels, timestamp)
            else:
                return await self._write_time_series_real(metric_type, value, labels, timestamp)
                
        except Exception as e:
            logger.error(f"Error writing time series for {metric_type}: {e}")
            return False
    
    async def _query_metric_mock(
        self,
        metric_name: str,
        time_range: str,
        labels: Optional[Dict[str, str]] = None,
        aggregation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mock implementation of metric querying using synthetic data."""
        
        # Load synthetic data if cache is stale
        await self._load_synthetic_data()
        
        # Parse time range
        end_time = datetime.utcnow()
        start_time = self._parse_time_range(time_range, end_time)
        
        # Generate realistic metric data based on metric type
        metric_data = self._generate_metric_data(metric_name, start_time, end_time, labels)
        
        # Apply aggregation if specified
        if aggregation:
            metric_data = self._apply_aggregation(metric_data, aggregation)
        
        # Add trend analysis
        trend = self._calculate_trend(metric_data.get("data_points", []))
        
        result = {
            "metric_name": metric_name,
            "resource_type": self._get_resource_type(metric_name),
            "labels": labels or {},
            "time_range": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "latest_value": metric_data.get("latest_value", 0),
            "data_points": metric_data.get("data_points", []),
            "aggregation": aggregation,
            "trend": trend,
            "unit": self.metric_configs.get(metric_name, {}).get("unit", ""),
            "metadata": {
                "query_time": datetime.utcnow().isoformat(),
                "is_mock": True,
                "data_source": "synthetic"
            }
        }
        
        logger.debug(f"Generated mock metric data for {metric_name}: latest_value={result['latest_value']}")
        return result
    
    async def _query_metric_real(
        self,
        metric_name: str,
        time_range: str,
        labels: Optional[Dict[str, str]] = None,
        aggregation: Optional[str] = None,
        interval: Optional[str] = None
    ) -> Dict[str, Any]:
        """Real implementation of metric querying using GCP Monitoring API."""
        
        if not self.client:
            raise Exception("GCP Monitoring client not initialized")
        
        # Parse time range
        end_time = datetime.utcnow()
        start_time = self._parse_time_range(time_range, end_time)
        
        # Convert to protobuf timestamps
        from google.protobuf.timestamp_pb2 import Timestamp
        start_timestamp = Timestamp()
        start_timestamp.FromDatetime(start_time)
        end_timestamp = Timestamp()
        end_timestamp.FromDatetime(end_time)
        
        # Build filter
        filter_str = f'metric.type="compute.googleapis.com/{metric_name}"'
        if labels:
            for key, value in labels.items():
                filter_str += f' AND resource.label.{key}="{value}"'
        
        # Create request
        interval = monitoring_v3.TimeInterval(
            end_time=end_timestamp,
            start_time=start_timestamp
        )
        
        request = monitoring_v3.ListTimeSeriesRequest(
            name=self.project_name,
            filter=filter_str,
            interval=interval,
            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
        )
        
        # Execute query
        response = self.client.list_time_series(request=request)
        
        # Process response
        data_points = []
        latest_value = 0
        
        for time_series in response:
            for point in time_series.points:
                timestamp = point.interval.end_time.ToDatetime()
                value = point.value.double_value or point.value.int64_value
                data_points.append({
                    "timestamp": timestamp.isoformat(),
                    "value": value
                })
                latest_value = value  # Last point becomes latest
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])
        
        result = {
            "metric_name": metric_name,
            "resource_type": self._get_resource_type(metric_name),
            "labels": labels or {},
            "time_range": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "latest_value": latest_value,
            "data_points": data_points,
            "aggregation": aggregation,
            "trend": self._calculate_trend(data_points),
            "metadata": {
                "query_time": datetime.utcnow().isoformat(),
                "is_mock": False,
                "data_source": "gcp_monitoring"
            }
        }
        
        return result
    
    async def _get_metric_descriptors_mock(
        self,
        metric_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Mock implementation of getting metric descriptors."""
        
        descriptors = []
        
        for metric_type, config in self.metric_configs.items():
            if metric_filter and metric_filter not in metric_type:
                continue
                
            descriptor = {
                "name": f"projects/{self.project_id}/metricDescriptors/{metric_type}",
                "type": metric_type,
                "labels": [],
                "metric_kind": config.get("metric_kind", "GAUGE"),
                "value_type": config.get("value_type", "DOUBLE"), 
                "unit": config.get("unit", ""),
                "description": config.get("description", ""),
                "display_name": config.get("display_name", metric_type),
                "metadata": {
                    "launch_stage": "GA",
                    "sample_period": {"seconds": 60}
                }
            }
            descriptors.append(descriptor)
        
        return descriptors
    
    async def _get_metric_descriptors_real(
        self,
        metric_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Real implementation of getting metric descriptors."""
        
        if not self.client:
            raise Exception("GCP Monitoring client not initialized")
        
        request = monitoring_v3.ListMetricDescriptorsRequest(
            name=self.project_name,
            filter=metric_filter
        )
        
        response = self.client.list_metric_descriptors(request=request)
        
        descriptors = []
        for descriptor in response:
            descriptors.append({
                "name": descriptor.name,
                "type": descriptor.type,
                "labels": [{"key": label.key, "description": label.description} for label in descriptor.labels],
                "metric_kind": descriptor.metric_kind.name,
                "value_type": descriptor.value_type.name,
                "unit": descriptor.unit,
                "description": descriptor.description,
                "display_name": descriptor.display_name
            })
        
        return descriptors
    
    async def _create_custom_metric_mock(
        self,
        metric_type: str,
        display_name: str,
        description: str,
        value_type: str,
        metric_kind: str,
        unit: str
    ) -> bool:
        """Mock implementation of creating custom metrics."""
        
        # Add to metric configs
        self.metric_configs[metric_type] = {
            "unit": unit,
            "value_type": value_type,
            "metric_kind": metric_kind,
            "display_name": display_name,
            "description": description
        }
        
        logger.info(f"Mock created custom metric: {metric_type}")
        return True
    
    async def _create_custom_metric_real(
        self,
        metric_type: str,
        display_name: str,
        description: str,
        value_type: str,
        metric_kind: str,
        unit: str
    ) -> bool:
        """Real implementation of creating custom metrics."""
        
        if not self.client:
            raise Exception("GCP Monitoring client not initialized")
        
        # Convert string enums to protobuf enums
        value_type_enum = getattr(monitoring_v3.MetricDescriptor.ValueType, value_type)
        metric_kind_enum = getattr(monitoring_v3.MetricDescriptor.MetricKind, metric_kind)
        
        descriptor = monitoring_v3.MetricDescriptor(
            type=f"custom.googleapis.com/{metric_type}",
            metric_kind=metric_kind_enum,
            value_type=value_type_enum,
            unit=unit,
            description=description,
            display_name=display_name
        )
        
        request = monitoring_v3.CreateMetricDescriptorRequest(
            name=self.project_name,
            metric_descriptor=descriptor
        )
        
        self.client.create_metric_descriptor(request=request)
        logger.info(f"Created custom metric: {metric_type}")
        return True
    
    async def _write_time_series_mock(
        self,
        metric_type: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Mock implementation of writing time series data."""
        
        timestamp = timestamp or datetime.utcnow()
        
        logger.info(f"Mock wrote time series: {metric_type}={value} at {timestamp}")
        return True
    
    async def _write_time_series_real(
        self,
        metric_type: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Real implementation of writing time series data."""
        
        if not self.client:
            raise Exception("GCP Monitoring client not initialized")
        
        timestamp = timestamp or datetime.utcnow()
        
        # Convert to protobuf timestamp
        from google.protobuf.timestamp_pb2 import Timestamp
        pb_timestamp = Timestamp()
        pb_timestamp.FromDatetime(timestamp)
        
        # Create time series
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_type}"
        
        # Add labels
        if labels:
            for key, val in labels.items():
                series.metric.labels[key] = val
        
        # Add resource labels (required)
        series.resource.type = "global"
        
        # Add data point
        point = monitoring_v3.Point()
        point.interval.end_time = pb_timestamp
        
        if isinstance(value, int):
            point.value.int64_value = value
        else:
            point.value.double_value = float(value)
        
        series.points = [point]
        
        # Write time series
        request = monitoring_v3.CreateTimeSeriesRequest(
            name=self.project_name,
            time_series=[series]
        )
        
        self.client.create_time_series(request=request)
        logger.info(f"Wrote time series: {metric_type}={value}")
        return True
    
    async def _load_synthetic_data(self) -> None:
        """Load synthetic data for mock responses."""
        
        # Check if cache needs refresh
        if (self._last_cache_update and 
            datetime.utcnow() - self._last_cache_update < self._cache_ttl and
            self._mock_data_cache):
            return
        
        try:
            # Load GCP observability data
            gcp_data_file = self.synthetic_data_path / "scenario_1_gcp_data.json"
            if gcp_data_file.exists():
                with open(gcp_data_file, 'r') as f:
                    # Read file in chunks for large files
                    content = ""
                    chunk_size = 8192
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        content += chunk
                        # Break if we have enough data for demo
                        if len(content) > 50000:  # ~50KB should be enough
                            break
                    
                    # Try to parse as much as possible
                    try:
                        gcp_data = json.loads(content)
                    except json.JSONDecodeError:
                        # If partial read, try to find valid JSON objects
                        lines = content.split('\n')
                        gcp_data = {"monitoring_data": []}
                        for line in lines:
                            if line.strip():
                                try:
                                    obj = json.loads(line)
                                    gcp_data["monitoring_data"].append(obj)
                                except:
                                    continue
                    
                    self._mock_data_cache["gcp_data"] = gcp_data
            
            # Load incidents data
            incidents_file = self.synthetic_data_path / "historical_incidents.json"
            if incidents_file.exists():
                with open(incidents_file, 'r') as f:
                    incidents_data = json.load(f)
                    self._mock_data_cache["incidents"] = incidents_data
            
            self._last_cache_update = datetime.utcnow()
            logger.debug("Loaded synthetic data for mock responses")
            
        except Exception as e:
            logger.warning(f"Error loading synthetic data: {e}")
            # Use fallback data
            self._mock_data_cache = {"gcp_data": {}, "incidents": []}
    
    def _generate_metric_data(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate realistic metric data for mock responses."""
        
        # Determine number of data points based on time range
        duration = (end_time - start_time).total_seconds()
        if duration <= 3600:  # 1 hour
            interval_seconds = 60  # 1 minute intervals
        elif duration <= 86400:  # 1 day
            interval_seconds = 300  # 5 minute intervals
        else:
            interval_seconds = 3600  # 1 hour intervals
        
        num_points = int(duration / interval_seconds)
        num_points = min(num_points, 100)  # Limit to 100 points for demo
        
        # Generate base values based on metric type
        base_values = self._get_metric_base_values(metric_name, labels)
        
        # Generate data points with realistic variations
        data_points = []
        current_time = start_time
        
        for i in range(num_points):
            # Add some realistic variation and trends
            variation = random.uniform(-0.1, 0.1)  # ±10% variation
            trend_factor = 1 + (i / num_points) * 0.05  # Slight upward trend
            
            # Apply incident-like spikes occasionally
            spike_factor = 1.0
            if random.random() < 0.05:  # 5% chance of spike
                spike_factor = random.uniform(1.5, 3.0)
            
            value = base_values["base_value"] * trend_factor * (1 + variation) * spike_factor
            
            # Apply metric-specific constraints
            value = self._apply_metric_constraints(metric_name, value)
            
            data_points.append({
                "timestamp": current_time.isoformat(),
                "value": round(value, 2)
            })
            
            current_time += timedelta(seconds=interval_seconds)
        
        # Latest value is the last data point
        latest_value = data_points[-1]["value"] if data_points else 0
        
        return {
            "latest_value": latest_value,
            "data_points": data_points,
            "base_value": base_values["base_value"],
            "normal_range": base_values["normal_range"]
        }
    
    def _get_metric_base_values(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get base values for different metric types."""
        
        # Check if this is related to an incident scenario
        service_name = labels.get("service") if labels else None
        is_incident_scenario = service_name in ["payment-api", "billing-service", "user-api"]
        
        base_values = {
            "compute/instance/cpu_utilization": {
                "base_value": 75.0 if is_incident_scenario else 45.0,
                "normal_range": (20, 70)
            },
            "compute/instance/memory_utilization": {
                "base_value": 68.0 if is_incident_scenario else 52.0,
                "normal_range": (30, 80)
            },
            "cloudsql/database/cpu_utilization": {
                "base_value": 82.0 if is_incident_scenario else 35.0,
                "normal_range": (15, 60)
            },
            "cloudsql/database/memory_utilization": {
                "base_value": 79.0 if is_incident_scenario else 48.0,
                "normal_range": (25, 70)
            },
            "cloudsql/database/connection_count": {
                "base_value": 190.0 if is_incident_scenario else 85.0,
                "normal_range": (50, 150)
            },
            "cloudsql/database/query_execution_time": {
                "base_value": 8500.0 if is_incident_scenario else 1200.0,  # milliseconds
                "normal_range": (100, 3000)
            },
            "redis/cache_hit_rate": {
                "base_value": 38.0 if is_incident_scenario else 92.0,  # percentage
                "normal_range": (85, 98)
            },
            "redis/memory_usage": {
                "base_value": 4200000000 if is_incident_scenario else 2100000000,  # bytes
                "normal_range": (1000000000, 3000000000)
            },
            "api/request_latency": {
                "base_value": 6200.0 if is_incident_scenario else 850.0,  # milliseconds
                "normal_range": (200, 2000)
            },
            "api/error_rate": {
                "base_value": 2.5 if is_incident_scenario else 0.15,  # percentage
                "normal_range": (0, 1)
            },
            "loadbalancer/latency": {
                "base_value": 1800.0 if is_incident_scenario else 120.0,  # milliseconds
                "normal_range": (50, 500)
            }
        }
        
        return base_values.get(metric_name, {
            "base_value": 50.0,
            "normal_range": (0, 100)
        })
    
    def _apply_metric_constraints(self, metric_name: str, value: float) -> float:
        """Apply realistic constraints to metric values."""
        
        # Percentage metrics
        if "utilization" in metric_name or "rate" in metric_name or "hit_rate" in metric_name:
            return max(0, min(100, value))
        
        # Count metrics
        if "count" in metric_name or "connections" in metric_name:
            return max(0, int(value))
        
        # Time metrics (milliseconds)
        if "latency" in metric_name or "time" in metric_name:
            return max(0, value)
        
        # Memory/disk metrics (bytes)
        if "memory" in metric_name or "bytes" in metric_name:
            return max(0, int(value))
        
        # Default: non-negative
        return max(0, value)
    
    def _apply_aggregation(
        self,
        metric_data: Dict[str, Any],
        aggregation: str
    ) -> Dict[str, Any]:
        """Apply aggregation to metric data."""
        
        data_points = metric_data.get("data_points", [])
        if not data_points:
            return metric_data
        
        values = [point["value"] for point in data_points]
        
        if aggregation.upper() == "MEAN":
            aggregated_value = sum(values) / len(values)
        elif aggregation.upper() == "MAX":
            aggregated_value = max(values)
        elif aggregation.upper() == "MIN":
            aggregated_value = min(values)
        elif aggregation.upper() == "SUM":
            aggregated_value = sum(values)
        elif aggregation.upper() == "COUNT":
            aggregated_value = len(values)
        else:
            aggregated_value = metric_data.get("latest_value", 0)
        
        # Update the metric data with aggregated value
        metric_data["aggregated_value"] = round(aggregated_value, 2)
        metric_data["aggregation_method"] = aggregation.upper()
        
        return metric_data
    
    def _calculate_trend(self, data_points: List[Dict[str, Any]]) -> str:
        """Calculate trend direction from data points."""
        
        if len(data_points) < 2:
            return "stable"
        
        values = [point["value"] for point in data_points]
        
        # Simple trend calculation using first and last quartile
        quartile_size = len(values) // 4
        if quartile_size < 1:
            first_quartile = values[0]
            last_quartile = values[-1]
        else:
            first_quartile = sum(values[:quartile_size]) / quartile_size
            last_quartile = sum(values[-quartile_size:]) / quartile_size
        
        change_percent = ((last_quartile - first_quartile) / first_quartile) * 100
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
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
    
    def _get_resource_type(self, metric_name: str) -> str:
        """Get resource type based on metric name."""
        
        if metric_name.startswith("compute/"):
            return "gce_instance"
        elif metric_name.startswith("cloudsql/"):
            return "cloudsql_database"
        elif metric_name.startswith("redis/"):
            return "redis_instance"
        elif metric_name.startswith("kubernetes/"):
            return "k8s_container"
        elif metric_name.startswith("loadbalancer/"):
            return "http_load_balancer"
        elif metric_name.startswith("api/"):
            return "api_gateway"
        else:
            return "global"
    
    async def get_monitoring_dashboard_url(
        self,
        metric_name: str,
        resource_labels: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate URL for GCP Monitoring dashboard."""
        
        base_url = f"https://console.cloud.google.com/monitoring/metrics-explorer"
        params = [
            f"project={self.project_id}",
            f"pageState=%7B%22xyChart%22:%7B%22dataSets%22:%5B%7B%22timeSeriesFilter%22:%7B%22filter%22:%22metric.type%3D%5C%22{metric_name}%5C%22%22%7D%7D%5D%7D%7D"
        ]
        
        return f"{base_url}?{'&'.join(params)}"
    
    async def get_service_health_summary(
        self,
        service_name: str,
        time_range: str = "1h"
    ) -> Dict[str, Any]:
        """Get overall health summary for a service."""
        
        # Key metrics for service health
        key_metrics = [
            "api/request_latency",
            "api/error_rate", 
            "compute/instance/cpu_utilization",
            "compute/instance/memory_utilization"
        ]
        
        # Query all metrics
        metric_results = await self.query_multiple_metrics(
            key_metrics,
            time_range=time_range,
            labels={"service": service_name}
        )
        
        # Calculate health score
        health_score = 100.0
        health_factors = []
        
        for metric_name, data in metric_results.items():
            if data and "latest_value" in data:
                value = data["latest_value"]
                
                # Apply health scoring rules
                if metric_name == "api/request_latency":
                    if value > 5000:  # > 5 seconds
                        health_score -= 30
                        health_factors.append(f"High latency: {value:.0f}ms")
                    elif value > 2000:  # > 2 seconds
                        health_score -= 15
                        health_factors.append(f"Elevated latency: {value:.0f}ms")
                
                elif metric_name == "api/error_rate":
                    if value > 5:  # > 5%
                        health_score -= 40
                        health_factors.append(f"High error rate: {value:.1f}%")
                    elif value > 1:  # > 1%
                        health_score -= 20
                        health_factors.append(f"Elevated error rate: {value:.1f}%")
                
                elif "cpu_utilization" in metric_name:
                    if value > 90:
                        health_score -= 25
                        health_factors.append(f"High CPU: {value:.0f}%")
                    elif value > 80:
                        health_score -= 10
                        health_factors.append(f"Elevated CPU: {value:.0f}%")
                
                elif "memory_utilization" in metric_name:
                    if value > 90:
                        health_score -= 20
                        health_factors.append(f"High memory: {value:.0f}%")
                    elif value > 80:
                        health_score -= 8
                        health_factors.append(f"Elevated memory: {value:.0f}%")
        
        health_score = max(0, health_score)
        
        # Determine health status
        if health_score >= 90:
            health_status = "healthy"
        elif health_score >= 70:
            health_status = "warning"
        elif health_score >= 50:
            health_status = "degraded"
        else:
            health_status = "critical"
        
        return {
            "service_name": service_name,
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "health_factors": health_factors,
            "metrics": metric_results,
            "timestamp": datetime.utcnow().isoformat(),
            "time_range": time_range
        }