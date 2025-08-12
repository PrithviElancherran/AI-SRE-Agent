"""
Utility functions for formatting confidence scores, durations, and other data display formats.

This module provides standardized formatting functions for the AI SRE Agent backend
to ensure consistent data presentation across all components.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any, List
from decimal import Decimal, ROUND_HALF_UP

def format_confidence_score(score: float, precision: int = 2, as_percentage: bool = True) -> str:
    """
    Format confidence score for display.
    
    Args:
        score: Confidence score between 0 and 1
        precision: Number of decimal places
        as_percentage: Whether to format as percentage
        
    Returns:
        Formatted confidence score string
    """
    if score is None:
        return "N/A"
    
    # Ensure score is within valid range
    score = max(0.0, min(1.0, float(score)))
    
    if as_percentage:
        percentage = score * 100
        return f"{percentage:.{precision}f}%"
    else:
        return f"{score:.{precision}f}"

def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format a decimal value as a percentage.
    
    Args:
        value: Decimal value (0.0 to 1.0)
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    percentage = float(value) * 100
    return f"{percentage:.{precision}f}%"

def format_duration(
    duration_seconds: Optional[Union[int, float]], 
    include_seconds: bool = True,
    compact: bool = False
) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        duration_seconds: Duration in seconds
        include_seconds: Whether to include seconds in output
        compact: Whether to use compact format (1h 2m vs 1 hour 2 minutes)
        
    Returns:
        Formatted duration string
    """
    if duration_seconds is None:
        return "N/A"
    
    duration_seconds = abs(float(duration_seconds))
    
    if duration_seconds == 0:
        return "0s" if compact else "0 seconds"
    
    # Calculate time components
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    milliseconds = int((duration_seconds % 1) * 1000)
    
    parts = []
    
    if hours > 0:
        if compact:
            parts.append(f"{hours}h")
        else:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    
    if minutes > 0:
        if compact:
            parts.append(f"{minutes}m")
        else:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    
    if include_seconds and (seconds > 0 or (not parts and milliseconds > 0)):
        if milliseconds > 0 and duration_seconds < 60:
            # Show milliseconds for durations under 1 minute
            total_ms = seconds * 1000 + milliseconds
            if compact:
                parts.append(f"{total_ms}ms")
            else:
                parts.append(f"{total_ms} milliseconds")
        else:
            if compact:
                parts.append(f"{seconds}s")
            else:
                parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    
    if not parts:
        if compact:
            return f"{milliseconds}ms"
        else:
            return f"{milliseconds} milliseconds"
    
    return " ".join(parts)

def format_bytes(bytes_value: Union[int, float], decimal_places: int = 2) -> str:
    """
    Format bytes to human-readable string with appropriate units.
    
    Args:
        bytes_value: Number of bytes
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted bytes string (e.g., "1.5 GB")
    """
    if bytes_value is None:
        return "N/A"
    
    bytes_value = abs(float(bytes_value))
    
    if bytes_value == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    
    while bytes_value >= 1024 and unit_index < len(units) - 1:
        bytes_value /= 1024
        unit_index += 1
    
    if unit_index == 0:  # Bytes
        return f"{int(bytes_value)} {units[unit_index]}"
    else:
        return f"{bytes_value:.{decimal_places}f} {units[unit_index]}"

def format_metric_value(
    value: Union[int, float, str], 
    unit: Optional[str] = None,
    precision: int = 2
) -> str:
    """
    Format metric value with appropriate precision and unit.
    
    Args:
        value: Metric value
        unit: Unit of measurement
        precision: Number of decimal places
        
    Returns:
        Formatted metric value string
    """
    if value is None:
        return "N/A"
    
    if isinstance(value, str):
        return f"{value} {unit}" if unit else value
    
    # Handle numeric values
    numeric_value = float(value)
    
    # Format based on magnitude
    if abs(numeric_value) >= 1000000:
        formatted = f"{numeric_value / 1000000:.{precision}f}M"
    elif abs(numeric_value) >= 1000:
        formatted = f"{numeric_value / 1000:.{precision}f}K"
    elif numeric_value == int(numeric_value):
        formatted = str(int(numeric_value))
    else:
        formatted = f"{numeric_value:.{precision}f}"
    
    return f"{formatted} {unit}" if unit else formatted

def format_timestamp(
    timestamp: Union[str, datetime], 
    format_type: str = "relative",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Timestamp string or datetime object
        format_type: "relative", "absolute", or "both"
        date_format: Format string for absolute timestamps
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        return "N/A"
    
    # Convert to datetime if string
    if isinstance(timestamp, str):
        try:
            # Try ISO format first
            if 'T' in timestamp:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return str(timestamp)
    
    now = datetime.utcnow().replace(tzinfo=timestamp.tzinfo) if timestamp.tzinfo else datetime.utcnow()
    
    if format_type == "relative":
        return format_time_ago(timestamp, now)
    elif format_type == "absolute":
        return timestamp.strftime(date_format)
    elif format_type == "both":
        relative = format_time_ago(timestamp, now)
        absolute = timestamp.strftime(date_format)
        return f"{relative} ({absolute})"
    else:
        return timestamp.strftime(date_format)

def format_time_ago(timestamp: datetime, now: Optional[datetime] = None) -> str:
    """
    Format timestamp as "time ago" string.
    
    Args:
        timestamp: Target timestamp
        now: Current timestamp (defaults to now)
        
    Returns:
        Relative time string (e.g., "2 minutes ago")
    """
    if now is None:
        now = datetime.utcnow().replace(tzinfo=timestamp.tzinfo) if timestamp.tzinfo else datetime.utcnow()
    
    diff = now - timestamp
    
    if diff.total_seconds() < 0:
        return "in the future"
    
    total_seconds = int(diff.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds} second{'s' if total_seconds != 1 else ''} ago"
    
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    
    days = hours // 24
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    
    years = months // 12
    return f"{years} year{'s' if years != 1 else ''} ago"

def format_error_rate(errors: int, total: int, precision: int = 2) -> str:
    """
    Format error rate as percentage.
    
    Args:
        errors: Number of errors
        total: Total number of requests
        precision: Number of decimal places
        
    Returns:
        Formatted error rate string
    """
    if total == 0:
        return "0.00%"
    
    rate = (errors / total) * 100
    return f"{rate:.{precision}f}%"

def format_incident_id(incident_id: str) -> str:
    """
    Format incident ID for consistent display.
    
    Args:
        incident_id: Raw incident ID
        
    Returns:
        Formatted incident ID
    """
    if not incident_id:
        return "N/A"
    
    # Ensure consistent format: INC-YYYY-NNN
    if not incident_id.startswith("INC-"):
        return f"INC-{incident_id}"
    
    return incident_id.upper()

def format_severity_badge(severity: str) -> Dict[str, str]:
    """
    Get formatting information for severity badge.
    
    Args:
        severity: Severity level
        
    Returns:
        Dictionary with color, background, and text formatting
    """
    severity_lower = severity.lower() if severity else "unknown"
    
    severity_formats = {
        "critical": {
            "color": "text-red-900",
            "background": "bg-red-100",
            "border": "border-red-300",
            "icon": "🔴"
        },
        "high": {
            "color": "text-orange-900",
            "background": "bg-orange-100",
            "border": "border-orange-300",
            "icon": "🟠"
        },
        "medium": {
            "color": "text-yellow-900",
            "background": "bg-yellow-100",
            "border": "border-yellow-300",
            "icon": "🟡"
        },
        "low": {
            "color": "text-green-900",
            "background": "bg-green-100",
            "border": "border-green-300",
            "icon": "🟢"
        },
        "unknown": {
            "color": "text-gray-900",
            "background": "bg-gray-100",
            "border": "border-gray-300",
            "icon": "⚪"
        }
    }
    
    return severity_formats.get(severity_lower, severity_formats["unknown"])

def format_status_badge(status: str) -> Dict[str, str]:
    """
    Get formatting information for status badge.
    
    Args:
        status: Status value
        
    Returns:
        Dictionary with color, background, and text formatting
    """
    status_lower = status.lower() if status else "unknown"
    
    status_formats = {
        "open": {
            "color": "text-red-900",
            "background": "bg-red-100",
            "border": "border-red-300",
            "icon": "🔴"
        },
        "investigating": {
            "color": "text-yellow-900",
            "background": "bg-yellow-100",
            "border": "border-yellow-300",
            "icon": "🔍"
        },
        "identified": {
            "color": "text-blue-900",
            "background": "bg-blue-100",
            "border": "border-blue-300",
            "icon": "🎯"
        },
        "monitoring": {
            "color": "text-purple-900",
            "background": "bg-purple-100",
            "border": "border-purple-300",
            "icon": "👁️"
        },
        "resolved": {
            "color": "text-green-900",
            "background": "bg-green-100",
            "border": "border-green-300",
            "icon": "✅"
        },
        "closed": {
            "color": "text-gray-900",
            "background": "bg-gray-100",
            "border": "border-gray-300",
            "icon": "✅"
        }
    }
    
    return status_formats.get(status_lower, status_formats.get("unknown", {
        "color": "text-gray-900",
        "background": "bg-gray-100",
        "border": "border-gray-300",
        "icon": "❓"
    }))

def format_list_display(items: List[Any], max_items: int = 3, separator: str = ", ") -> str:
    """
    Format list for display with truncation.
    
    Args:
        items: List of items to display
        max_items: Maximum number of items to show
        separator: Separator between items
        
    Returns:
        Formatted list string
    """
    if not items:
        return "None"
    
    str_items = [str(item) for item in items]
    
    if len(str_items) <= max_items:
        return separator.join(str_items)
    else:
        displayed = str_items[:max_items]
        remaining = len(str_items) - max_items
        return f"{separator.join(displayed)}, +{remaining} more"

def format_json_preview(data: Dict[Any, Any], max_length: int = 100) -> str:
    """
    Format JSON data for preview display.
    
    Args:
        data: Dictionary to format
        max_length: Maximum character length
        
    Returns:
        Formatted JSON preview string
    """
    if not data:
        return "{}"
    
    import json
    
    try:
        json_str = json.dumps(data, separators=(',', ':'))
        if len(json_str) <= max_length:
            return json_str
        else:
            return json_str[:max_length - 3] + "..."
    except (TypeError, ValueError):
        return str(data)[:max_length]

def sanitize_string(text: str, max_length: int = 500) -> str:
    """
    Sanitize string for safe display.
    
    Args:
        text: Input text
        max_length: Maximum length to allow
        
    Returns:
        Sanitized string
    """
    if not text:
        return ""
    
    # Remove any potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', str(text))
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 3] + "..."
    
    return sanitized

def format_url_display(url: str, max_length: int = 50) -> str:
    """
    Format URL for display with truncation.
    
    Args:
        url: URL to format
        max_length: Maximum display length
        
    Returns:
        Formatted URL string
    """
    if not url:
        return "N/A"
    
    if len(url) <= max_length:
        return url
    
    # Try to keep the important parts (domain and path)
    if url.startswith(('http://', 'https://')):
        protocol = url.split('://')[0] + '://'
        rest = url[len(protocol):]
        
        if len(rest) <= max_length - len(protocol) - 3:
            return url
        
        # Truncate middle part
        keep_start = (max_length - len(protocol) - 6) // 2
        keep_end = max_length - len(protocol) - keep_start - 6
        
        return f"{protocol}{rest[:keep_start]}...{rest[-keep_end:]}"
    
    # Non-HTTP URL
    if len(url) > max_length:
        return f"{url[:max_length-3]}..."
    
    return url

def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """
    Return pluralized form based on count.
    
    Args:
        count: Number to check
        singular: Singular form
        plural: Plural form (defaults to singular + 's')
        
    Returns:
        Appropriate form based on count
    """
    if count == 1:
        return f"{count} {singular}"
    else:
        plural_form = plural if plural is not None else f"{singular}s"
        return f"{count} {plural_form}"

def format_incident_response(incident_data: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Format incident data for API response.
    
    Args:
        incident_data: Raw incident data dictionary
        
    Returns:
        Formatted incident response dictionary
    """
    if not incident_data:
        return {}
    
    formatted = {
        'incident_id': incident_data.get('incident_id', 'N/A'),
        'title': incident_data.get('title', 'Untitled Incident'),
        'description': incident_data.get('description', ''),
        'severity': incident_data.get('severity', 'medium'),
        'status': incident_data.get('status', 'open'),
        'service_name': incident_data.get('service_name', 'unknown'),
        'region': incident_data.get('region', 'unknown'),
        'created_at': format_timestamp(incident_data.get('created_at')),
        'updated_at': format_timestamp(incident_data.get('updated_at')),
        'resolved_at': format_timestamp(incident_data.get('resolved_at')),
        'mttr_minutes': incident_data.get('mttr_minutes'),
        'affected_users': incident_data.get('affected_users'),
        'tags': incident_data.get('tags', []),
        'severity_display': format_severity_badge(incident_data.get('severity', 'medium')),
        'status_display': format_status_badge(incident_data.get('status', 'open'))
    }
    
    return formatted

def format_analysis_response(analysis_data: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Format analysis data for API response.
    
    Args:
        analysis_data: Raw analysis data dictionary
        
    Returns:
        Formatted analysis response dictionary
    """
    if not analysis_data:
        return {}
    
    formatted = {
        'analysis_id': analysis_data.get('analysis_id', 'N/A'),
        'incident_id': analysis_data.get('incident_id', 'N/A'),
        'analysis_type': analysis_data.get('analysis_type', 'unknown'),
        'confidence_score': format_confidence_score(analysis_data.get('confidence_score', 0)),
        'confidence_display': format_percentage(analysis_data.get('confidence_score', 0)),
        'status': analysis_data.get('status', 'pending'),
        'reasoning_steps': analysis_data.get('reasoning_steps', []),
        'recommendations': analysis_data.get('recommendations', []),
        'evidence_items': analysis_data.get('evidence_items', []),
        'root_cause': analysis_data.get('root_cause'),
        'created_at': format_timestamp(analysis_data.get('created_at')),
        'updated_at': format_timestamp(analysis_data.get('updated_at')),
        'analyzed_by': analysis_data.get('analyzed_by', 'ai_sre_agent'),
        'duration': format_duration(analysis_data.get('duration_seconds'))
    }
    
    return formatted

def format_playbook_response(playbook_data: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Format playbook data for API response.
    
    Args:
        playbook_data: Raw playbook data dictionary
        
    Returns:
        Formatted playbook response dictionary
    """
    if not playbook_data:
        return {}
    
    formatted = {
        'playbook_id': playbook_data.get('playbook_id', 'N/A'),
        'name': playbook_data.get('name', 'Untitled Playbook'),
        'description': playbook_data.get('description', ''),
        'version': playbook_data.get('version', '1.0'),
        'category': playbook_data.get('category', 'general'),
        'status': playbook_data.get('status', 'active'),
        'created_at': format_timestamp(playbook_data.get('created_at')),
        'updated_at': format_timestamp(playbook_data.get('updated_at')),
        'created_by': playbook_data.get('created_by', 'unknown'),
        'author': playbook_data.get('author', 'SRE Team'),
        'effectiveness_score': format_percentage(playbook_data.get('effectiveness_score', 0)),
        'success_rate': format_percentage(playbook_data.get('success_rate', 0)),
        'usage_count': playbook_data.get('usage_count', 0),
        'estimated_duration_minutes': playbook_data.get('estimated_duration_minutes', 0),
        'last_executed_at': format_timestamp(playbook_data.get('last_executed_at')),
        'steps_count': len(playbook_data.get('steps', [])),
        'tags': playbook_data.get('tags', []),
        'target_severity': playbook_data.get('target_severity', []),
        'is_active': playbook_data.get('is_active', True)
    }
    
    return formatted

def format_execution_response(execution_data: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Format playbook execution data for API response.
    
    Args:
        execution_data: Raw execution data dictionary
        
    Returns:
        Formatted execution response dictionary
    """
    if not execution_data:
        return {}
    
    formatted = {
        'execution_id': execution_data.get('execution_id', 'N/A'),
        'playbook_id': execution_data.get('playbook_id', 'N/A'),
        'incident_id': execution_data.get('incident_id', 'N/A'),
        'status': execution_data.get('status', 'pending'),
        'execution_mode': execution_data.get('execution_mode', 'manual'),
        'current_step_number': execution_data.get('current_step_number', 1),
        'total_steps': len(execution_data.get('step_results', [])),
        'progress': execution_data.get('progress', 0),
        'started_at': format_timestamp(execution_data.get('started_at')),
        'completed_at': format_timestamp(execution_data.get('completed_at')),
        'duration': format_duration(execution_data.get('duration_seconds')),
        'executed_by': execution_data.get('executed_by', 'unknown'),
        'root_cause_found': execution_data.get('root_cause_found', False),
        'confidence_score': format_confidence_score(execution_data.get('confidence_score', 0)),
        'confidence_display': format_percentage(execution_data.get('confidence_score', 0)),
        'recommendations': execution_data.get('recommendations', []),
        'step_results': execution_data.get('step_results', []),
        'error_message': execution_data.get('error_message'),
        'retry_count': execution_data.get('retry_count', 0),
        'escalation_triggered': execution_data.get('escalation_triggered', False)
    }
    
    return formatted
