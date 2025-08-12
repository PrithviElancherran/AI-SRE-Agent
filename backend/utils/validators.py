"""
Validation utility functions for data validation, input sanitization, and schema validation.

This module provides comprehensive validation functions for the AI SRE Agent backend
to ensure data integrity and security across all components.
"""

import re
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID
from email_validator import validate_email, EmailNotValidError

from pydantic import BaseModel, ValidationError
from models.incident import IncidentSeverity, IncidentStatus
from models.playbook import PlaybookStatus
from models.user import UserRole, Permission

class ValidationResult(BaseModel):
    """Result of validation operation."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    sanitized_data: Optional[Dict[str, Any]] = None

def validate_incident_data(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate incident data for creation or update.
    
    Args:
        data: Incident data dictionary
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    sanitized_data = data.copy()
    
    # Required fields validation
    required_fields = ['title', 'description', 'severity', 'service_name']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Required field '{field}' is missing or empty")
    
    # Title validation
    if 'title' in data:
        title = str(data['title']).strip()
        if len(title) < 3:
            errors.append("Title must be at least 3 characters long")
        elif len(title) > 200:
            errors.append("Title must be no more than 200 characters long")
        sanitized_data['title'] = title
    
    # Description validation
    if 'description' in data:
        description = str(data['description']).strip()
        if len(description) < 10:
            errors.append("Description must be at least 10 characters long")
        elif len(description) > 5000:
            errors.append("Description must be no more than 5000 characters long")
        sanitized_data['description'] = description
    
    # Severity validation
    if 'severity' in data:
        try:
            severity = IncidentSeverity(data['severity'])
            sanitized_data['severity'] = severity
        except ValueError:
            valid_severities = [s.value for s in IncidentSeverity]
            errors.append(f"Invalid severity. Must be one of: {valid_severities}")
    
    # Status validation
    if 'status' in data:
        try:
            status = IncidentStatus(data['status'])
            sanitized_data['status'] = status
        except ValueError:
            valid_statuses = [s.value for s in IncidentStatus]
            errors.append(f"Invalid status. Must be one of: {valid_statuses}")
    
    # Service name validation
    if 'service_name' in data:
        service_name = str(data['service_name']).strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', service_name):
            errors.append("Service name can only contain letters, numbers, hyphens, and underscores")
        elif len(service_name) > 100:
            errors.append("Service name must be no more than 100 characters long")
        sanitized_data['service_name'] = service_name
    
    # Region validation
    if 'region' in data:
        region = str(data['region']).strip()
        if region and not re.match(r'^[a-z0-9-]+$', region):
            errors.append("Region must contain only lowercase letters, numbers, and hyphens")
        sanitized_data['region'] = region
    
    # Tags validation
    if 'tags' in data:
        tags = data['tags']
        if isinstance(tags, list):
            sanitized_tags = []
            for tag in tags:
                tag_str = str(tag).strip()
                if tag_str:
                    if len(tag_str) > 50:
                        warnings.append(f"Tag '{tag_str}' is longer than 50 characters and will be truncated")
                        tag_str = tag_str[:50]
                    sanitized_tags.append(tag_str)
            sanitized_data['tags'] = sanitized_tags
        else:
            errors.append("Tags must be a list of strings")
    
    # URL validation
    if 'external_url' in data and data['external_url']:
        url = str(data['external_url']).strip()
        if not is_valid_url(url):
            errors.append("External URL is not valid")
        sanitized_data['external_url'] = url
    
    # Escalation level validation
    if 'escalation_level' in data:
        escalation_level = data['escalation_level']
        if not isinstance(escalation_level, int) or escalation_level < 0 or escalation_level > 5:
            errors.append("Escalation level must be an integer between 0 and 5")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized_data if len(errors) == 0 else None
    )

def validate_playbook_data(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate playbook data for creation or update.
    
    Args:
        data: Playbook data dictionary
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    sanitized_data = data.copy()
    
    # Required fields validation
    required_fields = ['name', 'description', 'steps']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Required field '{field}' is missing or empty")
    
    # Name validation
    if 'name' in data:
        name = str(data['name']).strip()
        if len(name) < 3:
            errors.append("Playbook name must be at least 3 characters long")
        elif len(name) > 100:
            errors.append("Playbook name must be no more than 100 characters long")
        sanitized_data['name'] = name
    
    # Description validation
    if 'description' in data:
        description = str(data['description']).strip()
        if len(description) < 10:
            errors.append("Description must be at least 10 characters long")
        elif len(description) > 1000:
            errors.append("Description must be no more than 1000 characters long")
        sanitized_data['description'] = description
    
    # Steps validation
    if 'steps' in data:
        steps = data['steps']
        if not isinstance(steps, list):
            errors.append("Steps must be a list")
        elif len(steps) == 0:
            errors.append("Playbook must have at least one step")
        elif len(steps) > 50:
            errors.append("Playbook cannot have more than 50 steps")
        else:
            sanitized_steps = []
            for i, step in enumerate(steps):
                step_result = validate_playbook_step_data(step, i + 1)
                if not step_result.is_valid:
                    errors.extend([f"Step {i + 1}: {error}" for error in step_result.errors])
                else:
                    sanitized_steps.append(step_result.sanitized_data)
            sanitized_data['steps'] = sanitized_steps
    
    # Status validation
    if 'status' in data:
        try:
            status = PlaybookStatus(data['status'])
            sanitized_data['status'] = status
        except ValueError:
            valid_statuses = [s.value for s in PlaybookStatus]
            errors.append(f"Invalid status. Must be one of: {valid_statuses}")
    
    # Version validation
    if 'version' in data:
        version = str(data['version']).strip()
        if not re.match(r'^\d+\.\d+(\.\d+)?$', version):
            errors.append("Version must follow semantic versioning (e.g., 1.0.0)")
        sanitized_data['version'] = version
    
    # Category validation
    if 'category' in data:
        category = str(data['category']).strip()
        if len(category) > 50:
            errors.append("Category must be no more than 50 characters long")
        sanitized_data['category'] = category
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized_data if len(errors) == 0 else None
    )

def validate_playbook_step_data(data: Dict[str, Any], step_number: int) -> ValidationResult:
    """
    Validate individual playbook step data.
    
    Args:
        data: Step data dictionary
        step_number: Step number for error reporting
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    sanitized_data = data.copy()
    
    # Required fields validation
    required_fields = ['description', 'action_type']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Required field '{field}' is missing or empty")
    
    # Description validation
    if 'description' in data:
        description = str(data['description']).strip()
        if len(description) < 5:
            errors.append("Step description must be at least 5 characters long")
        elif len(description) > 500:
            errors.append("Step description must be no more than 500 characters long")
        sanitized_data['description'] = description
    
    # Action type validation
    if 'action_type' in data:
        action_type = str(data['action_type']).strip().lower()
        valid_action_types = ['query', 'check', 'execute', 'validate', 'escalate', 'notify']
        if action_type not in valid_action_types:
            errors.append(f"Invalid action type. Must be one of: {valid_action_types}")
        sanitized_data['action_type'] = action_type
    
    # Command validation
    if 'command' in data and data['command']:
        command = str(data['command']).strip()
        if len(command) > 1000:
            errors.append("Command must be no more than 1000 characters long")
        sanitized_data['command'] = command
    
    # Expected result validation
    if 'expected_result' in data and data['expected_result']:
        expected_result = str(data['expected_result']).strip()
        if len(expected_result) > 500:
            errors.append("Expected result must be no more than 500 characters long")
        sanitized_data['expected_result'] = expected_result
    
    # Timeout validation
    if 'timeout_seconds' in data:
        timeout = data['timeout_seconds']
        if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 3600:
            errors.append("Timeout must be a positive number no greater than 3600 seconds")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized_data if len(errors) == 0 else None
    )

def validate_user_data(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate user data for creation or update.
    
    Args:
        data: User data dictionary
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    sanitized_data = data.copy()
    
    # Email validation
    if 'email' in data:
        email = str(data['email']).strip().lower()
        try:
            validate_email(email)
            sanitized_data['email'] = email
        except EmailNotValidError as e:
            errors.append(f"Invalid email address: {str(e)}")
    
    # Role validation
    if 'role' in data:
        try:
            role = UserRole(data['role'])
            sanitized_data['role'] = role
        except ValueError:
            valid_roles = [r.value for r in UserRole]
            errors.append(f"Invalid role. Must be one of: {valid_roles}")
    
    # Full name validation
    if 'full_name' in data:
        full_name = str(data['full_name']).strip()
        if len(full_name) < 2:
            errors.append("Full name must be at least 2 characters long")
        elif len(full_name) > 100:
            errors.append("Full name must be no more than 100 characters long")
        elif not re.match(r'^[a-zA-Z\s\-\.\']+$', full_name):
            errors.append("Full name can only contain letters, spaces, hyphens, dots, and apostrophes")
        sanitized_data['full_name'] = full_name
    
    # Password validation (if provided)
    if 'password' in data and data['password']:
        password = str(data['password'])
        password_errors = validate_password(password)
        if password_errors:
            errors.extend(password_errors)
    
    # Permissions validation
    if 'permissions' in data:
        permissions = data['permissions']
        if isinstance(permissions, list):
            valid_permissions = []
            for perm in permissions:
                try:
                    permission = Permission(perm)
                    valid_permissions.append(permission)
                except ValueError:
                    errors.append(f"Invalid permission: {perm}")
            sanitized_data['permissions'] = valid_permissions
        else:
            errors.append("Permissions must be a list")
    
    # Timezone validation
    if 'timezone' in data:
        timezone = str(data['timezone']).strip()
        if not is_valid_timezone(timezone):
            warnings.append(f"Timezone '{timezone}' may not be valid")
        sanitized_data['timezone'] = timezone
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized_data if len(errors) == 0 else None
    )

def validate_gcp_metric_data(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate GCP metric data.
    
    Args:
        data: GCP metric data dictionary
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    sanitized_data = data.copy()
    
    # Required fields validation
    required_fields = ['metric_name', 'value', 'timestamp']
    for field in required_fields:
        if field not in data:
            errors.append(f"Required field '{field}' is missing")
    
    # Metric name validation
    if 'metric_name' in data:
        metric_name = str(data['metric_name']).strip()
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_/\.]+$', metric_name):
            errors.append("Metric name must start with a letter and contain only letters, numbers, underscores, dots, and slashes")
        sanitized_data['metric_name'] = metric_name
    
    # Value validation
    if 'value' in data:
        try:
            value = float(data['value'])
            sanitized_data['value'] = value
        except (ValueError, TypeError):
            errors.append("Metric value must be a number")
    
    # Timestamp validation
    if 'timestamp' in data:
        if not is_valid_timestamp(data['timestamp']):
            errors.append("Invalid timestamp format")
    
    # Labels validation
    if 'labels' in data:
        labels = data['labels']
        if isinstance(labels, dict):
            sanitized_labels = {}
            for key, value in labels.items():
                key_str = str(key).strip()
                value_str = str(value).strip()
                if len(key_str) > 100:
                    warnings.append(f"Label key '{key_str}' is longer than 100 characters")
                if len(value_str) > 100:
                    warnings.append(f"Label value '{value_str}' is longer than 100 characters")
                sanitized_labels[key_str] = value_str
            sanitized_data['labels'] = sanitized_labels
        else:
            errors.append("Labels must be a dictionary")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized_data if len(errors) == 0 else None
    )

def validate_json_data(data: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate and parse JSON data.
    
    Args:
        data: JSON string to validate
        
    Returns:
        Tuple of (is_valid, parsed_data, error_message)
    """
    try:
        parsed_data = json.loads(data)
        return True, parsed_data, None
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"

def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID string format.
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid UUID format
    """
    try:
        UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False

def validate_password(password: str) -> List[str]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if len(password) > 128:
        errors.append("Password must be no more than 128 characters long")
    
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one digit")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    
    return errors

def is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL format
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None

def is_valid_timestamp(timestamp: Union[str, datetime, int, float]) -> bool:
    """
    Validate timestamp format.
    
    Args:
        timestamp: Timestamp to validate
        
    Returns:
        True if valid timestamp
    """
    if isinstance(timestamp, datetime):
        return True
    
    if isinstance(timestamp, (int, float)):
        try:
            datetime.fromtimestamp(timestamp)
            return True
        except (ValueError, OSError):
            return False
    
    if isinstance(timestamp, str):
        # Try ISO format
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except ValueError:
            pass
        
        # Try timestamp format
        try:
            float(timestamp)
            return True
        except ValueError:
            return False
    
    return False

def is_valid_timezone(timezone: str) -> bool:
    """
    Validate timezone string.
    
    Args:
        timezone: Timezone string to validate
        
    Returns:
        True if valid timezone
    """
    # Basic timezone validation - in production, use pytz for comprehensive validation
    common_timezones = [
        'UTC', 'GMT', 'EST', 'PST', 'CST', 'MST',
        'America/New_York', 'America/Los_Angeles', 'America/Chicago',
        'Europe/London', 'Europe/Paris', 'Asia/Tokyo', 'Asia/Shanghai'
    ]
    
    if timezone in common_timezones:
        return True
    
    # Check for offset format like +05:30 or -08:00
    if re.match(r'^[+-]\d{2}:\d{2}$', timezone):
        return True
    
    # Check for timezone name format
    if re.match(r'^[A-Za-z]+/[A-Za-z_]+$', timezone):
        return True
    
    return False

def sanitize_string(text: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """
    Sanitize string input.
    
    Args:
        text: String to sanitize
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags
        
    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip whitespace
    text = text.strip()
    
    # Remove HTML tags if not allowed
    if not allow_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
    
    return text

def validate_confidence_score(score: Union[int, float]) -> bool:
    """
    Validate confidence score is between 0 and 1.
    
    Args:
        score: Confidence score to validate
        
    Returns:
        True if valid confidence score
    """
    try:
        score_float = float(score)
        return 0.0 <= score_float <= 1.0
    except (ValueError, TypeError):
        return False

def validate_severity_level(severity: str) -> bool:
    """
    Validate incident severity level.
    
    Args:
        severity: Severity level to validate
        
    Returns:
        True if valid severity level
    """
    try:
        IncidentSeverity(severity)
        return True
    except ValueError:
        return False

def validate_metric_threshold(threshold: Dict[str, Any]) -> ValidationResult:
    """
    Validate metric threshold configuration.
    
    Args:
        threshold: Threshold configuration dictionary
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    sanitized_data = threshold.copy()
    
    # Required fields
    required_fields = ['metric_name', 'operator', 'value']
    for field in required_fields:
        if field not in threshold:
            errors.append(f"Required field '{field}' is missing")
    
    # Operator validation
    if 'operator' in threshold:
        valid_operators = ['>', '<', '>=', '<=', '==', '!=']
        if threshold['operator'] not in valid_operators:
            errors.append(f"Invalid operator. Must be one of: {valid_operators}")
    
    # Value validation
    if 'value' in threshold:
        try:
            float(threshold['value'])
        except (ValueError, TypeError):
            errors.append("Threshold value must be a number")
    
    # Duration validation
    if 'duration_minutes' in threshold:
        duration = threshold['duration_minutes']
        if not isinstance(duration, (int, float)) or duration <= 0:
            errors.append("Duration must be a positive number")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized_data if len(errors) == 0 else None
    )

def validate_analysis_request(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate analysis request data.
    
    Args:
        data: Analysis request data dictionary
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    sanitized_data = data.copy()
    
    # Required fields validation
    required_fields = ['incident_id', 'analysis_type']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Required field '{field}' is missing or empty")
    
    # Incident ID validation
    if 'incident_id' in data:
        incident_id = str(data['incident_id']).strip()
        if not incident_id:
            errors.append("Incident ID cannot be empty")
        elif len(incident_id) > 100:
            errors.append("Incident ID must be no more than 100 characters long")
        sanitized_data['incident_id'] = incident_id
    
    # Analysis type validation
    if 'analysis_type' in data:
        analysis_type = str(data['analysis_type']).strip().lower()
        valid_types = ['correlation', 'root_cause', 'playbook', 'trend', 'prediction']
        if analysis_type not in valid_types:
            errors.append(f"Invalid analysis type. Must be one of: {valid_types}")
        sanitized_data['analysis_type'] = analysis_type
    
    # Priority validation
    if 'priority' in data:
        priority = data['priority']
        if isinstance(priority, str):
            priority = priority.lower()
            valid_priorities = ['low', 'medium', 'high', 'urgent']
            if priority not in valid_priorities:
                errors.append(f"Invalid priority. Must be one of: {valid_priorities}")
        elif isinstance(priority, int):
            if priority < 1 or priority > 4:
                errors.append("Priority must be between 1 and 4")
        else:
            errors.append("Priority must be a string or integer")
    
    # Parameters validation
    if 'parameters' in data:
        parameters = data['parameters']
        if isinstance(parameters, dict):
            # Validate specific parameter types
            if 'similarity_threshold' in parameters:
                threshold = parameters['similarity_threshold']
                if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                    errors.append("Similarity threshold must be a number between 0 and 1")
            
            if 'time_range_hours' in parameters:
                time_range = parameters['time_range_hours']
                if not isinstance(time_range, (int, float)) or time_range <= 0 or time_range > 8760:
                    errors.append("Time range must be a positive number no greater than 8760 hours")
            
            if 'max_results' in parameters:
                max_results = parameters['max_results']
                if not isinstance(max_results, int) or max_results < 1 or max_results > 1000:
                    errors.append("Max results must be an integer between 1 and 1000")
        else:
            errors.append("Parameters must be a dictionary")
    
    # Auto-execute validation
    if 'auto_execute' in data:
        auto_execute = data['auto_execute']
        if not isinstance(auto_execute, bool):
            errors.append("Auto execute must be a boolean value")
    
    # Timeout validation
    if 'timeout_seconds' in data:
        timeout = data['timeout_seconds']
        if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 3600:
            errors.append("Timeout must be a positive number no greater than 3600 seconds")
    
    # Callback URL validation
    if 'callback_url' in data and data['callback_url']:
        callback_url = str(data['callback_url']).strip()
        if not is_valid_url(callback_url):
            errors.append("Callback URL is not valid")
        sanitized_data['callback_url'] = callback_url
    
    # Tags validation
    if 'tags' in data:
        tags = data['tags']
        if isinstance(tags, list):
            sanitized_tags = []
            for tag in tags:
                tag_str = str(tag).strip()
                if tag_str:
                    if len(tag_str) > 50:
                        warnings.append(f"Tag '{tag_str}' is longer than 50 characters and will be truncated")
                        tag_str = tag_str[:50]
                    sanitized_tags.append(tag_str)
            sanitized_data['tags'] = sanitized_tags
        else:
            errors.append("Tags must be a list of strings")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_data=sanitized_data if len(errors) == 0 else None
    )
