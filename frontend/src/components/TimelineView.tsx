/**
 * React TypeScript component for displaying incident timelines.
 * 
 * This component provides chronological events, analysis progression,
 * and interactive timeline visualization for the AI SRE Agent frontend.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ClockIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  PlayIcon,
  PauseIcon,
  UserIcon,
  CogIcon,
  DocumentTextIcon,
  BugAntIcon,
  ServerIcon,
  EyeIcon,
  ArrowRightIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  CalendarIcon,
  ListBulletIcon,
  Squares2X2Icon
} from '@heroicons/react/24/outline';
import {
  CheckCircleIcon as CheckCircleIconSolid,
  ExclamationTriangleIcon as ExclamationTriangleIconSolid,
  XCircleIcon as XCircleIconSolid
} from '@heroicons/react/24/solid';
import { format, formatDistanceToNow, parseISO, isValid } from 'date-fns';
import { Incident, IncidentTimelineEvent } from '../types/incident';
import ConfidenceMeter from './ConfidenceMeter';

interface TimelineViewProps {
  incident: Incident;
  detailed?: boolean;
  className?: string;
  onEventClick?: (event: IncidentTimelineEvent) => void;
  showFilters?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface TimelineFilter {
  eventTypes: string[];
  severity: string[];
  sources: string[];
  dateRange: 'all' | '1h' | '6h' | '24h' | '7d';
  searchTerm: string;
}

interface TimelineGroup {
  date: string;
  events: IncidentTimelineEvent[];
}

const TimelineView: React.FC<TimelineViewProps> = ({
  incident,
  detailed = false,
  className = '',
  onEventClick,
  showFilters = false,
  autoRefresh = false,
  refreshInterval = 30000
}) => {
  const [timelineEvents, setTimelineEvents] = useState<IncidentTimelineEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<TimelineFilter>({
    eventTypes: [],
    severity: [],
    sources: [],
    dateRange: 'all',
    searchTerm: ''
  });
  const [viewMode, setViewMode] = useState<'list' | 'grouped'>('grouped');
  const [selectedEvent, setSelectedEvent] = useState<IncidentTimelineEvent | null>(null);
  const [showEventDetails, setShowEventDetails] = useState(false);

  // Generate timeline events from incident data and analysis
  const generateTimelineEvents = useCallback((): IncidentTimelineEvent[] => {
    const events: IncidentTimelineEvent[] = [];

    // Incident creation event
    events.push({
      eventId: `incident_created_${incident.incidentId}`,
      timestamp: incident.createdAt,
      eventType: 'created',
      description: `Incident ${incident.incidentId} created: ${incident.title}`,
      actor: incident.createdBy,
      severity: incident.severity === 'critical' ? 'critical' : 
                incident.severity === 'high' ? 'error' :
                incident.severity === 'medium' ? 'warning' : 'info',
      source: 'incident_management',
      details: {
        incidentId: incident.incidentId,
        title: incident.title,
        description: incident.description,
        severity: incident.severity,
        serviceName: incident.serviceName,
        region: incident.region
      }
    });

    // Status change events
    if (incident.acknowledgedAt) {
      events.push({
        eventId: `incident_acknowledged_${incident.incidentId}`,
        timestamp: incident.acknowledgedAt,
        eventType: 'updated',
        description: 'Incident acknowledged and investigation started',
        actor: incident.assignedTo || 'system',
        severity: 'info',
        source: 'incident_management',
        details: {
          status: 'investigating',
          assignedTo: incident.assignedTo
        }
      });
    }

    // Resolution event
    if (incident.resolvedAt) {
      events.push({
        eventId: `incident_resolved_${incident.incidentId}`,
        timestamp: incident.resolvedAt,
        eventType: 'resolved',
        description: `Incident resolved: ${incident.resolution || 'Issue fixed'}`,
        actor: incident.assignedTo || 'system',
        severity: 'info',
        source: 'incident_management',
        details: {
          status: 'resolved',
          resolution: incident.resolution,
          rootCause: incident.rootCause,
          mttrMinutes: incident.mttrMinutes
        }
      });
    }

    // Analysis events (synthetic)
    const analysisStartTime = new Date(new Date(incident.createdAt).getTime() + 2 * 60 * 1000).toISOString();
    events.push({
      eventId: `analysis_started_${incident.incidentId}`,
      timestamp: analysisStartTime,
      eventType: 'analysis_started',
      description: 'AI analysis initiated - searching for similar incidents',
      actor: 'ai_sre_agent',
      severity: 'info',
      source: 'ai_analysis',
      details: {
        analysisType: 'correlation',
        status: 'started'
      }
    });

    // Evidence collection events
    const evidenceTime = new Date(new Date(analysisStartTime).getTime() + 30 * 1000).toISOString();
    events.push({
      eventId: `evidence_collected_${incident.incidentId}`,
      timestamp: evidenceTime,
      eventType: 'analysis_completed',
      description: 'Evidence collected from GCP observability tools',
      actor: 'ai_sre_agent',
      severity: 'info',
      source: 'gcp_monitoring',
      details: {
        evidenceTypes: ['metrics', 'logs', 'traces'],
        sources: ['monitoring', 'logging', 'error_reporting'],
        dataPoints: 156
      }
    });

    // Pattern detection
    const patternTime = new Date(new Date(evidenceTime).getTime() + 45 * 1000).toISOString();
    events.push({
      eventId: `pattern_detected_${incident.incidentId}`,
      timestamp: patternTime,
      eventType: 'analysis_completed',
      description: 'Similar incident patterns detected with 92% confidence',
      actor: 'ai_sre_agent',
      severity: 'info',
      source: 'pattern_analysis',
      details: {
        similarIncidents: ['INC-2024-045', 'INC-2024-072'],
        confidence: 0.92,
        patterns: ['redis_memory_pressure', 'cache_hit_rate_drop']
      }
    });

    // Root cause identification
    if (incident.rootCause) {
      const rootCauseTime = new Date(new Date(patternTime).getTime() + 60 * 1000).toISOString();
      events.push({
        eventId: `root_cause_identified_${incident.incidentId}`,
        timestamp: rootCauseTime,
        eventType: 'analysis_completed',
        description: `Root cause identified: ${incident.rootCause}`,
        actor: 'ai_sre_agent',
        severity: 'warning',
        source: 'root_cause_analysis',
        details: {
          rootCause: incident.rootCause,
          confidence: 0.89,
          evidence: ['memory_usage_spike', 'cache_eviction_increase', 'latency_correlation']
        }
      });
    }

    // System metrics events (synthetic)
    const metricsEvents = [
      {
        timestamp: new Date(new Date(incident.createdAt).getTime() + 1 * 60 * 1000).toISOString(),
        description: 'High memory usage detected: 95% utilization on Redis nodes',
        source: 'gcp_monitoring',
        severity: 'warning' as const,
        details: { metric: 'memory_utilization', value: 95, threshold: 80 }
      },
      {
        timestamp: new Date(new Date(incident.createdAt).getTime() + 3 * 60 * 1000).toISOString(),
        description: 'Cache hit rate dropped to 38% (normal: >90%)',
        source: 'gcp_monitoring',
        severity: 'error' as const,
        details: { metric: 'cache_hit_rate', value: 38, threshold: 90 }
      },
      {
        timestamp: new Date(new Date(incident.createdAt).getTime() + 4 * 60 * 1000).toISOString(),
        description: 'API latency P95 increased to 6.2s (threshold: 2.0s)',
        source: 'gcp_monitoring',
        severity: 'critical' as const,
        details: { metric: 'api_latency_p95', value: 6.2, threshold: 2.0 }
      }
    ];

    metricsEvents.forEach((metric, index) => {
      events.push({
        eventId: `metric_${incident.incidentId}_${index}`,
        timestamp: metric.timestamp,
        eventType: 'escalated',
        description: metric.description,
        actor: 'monitoring_system',
        severity: metric.severity,
        source: metric.source,
        details: metric.details
      });
    });

    // Sort events by timestamp
    return events.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }, [incident]);

  // Load timeline events
  useEffect(() => {
    setLoading(true);
    try {
      const events = generateTimelineEvents();
      setTimelineEvents(events);
    } catch (error) {
      console.error('Failed to generate timeline events:', error);
    } finally {
      setLoading(false);
    }
  }, [generateTimelineEvents]);

  // Auto-refresh timeline
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      const events = generateTimelineEvents();
      setTimelineEvents(events);
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, generateTimelineEvents]);

  // Filter timeline events
  const filteredEvents = useMemo(() => {
    return timelineEvents.filter(event => {
      // Event type filter
      if (filters.eventTypes.length > 0 && !filters.eventTypes.includes(event.eventType)) {
        return false;
      }

      // Severity filter
      if (filters.severity.length > 0 && !filters.severity.includes(event.severity || 'info')) {
        return false;
      }

      // Source filter
      if (filters.sources.length > 0 && !filters.sources.includes(event.source || 'unknown')) {
        return false;
      }

      // Date range filter
      if (filters.dateRange !== 'all') {
        const eventDate = new Date(event.timestamp);
        const now = new Date();
        const hours = {
          '1h': 1,
          '6h': 6,
          '24h': 24,
          '7d': 24 * 7
        }[filters.dateRange] || 0;

        const diff = (now.getTime() - eventDate.getTime()) / (1000 * 60 * 60);
        if (diff > hours) return false;
      }

      // Search filter
      if (filters.searchTerm) {
        const searchLower = filters.searchTerm.toLowerCase();
        const searchableText = `${event.description} ${event.actor} ${event.source}`.toLowerCase();
        if (!searchableText.includes(searchLower)) {
          return false;
        }
      }

      return true;
    });
  }, [timelineEvents, filters]);

  // Group events by date
  const groupedEvents = useMemo((): TimelineGroup[] => {
    const groups: { [key: string]: IncidentTimelineEvent[] } = {};

    filteredEvents.forEach(event => {
      const dateKey = format(new Date(event.timestamp), 'yyyy-MM-dd');
      if (!groups[dateKey]) {
        groups[dateKey] = [];
      }
      groups[dateKey].push(event);
    });

    return Object.entries(groups)
      .map(([date, events]) => ({ date, events }))
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  }, [filteredEvents]);

  // Get unique filter options
  const filterOptions = useMemo(() => {
    const eventTypes = Array.from(new Set(timelineEvents.map(e => e.eventType)));
    const severities = Array.from(new Set(timelineEvents.map(e => e.severity || 'info')));
    const sources = Array.from(new Set(timelineEvents.map(e => e.source || 'unknown')));

    return { eventTypes, severities, sources };
  }, [timelineEvents]);

  // Event icon mapping
  const getEventIcon = useCallback((event: IncidentTimelineEvent) => {
    const iconClass = "w-5 h-5";

    switch (event.eventType) {
      case 'created':
        return <ExclamationTriangleIconSolid className={`${iconClass} text-orange-500`} />;
      case 'updated':
        return <CogIcon className={`${iconClass} text-blue-500`} />;
      case 'assigned':
        return <UserIcon className={`${iconClass} text-purple-500`} />;
      case 'escalated':
        return <ExclamationTriangleIconSolid className={`${iconClass} text-red-500`} />;
      case 'resolved':
        return <CheckCircleIconSolid className={`${iconClass} text-green-500`} />;
      case 'closed':
        return <XCircleIconSolid className={`${iconClass} text-gray-500`} />;
      case 'analysis_started':
        return <ChartBarIcon className={`${iconClass} text-blue-500`} />;
      case 'analysis_completed':
        return <CheckCircleIcon className={`${iconClass} text-green-500`} />;
      case 'comment':
        return <DocumentTextIcon className={`${iconClass} text-gray-500`} />;
      default:
        return <ClockIcon className={`${iconClass} text-gray-400`} />;
    }
  }, []);

  // Event severity color
  const getSeverityColor = useCallback((severity?: string) => {
    switch (severity) {
      case 'critical':
        return 'border-red-500 bg-red-50';
      case 'error':
        return 'border-red-400 bg-red-50';
      case 'warning':
        return 'border-yellow-400 bg-yellow-50';
      case 'info':
      default:
        return 'border-blue-400 bg-blue-50';
    }
  }, []);

  // Format timestamp
  const formatTimestamp = useCallback((timestamp: string) => {
    try {
      const date = parseISO(timestamp);
      if (!isValid(date)) return 'Invalid date';
      
      return {
        absolute: format(date, 'MMM dd, yyyy HH:mm:ss'),
        relative: formatDistanceToNow(date, { addSuffix: true }),
        time: format(date, 'HH:mm:ss')
      };
    } catch {
      return {
        absolute: 'Invalid date',
        relative: 'Invalid date',
        time: 'Invalid'
      };
    }
  }, []);

  // Handle event click
  const handleEventClick = useCallback((event: IncidentTimelineEvent) => {
    setSelectedEvent(event);
    setShowEventDetails(true);
    onEventClick?.(event);
  }, [onEventClick]);

  // Handle filter change
  const handleFilterChange = useCallback((key: keyof TimelineFilter, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  // Clear filters
  const clearFilters = useCallback(() => {
    setFilters({
      eventTypes: [],
      severity: [],
      sources: [],
      dateRange: 'all',
      searchTerm: ''
    });
  }, []);

  // Render filter panel
  const renderFilterPanel = () => {
    if (!showFilters) return null;

    return (
      <div className="bg-white p-4 rounded-lg shadow-soft border border-gray-200 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Search */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Search</label>
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search events..."
                value={filters.searchTerm}
                onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>

          {/* Event Types */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Event Types</label>
            <select
              multiple
              value={filters.eventTypes}
              onChange={(e) => handleFilterChange('eventTypes', Array.from(e.target.selectedOptions, o => o.value))}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              size={3}
            >
              {filterOptions.eventTypes.map(type => (
                <option key={type} value={type} className="capitalize">
                  {type.replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>

          {/* Severity */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Severity</label>
            <select
              multiple
              value={filters.severity}
              onChange={(e) => handleFilterChange('severity', Array.from(e.target.selectedOptions, o => o.value))}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              size={3}
            >
              {filterOptions.severities.map(severity => (
                <option key={severity} value={severity} className="capitalize">
                  {severity}
                </option>
              ))}
            </select>
          </div>

          {/* Date Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
            <select
              value={filters.dateRange}
              onChange={(e) => handleFilterChange('dateRange', e.target.value)}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All Time</option>
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
            </select>
          </div>
        </div>

        <div className="mt-4 flex justify-between items-center">
          <span className="text-sm text-gray-500">
            Showing {filteredEvents.length} of {timelineEvents.length} events
          </span>
          <button onClick={clearFilters} className="btn-secondary text-sm">
            Clear Filters
          </button>
        </div>
      </div>
    );
  };

  // Render timeline event
  const renderEvent = (event: IncidentTimelineEvent, index: number) => {
    const timestamps = formatTimestamp(event.timestamp);
    const isLast = index === filteredEvents.length - 1;

    return (
      <div key={event.eventId} className="relative pb-8">
        {/* Timeline line */}
        {!isLast && (
          <div className="absolute left-4 top-8 w-0.5 h-full bg-gray-200" />
        )}

        {/* Event card */}
        <div className="flex items-start space-x-4">
          {/* Icon */}
          <div className="flex-shrink-0 w-8 h-8 bg-white border-2 border-gray-200 rounded-full flex items-center justify-center">
            {getEventIcon(event)}
          </div>

          {/* Content */}
          <div
            className={`flex-1 p-4 rounded-lg border-l-4 cursor-pointer transition-all hover:shadow-md ${getSeverityColor(event.severity)}`}
            onClick={() => handleEventClick(event)}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-gray-900 mb-1">
                  {event.description}
                </h3>
                <div className="flex items-center space-x-3 text-xs text-gray-500">
                  <span className="flex items-center space-x-1">
                    <UserIcon className="w-3 h-3" />
                    <span>{event.actor}</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <ServerIcon className="w-3 h-3" />
                    <span>{event.source}</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <ClockIcon className="w-3 h-3" />
                    <span title={typeof timestamps === 'object' ? timestamps.absolute : timestamps}>
                      {typeof timestamps === 'object' ? timestamps.relative : timestamps}
                    </span>
                  </span>
                </div>
              </div>

              {event.severity && (
                <span
                  className={`px-2 py-1 rounded-full text-xs font-medium ${
                    event.severity === 'critical' ? 'bg-red-100 text-red-800' :
                    event.severity === 'error' ? 'bg-red-100 text-red-700' :
                    event.severity === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-blue-100 text-blue-800'
                  }`}
                >
                  {event.severity}
                </span>
              )}
            </div>

            {/* Event details preview */}
            {detailed && event.details && (
              <div className="mt-3 p-3 bg-white rounded border">
                {event.details.confidence && (
                  <div className="mb-2">
                    <ConfidenceMeter
                      confidenceScore={event.details.confidence}
                      size="sm"
                      showLabel={true}
                    />
                  </div>
                )}
                
                {Object.entries(event.details).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600 capitalize">{key.replace('_', ' ')}:</span>
                    <span className="text-gray-900 font-medium">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Render grouped timeline
  const renderGroupedTimeline = () => {
    return (
      <div className="space-y-6">
        {groupedEvents.map(group => (
          <div key={group.date} className="bg-white rounded-lg shadow-soft border border-gray-200 p-6">
            <div className="flex items-center space-x-2 mb-4 pb-3 border-b border-gray-200">
              <CalendarIcon className="w-5 h-5 text-gray-400" />
              <h3 className="text-lg font-semibold text-gray-900">
                {format(new Date(group.date), 'EEEE, MMMM dd, yyyy')}
              </h3>
              <span className="bg-gray-100 text-gray-600 px-2 py-1 rounded-full text-xs">
                {group.events.length} events
              </span>
            </div>
            
            <div className="space-y-4">
              {group.events.map((event, index) => renderEvent(event, index))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Render list timeline
  const renderListTimeline = () => {
    return (
      <div className="bg-white rounded-lg shadow-soft border border-gray-200 p-6">
        <div className="space-y-4">
          {filteredEvents.map((event, index) => renderEvent(event, index))}
        </div>
      </div>
    );
  };

  // Render event details modal
  const renderEventDetails = () => {
    if (!showEventDetails || !selectedEvent) return null;

    const timestamps = formatTimestamp(selectedEvent.timestamp);

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-3">
                {getEventIcon(selectedEvent)}
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    Event Details
                  </h2>
                  <p className="text-sm text-gray-500 mt-1">
                    {selectedEvent.eventType.replace('_', ' ')} • {typeof timestamps === 'object' ? timestamps.absolute : timestamps}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowEventDetails(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <XCircleIcon className="w-6 h-6" />
              </button>
            </div>
          </div>

          <div className="p-6 space-y-4">
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Description</h3>
              <p className="text-gray-900">{selectedEvent.description}</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Actor</h3>
                <p className="text-gray-900">{selectedEvent.actor}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Source</h3>
                <p className="text-gray-900">{selectedEvent.source}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Severity</h3>
                <span
                  className={`px-2 py-1 rounded-full text-xs font-medium ${
                    selectedEvent.severity === 'critical' ? 'bg-red-100 text-red-800' :
                    selectedEvent.severity === 'error' ? 'bg-red-100 text-red-700' :
                    selectedEvent.severity === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-blue-100 text-blue-800'
                  }`}
                >
                  {selectedEvent.severity || 'info'}
                </span>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Time</h3>
                <p className="text-gray-900" title={typeof timestamps === 'object' ? timestamps.absolute : timestamps}>
                  {typeof timestamps === 'object' ? timestamps.relative : timestamps}
                </p>
              </div>
            </div>

            {selectedEvent.details && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Details</h3>
                <div className="bg-gray-50 rounded-lg p-4">
                  <pre className="text-sm text-gray-900 whitespace-pre-wrap">
                    {JSON.stringify(selectedEvent.details, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className={`flex items-center justify-center py-12 ${className}`}>
        <div className="text-center">
          <div className="spinner mx-auto mb-4"></div>
          <p className="text-gray-500">Loading timeline...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">
            Incident Timeline
          </h2>
          <p className="text-gray-600 mt-1">
            Chronological view of incident events and analysis progression
          </p>
        </div>
        <div className="flex items-center space-x-3">
          {showFilters && (
            <button
              onClick={() => setViewMode(viewMode === 'list' ? 'grouped' : 'list')}
              className="btn-secondary flex items-center space-x-2"
            >
              {viewMode === 'list' ? (
                <>
                  <Squares2X2Icon className="w-4 h-4" />
                  <span>Group by Date</span>
                </>
              ) : (
                <>
                  <ListBulletIcon className="w-4 h-4" />
                  <span>List View</span>
                </>
              )}
            </button>
          )}
          {showFilters && (
            <button
              onClick={() => setFilters(prev => ({ ...prev, searchTerm: '' }))}
              className="btn-secondary flex items-center space-x-2"
            >
              <FunnelIcon className="w-4 h-4" />
              <span>Filters</span>
              {(filters.eventTypes.length > 0 || filters.severity.length > 0 || filters.sources.length > 0 || filters.searchTerm) && (
                <span className="bg-primary-600 text-white text-xs rounded-full px-1.5 py-0.5">
                  {filters.eventTypes.length + filters.severity.length + filters.sources.length + (filters.searchTerm ? 1 : 0)}
                </span>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Summary */}
      <div className="bg-white p-4 rounded-lg shadow-soft border border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">{timelineEvents.length}</p>
            <p className="text-sm text-gray-500">Total Events</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-orange-600">
              {timelineEvents.filter(e => e.eventType === 'created' || e.eventType === 'escalated').length}
            </p>
            <p className="text-sm text-gray-500">Alerts</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {timelineEvents.filter(e => e.eventType.includes('analysis')).length}
            </p>
            <p className="text-sm text-gray-500">AI Actions</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {timelineEvents.filter(e => e.eventType === 'resolved').length}
            </p>
            <p className="text-sm text-gray-500">Resolved</p>
          </div>
        </div>
      </div>

      {/* Filters */}
      {renderFilterPanel()}

      {/* Timeline Content */}
      {filteredEvents.length === 0 ? (
        <div className="text-center py-12">
          <ClockIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No events found</h3>
          <p className="mt-1 text-sm text-gray-500">
            Try adjusting your filters or check back later.
          </p>
        </div>
      ) : viewMode === 'grouped' ? (
        renderGroupedTimeline()
      ) : (
        renderListTimeline()
      )}

      {/* Event Details Modal */}
      {renderEventDetails()}
    </div>
  );
};

export default TimelineView;