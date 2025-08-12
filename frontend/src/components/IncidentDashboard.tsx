/**
 * React TypeScript component for the incident dashboard.
 * 
 * This component provides incident list, metrics display, analysis results,
 * and real-time status updates for the AI SRE Agent frontend.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ExclamationTriangleIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  ChartBarIcon,
  PlayIcon,
  EyeIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  ArrowPathIcon,
  PlusIcon,
  FireIcon,
  SignalIcon,
  UserGroupIcon,
  ServerIcon
} from '@heroicons/react/24/outline';
import { format, formatDistanceToNow } from 'date-fns';
import {
  Incident,
  IncidentSeverity,
  IncidentStatus,
  IncidentMetrics,
  INCIDENT_SEVERITY_COLORS,
  INCIDENT_STATUS_COLORS,
  INCIDENT_SEVERITY_PRIORITY
} from '../types/incident';
import { AnalysisResult } from '../types/analysis';
import ConfidenceMeter from './ConfidenceMeter';

interface IncidentDashboardProps {
  incidents: Incident[];
  onIncidentSelect: (incident: Incident) => void;
  onAnalyzeIncident: (incidentId: string) => void;
  analysisResults: AnalysisResult[];
  isAnalyzing: boolean;
  selectedIncident?: Incident | null;
  metrics?: IncidentMetrics | null;
  className?: string;
}

interface FilterState {
  search: string;
  severity: IncidentSeverity[];
  status: IncidentStatus[];
  service: string[];
  assignee: string[];
  dateRange: 'all' | '1d' | '7d' | '30d';
}

const IncidentDashboard: React.FC<IncidentDashboardProps> = ({
  incidents,
  onIncidentSelect,
  onAnalyzeIncident,
  analysisResults,
  isAnalyzing,
  selectedIncident,
  metrics,
  className = ''
}) => {
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    severity: [],
    status: [],
    service: [],
    assignee: [],
    dateRange: 'all'
  });
  const [sortBy, setSortBy] = useState<'timestamp' | 'severity' | 'status'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showFilters, setShowFilters] = useState(false);
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');

  // Filter and sort incidents
  const filteredAndSortedIncidents = useMemo(() => {
    let filtered = incidents.filter(incident => {
      // Search filter
      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        const matchesSearch = 
          incident.title.toLowerCase().includes(searchLower) ||
          incident.description.toLowerCase().includes(searchLower) ||
          incident.incidentId.toLowerCase().includes(searchLower) ||
          incident.serviceName.toLowerCase().includes(searchLower);
        
        if (!matchesSearch) return false;
      }

      // Severity filter
      if (filters.severity.length > 0 && !filters.severity.includes(incident.severity)) {
        return false;
      }

      // Status filter
      if (filters.status.length > 0 && !filters.status.includes(incident.status)) {
        return false;
      }

      // Service filter
      if (filters.service.length > 0 && !filters.service.includes(incident.serviceName)) {
        return false;
      }

      // Date range filter
      if (filters.dateRange !== 'all') {
        const incidentDate = new Date(incident.timestamp);
        const now = new Date();
        const daysDiff = Math.floor((now.getTime() - incidentDate.getTime()) / (1000 * 60 * 60 * 24));
        
        const rangeDays = {
          '1d': 1,
          '7d': 7,
          '30d': 30
        }[filters.dateRange] || 0;
        
        if (daysDiff > rangeDays) return false;
      }

      return true;
    });

    // Sort incidents
    filtered.sort((a, b) => {
      let aValue, bValue;

      switch (sortBy) {
        case 'severity':
          aValue = INCIDENT_SEVERITY_PRIORITY[a.severity];
          bValue = INCIDENT_SEVERITY_PRIORITY[b.severity];
          break;
        case 'status':
          aValue = a.status;
          bValue = b.status;
          break;
        case 'timestamp':
        default:
          aValue = new Date(a.timestamp).getTime();
          bValue = new Date(b.timestamp).getTime();
          break;
      }

      const result = aValue > bValue ? 1 : aValue < bValue ? -1 : 0;
      return sortOrder === 'desc' ? -result : result;
    });

    return filtered;
  }, [incidents, filters, sortBy, sortOrder]);

  // Get unique services and assignees for filter options
  const filterOptions = useMemo(() => {
    const services = Array.from(new Set(incidents.map(i => i.serviceName))).sort();
    const assignees = Array.from(new Set(incidents.map(i => i.assignedTo).filter(Boolean))).sort();
    
    return { services, assignees };
  }, [incidents]);

  // Get analysis result for incident
  const getAnalysisForIncident = useCallback((incidentId: string) => {
    return analysisResults.find(result => result.incidentId === incidentId);
  }, [analysisResults]);

  // Format incident age
  const formatIncidentAge = useCallback((timestamp: string) => {
    try {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
    } catch {
      return 'Unknown';
    }
  }, []);

  // Get severity icon and color
  const getSeverityDisplay = useCallback((severity: IncidentSeverity) => {
    const icons = {
      [IncidentSeverity.LOW]: <SignalIcon className="w-4 h-4" />,
      [IncidentSeverity.MEDIUM]: <ExclamationTriangleIcon className="w-4 h-4" />,
      [IncidentSeverity.HIGH]: <FireIcon className="w-4 h-4" />,
      [IncidentSeverity.CRITICAL]: <FireIcon className="w-4 h-4" />
    };

    return {
      icon: icons[severity],
      color: INCIDENT_SEVERITY_COLORS[severity],
      label: severity.charAt(0).toUpperCase() + severity.slice(1)
    };
  }, []);

  // Get status display
  const getStatusDisplay = useCallback((status: IncidentStatus) => {
    const icons = {
      [IncidentStatus.OPEN]: <ExclamationTriangleIcon className="w-4 h-4" />,
      [IncidentStatus.INVESTIGATING]: <ClockIcon className="w-4 h-4" />,
      [IncidentStatus.IDENTIFIED]: <EyeIcon className="w-4 h-4" />,
      [IncidentStatus.MONITORING]: <ChartBarIcon className="w-4 h-4" />,
      [IncidentStatus.RESOLVED]: <CheckCircleIcon className="w-4 h-4" />,
      [IncidentStatus.CLOSED]: <XCircleIcon className="w-4 h-4" />
    };

    return {
      icon: icons[status],
      color: INCIDENT_STATUS_COLORS[status],
      label: status.charAt(0).toUpperCase() + status.slice(1)
    };
  }, []);

  // Handle filter changes
  const handleFilterChange = useCallback((key: keyof FilterState, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  // Clear all filters
  const clearFilters = useCallback(() => {
    setFilters({
      search: '',
      severity: [],
      status: [],
      service: [],
      assignee: [],
      dateRange: 'all'
    });
  }, []);

  // Render metrics cards
  const renderMetricsCards = useCallback(() => {
    if (!metrics) return null;

    const metricsCards = [
      {
        title: 'Total Incidents',
        value: metrics.totalIncidents,
        icon: <ChartBarIcon className="w-6 h-6" />,
        color: 'bg-blue-500',
        change: '+12%'
      },
      {
        title: 'Open Incidents',
        value: metrics.openIncidents,
        icon: <ExclamationTriangleIcon className="w-6 h-6" />,
        color: 'bg-orange-500',
        change: '-3%'
      },
      {
        title: 'Critical Incidents',
        value: metrics.criticalIncidents,
        icon: <FireIcon className="w-6 h-6" />,
        color: 'bg-red-500',
        change: '0%'
      },
      {
        title: 'Avg MTTR',
        value: `${metrics.averageMttr}m`,
        icon: <ClockIcon className="w-6 h-6" />,
        color: 'bg-green-500',
        change: '-15%'
      }
    ];

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {metricsCards.map((card, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-soft border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{card.title}</p>
                <p className="text-2xl font-semibold text-gray-900 mt-1">{card.value}</p>
                <p className="text-xs text-green-600 mt-1">{card.change} from last week</p>
              </div>
              <div className={`${card.color} p-3 rounded-lg text-white`}>
                {card.icon}
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }, [metrics]);

  // Render filter panel
  const renderFilterPanel = useCallback(() => {
    if (!showFilters) return null;

    return (
      <div className="bg-white p-4 rounded-lg shadow-soft border border-gray-200 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {/* Severity Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Severity</label>
            <div className="space-y-1">
              {Object.values(IncidentSeverity).map(severity => (
                <label key={severity} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.severity.includes(severity)}
                    onChange={(e) => {
                      const newSeverity = e.target.checked
                        ? [...filters.severity, severity]
                        : filters.severity.filter(s => s !== severity);
                      handleFilterChange('severity', newSeverity);
                    }}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-700 capitalize">{severity}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Status Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
            <div className="space-y-1">
              {Object.values(IncidentStatus).map(status => (
                <label key={status} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.status.includes(status)}
                    onChange={(e) => {
                      const newStatus = e.target.checked
                        ? [...filters.status, status]
                        : filters.status.filter(s => s !== status);
                      handleFilterChange('status', newStatus);
                    }}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-700 capitalize">{status.replace('_', ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Service Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Service</label>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {filterOptions.services.map(service => (
                <label key={service} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.service.includes(service)}
                    onChange={(e) => {
                      const newServices = e.target.checked
                        ? [...filters.service, service]
                        : filters.service.filter(s => s !== service);
                      handleFilterChange('service', newServices);
                    }}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">{service}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Date Range Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
            <select
              value={filters.dateRange}
              onChange={(e) => handleFilterChange('dateRange', e.target.value)}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              <option value="all">All Time</option>
              <option value="1d">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>

          {/* Sort Options */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
            <select
              value={`${sortBy}_${sortOrder}`}
              onChange={(e) => {
                const [field, order] = e.target.value.split('_');
                setSortBy(field as any);
                setSortOrder(order as any);
              }}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              <option value="timestamp_desc">Newest First</option>
              <option value="timestamp_asc">Oldest First</option>
              <option value="severity_desc">Highest Severity</option>
              <option value="severity_asc">Lowest Severity</option>
              <option value="status_asc">Status A-Z</option>
              <option value="status_desc">Status Z-A</option>
            </select>
          </div>

          {/* Clear Filters */}
          <div className="flex items-end">
            <button
              onClick={clearFilters}
              className="btn-secondary w-full"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>
    );
  }, [showFilters, filters, filterOptions, sortBy, sortOrder, handleFilterChange, clearFilters]);

  // Render incident card
  const renderIncidentCard = useCallback((incident: Incident) => {
    const severity = getSeverityDisplay(incident.severity);
    const status = getStatusDisplay(incident.status);
    const analysis = getAnalysisForIncident(incident.incidentId);
    const isSelected = selectedIncident?.incidentId === incident.incidentId;

    return (
      <div
        key={incident.incidentId}
        className={`bg-white p-6 rounded-lg shadow-soft border-2 transition-all cursor-pointer hover:shadow-medium ${
          isSelected ? 'border-primary-500 bg-primary-50' : 'border-gray-200 hover:border-gray-300'
        }`}
        onClick={() => onIncidentSelect(incident)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div style={{ color: severity.color }} className="flex items-center space-x-1">
              {severity.icon}
              <span className="text-sm font-medium">{severity.label}</span>
            </div>
            <span className="text-gray-300">•</span>
            <div style={{ color: status.color }} className="flex items-center space-x-1">
              {status.icon}
              <span className="text-sm">{status.label}</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {analysis && (
              <ConfidenceMeter
                confidenceScore={analysis.confidenceScore}
                size="sm"
                showLabel={false}
              />
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onAnalyzeIncident(incident.incidentId);
              }}
              disabled={isAnalyzing}
              className="p-2 text-primary-600 hover:bg-primary-100 rounded-lg transition-colors disabled:opacity-50"
              title="Analyze Incident"
            >
              {isAnalyzing ? (
                <ArrowPathIcon className="w-4 h-4 animate-spin" />
              ) : (
                <ChartBarIcon className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-start justify-between">
            <h3 className="text-lg font-semibold text-gray-900 line-clamp-2">{incident.title}</h3>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded ml-2 flex-shrink-0">
              {incident.incidentId}
            </span>
          </div>
          
          <p className="text-gray-600 text-sm line-clamp-2">{incident.description}</p>
          
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <ServerIcon className="w-4 h-4" />
                <span>{incident.serviceName}</span>
              </div>
              <div className="flex items-center space-x-1">
                <ClockIcon className="w-4 h-4" />
                <span>{formatIncidentAge(incident.timestamp)}</span>
              </div>
              {incident.assignedTo && (
                <div className="flex items-center space-x-1">
                  <UserGroupIcon className="w-4 h-4" />
                  <span>{incident.assignedTo}</span>
                </div>
              )}
            </div>
            
            {incident.mttrMinutes && (
              <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                MTTR: {incident.mttrMinutes}m
              </span>
            )}
          </div>

          {incident.symptoms && incident.symptoms.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {incident.symptoms.slice(0, 3).map((symptom, index) => (
                <span
                  key={index}
                  className="text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded-full"
                >
                  {symptom}
                </span>
              ))}
              {incident.symptoms.length > 3 && (
                <span className="text-xs text-gray-500">
                  +{incident.symptoms.length - 3} more
                </span>
              )}
            </div>
          )}

          {analysis && (
            <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-blue-900">AI Analysis Available</span>
                <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                  {Math.round(analysis.confidenceScore * 100)}% confidence
                </span>
              </div>
              {analysis.rootCause && (
                <p className="text-sm text-blue-800 line-clamp-2">
                  <strong>Root Cause:</strong> {analysis.rootCause.primaryCause}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }, [getSeverityDisplay, getStatusDisplay, getAnalysisForIncident, selectedIncident, onIncidentSelect, onAnalyzeIncident, isAnalyzing, formatIncidentAge]);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Incident Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Manage and analyze production incidents with AI assistance
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setViewMode(viewMode === 'list' ? 'grid' : 'list')}
            className="btn-secondary"
          >
            {viewMode === 'list' ? 'Grid View' : 'List View'}
          </button>
          <button className="btn-primary">
            <PlusIcon className="w-4 h-4 mr-2" />
            New Incident
          </button>
        </div>
      </div>

      {/* Metrics Cards */}
      {renderMetricsCards()}

      {/* Search and Filter Bar */}
      <div className="bg-white p-4 rounded-lg shadow-soft border border-gray-200">
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search incidents..."
              value={filters.search}
              onChange={(e) => handleFilterChange('search', e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`btn-secondary flex items-center space-x-2 ${showFilters ? 'bg-primary-100 text-primary-700' : ''}`}
          >
            <FunnelIcon className="w-4 h-4" />
            <span>Filters</span>
            {(filters.severity.length > 0 || filters.status.length > 0 || filters.service.length > 0) && (
              <span className="bg-primary-600 text-white text-xs rounded-full px-2 py-0.5 ml-1">
                {filters.severity.length + filters.status.length + filters.service.length}
              </span>
            )}
          </button>
          <div className="text-sm text-gray-500">
            {filteredAndSortedIncidents.length} of {incidents.length} incidents
          </div>
        </div>
      </div>

      {/* Filter Panel */}
      {renderFilterPanel()}

      {/* Incidents Grid/List */}
      <div className={`${viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' : 'space-y-4'}`}>
        {filteredAndSortedIncidents.length === 0 ? (
          <div className="col-span-full text-center py-12">
            <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No incidents found</h3>
            <p className="mt-1 text-sm text-gray-500">
              Try adjusting your search criteria or filters.
            </p>
            <div className="mt-6">
              <button onClick={clearFilters} className="btn-primary">
                <PlusIcon className="w-4 h-4 mr-2" />
                Clear Filters
              </button>
            </div>
          </div>
        ) : (
          filteredAndSortedIncidents.map(renderIncidentCard)
        )}
      </div>

      {/* Loading State */}
      {isAnalyzing && (
        <div className="fixed bottom-4 right-4 bg-primary-600 text-white px-4 py-2 rounded-lg shadow-lg">
          <div className="flex items-center space-x-2">
            <ArrowPathIcon className="w-4 h-4 animate-spin" />
            <span>Analyzing incident...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default IncidentDashboard;