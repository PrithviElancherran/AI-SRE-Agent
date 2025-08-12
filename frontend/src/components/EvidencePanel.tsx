/**
 * React TypeScript component for displaying analysis evidence.
 * 
 * This component provides GCP observability data, supporting metrics, logs,
 * and correlations for the AI SRE Agent frontend.
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  ChartBarIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  ServerIcon,
  EyeIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  LinkIcon,
  CloudIcon,
  CpuChipIcon,
  CircleStackIcon,
  SignalIcon,
  BugAntIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  XMarkIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import {
  ChartBarIcon as ChartBarIconSolid,
  ExclamationTriangleIcon as ExclamationTriangleIconSolid
} from '@heroicons/react/24/solid';
import { format } from 'date-fns';
import { Incident } from '../types/incident';
import { AnalysisResult, EvidenceItem, EvidenceType, EVIDENCE_TYPE_LABELS } from '../types/analysis';
import ConfidenceMeter from './ConfidenceMeter';

interface EvidencePanelProps {
  incident: Incident;
  analysisResults: AnalysisResult[];
  className?: string;
  onEvidenceClick?: (evidence: EvidenceItem) => void;
  showFilters?: boolean;
}

interface EvidenceFilter {
  types: EvidenceType[];
  sources: string[];
  relevanceThreshold: number;
  qualityThreshold: number;
  searchTerm: string;
}

interface EvidenceGroup {
  type: EvidenceType;
  label: string;
  items: EvidenceItem[];
  icon: React.ReactNode;
  color: string;
}

const EvidencePanel: React.FC<EvidencePanelProps> = ({
  incident,
  analysisResults,
  className = '',
  onEvidenceClick,
  showFilters = true
}) => {
  const [filters, setFilters] = useState<EvidenceFilter>({
    types: [],
    sources: [],
    relevanceThreshold: 0,
    qualityThreshold: 0,
    searchTerm: ''
  });
  const [expandedGroups, setExpandedGroups] = useState<Set<EvidenceType>>(new Set());
  const [selectedEvidence, setSelectedEvidence] = useState<EvidenceItem | null>(null);
  const [showEvidenceModal, setShowEvidenceModal] = useState(false);

  // Get all evidence items from analysis results
  const allEvidence = useMemo((): EvidenceItem[] => {
    const evidence: EvidenceItem[] = [];
    
    analysisResults.forEach(result => {
      if (result.evidenceItems) {
        evidence.push(...result.evidenceItems);
      }
    });

    // Add synthetic evidence for demo
    const syntheticEvidence: EvidenceItem[] = [
      {
        evidenceId: `gcp_metrics_${incident.incidentId}_1`,
        evidenceType: EvidenceType.GCP_MONITORING,
        description: 'Payment API P95 latency spike detected',
        data: {
          metric_name: 'api_latency_p95',
          current_value: 6.2,
          threshold: 2.0,
          unit: 'seconds',
          service: 'payment-api',
          region: 'us-central1',
          trend: 'increasing',
          change_percentage: 210
        },
        source: 'GCP Monitoring',
        relevanceScore: 0.95,
        qualityScore: 0.92,
        timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
        collectedAt: new Date().toISOString(),
        gcpDashboardUrl: 'https://console.cloud.google.com/monitoring/dashboards/custom/payment-api',
        correlationScore: 0.89,
        validationStatus: 'validated',
        tags: ['latency', 'performance', 'api']
      },
      {
        evidenceId: `gcp_metrics_${incident.incidentId}_2`,
        evidenceType: EvidenceType.GCP_MONITORING,
        description: 'Redis cache hit rate drop to 38%',
        data: {
          metric_name: 'redis_cache_hit_rate',
          current_value: 38,
          baseline: 95,
          unit: 'percentage',
          service: 'redis-cache',
          instance: 'redis-prod-001',
          trend: 'decreasing',
          change_percentage: -60
        },
        source: 'GCP Monitoring',
        relevanceScore: 0.92,
        qualityScore: 0.88,
        timestamp: new Date(Date.now() - 3 * 60 * 1000).toISOString(),
        collectedAt: new Date().toISOString(),
        gcpDashboardUrl: 'https://console.cloud.google.com/monitoring/dashboards/custom/redis-cache',
        correlationScore: 0.94,
        validationStatus: 'validated',
        tags: ['cache', 'performance', 'redis']
      },
      {
        evidenceId: `gcp_logs_${incident.incidentId}_1`,
        evidenceType: EvidenceType.GCP_LOGGING,
        description: 'Database connection timeout errors increasing',
        data: {
          log_entries: 47,
          error_rate: 0.12,
          time_window: '5 minutes',
          service: 'payment-api',
          error_type: 'connection_timeout',
          sample_message: 'Connection timeout: Unable to acquire connection from pool within 5000ms',
          affected_endpoints: ['/api/payments/process', '/api/payments/validate']
        },
        source: 'GCP Cloud Logging',
        relevanceScore: 0.87,
        qualityScore: 0.85,
        timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
        collectedAt: new Date().toISOString(),
        gcpDashboardUrl: 'https://console.cloud.google.com/logs/query',
        correlationScore: 0.76,
        validationStatus: 'validated',
        tags: ['errors', 'database', 'timeout']
      },
      {
        evidenceId: `gcp_errors_${incident.incidentId}_1`,
        evidenceType: EvidenceType.GCP_ERROR_REPORTING,
        description: 'Redis connection pool exhaustion errors',
        data: {
          error_count: 23,
          error_rate: 0.08,
          first_seen: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
          last_seen: new Date(Date.now() - 1 * 60 * 1000).toISOString(),
          affected_users: 156,
          error_message: 'RedisConnectionPoolExhaustedException: Pool exhausted',
          stack_trace_sample: 'at redis.clients.jedis.JedisPool.getResource',
          service: 'payment-api'
        },
        source: 'GCP Error Reporting',
        relevanceScore: 0.91,
        qualityScore: 0.89,
        timestamp: new Date(Date.now() - 1 * 60 * 1000).toISOString(),
        collectedAt: new Date().toISOString(),
        gcpDashboardUrl: 'https://console.cloud.google.com/errors',
        correlationScore: 0.88,
        validationStatus: 'validated',
        tags: ['errors', 'redis', 'connection_pool']
      },
      {
        evidenceId: `gcp_traces_${incident.incidentId}_1`,
        evidenceType: EvidenceType.GCP_TRACING,
        description: 'Slow database queries in payment processing',
        data: {
          trace_count: 34,
          avg_duration: 8.5,
          max_duration: 12.3,
          unit: 'seconds',
          service: 'payment-api',
          operation: 'process_payment',
          slowest_spans: [
            { name: 'database.query.user_profile', duration: 7.2 },
            { name: 'database.query.payment_validation', duration: 5.8 },
            { name: 'redis.get.user_cache', duration: 2.1 }
          ]
        },
        source: 'GCP Cloud Trace',
        relevanceScore: 0.83,
        qualityScore: 0.86,
        timestamp: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
        collectedAt: new Date().toISOString(),
        gcpDashboardUrl: 'https://console.cloud.google.com/traces',
        correlationScore: 0.79,
        validationStatus: 'validated',
        tags: ['tracing', 'performance', 'database']
      },
      {
        evidenceId: `historical_${incident.incidentId}_1`,
        evidenceType: EvidenceType.HISTORICAL_CORRELATION,
        description: 'Similar incident pattern from INC-2024-045',
        data: {
          similar_incident_id: 'INC-2024-045',
          similarity_score: 0.92,
          matching_symptoms: ['redis_cache_miss', 'api_latency_spike', 'db_timeout'],
          resolution_time: 25,
          resolution_method: 'Redis node scaling + cache warming',
          success_rate: 0.96,
          confidence: 0.89
        },
        source: 'Historical Analysis',
        relevanceScore: 0.94,
        qualityScore: 0.91,
        timestamp: new Date(Date.now() - 6 * 60 * 1000).toISOString(),
        collectedAt: new Date().toISOString(),
        correlationScore: 0.92,
        validationStatus: 'validated',
        tags: ['historical', 'correlation', 'pattern']
      }
    ];

    return [...evidence, ...syntheticEvidence];
  }, [analysisResults, incident.incidentId]);

  // Filter evidence based on current filters
  const filteredEvidence = useMemo(() => {
    return allEvidence.filter(item => {
      // Type filter
      if (filters.types.length > 0 && !filters.types.includes(item.evidenceType)) {
        return false;
      }

      // Source filter
      if (filters.sources.length > 0 && !filters.sources.includes(item.source)) {
        return false;
      }

      // Relevance threshold
      if (item.relevanceScore < filters.relevanceThreshold) {
        return false;
      }

      // Quality threshold
      if (item.qualityScore < filters.qualityThreshold) {
        return false;
      }

      // Search term
      if (filters.searchTerm) {
        const searchLower = filters.searchTerm.toLowerCase();
        const searchableText = `${item.description} ${item.source} ${item.tags?.join(' ') || ''}`.toLowerCase();
        if (!searchableText.includes(searchLower)) {
          return false;
        }
      }

      return true;
    });
  }, [allEvidence, filters]);

  // Group evidence by type
  const evidenceGroups = useMemo((): EvidenceGroup[] => {
    const groups: { [key: string]: EvidenceItem[] } = {};
    
    filteredEvidence.forEach(item => {
      const type = item.evidenceType;
      if (!groups[type]) {
        groups[type] = [];
      }
      groups[type].push(item);
    });

    const typeIcons = {
      [EvidenceType.GCP_MONITORING]: <ChartBarIconSolid className="w-5 h-5" />,
      [EvidenceType.GCP_LOGGING]: <DocumentTextIcon className="w-5 h-5" />,
      [EvidenceType.GCP_ERROR_REPORTING]: <ExclamationTriangleIconSolid className="w-5 h-5" />,
      [EvidenceType.GCP_TRACING]: <SignalIcon className="w-5 h-5" />,
      [EvidenceType.HISTORICAL_CORRELATION]: <ClockIcon className="w-5 h-5" />,
      [EvidenceType.USER_INPUT]: <EyeIcon className="w-5 h-5" />,
      [EvidenceType.EXTERNAL_API]: <LinkIcon className="w-5 h-5" />,
      [EvidenceType.INFRASTRUCTURE_DATA]: <ServerIcon className="w-5 h-5" />,
      [EvidenceType.APPLICATION_METRICS]: <CpuChipIcon className="w-5 h-5" />,
      [EvidenceType.SYSTEM_LOGS]: <CircleStackIcon className="w-5 h-5" />
    };

    const typeColors = {
      [EvidenceType.GCP_MONITORING]: 'text-blue-600',
      [EvidenceType.GCP_LOGGING]: 'text-green-600',
      [EvidenceType.GCP_ERROR_REPORTING]: 'text-red-600',
      [EvidenceType.GCP_TRACING]: 'text-purple-600',
      [EvidenceType.HISTORICAL_CORRELATION]: 'text-orange-600',
      [EvidenceType.USER_INPUT]: 'text-gray-600',
      [EvidenceType.EXTERNAL_API]: 'text-indigo-600',
      [EvidenceType.INFRASTRUCTURE_DATA]: 'text-teal-600',
      [EvidenceType.APPLICATION_METRICS]: 'text-pink-600',
      [EvidenceType.SYSTEM_LOGS]: 'text-yellow-600'
    };

    return Object.entries(groups).map(([type, items]) => ({
      type: type as EvidenceType,
      label: EVIDENCE_TYPE_LABELS[type as EvidenceType],
      items: items.sort((a, b) => b.relevanceScore - a.relevanceScore),
      icon: typeIcons[type as EvidenceType],
      color: typeColors[type as EvidenceType]
    })).sort((a, b) => b.items.length - a.items.length);
  }, [filteredEvidence]);

  // Get unique filter options
  const filterOptions = useMemo(() => {
    const types = Array.from(new Set(allEvidence.map(e => e.evidenceType)));
    const sources = Array.from(new Set(allEvidence.map(e => e.source)));
    return { types, sources };
  }, [allEvidence]);

  // Handle group toggle
  const toggleGroup = useCallback((type: EvidenceType) => {
    setExpandedGroups(prev => {
      const newSet = new Set(prev);
      if (newSet.has(type)) {
        newSet.delete(type);
      } else {
        newSet.add(type);
      }
      return newSet;
    });
  }, []);

  // Handle evidence click
  const handleEvidenceClick = useCallback((evidence: EvidenceItem) => {
    setSelectedEvidence(evidence);
    setShowEvidenceModal(true);
    onEvidenceClick?.(evidence);
  }, [onEvidenceClick]);

  // Handle filter change
  const handleFilterChange = useCallback((key: keyof EvidenceFilter, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  // Clear filters
  const clearFilters = useCallback(() => {
    setFilters({
      types: [],
      sources: [],
      relevanceThreshold: 0,
      qualityThreshold: 0,
      searchTerm: ''
    });
  }, []);

  // Format timestamp
  const formatTimestamp = useCallback((timestamp: string) => {
    try {
      return format(new Date(timestamp), 'HH:mm:ss');
    } catch {
      return 'Invalid';
    }
  }, []);

  // Get relevance color
  const getRelevanceColor = useCallback((score: number) => {
    if (score >= 0.9) return 'text-green-600 bg-green-50';
    if (score >= 0.7) return 'text-blue-600 bg-blue-50';
    if (score >= 0.5) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  }, []);

  // Get quality color
  const getQualityColor = useCallback((score: number) => {
    if (score >= 0.9) return 'text-green-600';
    if (score >= 0.7) return 'text-blue-600';
    if (score >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
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
                placeholder="Search evidence..."
                value={filters.searchTerm}
                onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>

          {/* Evidence Types */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Types</label>
            <select
              multiple
              value={filters.types}
              onChange={(e) => handleFilterChange('types', Array.from(e.target.selectedOptions, o => o.value as EvidenceType))}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              size={3}
            >
              {filterOptions.types.map(type => (
                <option key={type} value={type}>
                  {EVIDENCE_TYPE_LABELS[type as EvidenceType]}
                </option>
              ))}
            </select>
          </div>

          {/* Sources */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sources</label>
            <select
              multiple
              value={filters.sources}
              onChange={(e) => handleFilterChange('sources', Array.from(e.target.selectedOptions, o => o.value))}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              size={3}
            >
              {filterOptions.sources.map(source => (
                <option key={source} value={source}>
                  {source}
                </option>
              ))}
            </select>
          </div>

          {/* Thresholds */}
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Relevance ≥ {Math.round(filters.relevanceThreshold * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filters.relevanceThreshold}
                onChange={(e) => handleFilterChange('relevanceThreshold', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Quality ≥ {Math.round(filters.qualityThreshold * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filters.qualityThreshold}
                onChange={(e) => handleFilterChange('qualityThreshold', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        <div className="mt-4 flex justify-between items-center">
          <span className="text-sm text-gray-500">
            Showing {filteredEvidence.length} of {allEvidence.length} evidence items
          </span>
          <button onClick={clearFilters} className="btn-secondary text-sm">
            Clear Filters
          </button>
        </div>
      </div>
    );
  };

  // Render evidence item
  const renderEvidenceItem = (evidence: EvidenceItem) => {
    return (
      <div
        key={evidence.evidenceId}
        className="p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all cursor-pointer"
        onClick={() => handleEvidenceClick(evidence)}
      >
        <div className="flex items-start justify-between mb-2">
          <h4 className="text-sm font-semibold text-gray-900 line-clamp-2">
            {evidence.description}
          </h4>
          <div className="flex items-center space-x-2 ml-2">
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRelevanceColor(evidence.relevanceScore)}`}>
              {Math.round(evidence.relevanceScore * 100)}%
            </span>
            {evidence.gcpDashboardUrl && (
              <a
                href={evidence.gcpDashboardUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-primary-600 hover:text-primary-800"
                title="View in GCP Console"
              >
                <LinkIcon className="w-4 h-4" />
              </a>
            )}
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
          <div className="flex items-center space-x-3">
            <span>{evidence.source}</span>
            <span>•</span>
            <span>{formatTimestamp(evidence.timestamp)}</span>
            {evidence.correlationScore && (
              <>
                <span>•</span>
                <span className="text-blue-600">
                  {Math.round(evidence.correlationScore * 100)}% correlation
                </span>
              </>
            )}
          </div>
          <div className={`flex items-center space-x-1 ${getQualityColor(evidence.qualityScore)}`}>
            <span>Quality: {Math.round(evidence.qualityScore * 100)}%</span>
          </div>
        </div>

        {/* Evidence preview */}
        <div className="bg-gray-50 rounded p-3 text-sm">
          {evidence.evidenceType === EvidenceType.GCP_MONITORING && evidence.data.metric_name && (
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-600">Metric:</span>
                <span className="font-medium">{evidence.data.metric_name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Current:</span>
                <span className="font-medium flex items-center">
                  {evidence.data.current_value} {evidence.data.unit}
                  {evidence.data.trend === 'increasing' ? (
                    <ArrowTrendingUpIcon className="w-3 h-3 text-red-500 ml-1" />
                  ) : evidence.data.trend === 'decreasing' ? (
                    <ArrowTrendingDownIcon className="w-3 h-3 text-red-500 ml-1" />
                  ) : null}
                </span>
              </div>
              {evidence.data.threshold && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Threshold:</span>
                  <span className="font-medium">{evidence.data.threshold} {evidence.data.unit}</span>
                </div>
              )}
            </div>
          )}

          {evidence.evidenceType === EvidenceType.GCP_LOGGING && evidence.data.log_entries && (
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-600">Log Entries:</span>
                <span className="font-medium">{evidence.data.log_entries}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Error Rate:</span>
                <span className="font-medium">{(evidence.data.error_rate * 100).toFixed(1)}%</span>
              </div>
              {evidence.data.sample_message && (
                <div className="mt-2 p-2 bg-white rounded border text-xs text-gray-700">
                  {evidence.data.sample_message}
                </div>
              )}
            </div>
          )}

          {evidence.evidenceType === EvidenceType.GCP_ERROR_REPORTING && evidence.data.error_count && (
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-600">Error Count:</span>
                <span className="font-medium">{evidence.data.error_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Affected Users:</span>
                <span className="font-medium">{evidence.data.affected_users}</span>
              </div>
              {evidence.data.error_message && (
                <div className="mt-2 p-2 bg-white rounded border text-xs text-gray-700">
                  {evidence.data.error_message}
                </div>
              )}
            </div>
          )}

          {evidence.evidenceType === EvidenceType.HISTORICAL_CORRELATION && evidence.data.similar_incident_id && (
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-600">Similar Incident:</span>
                <span className="font-medium">{evidence.data.similar_incident_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Similarity:</span>
                <span className="font-medium">{Math.round(evidence.data.similarity_score * 100)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Resolution Time:</span>
                <span className="font-medium">{evidence.data.resolution_time}m</span>
              </div>
            </div>
          )}

          {evidence.evidenceType === EvidenceType.GCP_TRACING && evidence.data.trace_count && (
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-600">Traces:</span>
                <span className="font-medium">{evidence.data.trace_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Duration:</span>
                <span className="font-medium">{evidence.data.avg_duration}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Max Duration:</span>
                <span className="font-medium">{evidence.data.max_duration}s</span>
              </div>
            </div>
          )}
        </div>

        {/* Tags */}
        {evidence.tags && evidence.tags.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1">
            {evidence.tags.map((tag, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Render evidence group
  const renderEvidenceGroup = (group: EvidenceGroup) => {
    const isExpanded = expandedGroups.has(group.type);

    return (
      <div key={group.type} className="bg-white rounded-lg shadow-soft border border-gray-200 mb-4">
        <button
          onClick={() => toggleGroup(group.type)}
          className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center space-x-3">
            <div className={group.color}>
              {group.icon}
            </div>
            <div className="text-left">
              <h3 className="text-lg font-semibold text-gray-900">{group.label}</h3>
              <p className="text-sm text-gray-500">{group.items.length} evidence items</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className="text-sm text-gray-500">
              Avg Relevance: {Math.round((group.items.reduce((sum, item) => sum + item.relevanceScore, 0) / group.items.length) * 100)}%
            </div>
            {isExpanded ? (
              <ChevronDownIcon className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronRightIcon className="w-5 h-5 text-gray-400" />
            )}
          </div>
        </button>

        {isExpanded && (
          <div className="px-6 pb-6">
            <div className="grid grid-cols-1 gap-4">
              {group.items.map(renderEvidenceItem)}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render evidence modal
  const renderEvidenceModal = () => {
    if (!showEvidenceModal || !selectedEvidence) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-y-auto">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-3">
                <div className={evidenceGroups.find(g => g.type === selectedEvidence.evidenceType)?.color || 'text-gray-600'}>
                  {evidenceGroups.find(g => g.type === selectedEvidence.evidenceType)?.icon}
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    Evidence Details
                  </h2>
                  <p className="text-sm text-gray-500 mt-1">
                    {EVIDENCE_TYPE_LABELS[selectedEvidence.evidenceType]} • {selectedEvidence.source}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowEvidenceModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <XMarkIcon className="w-6 h-6" />
              </button>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* Description */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Description</h3>
              <p className="text-gray-900">{selectedEvidence.description}</p>
            </div>

            {/* Scores */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Relevance Score</h3>
                <ConfidenceMeter
                  confidenceScore={selectedEvidence.relevanceScore}
                  analysisType="incident_analysis"
                  size="sm"
                />
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Quality Score</h3>
                <ConfidenceMeter
                  confidenceScore={selectedEvidence.qualityScore}
                  analysisType="incident_analysis"
                  size="sm"
                />
              </div>
              {selectedEvidence.correlationScore && (
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Correlation Score</h3>
                  <ConfidenceMeter
                    confidenceScore={selectedEvidence.correlationScore}
                    analysisType="incident_analysis"
                    size="sm"
                  />
                </div>
              )}
            </div>

            {/* Metadata */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Timestamp</h3>
                <p className="text-gray-900">{format(new Date(selectedEvidence.timestamp), 'PPpp')}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Collected At</h3>
                <p className="text-gray-900">{format(new Date(selectedEvidence.collectedAt), 'PPpp')}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Validation Status</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  selectedEvidence.validationStatus === 'validated' ? 'bg-green-100 text-green-800' :
                  selectedEvidence.validationStatus === 'rejected' ? 'bg-red-100 text-red-800' :
                  selectedEvidence.validationStatus === 'expired' ? 'bg-gray-100 text-gray-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {selectedEvidence.validationStatus || 'pending'}
                </span>
              </div>
              {selectedEvidence.gcpDashboardUrl && (
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">GCP Dashboard</h3>
                  <a
                    href={selectedEvidence.gcpDashboardUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary-600 hover:text-primary-800 text-sm flex items-center space-x-1"
                  >
                    <LinkIcon className="w-4 h-4" />
                    <span>View in Console</span>
                  </a>
                </div>
              )}
            </div>

            {/* Raw Data */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Raw Data</h3>
              <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm text-gray-900 whitespace-pre-wrap">
                  {JSON.stringify(selectedEvidence.data, null, 2)}
                </pre>
              </div>
            </div>

            {/* Tags */}
            {selectedEvidence.tags && selectedEvidence.tags.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Tags</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedEvidence.tags.map((tag, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Analysis Evidence</h2>
          <p className="text-gray-600 mt-1">
            Supporting evidence from GCP observability tools and historical analysis
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setExpandedGroups(new Set(evidenceGroups.map(g => g.type)))}
            className="btn-secondary text-sm"
          >
            Expand All
          </button>
          <button
            onClick={() => setExpandedGroups(new Set())}
            className="btn-secondary text-sm"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-white p-4 rounded-lg shadow-soft border border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">{allEvidence.length}</p>
            <p className="text-sm text-gray-500">Total Evidence</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {allEvidence.filter(e => e.validationStatus === 'validated').length}
            </p>
            <p className="text-sm text-gray-500">Validated</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {allEvidence.filter(e => e.relevanceScore >= 0.8).length}
            </p>
            <p className="text-sm text-gray-500">High Relevance</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">
              {evidenceGroups.length}
            </p>
            <p className="text-sm text-gray-500">Evidence Types</p>
          </div>
        </div>
      </div>

      {/* Filters */}
      {renderFilterPanel()}

      {/* Evidence Groups */}
      <div className="space-y-4">
        {evidenceGroups.length === 0 ? (
          <div className="text-center py-12">
            <InformationCircleIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No evidence found</h3>
            <p className="mt-1 text-sm text-gray-500">
              Try adjusting your filters or check back after analysis completes.
            </p>
          </div>
        ) : (
          evidenceGroups.map(renderEvidenceGroup)
        )}
      </div>

      {/* Evidence Modal */}
      {renderEvidenceModal()}
    </div>
  );
};

export default EvidencePanel;