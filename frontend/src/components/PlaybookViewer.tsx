/**
 * React TypeScript component for displaying and managing playbooks.
 * 
 * This component provides step-by-step execution, progress tracking,
 * and real-time updates for the AI SRE Agent frontend.
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  ArrowPathIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  DocumentTextIcon,
  CommandLineIcon,
  CogIcon,
  UserIcon,
  EyeIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
  ShareIcon,
  StarIcon,
  ClipboardDocumentIcon,
  BoltIcon,
  ServerIcon,
  CircleStackIcon,
  LinkIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import {
  PlayIcon as PlayIconSolid,
  CheckCircleIcon as CheckCircleIconSolid,
  XCircleIcon as XCircleIconSolid,
  ClockIcon as ClockIconSolid,
  StarIcon as StarIconSolid
} from '@heroicons/react/24/solid';
import { format, formatDistanceToNow } from 'date-fns';
import {
  Playbook,
  PlaybookExecution,
  PlaybookStep,
  PlaybookStepResult,
  ExecutionStatus,
  StepStatus,
  StepType,
  STEP_TYPE_LABELS,
  STEP_TYPE_ICONS,
  EXECUTION_STATUS_COLORS,
  STEP_STATUS_COLORS
} from '../types/playbook';
import ConfidenceMeter from './ConfidenceMeter';

interface PlaybookViewerProps {
  playbooks: Playbook[];
  executions: PlaybookExecution[];
  onExecutionSelect: (execution: PlaybookExecution) => void;
  onExecutePlaybook: (playbookId: string, incidentId: string) => void;
  selectedExecution?: PlaybookExecution | null;
  className?: string;
}

interface PlaybookFilter {
  search: string;
  category: string[];
  tags: string[];
  isActive: boolean | null;
  effectivenessScore: number;
}

const PlaybookViewer: React.FC<PlaybookViewerProps> = ({
  playbooks,
  executions,
  onExecutionSelect,
  onExecutePlaybook,
  selectedExecution,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'playbooks' | 'executions'>('playbooks');
  const [selectedPlaybook, setSelectedPlaybook] = useState<Playbook | null>(null);
  const [showPlaybookModal, setShowPlaybookModal] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [filters, setFilters] = useState<PlaybookFilter>({
    search: '',
    category: [],
    tags: [],
    isActive: null,
    effectivenessScore: 0
  });
  const [showFilters, setShowFilters] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Filter playbooks
  const filteredPlaybooks = useMemo(() => {
    return playbooks.filter(playbook => {
      // Search filter
      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        const searchableText = `${playbook.name} ${playbook.description} ${playbook.category}`.toLowerCase();
        if (!searchableText.includes(searchLower)) {
          return false;
        }
      }

      // Category filter
      if (filters.category.length > 0 && !filters.category.includes(playbook.category)) {
        return false;
      }

      // Tags filter
      if (filters.tags.length > 0) {
        const playbookTags = playbook.tags || [];
        if (!filters.tags.some(tag => playbookTags.includes(tag))) {
          return false;
        }
      }

      // Active status filter
      if (filters.isActive !== null && playbook.isActive !== filters.isActive) {
        return false;
      }

      // Effectiveness score filter
      if ((playbook.effectivenessScore || 0) < filters.effectivenessScore) {
        return false;
      }

      return true;
    });
  }, [playbooks, filters]);

  // Recent executions
  const recentExecutions = useMemo(() => {
    return executions
      .sort((a, b) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime())
      .slice(0, 10);
  }, [executions]);

  // Get unique filter options
  const filterOptions = useMemo(() => {
    const categories = Array.from(new Set(playbooks.map(p => p.category))).sort();
    const tags = Array.from(new Set(playbooks.flatMap(p => p.tags || []))).sort();
    return { categories, tags };
  }, [playbooks]);

  // Toggle step expansion
  const toggleStep = useCallback((stepId: string) => {
    setExpandedSteps(prev => {
      const newSet = new Set(prev);
      if (newSet.has(stepId)) {
        newSet.delete(stepId);
      } else {
        newSet.add(stepId);
      }
      return newSet;
    });
  }, []);

  // Handle playbook click
  const handlePlaybookClick = useCallback((playbook: Playbook) => {
    setSelectedPlaybook(playbook);
    setShowPlaybookModal(true);
  }, []);

  // Handle filter change
  const handleFilterChange = useCallback((key: keyof PlaybookFilter, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  // Clear filters
  const clearFilters = useCallback(() => {
    setFilters({
      search: '',
      category: [],
      tags: [],
      isActive: null,
      effectivenessScore: 0
    });
  }, []);

  // Format execution duration
  const formatExecutionDuration = useCallback((execution: PlaybookExecution) => {
    if (execution.completedAt) {
      const start = new Date(execution.startedAt);
      const end = new Date(execution.completedAt);
      const durationMs = end.getTime() - start.getTime();
      const minutes = Math.floor(durationMs / 60000);
      const seconds = Math.floor((durationMs % 60000) / 1000);
      return `${minutes}m ${seconds}s`;
    } else {
      return formatDistanceToNow(new Date(execution.startedAt), { addSuffix: true });
    }
  }, []);

  // Get step icon
  const getStepIcon = useCallback((step: PlaybookStep, result?: PlaybookStepResult) => {
    const iconClass = "w-5 h-5";
    
    if (result) {
      switch (result.status) {
        case StepStatus.COMPLETED:
          return <CheckCircleIconSolid className={`${iconClass} text-green-500`} />;
        case StepStatus.FAILED:
          return <XCircleIconSolid className={`${iconClass} text-red-500`} />;
        case StepStatus.RUNNING:
          return <ClockIconSolid className={`${iconClass} text-blue-500 animate-spin`} />;
        case StepStatus.WAITING_APPROVAL:
          return <ExclamationTriangleIcon className={`${iconClass} text-yellow-500`} />;
        case StepStatus.SKIPPED:
          return <div className={`${iconClass} bg-gray-400 rounded-full`} />;
        default:
          return <div className={`${iconClass} bg-gray-300 rounded-full`} />;
      }
    }

    return (
      <div className="text-gray-400">
        {STEP_TYPE_ICONS[step.stepType] ? (
          <span className="text-lg">{STEP_TYPE_ICONS[step.stepType]}</span>
        ) : (
          <CogIcon className={iconClass} />
        )}
      </div>
    );
  }, []);

  // Get execution status display
  const getExecutionStatusDisplay = useCallback((status: ExecutionStatus) => {
    const color = EXECUTION_STATUS_COLORS[status];
    const icons = {
      [ExecutionStatus.PENDING]: <ClockIcon className="w-4 h-4" />,
      [ExecutionStatus.RUNNING]: <PlayIconSolid className="w-4 h-4" />,
      [ExecutionStatus.PAUSED]: <PauseIcon className="w-4 h-4" />,
      [ExecutionStatus.COMPLETED]: <CheckCircleIconSolid className="w-4 h-4" />,
      [ExecutionStatus.FAILED]: <XCircleIconSolid className="w-4 h-4" />,
      [ExecutionStatus.CANCELLED]: <StopIcon className="w-4 h-4" />,
      [ExecutionStatus.WAITING_APPROVAL]: <ExclamationTriangleIcon className="w-4 h-4" />
    };

    return {
      icon: icons[status],
      color,
      label: status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')
    };
  }, []);

  // Render filter panel
  const renderFilterPanel = () => {
    if (!showFilters) return null;

    return (
      <div className="bg-white p-4 rounded-lg shadow-soft border border-gray-200 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* Search */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Search</label>
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search playbooks..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>

          {/* Category */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Category</label>
            <select
              multiple
              value={filters.category}
              onChange={(e) => handleFilterChange('category', Array.from(e.target.selectedOptions, o => o.value))}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              size={3}
            >
              {filterOptions.categories.map(category => (
                <option key={category} value={category} className="capitalize">
                  {category.replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Tags</label>
            <select
              multiple
              value={filters.tags}
              onChange={(e) => handleFilterChange('tags', Array.from(e.target.selectedOptions, o => o.value))}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              size={3}
            >
              {filterOptions.tags.map(tag => (
                <option key={tag} value={tag}>
                  {tag}
                </option>
              ))}
            </select>
          </div>

          {/* Status */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
            <select
              value={filters.isActive === null ? 'all' : filters.isActive ? 'active' : 'inactive'}
              onChange={(e) => {
                const value = e.target.value;
                handleFilterChange('isActive', value === 'all' ? null : value === 'active');
              }}
              className="w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All</option>
              <option value="active">Active Only</option>
              <option value="inactive">Inactive Only</option>
            </select>
          </div>

          {/* Effectiveness */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Min Effectiveness: {Math.round(filters.effectivenessScore * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.effectivenessScore}
              onChange={(e) => handleFilterChange('effectivenessScore', parseFloat(e.target.value))}
              className="w-full mt-2"
            />
          </div>
        </div>

        <div className="mt-4 flex justify-between items-center">
          <span className="text-sm text-gray-500">
            Showing {filteredPlaybooks.length} of {playbooks.length} playbooks
          </span>
          <button onClick={clearFilters} className="btn-secondary text-sm">
            Clear Filters
          </button>
        </div>
      </div>
    );
  };

  // Render playbook card
  const renderPlaybookCard = (playbook: Playbook) => {
    return (
      <div
        key={playbook.playbookId}
        className="bg-white p-6 rounded-lg shadow-soft border border-gray-200 hover:shadow-medium transition-all cursor-pointer dashboard-card"
        onClick={() => handlePlaybookClick(playbook)}
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <DocumentTextIcon className="w-6 h-6 text-primary-600" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-gray-900 truncate">
                {playbook.name}
              </h3>
              <p className="text-sm text-gray-500 capitalize">
                {playbook.category.replace('_', ' ')}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {playbook.effectivenessScore && (
              <div className="flex items-center space-x-1">
                <StarIconSolid className="w-4 h-4 text-yellow-500" />
                <span className="text-sm font-medium text-gray-700">
                  {Math.round(playbook.effectivenessScore * 100)}%
                </span>
              </div>
            )}
            <div className={`w-2 h-2 rounded-full ${playbook.isActive ? 'bg-green-500' : 'bg-gray-400'}`} />
          </div>
        </div>

        <p className="text-gray-600 text-sm mb-4 line-clamp-2">
          {playbook.description}
        </p>

        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Steps:</span>
            <span className="font-medium">{playbook.steps.length}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Est. Duration:</span>
            <span className="font-medium">{playbook.estimatedDurationMinutes}m</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Success Rate:</span>
            <span className="font-medium text-green-600">
              {playbook.successRate ? `${Math.round(playbook.successRate * 100)}%` : 'N/A'}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Usage Count:</span>
            <span className="font-medium">{playbook.usageCount || 0}</span>
          </div>
        </div>

        {playbook.tags && playbook.tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1">
            {playbook.tags.slice(0, 3).map((tag, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
              >
                {tag}
              </span>
            ))}
            {playbook.tags.length > 3 && (
              <span className="text-xs text-gray-500">
                +{playbook.tags.length - 3} more
              </span>
            )}
          </div>
        )}

        <div className="mt-4 pt-4 border-t border-gray-200 flex items-center justify-between">
          <div className="text-xs text-gray-500">
            Updated {formatDistanceToNow(new Date(playbook.updatedAt || playbook.createdAt), { addSuffix: true })}
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onExecutePlaybook(playbook.playbookId, 'demo_incident');
            }}
            className="btn-primary text-sm py-1 px-3"
          >
            <PlayIcon className="w-3 h-3 mr-1" />
            Execute
          </button>
        </div>
      </div>
    );
  };

  // Render execution item
  const renderExecutionItem = (execution: PlaybookExecution) => {
    const status = getExecutionStatusDisplay(execution.status);
    const playbook = playbooks.find(p => p.playbookId === execution.playbookId);

    return (
      <div
        key={execution.executionId}
        className={`p-4 bg-white border rounded-lg hover:shadow-md transition-all cursor-pointer ${
          selectedExecution?.executionId === execution.executionId
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-200'
        }`}
        onClick={() => onExecutionSelect(execution)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-3">
            <div style={{ color: status.color }}>
              {status.icon}
            </div>
            <div>
              <h4 className="font-semibold text-gray-900">
                {playbook?.name || execution.playbookId}
              </h4>
              <p className="text-sm text-gray-500">
                Execution ID: {execution.executionId}
              </p>
            </div>
          </div>
          <div className="text-right">
            <span
              className="px-2 py-1 rounded-full text-xs font-medium"
              style={{
                backgroundColor: `${status.color}20`,
                color: status.color
              }}
            >
              {status.label}
            </span>
            {execution.confidenceScore && (
              <div className="mt-1">
                <ConfidenceMeter
                  confidenceScore={execution.confidenceScore}
                  size="sm"
                  showLabel={false}
                />
              </div>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Progress:</span>
            <span className="font-medium">{execution.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-primary-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${execution.progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Duration:</span>
            <span className="font-medium">{formatExecutionDuration(execution)}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Steps:</span>
            <span className="font-medium">
              {execution.stepResults.length} / {playbook?.steps.length || 0}
            </span>
          </div>
        </div>

        {execution.recommendations && execution.recommendations.length > 0 && (
          <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200">
            <div className="text-xs font-medium text-blue-900 mb-1">Recommendations:</div>
            <div className="text-xs text-blue-800">
              {execution.recommendations.slice(0, 2).join(', ')}
              {execution.recommendations.length > 2 && '...'}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render playbook details modal
  const renderPlaybookModal = () => {
    if (!showPlaybookModal || !selectedPlaybook) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                  <DocumentTextIcon className="w-8 h-8 text-primary-600" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    {selectedPlaybook.name}
                  </h2>
                  <p className="text-gray-600 mt-1 capitalize">
                    {selectedPlaybook.category.replace('_', ' ')} • Version {selectedPlaybook.version}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => onExecutePlaybook(selectedPlaybook.playbookId, 'demo_incident')}
                  className="btn-primary"
                >
                  <PlayIcon className="w-4 h-4 mr-2" />
                  Execute Playbook
                </button>
                <button
                  onClick={() => setShowPlaybookModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <XCircleIcon className="w-6 h-6" />
                </button>
              </div>
            </div>
          </div>

          <div className="p-6">
            {/* Description */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Description</h3>
              <p className="text-gray-700">{selectedPlaybook.description}</p>
            </div>

            {/* Metadata */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">Execution Info</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Duration:</span>
                    <span className="font-medium">{selectedPlaybook.estimatedDurationMinutes}m</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Steps:</span>
                    <span className="font-medium">{selectedPlaybook.steps.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status:</span>
                    <span className={`font-medium ${selectedPlaybook.isActive ? 'text-green-600' : 'text-gray-600'}`}>
                      {selectedPlaybook.isActive ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">Performance</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Effectiveness:</span>
                    <span className="font-medium">
                      {selectedPlaybook.effectivenessScore 
                        ? `${Math.round(selectedPlaybook.effectivenessScore * 100)}%`
                        : 'N/A'
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Success Rate:</span>
                    <span className="font-medium text-green-600">
                      {selectedPlaybook.successRate 
                        ? `${Math.round(selectedPlaybook.successRate * 100)}%`
                        : 'N/A'
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Usage Count:</span>
                    <span className="font-medium">{selectedPlaybook.usageCount || 0}</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">Metadata</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Author:</span>
                    <span className="font-medium">{selectedPlaybook.author}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Created:</span>
                    <span className="font-medium">
                      {format(new Date(selectedPlaybook.createdAt), 'MMM dd, yyyy')}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Last Used:</span>
                    <span className="font-medium">
                      {selectedPlaybook.lastExecutedAt 
                        ? formatDistanceToNow(new Date(selectedPlaybook.lastExecutedAt), { addSuffix: true })
                        : 'Never'
                      }
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Prerequisites */}
            {selectedPlaybook.prerequisites && selectedPlaybook.prerequisites.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Prerequisites</h3>
                <ul className="list-disc list-inside space-y-1 text-gray-700">
                  {selectedPlaybook.prerequisites.map((prereq, index) => (
                    <li key={index}>{prereq}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Tags */}
            {selectedPlaybook.tags && selectedPlaybook.tags.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Tags</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedPlaybook.tags.map((tag, index) => (
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

            {/* Steps */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Execution Steps</h3>
              <div className="space-y-3">
                {selectedPlaybook.steps.map((step, index) => {
                  const isExpanded = expandedSteps.has(step.stepId);
                  
                  return (
                    <div key={step.stepId} className="border border-gray-200 rounded-lg">
                      <button
                        onClick={() => toggleStep(step.stepId)}
                        className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-center space-x-4">
                          <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center text-primary-600 font-semibold text-sm">
                            {step.stepNumber}
                          </div>
                          <div className="text-left">
                            <h4 className="font-medium text-gray-900">{step.title}</h4>
                            <p className="text-sm text-gray-500 capitalize">
                              {STEP_TYPE_LABELS[step.stepType]} • {step.timeoutSeconds}s timeout
                              {step.requiresApproval && ' • Requires approval'}
                              {step.isCritical && ' • Critical step'}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-lg">{STEP_TYPE_ICONS[step.stepType]}</span>
                          {isExpanded ? (
                            <ChevronDownIcon className="w-5 h-5 text-gray-400" />
                          ) : (
                            <ChevronRightIcon className="w-5 h-5 text-gray-400" />
                          )}
                        </div>
                      </button>

                      {isExpanded && (
                        <div className="px-4 pb-4 border-t border-gray-200 bg-gray-50">
                          <div className="mt-4 space-y-3">
                            <div>
                              <h5 className="text-sm font-medium text-gray-700 mb-1">Description</h5>
                              <p className="text-sm text-gray-600">{step.description}</p>
                            </div>

                            {step.command && (
                              <div>
                                <h5 className="text-sm font-medium text-gray-700 mb-1">Command</h5>
                                <div className="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm">
                                  {step.command}
                                </div>
                              </div>
                            )}

                            {step.expectedOutput && (
                              <div>
                                <h5 className="text-sm font-medium text-gray-700 mb-1">Expected Output</h5>
                                <p className="text-sm text-gray-600">{step.expectedOutput}</p>
                              </div>
                            )}

                            {step.dependencies && step.dependencies.length > 0 && (
                              <div>
                                <h5 className="text-sm font-medium text-gray-700 mb-1">Dependencies</h5>
                                <div className="flex flex-wrap gap-1">
                                  {step.dependencies.map((dep, depIndex) => (
                                    <span
                                      key={depIndex}
                                      className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded"
                                    >
                                      Step {dep}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-gray-500">Timeout:</span>
                                <span className="ml-2 font-medium">{step.timeoutSeconds}s</span>
                              </div>
                              <div>
                                <span className="text-gray-500">Critical:</span>
                                <span className={`ml-2 font-medium ${step.isCritical ? 'text-red-600' : 'text-gray-600'}`}>
                                  {step.isCritical ? 'Yes' : 'No'}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500">Requires Approval:</span>
                                <span className={`ml-2 font-medium ${step.requiresApproval ? 'text-yellow-600' : 'text-gray-600'}`}>
                                  {step.requiresApproval ? 'Yes' : 'No'}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500">Retry Policy:</span>
                                <span className="ml-2 font-medium">
                                  {step.retryPolicy ? `${step.retryPolicy.maxRetries} retries` : 'None'}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
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
          <h1 className="text-2xl font-bold text-gray-900">Playbook Management</h1>
          <p className="text-gray-600 mt-1">
            Manage and execute automated troubleshooting procedures
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button className="btn-secondary">
            <PlusIcon className="w-4 h-4 mr-2" />
            New Playbook
          </button>
          <button
            onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
            className="btn-secondary"
          >
            {viewMode === 'grid' ? 'List View' : 'Grid View'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('playbooks')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'playbooks'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Playbooks ({playbooks.length})
          </button>
          <button
            onClick={() => setActiveTab('executions')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'executions'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Recent Executions ({executions.length})
          </button>
        </nav>
      </div>

      {/* Content */}
      {activeTab === 'playbooks' ? (
        <div className="space-y-6">
          {/* Search and Filter Bar */}
          <div className="bg-white p-4 rounded-lg shadow-soft border border-gray-200">
            <div className="flex items-center space-x-4">
              <div className="flex-1 relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search playbooks..."
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
                {(filters.category.length > 0 || filters.tags.length > 0 || filters.isActive !== null) && (
                  <span className="bg-primary-600 text-white text-xs rounded-full px-2 py-0.5 ml-1">
                    {filters.category.length + filters.tags.length + (filters.isActive !== null ? 1 : 0)}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Filter Panel */}
          {renderFilterPanel()}

          {/* Playbooks Grid/List */}
          {filteredPlaybooks.length === 0 ? (
            <div className="text-center py-12">
              <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No playbooks found</h3>
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
            <div className={viewMode === 'grid' 
              ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' 
              : 'space-y-4'
            }>
              {filteredPlaybooks.map(renderPlaybookCard)}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {/* Executions Header */}
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Recent Executions</h2>
            <div className="text-sm text-gray-500">
              {recentExecutions.length} executions shown
            </div>
          </div>

          {/* Executions List */}
          {recentExecutions.length === 0 ? (
            <div className="text-center py-12">
              <ClockIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No executions found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Execute a playbook to see execution history here.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {recentExecutions.map(renderExecutionItem)}
            </div>
          )}
        </div>
      )}

      {/* Playbook Details Modal */}
      {renderPlaybookModal()}
    </div>
  );
};

export default PlaybookViewer;