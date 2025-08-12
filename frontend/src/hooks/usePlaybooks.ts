/**
 * Custom React hook for playbook data management.
 * 
 * This hook provides playbook data management including fetching, executing playbooks,
 * tracking execution status, and managing playbook state with API integration for the AI SRE Agent frontend.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Playbook, 
  PlaybookCreate, 
  PlaybookUpdate, 
  PlaybookExecution,
  PlaybookExecutionRequest,
  PlaybookExecutionResponse,
  PlaybookStepResult,
  ExecutionStatus,
  StepStatus,
  PlaybookEffectiveness,
  PlaybookSearchFilters,
  PlaybookSearchResult,
  PlaybookAnalytics,
  PlaybookApiResponse,
  PlaybookListResponse,
  PlaybookExecutionListResponse
} from '../types/playbook';

interface UsePlaybooksOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  pageSize?: number;
  enableCaching?: boolean;
  cacheExpiryMinutes?: number;
  trackExecutions?: boolean;
}

interface UsePlaybooksReturn {
  playbooks: Playbook[];
  executions: PlaybookExecution[];
  selectedPlaybook: Playbook | null;
  selectedExecution: PlaybookExecution | null;
  loading: boolean;
  executionLoading: boolean;
  error: string | null;
  hasMore: boolean;
  totalCount: number;
  effectiveness: Record<string, PlaybookEffectiveness>;
  filters: PlaybookSearchFilters;
  
  // CRUD operations
  createPlaybook: (playbook: PlaybookCreate) => Promise<Playbook | null>;
  updatePlaybook: (playbookId: string, update: PlaybookUpdate) => Promise<Playbook | null>;
  deletePlaybook: (playbookId: string) => Promise<boolean>;
  getPlaybook: (playbookId: string) => Promise<Playbook | null>;
  
  // Execution operations
  executePlaybook: (request: PlaybookExecutionRequest) => Promise<PlaybookExecution | null>;
  getExecutionStatus: (executionId: string) => Promise<PlaybookExecution | null>;
  pauseExecution: (executionId: string) => Promise<boolean>;
  resumeExecution: (executionId: string) => Promise<boolean>;
  cancelExecution: (executionId: string) => Promise<boolean>;
  executeNextStep: (executionId: string, parameters?: Record<string, any>) => Promise<PlaybookStepResult | null>;
  approveExecutionStep: (executionId: string, stepId: string, approved: boolean, reason?: string) => Promise<boolean>;
  
  // List operations
  refreshPlaybooks: () => Promise<void>;
  refreshExecutions: () => Promise<void>;
  loadMorePlaybooks: () => Promise<void>;
  searchPlaybooks: (filters: Partial<PlaybookSearchFilters>) => Promise<void>;
  clearFilters: () => void;
  
  // Selection and navigation
  selectPlaybook: (playbook: Playbook | null) => void;
  selectExecution: (execution: PlaybookExecution | null) => void;
  
  // Analytics and effectiveness
  getPlaybookEffectiveness: (playbookId: string) => Promise<PlaybookEffectiveness | null>;
  getPlaybookAnalytics: (playbookId: string, timeRange?: string) => Promise<PlaybookAnalytics | null>;
  
  // Bulk operations
  bulkUpdatePlaybooks: (playbookIds: string[], update: Partial<PlaybookUpdate>) => Promise<boolean>;
  bulkDeletePlaybooks: (playbookIds: string[]) => Promise<boolean>;
  
  // Real-time updates
  subscribeToExecutionUpdates: (executionId: string, callback: (execution: PlaybookExecution) => void) => () => void;
}

const DEFAULT_OPTIONS: UsePlaybooksOptions = {
  autoRefresh: false,
  refreshInterval: 30000, // 30 seconds
  pageSize: 20,
  enableCaching: true,
  cacheExpiryMinutes: 5,
  trackExecutions: true
};

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const usePlaybooks = (options: UsePlaybooksOptions = {}): UsePlaybooksReturn => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // State management
  const [playbooks, setPlaybooks] = useState<Playbook[]>([]);
  const [executions, setExecutions] = useState<PlaybookExecution[]>([]);
  const [selectedPlaybook, setSelectedPlaybook] = useState<Playbook | null>(null);
  const [selectedExecution, setSelectedExecution] = useState<PlaybookExecution | null>(null);
  const [loading, setLoading] = useState(false);
  const [executionLoading, setExecutionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [totalCount, setTotalCount] = useState(0);
  const [effectiveness, setEffectiveness] = useState<Record<string, PlaybookEffectiveness>>({});
  const [filters, setFilters] = useState<PlaybookSearchFilters>({
    limit: opts.pageSize,
    offset: 0,
    sortBy: 'name',
    sortOrder: 'asc'
  });

  // Cache management
  const cacheRef = useRef<Map<string, { data: any; timestamp: number }>>(new Map());
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const executionSubscribersRef = useRef<Map<string, Set<(execution: PlaybookExecution) => void>>>(new Map());

  // Helper function to check cache validity
  const isCacheValid = useCallback((key: string): boolean => {
    if (!opts.enableCaching) return false;
    
    const cached = cacheRef.current.get(key);
    if (!cached) return false;
    
    const expiryTime = cached.timestamp + (opts.cacheExpiryMinutes! * 60 * 1000);
    return Date.now() < expiryTime;
  }, [opts.enableCaching, opts.cacheExpiryMinutes]);

  // Helper function to set cache
  const setCache = useCallback((key: string, data: any): void => {
    if (opts.enableCaching) {
      cacheRef.current.set(key, {
        data,
        timestamp: Date.now()
      });
    }
  }, [opts.enableCaching]);

  // Helper function to get cache
  const getCache = useCallback((key: string): any => {
    if (!opts.enableCaching) return null;
    
    const cached = cacheRef.current.get(key);
    return cached ? cached.data : null;
  }, [opts.enableCaching]);

  // API call helper with error handling
  const makeApiCall = useCallback(async <T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': 'demo_user',
          ...options.headers
        },
        ...options
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      console.error(`API call failed for ${endpoint}:`, errorMessage);
      setError(errorMessage);
      return null;
    }
  }, []);

  // Fetch playbooks with filters
  const fetchPlaybooks = useCallback(async (
    searchFilters: PlaybookSearchFilters = filters,
    append: boolean = false
  ): Promise<void> => {
    setLoading(true);
    setError(null);

    try {
      // Check cache first
      const cacheKey = `playbooks_${JSON.stringify(searchFilters)}`;
      if (!append && isCacheValid(cacheKey)) {
        const cachedData = getCache(cacheKey);
        if (cachedData) {
          setPlaybooks(cachedData.playbooks);
          setTotalCount(cachedData.totalCount);
          setHasMore(cachedData.hasMore);
          setLoading(false);
          return;
        }
      }

      // Build query parameters
      const params = new URLSearchParams();
      if (searchFilters.category?.length) {
        searchFilters.category.forEach(cat => params.append('category', cat));
      }
      if (searchFilters.targetSeverity?.length) {
        searchFilters.targetSeverity.forEach(sev => params.append('severity', sev));
      }
      if (searchFilters.tags?.length) {
        searchFilters.tags.forEach(tag => params.append('tags', tag));
      }
      if (searchFilters.author?.length) {
        searchFilters.author.forEach(author => params.append('author', author));
      }
      if (searchFilters.isActive !== undefined) {
        params.append('active_only', searchFilters.isActive.toString());
      }
      if (searchFilters.effectivenessScore) {
        params.append('min_effectiveness', searchFilters.effectivenessScore.min.toString());
        params.append('max_effectiveness', searchFilters.effectivenessScore.max.toString());
      }
      if (searchFilters.lastExecuted) {
        params.append('start_date', searchFilters.lastExecuted.start);
        params.append('end_date', searchFilters.lastExecuted.end);
      }
      if (searchFilters.searchTerm) {
        params.append('search', searchFilters.searchTerm);
      }
      if (searchFilters.sortBy) {
        params.append('sort_by', searchFilters.sortBy);
      }
      if (searchFilters.sortOrder) {
        params.append('sort_order', searchFilters.sortOrder);
      }
      params.append('limit', (searchFilters.limit || opts.pageSize!).toString());
      params.append('offset', (searchFilters.offset || 0).toString());

      const response = await makeApiCall<PlaybookListResponse>(
        `/api/v1/playbooks?${params.toString()}`
      );

      if (response?.success && response.data) {
        const newPlaybooks = response.data.playbooks;
        const newTotalCount = response.data.totalCount;
        const newHasMore = response.data.hasMore;

        if (append) {
          setPlaybooks(prev => [...prev, ...newPlaybooks]);
        } else {
          setPlaybooks(newPlaybooks);
        }
        
        setTotalCount(newTotalCount);
        setHasMore(newHasMore);

        // Cache the results
        if (!append) {
          setCache(cacheKey, {
            playbooks: newPlaybooks,
            totalCount: newTotalCount,
            hasMore: newHasMore
          });
        }
      }
    } catch (err) {
      console.error('Failed to fetch playbooks:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch playbooks');
    } finally {
      setLoading(false);
    }
  }, [filters, opts.pageSize, makeApiCall, isCacheValid, getCache, setCache]);

  // Fetch executions
  const fetchExecutions = useCallback(async (): Promise<void> => {
    if (!opts.trackExecutions) return;

    try {
      const cacheKey = 'executions';
      if (isCacheValid(cacheKey)) {
        const cachedExecutions = getCache(cacheKey);
        if (cachedExecutions) {
          setExecutions(cachedExecutions);
          return;
        }
      }

      const response = await makeApiCall<PlaybookExecutionListResponse>(
        '/api/v1/playbooks/executions/?limit=50&sort_by=startedAt&sort_order=desc'
      );

      if (response?.success && response.data) {
        const newExecutions = response.data.executions;
        setExecutions(newExecutions);
        setCache(cacheKey, newExecutions);
      }
    } catch (err) {
      console.error('Failed to fetch executions:', err);
    }
  }, [opts.trackExecutions, makeApiCall, isCacheValid, getCache, setCache]);

  // Create new playbook
  const createPlaybook = useCallback(async (playbookData: PlaybookCreate): Promise<Playbook | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await makeApiCall<PlaybookApiResponse<Playbook>>(
        '/api/v1/playbooks',
        {
          method: 'POST',
          body: JSON.stringify(playbookData)
        }
      );

      if (response?.success && response.data) {
        const newPlaybook = response.data;
        
        // Add to local state
        setPlaybooks(prev => [newPlaybook, ...prev]);
        setTotalCount(prev => prev + 1);
        
        // Clear cache to force refresh
        cacheRef.current.clear();
        
        return newPlaybook;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to create playbook:', err);
      setError(err instanceof Error ? err.message : 'Failed to create playbook');
      return null;
    } finally {
      setLoading(false);
    }
  }, [makeApiCall]);

  // Update existing playbook
  const updatePlaybook = useCallback(async (
    playbookId: string, 
    updateData: PlaybookUpdate
  ): Promise<Playbook | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await makeApiCall<PlaybookApiResponse<Playbook>>(
        `/api/v1/playbooks/${playbookId}`,
        {
          method: 'PUT',
          body: JSON.stringify(updateData)
        }
      );

      if (response?.success && response.data) {
        const updatedPlaybook = response.data;
        
        // Update local state
        setPlaybooks(prev => 
          prev.map(playbook => 
            playbook.playbookId === playbookId ? updatedPlaybook : playbook
          )
        );
        
        // Update selected playbook if it matches
        if (selectedPlaybook?.playbookId === playbookId) {
          setSelectedPlaybook(updatedPlaybook);
        }
        
        // Clear cache to force refresh
        cacheRef.current.clear();
        
        return updatedPlaybook;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to update playbook:', err);
      setError(err instanceof Error ? err.message : 'Failed to update playbook');
      return null;
    } finally {
      setLoading(false);
    }
  }, [makeApiCall, selectedPlaybook]);

  // Delete playbook
  const deletePlaybook = useCallback(async (playbookId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const response = await makeApiCall<PlaybookApiResponse>(
        `/api/v1/playbooks/${playbookId}`,
        { method: 'DELETE' }
      );

      if (response?.success) {
        // Remove from local state
        setPlaybooks(prev => prev.filter(playbook => playbook.playbookId !== playbookId));
        setTotalCount(prev => Math.max(0, prev - 1));
        
        // Clear selected playbook if it matches
        if (selectedPlaybook?.playbookId === playbookId) {
          setSelectedPlaybook(null);
        }
        
        // Clear cache to force refresh
        cacheRef.current.clear();
        
        return true;
      }
      
      return false;
    } catch (err) {
      console.error('Failed to delete playbook:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete playbook');
      return false;
    } finally {
      setLoading(false);
    }
  }, [makeApiCall, selectedPlaybook]);

  // Get single playbook
  const getPlaybook = useCallback(async (playbookId: string): Promise<Playbook | null> => {
    setError(null);

    try {
      // Check cache first
      const cacheKey = `playbook_${playbookId}`;
      if (isCacheValid(cacheKey)) {
        const cachedPlaybook = getCache(cacheKey);
        if (cachedPlaybook) {
          return cachedPlaybook;
        }
      }

      const response = await makeApiCall<PlaybookApiResponse<Playbook>>(
        `/api/v1/playbooks/${playbookId}?include_steps=true&include_effectiveness=true`
      );

      if (response?.success && response.data) {
        const playbook = response.data;
        
        // Update local state if playbook exists in list
        setPlaybooks(prev => 
          prev.map(existing => 
            existing.playbookId === playbookId ? playbook : existing
          )
        );
        
        // Cache the result
        setCache(cacheKey, playbook);
        
        return playbook;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to get playbook:', err);
      setError(err instanceof Error ? err.message : 'Failed to get playbook');
      return null;
    }
  }, [makeApiCall, isCacheValid, getCache, setCache]);

  // Execute playbook
  const executePlaybook = useCallback(async (request: PlaybookExecutionRequest): Promise<PlaybookExecution | null> => {
    setExecutionLoading(true);
    setError(null);

    try {
      const response = await makeApiCall<PlaybookExecutionResponse>(
        '/api/v1/playbooks/execute',
        {
          method: 'POST',
          body: JSON.stringify(request)
        }
      );

      if (response?.executionId) {
        const execution: PlaybookExecution = {
          executionId: response.executionId,
          playbookId: response.playbookId,
          incidentId: response.incidentId,
          status: response.status as ExecutionStatus,
          executionMode: request.executionMode,
          currentStepNumber: response.currentStep,
          progress: response.progress,
          startedAt: new Date().toISOString(),
          executedBy: 'demo_user',
          stepResults: (response.results || []).map(result => ({
            stepId: result.stepId,
            stepNumber: result.stepNumber,
            stepType: result.stepType as any,
            status: result.status as StepStatus,
            success: result.success,
            durationSeconds: result.durationSeconds,
            resultData: result.resultData,
            evidence: result.evidence,
            errorMessage: result.errorMessage,
            escalationTriggered: result.escalationTriggered || false
          })),
          rootCauseFound: response.rootCauseFound,
          confidenceScore: response.confidenceScore,
          recommendations: response.recommendations || []
        };

        // Add to executions list
        setExecutions(prev => [execution, ...prev]);
        
        // Clear cache
        cacheRef.current.delete('executions');
        
        return execution;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to execute playbook:', err);
      setError(err instanceof Error ? err.message : 'Failed to execute playbook');
      return null;
    } finally {
      setExecutionLoading(false);
    }
  }, [makeApiCall]);

  // Get execution status
  const getExecutionStatus = useCallback(async (executionId: string): Promise<PlaybookExecution | null> => {
    try {
      const response = await makeApiCall<any>(
        `/api/v1/playbooks/execute/${executionId}/status`
      );

      if (response?.success) {
        // Update execution in local state
        setExecutions(prev => 
          prev.map(exec => 
            exec.executionId === executionId 
              ? { ...exec, status: response.status, progress: response.progress }
              : exec
          )
        );

        // Find and return updated execution
        const execution = executions.find(e => e.executionId === executionId);
        if (execution) {
          const updatedExecution = {
            ...execution,
            status: response.status,
            progress: response.progress
          };

          // Notify subscribers
          const subscribers = executionSubscribersRef.current.get(executionId);
          if (subscribers) {
            subscribers.forEach(callback => callback(updatedExecution));
          }

          return updatedExecution;
        }
      }
      
      return null;
    } catch (err) {
      console.error('Failed to get execution status:', err);
      return null;
    }
  }, [makeApiCall, executions]);

  // Pause execution
  const pauseExecution = useCallback(async (executionId: string): Promise<boolean> => {
    try {
      const response = await makeApiCall<PlaybookApiResponse>(
        `/api/v1/playbooks/execute/${executionId}/pause`,
        { method: 'POST' }
      );

      if (response?.success) {
        setExecutions(prev => 
          prev.map(exec => 
            exec.executionId === executionId 
              ? { ...exec, status: ExecutionStatus.PAUSED, pausedAt: new Date().toISOString() }
              : exec
          )
        );
        return true;
      }
      
      return false;
    } catch (err) {
      console.error('Failed to pause execution:', err);
      return false;
    }
  }, [makeApiCall]);

  // Resume execution
  const resumeExecution = useCallback(async (executionId: string): Promise<boolean> => {
    try {
      const response = await makeApiCall<PlaybookApiResponse>(
        `/api/v1/playbooks/execute/${executionId}/resume`,
        { method: 'POST' }
      );

      if (response?.success) {
        setExecutions(prev => 
          prev.map(exec => 
            exec.executionId === executionId 
              ? { ...exec, status: ExecutionStatus.RUNNING, pausedAt: undefined }
              : exec
          )
        );
        return true;
      }
      
      return false;
    } catch (err) {
      console.error('Failed to resume execution:', err);
      return false;
    }
  }, [makeApiCall]);

  // Cancel execution
  const cancelExecution = useCallback(async (executionId: string): Promise<boolean> => {
    try {
      const response = await makeApiCall<PlaybookApiResponse>(
        `/api/v1/playbooks/execute/${executionId}/cancel`,
        { method: 'POST' }
      );

      if (response?.success) {
        setExecutions(prev => 
          prev.map(exec => 
            exec.executionId === executionId 
              ? { ...exec, status: ExecutionStatus.CANCELLED, cancelledAt: new Date().toISOString() }
              : exec
          )
        );
        return true;
      }
      
      return false;
    } catch (err) {
      console.error('Failed to cancel execution:', err);
      return false;
    }
  }, [makeApiCall]);

  // Execute next step
  const executeNextStep = useCallback(async (
    executionId: string, 
    parameters?: Record<string, any>
  ): Promise<PlaybookStepResult | null> => {
    try {
      const response = await makeApiCall<any>(
        `/api/v1/playbooks/execute/${executionId}/step`,
        {
          method: 'POST',
          body: JSON.stringify({ step_parameters: parameters || {} })
        }
      );

      if (response?.success) {
        const stepResult: PlaybookStepResult = {
          stepId: response.stepId,
          stepNumber: response.stepNumber,
          stepType: response.stepType,
          status: response.status,
          success: response.success,
          durationSeconds: response.durationSeconds,
          resultData: response.resultData,
          evidence: response.evidence,
          errorMessage: response.errorMessage,
          escalationTriggered: response.escalationTriggered || false
        };

        // Update execution with new step result
        setExecutions(prev => 
          prev.map(exec => 
            exec.executionId === executionId 
              ? { 
                  ...exec, 
                  stepResults: [...exec.stepResults, stepResult],
                  currentStepNumber: response.stepNumber + 1
                }
              : exec
          )
        );

        return stepResult;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to execute next step:', err);
      return null;
    }
  }, [makeApiCall]);

  // Approve execution step
  const approveExecutionStep = useCallback(async (
    executionId: string, 
    stepId: string, 
    approved: boolean, 
    reason?: string
  ): Promise<boolean> => {
    try {
      const response = await makeApiCall<PlaybookApiResponse>(
        `/api/v1/playbooks/execute/${executionId}/approve`,
        {
          method: 'POST',
          body: JSON.stringify({
            action: approved ? 'approve' : 'reject',
            step_id: stepId,
            reason: reason || ''
          })
        }
      );

      if (response?.success) {
        // Update execution with approval
        setExecutions(prev => 
          prev.map(exec => 
            exec.executionId === executionId 
              ? { 
                  ...exec,
                  approvals: [
                    ...(exec.approvals || []),
                    {
                      stepId,
                      approvedBy: 'demo_user',
                      approvedAt: new Date().toISOString(),
                      approved,
                      reason
                    }
                  ]
                }
              : exec
          )
        );
        return true;
      }
      
      return false;
    } catch (err) {
      console.error('Failed to approve execution step:', err);
      return false;
    }
  }, [makeApiCall]);

  // Refresh playbooks
  const refreshPlaybooks = useCallback(async (): Promise<void> => {
    // Clear cache and refetch
    cacheRef.current.clear();
    await fetchPlaybooks({ ...filters, offset: 0 }, false);
  }, [filters, fetchPlaybooks]);

  // Refresh executions
  const refreshExecutions = useCallback(async (): Promise<void> => {
    cacheRef.current.delete('executions');
    await fetchExecutions();
  }, [fetchExecutions]);

  // Load more playbooks (pagination)
  const loadMorePlaybooks = useCallback(async (): Promise<void> => {
    if (!hasMore || loading) return;
    
    const nextOffset = (filters.offset || 0) + (filters.limit || opts.pageSize!);
    const nextFilters = { ...filters, offset: nextOffset };
    
    await fetchPlaybooks(nextFilters, true);
    setFilters(nextFilters);
  }, [hasMore, loading, filters, opts.pageSize, fetchPlaybooks]);

  // Search playbooks with filters
  const searchPlaybooks = useCallback(async (searchFilters: Partial<PlaybookSearchFilters>): Promise<void> => {
    const newFilters = {
      ...filters,
      ...searchFilters,
      offset: 0 // Reset offset for new search
    };
    
    setFilters(newFilters);
    await fetchPlaybooks(newFilters, false);
  }, [filters, fetchPlaybooks]);

  // Clear filters
  const clearFilters = useCallback(() => {
    const defaultFilters: PlaybookSearchFilters = {
      limit: opts.pageSize,
      offset: 0,
      sortBy: 'name',
      sortOrder: 'asc'
    };
    
    setFilters(defaultFilters);
    fetchPlaybooks(defaultFilters, false);
  }, [opts.pageSize, fetchPlaybooks]);

  // Select playbook
  const selectPlaybook = useCallback((playbook: Playbook | null) => {
    setSelectedPlaybook(playbook);
  }, []);

  // Select execution
  const selectExecution = useCallback((execution: PlaybookExecution | null) => {
    setSelectedExecution(execution);
  }, []);

  // Get playbook effectiveness
  const getPlaybookEffectiveness = useCallback(async (playbookId: string): Promise<PlaybookEffectiveness | null> => {
    try {
      const cacheKey = `effectiveness_${playbookId}`;
      if (isCacheValid(cacheKey)) {
        const cachedEffectiveness = getCache(cacheKey);
        if (cachedEffectiveness) {
          return cachedEffectiveness;
        }
      }

      const response = await makeApiCall<any>(
        `/api/v1/playbooks/${playbookId}/effectiveness`
      );

      if (response?.success) {
        const effectivenessData: PlaybookEffectiveness = {
          playbookId: response.playbookId,
          effectivenessScore: response.effectivenessScore,
          totalExecutions: response.totalExecutions,
          successfulExecutions: response.successfulExecutions,
          averageExecutionTime: response.averageExecutionTime,
          successRate: response.successRate,
          confidenceDistribution: {},
          commonFailures: response.commonFailures || [],
          improvementSuggestions: response.improvementSuggestions || [],
          trendsOverTime: [],
          performanceByService: {},
          lastUpdated: response.lastUpdated
        };

        // Update local state
        setEffectiveness(prev => ({
          ...prev,
          [playbookId]: effectivenessData
        }));

        // Cache the result
        setCache(cacheKey, effectivenessData);
        
        return effectivenessData;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to get playbook effectiveness:', err);
      return null;
    }
  }, [makeApiCall, isCacheValid, getCache, setCache]);

  // Get playbook analytics
  const getPlaybookAnalytics = useCallback(async (
    playbookId: string, 
    timeRange: string = '30d'
  ): Promise<PlaybookAnalytics | null> => {
    try {
      const cacheKey = `analytics_${playbookId}_${timeRange}`;
      if (isCacheValid(cacheKey)) {
        const cachedAnalytics = getCache(cacheKey);
        if (cachedAnalytics) {
          return cachedAnalytics;
        }
      }

      // Mock analytics for demo
      const mockAnalytics: PlaybookAnalytics = {
        playbookId,
        timeRange,
        executionMetrics: {
          totalExecutions: 24,
          successfulExecutions: 22,
          failedExecutions: 2,
          averageExecutionTime: 12.5,
          successRate: 0.92,
          failureRate: 0.08
        },
        stepAnalytics: [],
        performanceTrends: [],
        errorAnalysis: {
          commonErrors: [],
          errorTrends: []
        },
        recommendations: {
          optimization: ['Optimize step timeouts'],
          reliability: ['Add retry logic'],
          performance: ['Parallel execution'],
          maintenance: ['Update documentation']
        }
      };

      // Cache the result
      setCache(cacheKey, mockAnalytics);
      
      return mockAnalytics;
    } catch (err) {
      console.error('Failed to get playbook analytics:', err);
      return null;
    }
  }, [isCacheValid, getCache, setCache]);

  // Bulk update playbooks
  const bulkUpdatePlaybooks = useCallback(async (
    playbookIds: string[], 
    updateData: Partial<PlaybookUpdate>
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const updatePromises = playbookIds.map(id => 
        updatePlaybook(id, updateData as PlaybookUpdate)
      );
      
      const results = await Promise.allSettled(updatePromises);
      const successCount = results.filter(r => r.status === 'fulfilled' && r.value !== null).length;
      
      return successCount === playbookIds.length;
    } catch (err) {
      console.error('Failed to bulk update playbooks:', err);
      setError(err instanceof Error ? err.message : 'Failed to bulk update playbooks');
      return false;
    } finally {
      setLoading(false);
    }
  }, [updatePlaybook]);

  // Bulk delete playbooks
  const bulkDeletePlaybooks = useCallback(async (playbookIds: string[]): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const deletePromises = playbookIds.map(id => deletePlaybook(id));
      const results = await Promise.allSettled(deletePromises);
      const successCount = results.filter(r => r.status === 'fulfilled' && r.value === true).length;
      
      return successCount === playbookIds.length;
    } catch (err) {
      console.error('Failed to bulk delete playbooks:', err);
      setError(err instanceof Error ? err.message : 'Failed to bulk delete playbooks');
      return false;
    } finally {
      setLoading(false);
    }
  }, [deletePlaybook]);

  // Subscribe to execution updates
  const subscribeToExecutionUpdates = useCallback((
    executionId: string, 
    callback: (execution: PlaybookExecution) => void
  ): (() => void) => {
    if (!executionSubscribersRef.current.has(executionId)) {
      executionSubscribersRef.current.set(executionId, new Set());
    }
    
    executionSubscribersRef.current.get(executionId)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const subscribers = executionSubscribersRef.current.get(executionId);
      if (subscribers) {
        subscribers.delete(callback);
        if (subscribers.size === 0) {
          executionSubscribersRef.current.delete(executionId);
        }
      }
    };
  }, []);

  // Auto-refresh setup
  useEffect(() => {
    if (opts.autoRefresh) {
      refreshTimerRef.current = setInterval(() => {
        refreshPlaybooks();
        if (opts.trackExecutions) {
          refreshExecutions();
        }
      }, opts.refreshInterval);
      
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [opts.autoRefresh, opts.refreshInterval, opts.trackExecutions, refreshPlaybooks, refreshExecutions]);

  // Initial data load
  useEffect(() => {
    fetchPlaybooks();
    if (opts.trackExecutions) {
      fetchExecutions();
    }
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, []);

  return {
    playbooks,
    executions,
    selectedPlaybook,
    selectedExecution,
    loading,
    executionLoading,
    error,
    hasMore,
    totalCount,
    effectiveness,
    filters,
    
    // CRUD operations
    createPlaybook,
    updatePlaybook,
    deletePlaybook,
    getPlaybook,
    
    // Execution operations
    executePlaybook,
    getExecutionStatus,
    pauseExecution,
    resumeExecution,
    cancelExecution,
    executeNextStep,
    approveExecutionStep,
    
    // List operations
    refreshPlaybooks,
    refreshExecutions,
    loadMorePlaybooks,
    searchPlaybooks,
    clearFilters,
    
    // Selection and navigation
    selectPlaybook,
    selectExecution,
    
    // Analytics and effectiveness
    getPlaybookEffectiveness,
    getPlaybookAnalytics,
    
    // Bulk operations
    bulkUpdatePlaybooks,
    bulkDeletePlaybooks,
    
    // Real-time updates
    subscribeToExecutionUpdates
  };
};

export default usePlaybooks;