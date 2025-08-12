/**
 * Custom React hook for incident data management.
 * 
 * This hook provides incident data management including fetching, creating, updating incidents,
 * and managing incident state with API integration for the AI SRE Agent frontend.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Incident, 
  IncidentCreate, 
  IncidentUpdate, 
  IncidentSeverity, 
  IncidentStatus,
  IncidentSearchFilters,
  IncidentSearchResult,
  IncidentMetrics,
  IncidentTimelineEvent,
  IncidentApiResponse,
  IncidentListResponse
} from '../types/incident';
import { apiClient } from '../services/api';

interface UseIncidentsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  pageSize?: number;
  enableCaching?: boolean;
  cacheExpiryMinutes?: number;
}

interface UseIncidentsReturn {
  incidents: Incident[];
  selectedIncident: Incident | null;
  loading: boolean;
  error: string | null;
  hasMore: boolean;
  totalCount: number;
  metrics: IncidentMetrics | null;
  filters: IncidentSearchFilters;
  
  // CRUD operations
  createIncident: (incident: IncidentCreate) => Promise<Incident | null>;
  updateIncident: (incidentId: string, update: IncidentUpdate) => Promise<Incident | null>;
  deleteIncident: (incidentId: string) => Promise<boolean>;
  getIncident: (incidentId: string) => Promise<Incident | null>;
  
  // List operations
  refreshIncidents: () => Promise<void>;
  loadMoreIncidents: () => Promise<void>;
  searchIncidents: (filters: Partial<IncidentSearchFilters>) => Promise<void>;
  clearFilters: () => void;
  
  // Selection and navigation
  selectIncident: (incident: Incident | null) => void;
  
  // Timeline and details
  getIncidentTimeline: (incidentId: string) => Promise<IncidentTimelineEvent[]>;
  getIncidentMetrics: () => Promise<IncidentMetrics | null>;
  
  // Bulk operations
  bulkUpdateIncidents: (incidentIds: string[], update: Partial<IncidentUpdate>) => Promise<boolean>;
  bulkDeleteIncidents: (incidentIds: string[]) => Promise<boolean>;
  
  // Real-time updates
  subscribeToIncidentUpdates: (incidentId: string, callback: (incident: Incident) => void) => () => void;
}

const DEFAULT_OPTIONS: UseIncidentsOptions = {
  autoRefresh: false,
  refreshInterval: 30000, // 30 seconds
  pageSize: 20,
  enableCaching: true,
  cacheExpiryMinutes: 5
};

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const useIncidents = (options: UseIncidentsOptions = {}): UseIncidentsReturn => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // State management
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [totalCount, setTotalCount] = useState(0);
  const [metrics, setMetrics] = useState<IncidentMetrics | null>(null);
  const [filters, setFilters] = useState<IncidentSearchFilters>({
    limit: opts.pageSize,
    offset: 0,
    sortBy: 'timestamp',
    sortOrder: 'desc'
  });

  // Cache management
  const cacheRef = useRef<Map<string, { data: any; timestamp: number }>>(new Map());
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  const subscribersRef = useRef<Map<string, Set<(incident: Incident) => void>>>(new Map());

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

  // Helper function to convert backend incident format to frontend format
  const convertBackendIncident = useCallback((backendIncident: any): Incident => {
    return {
      id: backendIncident.incident_id,
      incidentId: backendIncident.incident_id,
      title: backendIncident.title,
      description: backendIncident.description || '',
      severity: backendIncident.severity as IncidentSeverity,
      status: backendIncident.status as IncidentStatus,
      serviceName: backendIncident.service_name,
      region: backendIncident.region,
      symptoms: [],
      timestamp: backendIncident.timestamp,
      mttrMinutes: backendIncident.mttr_minutes,
      createdBy: backendIncident.created_by,
      createdAt: backendIncident.timestamp
    };
  }, []);

  // Fetch incidents with filters
  const fetchIncidents = useCallback(async (
    searchFilters: IncidentSearchFilters = filters,
    append: boolean = false
  ): Promise<void> => {
    setLoading(true);
    setError(null);

    try {
      // Check cache first
      const cacheKey = `incidents_${JSON.stringify(searchFilters)}`;
      if (!append && isCacheValid(cacheKey)) {
        const cachedData = getCache(cacheKey);
        if (cachedData) {
          setIncidents(cachedData.incidents);
          setTotalCount(cachedData.totalCount);
          setHasMore(cachedData.hasMore);
          setLoading(false);
          return;
        }
      }

      // Build query parameters
      const params = new URLSearchParams();
      if (searchFilters.status?.length) {
        searchFilters.status.forEach(status => params.append('status', status));
      }
      if (searchFilters.severity?.length) {
        searchFilters.severity.forEach(severity => params.append('severity', severity));
      }
      if (searchFilters.serviceName?.length) {
        searchFilters.serviceName.forEach(service => params.append('service', service));
      }
      if (searchFilters.assignedTo?.length) {
        searchFilters.assignedTo.forEach(assignee => params.append('assigned_to', assignee));
      }
      if (searchFilters.tags?.length) {
        searchFilters.tags.forEach(tag => params.append('tags', tag));
      }
      if (searchFilters.searchTerm) {
        params.append('search', searchFilters.searchTerm);
      }
      if (searchFilters.dateRange) {
        params.append('start_date', searchFilters.dateRange.start);
        params.append('end_date', searchFilters.dateRange.end);
      }
      if (searchFilters.sortBy) {
        params.append('sort_by', searchFilters.sortBy);
      }
      if (searchFilters.sortOrder) {
        params.append('sort_order', searchFilters.sortOrder);
      }
      params.append('limit', (searchFilters.limit || opts.pageSize!).toString());
      params.append('offset', (searchFilters.offset || 0).toString());

      const response = await apiClient.getIncidents(Object.fromEntries(params));

      if (response && 'incidents' in response) {
        const newIncidents = (response as any).incidents.map(convertBackendIncident);
        const newTotalCount = (response as any).total_count || (response as any).incidents.length;
        const newHasMore = (searchFilters.offset || 0) + newIncidents.length < newTotalCount;

        if (append) {
          setIncidents(prev => [...prev, ...newIncidents]);
        } else {
          setIncidents(newIncidents);
        }
        
        setTotalCount(newTotalCount);
        setHasMore(newHasMore);

        // Cache the results
        if (!append) {
          setCache(cacheKey, {
            incidents: newIncidents,
            totalCount: newTotalCount,
            hasMore: newHasMore
          });
        }
      }
    } catch (err) {
      console.error('Failed to fetch incidents:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch incidents');
    } finally {
      setLoading(false);
    }
  }, [filters, opts.pageSize, convertBackendIncident, isCacheValid, getCache, setCache]);

  // Create new incident
  const createIncident = useCallback(async (incidentData: IncidentCreate): Promise<Incident | null> => {
    setLoading(true);
    setError(null);

    try {
      const backendIncidentData = {
        title: incidentData.title,
        description: incidentData.description,
        service_name: incidentData.serviceName,
        severity: incidentData.severity,
        region: incidentData.region,
        symptoms: incidentData.symptoms
      };

      const response = await apiClient.createIncident(backendIncidentData);

      if (response?.incident_id) {
        const newIncident = convertBackendIncident(response);
        
        // Add to local state
        setIncidents(prev => [newIncident, ...prev]);
        setTotalCount(prev => prev + 1);
        
        // Clear cache to force refresh
        cacheRef.current.clear();
        
        return newIncident;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to create incident:', err);
      setError(err instanceof Error ? err.message : 'Failed to create incident');
      return null;
    } finally {
      setLoading(false);
    }
  }, [convertBackendIncident]);

  // Update existing incident
  const updateIncident = useCallback(async (
    incidentId: string, 
    updateData: IncidentUpdate
  ): Promise<Incident | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.updateIncident(incidentId, updateData);

      if (response?.incident_id) {
        const updatedIncident = convertBackendIncident(response);
        
        // Update local state
        setIncidents(prev => 
          prev.map(incident => 
            incident.incidentId === incidentId ? updatedIncident : incident
          )
        );
        
        // Update selected incident if it matches
        if (selectedIncident?.incidentId === incidentId) {
          setSelectedIncident(updatedIncident);
        }
        
        // Notify subscribers
        const subscribers = subscribersRef.current.get(incidentId);
        if (subscribers) {
          subscribers.forEach(callback => callback(updatedIncident));
        }
        
        // Clear cache to force refresh
        cacheRef.current.clear();
        
        return updatedIncident;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to update incident:', err);
      setError(err instanceof Error ? err.message : 'Failed to update incident');
      return null;
    } finally {
      setLoading(false);
    }
  }, [convertBackendIncident, selectedIncident]);

  // Delete incident
  const deleteIncident = useCallback(async (incidentId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.deleteIncident(incidentId);

      if (response?.status_code === 403) {
        throw new Error('Admin permissions required');
      } else {
        // Remove from local state
        setIncidents(prev => prev.filter(incident => incident.incidentId !== incidentId));
        setTotalCount(prev => Math.max(0, prev - 1));
        
        // Clear selected incident if it matches
        if (selectedIncident?.incidentId === incidentId) {
          setSelectedIncident(null);
        }
        
        // Clear cache to force refresh
        cacheRef.current.clear();
        
        return true;
      }
      
      return false;
    } catch (err) {
      console.error('Failed to delete incident:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete incident');
      return false;
    } finally {
      setLoading(false);
    }
  }, [selectedIncident]);

  // Get single incident
  const getIncident = useCallback(async (incidentId: string): Promise<Incident | null> => {
    setError(null);

    try {
      // Check cache first
      const cacheKey = `incident_${incidentId}`;
      if (isCacheValid(cacheKey)) {
        const cachedIncident = getCache(cacheKey);
        if (cachedIncident) {
          return cachedIncident;
        }
      }

      const response = await apiClient.getIncident(incidentId);

      if (response?.incident_id) {
        const incident = convertBackendIncident(response);
        
        // Update local state if incident exists in list
        setIncidents(prev => 
          prev.map(existing => 
            existing.incidentId === incidentId ? incident : existing
          )
        );
        
        // Cache the result
        setCache(cacheKey, incident);
        
        return incident;
      }
      
      return null;
    } catch (err) {
      console.error('Failed to get incident:', err);
      setError(err instanceof Error ? err.message : 'Failed to get incident');
      return null;
    }
  }, [convertBackendIncident, isCacheValid, getCache, setCache]);

  // Refresh incidents
  const refreshIncidents = useCallback(async (): Promise<void> => {
    // Clear cache and refetch
    cacheRef.current.clear();
    await fetchIncidents({ ...filters, offset: 0 }, false);
  }, [filters, fetchIncidents]);

  // Load more incidents (pagination)
  const loadMoreIncidents = useCallback(async (): Promise<void> => {
    if (!hasMore || loading) return;
    
    const nextOffset = (filters.offset || 0) + (filters.limit || opts.pageSize!);
    const nextFilters = { ...filters, offset: nextOffset };
    
    await fetchIncidents(nextFilters, true);
    setFilters(nextFilters);
  }, [hasMore, loading, filters, opts.pageSize, fetchIncidents]);

  // Search incidents with filters
  const searchIncidents = useCallback(async (searchFilters: Partial<IncidentSearchFilters>): Promise<void> => {
    const newFilters = {
      ...filters,
      ...searchFilters,
      offset: 0 // Reset offset for new search
    };
    
    setFilters(newFilters);
    await fetchIncidents(newFilters, false);
  }, [filters, fetchIncidents]);

  // Clear filters
  const clearFilters = useCallback(() => {
    const defaultFilters: IncidentSearchFilters = {
      limit: opts.pageSize,
      offset: 0,
      sortBy: 'timestamp',
      sortOrder: 'desc'
    };
    
    setFilters(defaultFilters);
    fetchIncidents(defaultFilters, false);
  }, [opts.pageSize, fetchIncidents]);

  // Select incident
  const selectIncident = useCallback((incident: Incident | null) => {
    setSelectedIncident(incident);
  }, []);

  // Get incident timeline
  const getIncidentTimeline = useCallback(async (incidentId: string): Promise<IncidentTimelineEvent[]> => {
    try {
      const cacheKey = `timeline_${incidentId}`;
      if (isCacheValid(cacheKey)) {
        const cachedTimeline = getCache(cacheKey);
        if (cachedTimeline) {
          return cachedTimeline;
        }
      }

      const response = await apiClient.getIncidentTimeline(incidentId);

      if (response?.timeline) {
        const timelineEvents: IncidentTimelineEvent[] = response.timeline.map((event: any) => ({
          eventId: event.timestamp + Math.random(),
          timestamp: event.timestamp,
          eventType: event.type === 'metric' ? 'updated' : 
                     event.type === 'log' ? 'updated' :
                     event.type === 'error' ? 'escalated' :
                     event.type === 'incident' ? 'created' : 'updated',
          description: event.title,
          actor: event.source || 'system',
          details: event.data,
          severity: event.severity as any || 'info',
          source: event.source,
          metadata: event
        }));
        
        // Cache the result
        setCache(cacheKey, timelineEvents);
        
        return timelineEvents;
      }
      
      return [];
    } catch (err) {
      console.error('Failed to get incident timeline:', err);
      return [];
    }
  }, [isCacheValid, getCache, setCache]);

  // Get incident metrics
  const getIncidentMetrics = useCallback(async (): Promise<IncidentMetrics | null> => {
    try {
      const cacheKey = 'incident_metrics';
      if (isCacheValid(cacheKey)) {
        const cachedMetrics = getCache(cacheKey);
        if (cachedMetrics) {
          setMetrics(cachedMetrics);
          return cachedMetrics;
        }
      }

      // Mock metrics for demo
      const mockMetrics: IncidentMetrics = {
        totalIncidents: incidents.length || 24,
        openIncidents: incidents.filter(i => i.status === IncidentStatus.OPEN || i.status === IncidentStatus.INVESTIGATING).length || 3,
        criticalIncidents: incidents.filter(i => i.severity === IncidentSeverity.CRITICAL).length || 1,
        averageMttr: 45.5,
        incidentsByService: {
          'payment-api': 8,
          'billing-service': 6,
          'user-api': 5,
          'order-service': 3,
          'notification-service': 2
        },
        incidentsBySeverity: {
          [IncidentSeverity.LOW]: 8,
          [IncidentSeverity.MEDIUM]: 10,
          [IncidentSeverity.HIGH]: 5,
          [IncidentSeverity.CRITICAL]: 1
        },
        incidentsByStatus: {
          [IncidentStatus.OPEN]: 1,
          [IncidentStatus.INVESTIGATING]: 2,
          [IncidentStatus.IDENTIFIED]: 0,
          [IncidentStatus.MONITORING]: 0,
          [IncidentStatus.RESOLVED]: 18,
          [IncidentStatus.CLOSED]: 3
        },
        resolutionTrends: [
          { date: '2024-07-01', resolved: 4, created: 3, avgMttr: 42 },
          { date: '2024-07-02', resolved: 2, created: 5, avgMttr: 52 },
          { date: '2024-07-03', resolved: 6, created: 4, avgMttr: 38 },
          { date: '2024-07-04', resolved: 3, created: 2, avgMttr: 35 },
          { date: '2024-07-05', resolved: 5, created: 6, avgMttr: 48 },
          { date: '2024-07-06', resolved: 2, created: 4, avgMttr: 40 }
        ],
        topAffectedServices: [
          { serviceName: 'payment-api', incidentCount: 8, avgSeverity: 2.5 },
          { serviceName: 'billing-service', incidentCount: 6, avgSeverity: 2.8 },
          { serviceName: 'user-api', incidentCount: 5, avgSeverity: 2.2 },
          { serviceName: 'order-service', incidentCount: 3, avgSeverity: 2.0 }
        ]
      };

      // Cache the result
      setCache(cacheKey, mockMetrics);
      setMetrics(mockMetrics);
      
      return mockMetrics;
    } catch (err) {
      console.error('Failed to get incident metrics:', err);
      return null;
    }
  }, [incidents, isCacheValid, getCache, setCache]);

  // Bulk update incidents
  const bulkUpdateIncidents = useCallback(async (
    incidentIds: string[], 
    updateData: Partial<IncidentUpdate>
  ): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const updatePromises = incidentIds.map(id => 
        updateIncident(id, updateData as IncidentUpdate)
      );
      
      const results = await Promise.allSettled(updatePromises);
      const successCount = results.filter(r => r.status === 'fulfilled' && r.value !== null).length;
      
      return successCount === incidentIds.length;
    } catch (err) {
      console.error('Failed to bulk update incidents:', err);
      setError(err instanceof Error ? err.message : 'Failed to bulk update incidents');
      return false;
    } finally {
      setLoading(false);
    }
  }, [updateIncident]);

  // Bulk delete incidents
  const bulkDeleteIncidents = useCallback(async (incidentIds: string[]): Promise<boolean> => {
    setLoading(true);
    setError(null);

    try {
      const deletePromises = incidentIds.map(id => deleteIncident(id));
      const results = await Promise.allSettled(deletePromises);
      const successCount = results.filter(r => r.status === 'fulfilled' && r.value === true).length;
      
      return successCount === incidentIds.length;
    } catch (err) {
      console.error('Failed to bulk delete incidents:', err);
      setError(err instanceof Error ? err.message : 'Failed to bulk delete incidents');
      return false;
    } finally {
      setLoading(false);
    }
  }, [deleteIncident]);

  // Subscribe to incident updates
  const subscribeToIncidentUpdates = useCallback((
    incidentId: string, 
    callback: (incident: Incident) => void
  ): (() => void) => {
    if (!subscribersRef.current.has(incidentId)) {
      subscribersRef.current.set(incidentId, new Set());
    }
    
    subscribersRef.current.get(incidentId)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const subscribers = subscribersRef.current.get(incidentId);
      if (subscribers) {
        subscribers.delete(callback);
        if (subscribers.size === 0) {
          subscribersRef.current.delete(incidentId);
        }
      }
    };
  }, []);

  // Auto-refresh setup
  useEffect(() => {
    if (opts.autoRefresh) {
      refreshTimerRef.current = setInterval(() => {
        refreshIncidents();
      }, opts.refreshInterval);
      
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [opts.autoRefresh, opts.refreshInterval, refreshIncidents]);

  // Initial data load
  useEffect(() => {
    fetchIncidents();
    getIncidentMetrics();
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
    incidents,
    selectedIncident,
    loading,
    error,
    hasMore,
    totalCount,
    metrics,
    filters,
    
    // CRUD operations
    createIncident,
    updateIncident,
    deleteIncident,
    getIncident,
    
    // List operations
    refreshIncidents,
    loadMoreIncidents,
    searchIncidents,
    clearFilters,
    
    // Selection and navigation
    selectIncident,
    
    // Timeline and details
    getIncidentTimeline,
    getIncidentMetrics,
    
    // Bulk operations
    bulkUpdateIncidents,
    bulkDeleteIncidents,
    
    // Real-time updates
    subscribeToIncidentUpdates
  };
};

export default useIncidents;