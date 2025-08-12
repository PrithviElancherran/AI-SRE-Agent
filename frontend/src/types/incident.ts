/**
 * TypeScript type definitions for incident-related data structures.
 * 
 * This module defines the Incident model, status enums, severity levels,
 * and incident management interfaces used throughout the React frontend.
 */

export enum IncidentSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum IncidentStatus {
  OPEN = 'open',
  INVESTIGATING = 'investigating',
  IDENTIFIED = 'identified',
  MONITORING = 'monitoring',
  RESOLVED = 'resolved',
  CLOSED = 'closed'
}

export interface IncidentSymptom {
  symptomId: string;
  description: string;
  severity: IncidentSeverity;
  observedAt: string;
  source: string;
  metadata?: Record<string, any>;
}

export interface IncidentCorrelation {
  correlationId: string;
  similarIncidentId: string;
  similarityScore: number;
  correlationFactors: string[];
  confidence: number;
  timestamp: string;
}

export interface Incident {
  id: string;
  incidentId: string;
  title: string;
  description: string;
  severity: IncidentSeverity;
  status: IncidentStatus;
  serviceName: string;
  region: string;
  symptoms: string[];
  timestamp: string;
  detectedAt?: string;
  acknowledgedAt?: string;
  resolvedAt?: string;
  closedAt?: string;
  mttrMinutes?: number;
  rootCause?: string;
  resolution?: string;
  impactedUsers?: number;
  estimatedDowntime?: number;
  tags?: string[];
  assignedTo?: string;
  createdBy: string;
  createdAt: string;
  updatedBy?: string;
  updatedAt?: string;
  deletedBy?: string;
  deletedAt?: string;
  correlations?: IncidentCorrelation[];
  relatedIncidents?: string[];
  escalationLevel?: number;
  communicationStatus?: 'pending' | 'notified' | 'escalated';
  businessImpact?: 'low' | 'medium' | 'high' | 'critical';
  customerImpact?: 'none' | 'minimal' | 'moderate' | 'significant' | 'severe';
}

export interface IncidentCreate {
  title: string;
  description: string;
  severity: IncidentSeverity;
  serviceName: string;
  region: string;
  symptoms: string[];
  tags?: string[];
  assignedTo?: string;
  autoAnalyze?: boolean;
  businessImpact?: 'low' | 'medium' | 'high' | 'critical';
  customerImpact?: 'none' | 'minimal' | 'moderate' | 'significant' | 'severe';
}

export interface IncidentUpdate {
  title?: string;
  description?: string;
  severity?: IncidentSeverity;
  status?: IncidentStatus;
  serviceName?: string;
  region?: string;
  symptoms?: string[];
  rootCause?: string;
  resolution?: string;
  assignedTo?: string;
  tags?: string[];
  escalationLevel?: number;
  communicationStatus?: 'pending' | 'notified' | 'escalated';
  businessImpact?: 'low' | 'medium' | 'high' | 'critical';
  customerImpact?: 'none' | 'minimal' | 'moderate' | 'significant' | 'severe';
}

export interface IncidentTimelineEvent {
  eventId: string;
  timestamp: string;
  eventType: 'created' | 'updated' | 'assigned' | 'escalated' | 'resolved' | 'closed' | 'comment' | 'analysis_started' | 'analysis_completed';
  description: string;
  actor: string;
  details?: Record<string, any>;
  severity?: 'info' | 'warning' | 'error' | 'critical';
  source?: string;
  metadata?: Record<string, any>;
}

export interface IncidentMetrics {
  totalIncidents: number;
  openIncidents: number;
  criticalIncidents: number;
  averageMttr: number;
  incidentsByService: Record<string, number>;
  incidentsBySeverity: Record<IncidentSeverity, number>;
  incidentsByStatus: Record<IncidentStatus, number>;
  resolutionTrends: Array<{
    date: string;
    resolved: number;
    created: number;
    avgMttr: number;
  }>;
  topAffectedServices: Array<{
    serviceName: string;
    incidentCount: number;
    avgSeverity: number;
  }>;
}

export interface IncidentAnalysisRequest {
  incidentId: string;
  analysisType: 'quick' | 'comprehensive' | 'correlation';
  includeHistorical?: boolean;
  timeRangeHours?: number;
  context?: Record<string, any>;
}

export interface IncidentSearchFilters {
  status?: IncidentStatus[];
  severity?: IncidentSeverity[];
  serviceName?: string[];
  assignedTo?: string[];
  dateRange?: {
    start: string;
    end: string;
  };
  tags?: string[];
  searchTerm?: string;
  sortBy?: 'timestamp' | 'severity' | 'status' | 'mttr';
  sortOrder?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

export interface IncidentSearchResult {
  incidents: Incident[];
  totalCount: number;
  filters: IncidentSearchFilters;
  aggregations: {
    severityDistribution: Record<IncidentSeverity, number>;
    statusDistribution: Record<IncidentStatus, number>;
    serviceDistribution: Record<string, number>;
  };
}

export interface IncidentEscalation {
  escalationId: string;
  incidentId: string;
  level: number;
  escalatedTo: string;
  escalatedBy: string;
  escalatedAt: string;
  reason: string;
  acknowledged: boolean;
  acknowledgedAt?: string;
  acknowledgedBy?: string;
  resolved: boolean;
  resolvedAt?: string;
  notes?: string;
}

export interface IncidentComment {
  commentId: string;
  incidentId: string;
  content: string;
  author: string;
  createdAt: string;
  updatedAt?: string;
  commentType: 'update' | 'investigation' | 'resolution' | 'escalation' | 'general';
  visibility: 'internal' | 'customer' | 'stakeholder';
  attachments?: Array<{
    fileName: string;
    fileUrl: string;
    fileSize: number;
    uploadedAt: string;
  }>;
}

export interface IncidentNotification {
  notificationId: string;
  incidentId: string;
  recipientType: 'individual' | 'team' | 'stakeholder' | 'customer';
  recipients: string[];
  channel: 'email' | 'slack' | 'sms' | 'webhook' | 'in_app';
  template: string;
  content: string;
  sentAt?: string;
  deliveredAt?: string;
  readAt?: string;
  status: 'pending' | 'sent' | 'delivered' | 'failed' | 'read';
  errorMessage?: string;
  retryCount?: number;
}

export interface IncidentSummary {
  incidentId: string;
  title: string;
  severity: IncidentSeverity;
  status: IncidentStatus;
  serviceName: string;
  duration: number;
  impactedUsers: number;
  rootCause: string;
  resolution: string;
  lessonsLearned: string[];
  preventiveMeasures: string[];
  followUpActions: Array<{
    action: string;
    assignee: string;
    dueDate: string;
    status: 'pending' | 'in_progress' | 'completed';
  }>;
  timeline: IncidentTimelineEvent[];
  metrics: {
    detectionTime: number;
    acknowledgmentTime: number;
    resolutionTime: number;
    communicationDelay: number;
  };
}

// Utility types for API responses
export interface IncidentApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface IncidentListResponse extends IncidentApiResponse {
  data: {
    incidents: Incident[];
    totalCount: number;
    hasMore: boolean;
    nextOffset?: number;
  };
}

export interface IncidentAnalysisResponse extends IncidentApiResponse {
  data: {
    analysisId: string;
    incidentId: string;
    status: 'started' | 'in_progress' | 'completed' | 'failed';
    progress: number;
    rootCause?: string;
    confidence?: number;
    recommendations: string[];
    similarIncidents: Array<{
      incidentId: string;
      similarity: number;
      title: string;
    }>;
    evidence: Array<{
      type: string;
      description: string;
      relevance: number;
      data: any;
    }>;
  };
}

// WebSocket event types for real-time updates
export interface IncidentWebSocketEvent {
  type: 'incident_created' | 'incident_updated' | 'incident_assigned' | 'incident_escalated' | 'incident_resolved' | 'analysis_update';
  incidentId: string;
  data: any;
  timestamp: string;
  userId?: string;
}

// Incident validation schemas (for form validation)
export interface IncidentValidationError {
  field: string;
  message: string;
  code: string;
}

export interface IncidentFormState {
  data: Partial<IncidentCreate>;
  errors: IncidentValidationError[];
  isValid: boolean;
  isSubmitting: boolean;
  isDirty: boolean;
}

// Constants for incident management
export const INCIDENT_SEVERITY_COLORS: Record<IncidentSeverity, string> = {
  [IncidentSeverity.LOW]: '#22c55e',
  [IncidentSeverity.MEDIUM]: '#f59e0b',
  [IncidentSeverity.HIGH]: '#ef4444',
  [IncidentSeverity.CRITICAL]: '#dc2626'
};

export const INCIDENT_STATUS_COLORS: Record<IncidentStatus, string> = {
  [IncidentStatus.OPEN]: '#ef4444',
  [IncidentStatus.INVESTIGATING]: '#f59e0b',
  [IncidentStatus.IDENTIFIED]: '#3b82f6',
  [IncidentStatus.MONITORING]: '#8b5cf6',
  [IncidentStatus.RESOLVED]: '#22c55e',
  [IncidentStatus.CLOSED]: '#6b7280'
};

export const INCIDENT_SEVERITY_PRIORITY: Record<IncidentSeverity, number> = {
  [IncidentSeverity.CRITICAL]: 4,
  [IncidentSeverity.HIGH]: 3,
  [IncidentSeverity.MEDIUM]: 2,
  [IncidentSeverity.LOW]: 1
};

export const INCIDENT_STATUS_ORDER: Record<IncidentStatus, number> = {
  [IncidentStatus.OPEN]: 1,
  [IncidentStatus.INVESTIGATING]: 2,
  [IncidentStatus.IDENTIFIED]: 3,
  [IncidentStatus.MONITORING]: 4,
  [IncidentStatus.RESOLVED]: 5,
  [IncidentStatus.CLOSED]: 6
};