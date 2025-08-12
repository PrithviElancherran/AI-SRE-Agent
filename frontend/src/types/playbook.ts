/**
 * TypeScript type definitions for playbook-related data structures.
 * 
 * This module defines the Playbook model, execution status, step types,
 * and playbook management interfaces used throughout the React frontend.
 */

export enum StepType {
  METRIC_CHECK = 'metric_check',
  LOG_ANALYSIS = 'log_analysis',
  COMMAND_EXECUTION = 'command_execution',
  API_CALL = 'api_call',
  MANUAL_VERIFICATION = 'manual_verification',
  CONDITIONAL_BRANCH = 'conditional_branch',
  ESCALATION = 'escalation',
  NOTIFICATION = 'notification'
}

export enum ExecutionStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  WAITING_APPROVAL = 'waiting_approval'
}

export enum StepStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  SKIPPED = 'skipped',
  WAITING_APPROVAL = 'waiting_approval',
  CANCELLED = 'cancelled'
}

export interface PlaybookStep {
  stepId: string;
  stepNumber: number;
  stepType: StepType;
  title: string;
  description: string;
  command?: string;
  expectedOutput?: string;
  successCriteria?: Record<string, any>;
  escalationCondition?: Record<string, any>;
  timeoutSeconds: number;
  requiresApproval: boolean;
  isCritical: boolean;
  dependencies?: string[];
  parameters?: Record<string, any>;
  retryPolicy?: {
    maxRetries: number;
    retryDelay: number;
    backoffMultiplier: number;
  };
  conditionalLogic?: {
    condition: string;
    onTrue?: string;
    onFalse?: string;
  };
  validationRules?: Array<{
    rule: string;
    message: string;
    severity: 'warning' | 'error';
  }>;
}

export interface Playbook {
  id: string;
  playbookId: string;
  name: string;
  description: string;
  category: string;
  targetSeverity?: string;
  steps: PlaybookStep[];
  estimatedDurationMinutes: number;
  version: number;
  isActive: boolean;
  tags?: string[];
  prerequisites?: string[];
  supportedServices?: string[];
  author: string;
  createdBy: string;
  createdAt: string;
  updatedBy?: string;
  updatedAt?: string;
  deletedBy?: string;
  deletedAt?: string;
  approvedBy?: string;
  approvedAt?: string;
  effectivenessScore?: number;
  usageCount?: number;
  successRate?: number;
  averageExecutionTime?: number;
  lastExecutedAt?: string;
  metadata?: Record<string, any>;
}

export interface PlaybookCreate {
  name: string;
  description: string;
  category: string;
  targetSeverity?: string;
  steps: Omit<PlaybookStep, 'stepId'>[];
  estimatedDurationMinutes: number;
  tags?: string[];
  prerequisites?: string[];
  supportedServices?: string[];
}

export interface PlaybookUpdate {
  name?: string;
  description?: string;
  category?: string;
  targetSeverity?: string;
  steps?: PlaybookStep[];
  estimatedDurationMinutes?: number;
  isActive?: boolean;
  tags?: string[];
  prerequisites?: string[];
  supportedServices?: string[];
}

export interface PlaybookStepResult {
  stepId: string;
  stepNumber: number;
  stepType: StepType;
  status: StepStatus;
  success?: boolean;
  startedAt?: string;
  completedAt?: string;
  durationSeconds?: number;
  resultData?: Record<string, any>;
  evidence?: Record<string, any>;
  errorMessage?: string;
  warningMessages?: string[];
  outputData?: any;
  metricsCollected?: Record<string, number>;
  thresholdMet?: boolean;
  escalationTriggered: boolean;
  approvalRequired?: boolean;
  approvedBy?: string;
  approvedAt?: string;
  retryCount?: number;
  nextStepOverride?: string;
}

export interface PlaybookExecution {
  executionId: string;
  playbookId: string;
  incidentId: string;
  status: ExecutionStatus;
  executionMode: 'automatic' | 'manual' | 'step_by_step';
  currentStepNumber?: number;
  progress: number;
  startedAt: string;
  completedAt?: string;
  pausedAt?: string;
  cancelledAt?: string;
  executionTimeSeconds?: number;
  executedBy: string;
  approvals?: Array<{
    stepId: string;
    approvedBy: string;
    approvedAt: string;
    approved: boolean;
    reason?: string;
  }>;
  stepResults: PlaybookStepResult[];
  rootCauseFound: boolean;
  confidenceScore?: number;
  recommendations: string[];
  failureReason?: string;
  context?: Record<string, any>;
  metadata?: Record<string, any>;
  logs?: Array<{
    timestamp: string;
    level: 'debug' | 'info' | 'warning' | 'error';
    message: string;
    data?: any;
  }>;
}

export interface PlaybookEffectiveness {
  playbookId: string;
  effectivenessScore: number;
  totalExecutions: number;
  successfulExecutions: number;
  averageExecutionTime: number;
  successRate: number;
  confidenceDistribution: Record<string, number>;
  commonFailures: Array<{
    step: string;
    failureType: string;
    count: number;
    percentage: number;
    description: string;
  }>;
  improvementSuggestions: string[];
  trendsOverTime: Array<{
    date: string;
    executions: number;
    successRate: number;
    avgExecutionTime: number;
    avgConfidence: number;
  }>;
  performanceByService: Record<string, {
    executions: number;
    successRate: number;
    avgExecutionTime: number;
  }>;
  lastUpdated: string;
}

export interface PlaybookTemplate {
  templateId: string;
  name: string;
  description: string;
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  estimatedTime: number;
  stepTemplates: Array<{
    stepType: StepType;
    title: string;
    description: string;
    template: string;
    parameters: Record<string, any>;
    customizable: boolean;
  }>;
  useCases: string[];
  prerequisites: string[];
  tags: string[];
  createdBy: string;
  createdAt: string;
  downloadCount: number;
  rating: number;
  reviews: Array<{
    userId: string;
    rating: number;
    comment: string;
    createdAt: string;
  }>;
}

export interface PlaybookLibrary {
  categories: Array<{
    name: string;
    description: string;
    playbookCount: number;
    subcategories?: string[];
  }>;
  featuredPlaybooks: Playbook[];
  recentlyUsed: Playbook[];
  mostEffective: Array<{
    playbook: Playbook;
    effectivenessScore: number;
    usageCount: number;
  }>;
  templates: PlaybookTemplate[];
  statistics: {
    totalPlaybooks: number;
    totalExecutions: number;
    averageSuccessRate: number;
    topCategories: Array<{
      category: string;
      playbookCount: number;
      executionCount: number;
    }>;
  };
}

export interface PlaybookExecutionRequest {
  playbookId: string;
  incidentId: string;
  executionMode: 'automatic' | 'manual' | 'step_by_step';
  parameters?: Record<string, any>;
  userContext?: Record<string, any>;
  overrides?: {
    skipSteps?: string[];
    modifySteps?: Record<string, Partial<PlaybookStep>>;
    addSteps?: PlaybookStep[];
  };
  approvalSettings?: {
    requireApprovalForCritical: boolean;
    approvers: string[];
    autoApproveTimeout: number;
  };
}

export interface PlaybookExecutionResponse {
  executionId: string;
  playbookId: string;
  incidentId: string;
  status: ExecutionStatus;
  progress: number;
  currentStep?: number;
  stepsCompleted: number;
  totalSteps: number;
  executionTimeSeconds: number;
  rootCauseFound: boolean;
  confidenceScore?: number;
  results: PlaybookStepResult[];
  recommendations: string[];
  timestamp: string;
}

export interface PlaybookStepExecutionResponse {
  stepId: string;
  stepNumber: number;
  stepType: StepType;
  status: StepStatus;
  success?: boolean;
  durationSeconds?: number;
  resultData?: Record<string, any>;
  evidence?: Record<string, any>;
  thresholdMet?: boolean;
  escalationTriggered: boolean;
  errorMessage?: string;
  timestamp: string;
}

export interface PlaybookAnalytics {
  playbookId: string;
  timeRange: string;
  executionMetrics: {
    totalExecutions: number;
    successfulExecutions: number;
    failedExecutions: number;
    averageExecutionTime: number;
    successRate: number;
    failureRate: number;
  };
  stepAnalytics: Array<{
    stepId: string;
    stepNumber: number;
    stepType: StepType;
    executions: number;
    successRate: number;
    averageExecutionTime: number;
    failureReasons: Record<string, number>;
    thresholdMetRate: number;
    escalationRate: number;
  }>;
  performanceTrends: Array<{
    date: string;
    executions: number;
    successRate: number;
    averageTime: number;
    confidence: number;
  }>;
  errorAnalysis: {
    commonErrors: Array<{
      error: string;
      count: number;
      percentage: number;
      affectedSteps: string[];
      recommendations: string[];
    }>;
    errorTrends: Array<{
      date: string;
      errorCount: number;
      errorRate: number;
    }>;
  };
  recommendations: {
    optimization: string[];
    reliability: string[];
    performance: string[];
    maintenance: string[];
  };
}

export interface PlaybookSearchFilters {
  category?: string[];
  targetSeverity?: string[];
  tags?: string[];
  author?: string[];
  isActive?: boolean;
  effectivenessScore?: {
    min: number;
    max: number;
  };
  lastExecuted?: {
    start: string;
    end: string;
  };
  searchTerm?: string;
  sortBy?: 'name' | 'effectivenessScore' | 'usageCount' | 'lastExecuted' | 'createdAt';
  sortOrder?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

export interface PlaybookSearchResult {
  playbooks: Playbook[];
  totalCount: number;
  filters: PlaybookSearchFilters;
  aggregations: {
    categoryDistribution: Record<string, number>;
    severityDistribution: Record<string, number>;
    tagDistribution: Record<string, number>;
    effectivenessDistribution: {
      excellent: number;
      good: number;
      average: number;
      poor: number;
    };
  };
}

export interface PlaybookValidationResult {
  isValid: boolean;
  errors: Array<{
    field: string;
    message: string;
    severity: 'error' | 'warning';
    suggestions?: string[];
  }>;
  warnings: Array<{
    field: string;
    message: string;
    impact: 'low' | 'medium' | 'high';
    suggestions?: string[];
  }>;
  stepValidation: Array<{
    stepId: string;
    stepNumber: number;
    isValid: boolean;
    errors: string[];
    warnings: string[];
  }>;
  recommendations: string[];
  estimatedReliability: number;
}

// WebSocket event types for real-time playbook updates
export interface PlaybookWebSocketEvent {
  type: 'execution_started' | 'execution_progress' | 'step_completed' | 'execution_completed' | 'execution_failed' | 'approval_required';
  executionId: string;
  playbookId: string;
  incidentId?: string;
  data: any;
  timestamp: string;
  userId?: string;
}

// API response types
export interface PlaybookApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface PlaybookListResponse extends PlaybookApiResponse {
  data: {
    playbooks: Playbook[];
    totalCount: number;
    hasMore: boolean;
    nextOffset?: number;
  };
}

export interface PlaybookExecutionListResponse extends PlaybookApiResponse {
  data: {
    executions: PlaybookExecution[];
    totalCount: number;
    hasMore: boolean;
    nextOffset?: number;
  };
}

// Form state management
export interface PlaybookFormState {
  data: Partial<PlaybookCreate>;
  errors: Array<{
    field: string;
    message: string;
  }>;
  isValid: boolean;
  isSubmitting: boolean;
  isDirty: boolean;
}

export interface PlaybookExecutionFormState {
  data: Partial<PlaybookExecutionRequest>;
  errors: Array<{
    field: string;
    message: string;
  }>;
  isValid: boolean;
  isSubmitting: boolean;
}

// Constants and utility types
export const STEP_TYPE_LABELS: Record<StepType, string> = {
  [StepType.METRIC_CHECK]: 'Metric Check',
  [StepType.LOG_ANALYSIS]: 'Log Analysis',
  [StepType.COMMAND_EXECUTION]: 'Command Execution',
  [StepType.API_CALL]: 'API Call',
  [StepType.MANUAL_VERIFICATION]: 'Manual Verification',
  [StepType.CONDITIONAL_BRANCH]: 'Conditional Branch',
  [StepType.ESCALATION]: 'Escalation',
  [StepType.NOTIFICATION]: 'Notification'
};

export const EXECUTION_STATUS_COLORS: Record<ExecutionStatus, string> = {
  [ExecutionStatus.PENDING]: '#6b7280',
  [ExecutionStatus.RUNNING]: '#3b82f6',
  [ExecutionStatus.PAUSED]: '#f59e0b',
  [ExecutionStatus.COMPLETED]: '#22c55e',
  [ExecutionStatus.FAILED]: '#ef4444',
  [ExecutionStatus.CANCELLED]: '#6b7280',
  [ExecutionStatus.WAITING_APPROVAL]: '#8b5cf6'
};

export const STEP_STATUS_COLORS: Record<StepStatus, string> = {
  [StepStatus.PENDING]: '#6b7280',
  [StepStatus.RUNNING]: '#3b82f6',
  [StepStatus.COMPLETED]: '#22c55e',
  [StepStatus.FAILED]: '#ef4444',
  [StepStatus.SKIPPED]: '#9ca3af',
  [StepStatus.WAITING_APPROVAL]: '#8b5cf6',
  [StepStatus.CANCELLED]: '#6b7280'
};

export const STEP_TYPE_ICONS: Record<StepType, string> = {
  [StepType.METRIC_CHECK]: '📊',
  [StepType.LOG_ANALYSIS]: '📋',
  [StepType.COMMAND_EXECUTION]: '⚡',
  [StepType.API_CALL]: '🔗',
  [StepType.MANUAL_VERIFICATION]: '👤',
  [StepType.CONDITIONAL_BRANCH]: '🔀',
  [StepType.ESCALATION]: '⬆️',
  [StepType.NOTIFICATION]: '📢'
};

export const PLAYBOOK_CATEGORIES = [
  'database',
  'networking',
  'performance',
  'security',
  'infrastructure',
  'application',
  'monitoring',
  'deployment',
  'backup_recovery',
  'compliance',
  'general'
] as const;

export type PlaybookCategory = typeof PLAYBOOK_CATEGORIES[number];

// Utility functions type definitions
export type PlaybookValidator = (playbook: Partial<PlaybookCreate>) => PlaybookValidationResult;
export type StepExecutor = (step: PlaybookStep, context: Record<string, any>) => Promise<PlaybookStepResult>;
export type PlaybookAnalyzer = (executions: PlaybookExecution[]) => PlaybookAnalytics;