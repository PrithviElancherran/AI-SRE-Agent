/**
 * TypeScript type definitions for analysis-related data structures.
 * 
 * This module defines the AnalysisResult model, confidence scoring, evidence collection,
 * and reasoning trail interfaces used throughout the React frontend for AI analysis.
 */

export enum AnalysisType {
  QUICK = 'quick',
  COMPREHENSIVE = 'comprehensive',
  CORRELATION = 'correlation',
  ROOT_CAUSE = 'root_cause',
  PATTERN_DETECTION = 'pattern_detection',
  PREDICTIVE = 'predictive'
}

export enum AnalysisStatus {
  NOT_STARTED = 'not_started',
  INITIALIZING = 'initializing',
  IN_PROGRESS = 'in_progress',
  COLLECTING_EVIDENCE = 'collecting_evidence',
  ANALYZING_PATTERNS = 'analyzing_patterns',
  CORRELATING_INCIDENTS = 'correlating_incidents',
  CALCULATING_CONFIDENCE = 'calculating_confidence',
  GENERATING_RECOMMENDATIONS = 'generating_recommendations',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  TIMEOUT = 'timeout'
}

export enum EvidenceType {
  GCP_MONITORING = 'gcp_monitoring',
  GCP_LOGGING = 'gcp_logging',
  GCP_ERROR_REPORTING = 'gcp_error_reporting',
  GCP_TRACING = 'gcp_tracing',
  HISTORICAL_CORRELATION = 'historical_correlation',
  USER_INPUT = 'user_input',
  EXTERNAL_API = 'external_api',
  INFRASTRUCTURE_DATA = 'infrastructure_data',
  APPLICATION_METRICS = 'application_metrics',
  SYSTEM_LOGS = 'system_logs'
}

export enum ConfidenceLevel {
  VERY_LOW = 'very_low',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  VERY_HIGH = 'very_high'
}

export interface EvidenceItem {
  evidenceId: string;
  evidenceType: EvidenceType;
  description: string;
  data: Record<string, any>;
  source: string;
  relevanceScore: number;
  qualityScore: number;
  timestamp: string;
  collectedAt: string;
  expiresAt?: string;
  metadata?: Record<string, any>;
  gcpDashboardUrl?: string;
  correlationScore?: number;
  validationStatus?: 'pending' | 'validated' | 'rejected' | 'expired';
  tags?: string[];
}

export interface ConfidenceFactor {
  factorName: string;
  factorType: 'historical_accuracy' | 'evidence_strength' | 'pattern_consistency' | 'data_quality' | 'model_confidence' | 'expert_validation';
  weight: number;
  score: number;
  contribution: number;
  explanation: string;
  details?: Record<string, any>;
  impactLevel: 'low' | 'medium' | 'high' | 'critical';
  reliability: number;
}

export interface ConfidenceScore {
  overallScore: number;
  reliabilityAssessment: ConfidenceLevel;
  factors: ConfidenceFactor[];
  qualityIndicators: Record<string, number>;
  calculationMethod: string;
  calculatedAt: string;
  validUntil?: string;
  basedOnSamples: number;
  historicalAccuracy?: number;
  uncertaintyRange?: {
    lower: number;
    upper: number;
  };
  recommendedActions: string[];
  improvementSuggestions: string[];
}

export interface ReasoningStep {
  stepNumber: number;
  stepType: 'initialization' | 'data_collection' | 'pattern_analysis' | 'correlation' | 'hypothesis_testing' | 'validation' | 'conclusion';
  title: string;
  description: string;
  action: string;
  reasoning: string;
  evidence: string[];
  confidence: number;
  timestamp: string;
  duration?: number;
  inputData?: Record<string, any>;
  outputData?: Record<string, any>;
  decisionPoints?: Array<{
    question: string;
    options: string[];
    selected: string;
    rationale: string;
  }>;
  validationChecks?: Array<{
    check: string;
    passed: boolean;
    details: string;
  }>;
  nextSteps?: string[];
}

export interface AnalysisResult {
  analysisId: string;
  incidentId: string;
  analysisType: AnalysisType;
  status: AnalysisStatus;
  progress: number;
  startedAt: string;
  completedAt?: string;
  durationSeconds?: number;
  performedBy: string;
  confidenceScore: number;
  confidenceLevel: ConfidenceLevel;
  confidenceDetails?: ConfidenceScore;
  rootCause?: {
    primaryCause: string;
    contributingFactors: string[];
    evidenceSupport: number;
    categoryType: 'infrastructure' | 'application' | 'external' | 'configuration' | 'capacity' | 'security' | 'human_error';
    severity: 'low' | 'medium' | 'high' | 'critical';
    impactScope: string[];
    timeToDetection?: number;
    preventable: boolean;
  };
  similarIncidents: Array<{
    incidentId: string;
    similarityScore: number;
    title: string;
    rootCause: string;
    resolution: string;
    resolutionTime: number;
    matchingFactors: string[];
  }>;
  recommendations: string[];
  evidenceItems: EvidenceItem[];
  reasoningTrail: ReasoningStep[];
  correlations: Array<{
    correlationId: string;
    correlationType: 'causal' | 'temporal' | 'pattern' | 'statistical';
    strength: number;
    description: string;
    evidence: string[];
    confidence: number;
  }>;
  patterns: Array<{
    patternId: string;
    patternType: 'recurring' | 'seasonal' | 'trending' | 'anomalous';
    description: string;
    frequency: number;
    significance: number;
    examples: string[];
  }>;
  metrics: {
    evidenceQuality: number;
    analysisDepth: number;
    correlationStrength: number;
    patternReliability: number;
    recommendationRelevance: number;
  };
  limitations: string[];
  assumptions: string[];
  alternativeHypotheses?: Array<{
    hypothesis: string;
    likelihood: number;
    evidence: string[];
    reasoning: string;
  }>;
  followUpActions?: Array<{
    action: string;
    priority: 'low' | 'medium' | 'high' | 'urgent';
    assignee?: string;
    dueDate?: string;
    status: 'pending' | 'in_progress' | 'completed';
  }>;
  validationResults?: {
    humanValidated: boolean;
    validatedBy?: string;
    validatedAt?: string;
    validationNotes?: string;
    accuracy?: number;
  };
  timestamp: string;
  version: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface AnalysisRequest {
  incidentId: string;
  analysisType: AnalysisType;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  scope?: 'quick' | 'standard' | 'comprehensive' | 'deep_dive';
  timeRange?: {
    start: string;
    end: string;
  };
  includeServices?: string[];
  excludeServices?: string[];
  evidenceTypes?: EvidenceType[];
  correlationThreshold?: number;
  confidenceThreshold?: number;
  maxSimilarIncidents?: number;
  context?: Record<string, any>;
  parameters?: {
    includeHistoricalData?: boolean;
    performDeepAnalysis?: boolean;
    generatePredictions?: boolean;
    validateWithExperts?: boolean;
    realTimeUpdates?: boolean;
  };
  userPreferences?: {
    notificationLevel: 'minimal' | 'standard' | 'detailed' | 'verbose';
    autoApprove?: boolean;
    requireValidation?: boolean;
  };
  requestedBy: string;
  requestedAt: string;
}

export interface AnalysisConfiguration {
  analysisType: AnalysisType;
  defaultTimeRange: number;
  evidenceCollectionTimeout: number;
  correlationThresholds: {
    weak: number;
    moderate: number;
    strong: number;
    veryStrong: number;
  };
  confidenceThresholds: {
    low: number;
    medium: number;
    high: number;
    veryHigh: number;
  };
  maxEvidenceItems: number;
  maxSimilarIncidents: number;
  enabledEvidenceTypes: EvidenceType[];
  qualityFilters: {
    minRelevanceScore: number;
    minQualityScore: number;
    maxAgeHours: number;
  };
  modelParameters: Record<string, any>;
  performanceSettings: {
    enableCaching: boolean;
    cacheExpiryHours: number;
    enableParallelProcessing: boolean;
    maxConcurrentQueries: number;
  };
}

export interface AnalysisProgress {
  analysisId: string;
  currentStage: string;
  stageProgress: number;
  overallProgress: number;
  estimatedTimeRemaining?: number;
  currentTask: string;
  completedTasks: string[];
  upcomingTasks: string[];
  stagesCompleted: number;
  totalStages: number;
  lastUpdate: string;
  issues?: Array<{
    type: 'warning' | 'error' | 'info';
    message: string;
    timestamp: string;
    resolved?: boolean;
  }>;
}

export interface AnalysisComparison {
  comparisonId: string;
  primaryAnalysisId: string;
  comparisonAnalysisIds: string[];
  comparisonType: 'confidence' | 'root_cause' | 'recommendations' | 'evidence' | 'comprehensive';
  results: {
    consensusFindings: string[];
    divergentFindings: string[];
    confidenceAlignment: number;
    rootCauseConsensus?: string;
    recommendationOverlap: number;
    evidenceCorrelation: number;
  };
  similarityScores: Record<string, number>;
  detailedComparison: Array<{
    analysisId: string;
    differences: string[];
    similarities: string[];
    uniqueFindings: string[];
    confidence: number;
  }>;
  recommendations: string[];
  timestamp: string;
}

export interface AnalysisTemplate {
  templateId: string;
  name: string;
  description: string;
  analysisType: AnalysisType;
  category: string;
  applicableScenarios: string[];
  requiredEvidenceTypes: EvidenceType[];
  optionalEvidenceTypes: EvidenceType[];
  defaultConfiguration: Partial<AnalysisConfiguration>;
  stepTemplates: Array<{
    stepType: string;
    title: string;
    description: string;
    parameters: Record<string, any>;
    estimatedDuration: number;
  }>;
  successCriteria: string[];
  qualityMetrics: string[];
  usageCount: number;
  averageSuccessRate: number;
  createdBy: string;
  createdAt: string;
  lastUsed?: string;
  tags: string[];
}

export interface AnalysisMetrics {
  analysisId: string;
  performanceMetrics: {
    totalDuration: number;
    evidenceCollectionTime: number;
    analysisProcessingTime: number;
    correlationTime: number;
    confidenceCalculationTime: number;
    reportGenerationTime: number;
  };
  qualityMetrics: {
    evidenceQualityScore: number;
    analysisDepthScore: number;
    confidenceAccuracy: number;
    recommendationRelevance: number;
    overallQualityScore: number;
  };
  resourceUtilization: {
    cpuUsage: number;
    memoryUsage: number;
    apiCallsCount: number;
    dataProcessedMB: number;
    cacheHitRate: number;
  };
  accuracyMetrics?: {
    humanValidationScore?: number;
    predictionAccuracy?: number;
    falsePositiveRate?: number;
    falseNegativeRate?: number;
  };
  userEngagement: {
    viewCount: number;
    actionsTaken: number;
    feedbackProvided: boolean;
    sharingCount: number;
  };
}

export interface AnalysisReport {
  reportId: string;
  analysisId: string;
  incidentId: string;
  reportType: 'executive_summary' | 'technical_details' | 'root_cause_analysis' | 'lessons_learned' | 'comprehensive';
  title: string;
  summary: string;
  sections: Array<{
    sectionId: string;
    title: string;
    content: string;
    charts?: Array<{
      type: 'line' | 'bar' | 'pie' | 'scatter' | 'timeline';
      data: any;
      title: string;
      description?: string;
    }>;
    tables?: Array<{
      title: string;
      headers: string[];
      rows: string[][];
    }>;
    attachments?: Array<{
      fileName: string;
      fileType: string;
      url: string;
      description: string;
    }>;
  }>;
  recommendations: Array<{
    priority: 'low' | 'medium' | 'high' | 'critical';
    category: 'immediate' | 'short_term' | 'long_term' | 'preventive';
    description: string;
    rationale: string;
    estimatedEffort: string;
    expectedImpact: string;
    assignee?: string;
    dueDate?: string;
  }>;
  appendices: Array<{
    title: string;
    content: string;
    type: 'technical_details' | 'raw_data' | 'methodology' | 'references';
  }>;
  generatedAt: string;
  generatedBy: string;
  version: string;
  distribution: string[];
  confidentialityLevel: 'public' | 'internal' | 'confidential' | 'restricted';
}

// WebSocket event types for real-time analysis updates
export interface AnalysisWebSocketEvent {
  type: 'analysis_started' | 'analysis_progress' | 'evidence_collected' | 'pattern_detected' | 'analysis_completed' | 'analysis_failed';
  analysisId: string;
  incidentId?: string;
  data: any;
  timestamp: string;
  userId?: string;
}

// API response types
export interface AnalysisApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface AnalysisListResponse extends AnalysisApiResponse {
  data: {
    analyses: AnalysisResult[];
    totalCount: number;
    hasMore: boolean;
    nextOffset?: number;
  };
}

export interface EvidenceCollectionResponse extends AnalysisApiResponse {
  data: {
    collectionId: string;
    incidentId: string;
    evidenceItems: EvidenceItem[];
    collectionSummary: {
      totalItems: number;
      sources: Record<string, number>;
      qualityScores: Record<string, number>;
      collectionTimeSeconds: number;
    };
    correlationsFound: Array<{
      correlationType: string;
      strength: number;
      description: string;
      evidenceIds: string[];
    }>;
    qualityScore: number;
    timestamp: string;
  };
}

export interface ConfidenceAnalysisResponse extends AnalysisApiResponse {
  data: {
    analysisId: string;
    incidentId: string;
    overallConfidence: number;
    reliabilityAssessment: ConfidenceLevel;
    confidenceFactors: Array<{
      factorName: string;
      factorType: string;
      weight: number;
      score: number;
      contribution: number;
      explanation: string;
    }>;
    qualityIndicators: Record<string, number>;
    calculationMethod: string;
    recommendations: string[];
    timestamp: string;
  };
}

// Form state and validation
export interface AnalysisRequestFormState {
  data: Partial<AnalysisRequest>;
  errors: Array<{
    field: string;
    message: string;
  }>;
  isValid: boolean;
  isSubmitting: boolean;
  isDirty: boolean;
}

export interface AnalysisConfigurationFormState {
  data: Partial<AnalysisConfiguration>;
  errors: Array<{
    field: string;
    message: string;
  }>;
  isValid: boolean;
  isSubmitting: boolean;
  isDirty: boolean;
}

// Constants and utility types
export const ANALYSIS_TYPE_LABELS: Record<AnalysisType, string> = {
  [AnalysisType.QUICK]: 'Quick Analysis',
  [AnalysisType.COMPREHENSIVE]: 'Comprehensive Analysis',
  [AnalysisType.CORRELATION]: 'Correlation Analysis',
  [AnalysisType.ROOT_CAUSE]: 'Root Cause Analysis',
  [AnalysisType.PATTERN_DETECTION]: 'Pattern Detection',
  [AnalysisType.PREDICTIVE]: 'Predictive Analysis'
};

export const ANALYSIS_STATUS_COLORS: Record<AnalysisStatus, string> = {
  [AnalysisStatus.NOT_STARTED]: '#6b7280',
  [AnalysisStatus.INITIALIZING]: '#8b5cf6',
  [AnalysisStatus.IN_PROGRESS]: '#3b82f6',
  [AnalysisStatus.COLLECTING_EVIDENCE]: '#06b6d4',
  [AnalysisStatus.ANALYZING_PATTERNS]: '#10b981',
  [AnalysisStatus.CORRELATING_INCIDENTS]: '#f59e0b',
  [AnalysisStatus.CALCULATING_CONFIDENCE]: '#8b5cf6',
  [AnalysisStatus.GENERATING_RECOMMENDATIONS]: '#06b6d4',
  [AnalysisStatus.COMPLETED]: '#22c55e',
  [AnalysisStatus.FAILED]: '#ef4444',
  [AnalysisStatus.CANCELLED]: '#6b7280',
  [AnalysisStatus.TIMEOUT]: '#f59e0b'
};

export const EVIDENCE_TYPE_LABELS: Record<EvidenceType, string> = {
  [EvidenceType.GCP_MONITORING]: 'GCP Monitoring',
  [EvidenceType.GCP_LOGGING]: 'GCP Logging',
  [EvidenceType.GCP_ERROR_REPORTING]: 'GCP Error Reporting',
  [EvidenceType.GCP_TRACING]: 'GCP Cloud Trace',
  [EvidenceType.HISTORICAL_CORRELATION]: 'Historical Data',
  [EvidenceType.USER_INPUT]: 'User Input',
  [EvidenceType.EXTERNAL_API]: 'External APIs',
  [EvidenceType.INFRASTRUCTURE_DATA]: 'Infrastructure Data',
  [EvidenceType.APPLICATION_METRICS]: 'Application Metrics',
  [EvidenceType.SYSTEM_LOGS]: 'System Logs'
};

export const CONFIDENCE_LEVEL_COLORS: Record<ConfidenceLevel, string> = {
  [ConfidenceLevel.VERY_LOW]: '#ef4444',
  [ConfidenceLevel.LOW]: '#f59e0b',
  [ConfidenceLevel.MEDIUM]: '#3b82f6',
  [ConfidenceLevel.HIGH]: '#10b981',
  [ConfidenceLevel.VERY_HIGH]: '#22c55e'
};

export const CONFIDENCE_LEVEL_THRESHOLDS: Record<ConfidenceLevel, [number, number]> = {
  [ConfidenceLevel.VERY_LOW]: [0, 0.3],
  [ConfidenceLevel.LOW]: [0.3, 0.5],
  [ConfidenceLevel.MEDIUM]: [0.5, 0.7],
  [ConfidenceLevel.HIGH]: [0.7, 0.85],
  [ConfidenceLevel.VERY_HIGH]: [0.85, 1.0]
};

// Utility functions type definitions
export type AnalysisValidator = (request: Partial<AnalysisRequest>) => { isValid: boolean; errors: string[] };
export type EvidenceCollector = (incidentId: string, evidenceTypes: EvidenceType[]) => Promise<EvidenceItem[]>;
export type ConfidenceCalculator = (evidence: EvidenceItem[], context: Record<string, any>) => Promise<ConfidenceScore>;
export type PatternDetector = (data: any[], options: Record<string, any>) => Promise<Array<{ pattern: string; confidence: number }>>;
export type RecommendationGenerator = (analysisResult: AnalysisResult) => Promise<string[]>;