/**
 * API client for the AI SRE Agent backend.
 * 
 * This module provides HTTP client functionality to interact with the backend API,
 * handling authentication, error handling, and response formatting.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const AUTH_TOKEN = process.env.REACT_APP_AUTH_TOKEN || 'demo_token';

interface ApiResponse<T = any> {
  success?: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp?: string;
  [key: string]: any;
}

// Specific response types based on actual backend responses
interface IncidentsListResponse {
  incidents: any[];
  total_count: number;
  limit: number;
  offset: number;
  filters: any;
  timestamp: string;
}

interface IncidentResponse {
  incident_id: string;
  title: string;
  description?: string;
  severity: string;
  status: string;
  service_name: string;
  region: string;
  timestamp: string;
  mttr_minutes?: number;
  created_by: string;
}

interface TimelineResponse {
  incident_id: string;
  timeline: any[];
}

interface ErrorResponse {
  error: string;
  status_code: number;
  message: string;
  timestamp: string;
  path: string;
}

class ApiClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor() {
    this.baseUrl = API_BASE_URL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${AUTH_TOKEN}`,
    };
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Incident API methods
  async getIncidents(params: Record<string, any> = {}): Promise<IncidentsListResponse> {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach(v => searchParams.append(key, v));
      } else if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString());
      }
    });

    return this.makeRequest<IncidentsListResponse>(`/api/v1/incidents/?${searchParams.toString()}`);
  }

  async createIncident(incident: any): Promise<IncidentResponse> {
    return this.makeRequest<IncidentResponse>('/api/v1/incidents/', {
      method: 'POST',
      body: JSON.stringify(incident),
    });
  }

  async getIncident(incidentId: string): Promise<IncidentResponse> {
    return this.makeRequest<IncidentResponse>(`/api/v1/incidents/${incidentId}`);
  }

  async updateIncident(incidentId: string, update: any): Promise<IncidentResponse> {
    return this.makeRequest<IncidentResponse>(`/api/v1/incidents/${incidentId}`, {
      method: 'PUT',
      body: JSON.stringify(update),
    });
  }

  async deleteIncident(incidentId: string): Promise<any> {
    return this.makeRequest(`/api/v1/incidents/${incidentId}`, {
      method: 'DELETE',
    });
  }

  async getIncidentStatus(incidentId: string) {
    return this.makeRequest(`/api/v1/incidents/${incidentId}/status`);
  }

  async getIncidentTimeline(incidentId: string): Promise<TimelineResponse> {
    return this.makeRequest<TimelineResponse>(`/api/v1/incidents/${incidentId}/timeline`);
  }

  async analyzeIncident(analysisRequest: any) {
    return this.makeRequest('/api/v1/incidents/analyze', {
      method: 'POST',
      body: JSON.stringify(analysisRequest),
    });
  }

  async correlateIncident(incidentId: string) {
    return this.makeRequest(`/api/v1/incidents/${incidentId}/correlate`, {
      method: 'POST',
      body: JSON.stringify({ incident_id: incidentId }),
    });
  }

  // Playbook API methods
  async getPlaybooks(params: Record<string, any> = {}) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString());
      }
    });

    return this.makeRequest(`/api/v1/playbooks/?${searchParams.toString()}`);
  }

  async createPlaybook(playbook: any) {
    return this.makeRequest('/api/v1/playbooks/', {
      method: 'POST',
      body: JSON.stringify(playbook),
    });
  }

  async getPlaybook(playbookId: string) {
    return this.makeRequest(`/api/v1/playbooks/${playbookId}`);
  }

  async updatePlaybook(playbookId: string, update: any) {
    return this.makeRequest(`/api/v1/playbooks/${playbookId}`, {
      method: 'PUT',
      body: JSON.stringify(update),
    });
  }

  async deletePlaybook(playbookId: string) {
    return this.makeRequest(`/api/v1/playbooks/${playbookId}`, {
      method: 'DELETE',
    });
  }

  async getPlaybookEffectiveness(playbookId: string) {
    return this.makeRequest(`/api/v1/playbooks/${playbookId}/effectiveness`);
  }

  async executePlaybook(executionRequest: any) {
    return this.makeRequest('/api/v1/playbooks/execute', {
      method: 'POST',
      body: JSON.stringify(executionRequest),
    });
  }

  async getPlaybookExecutions(params: Record<string, any> = {}) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString());
      }
    });

    return this.makeRequest(`/api/v1/playbooks/executions/?${searchParams.toString()}`);
  }

  async getExecutionStatus(executionId: string) {
    return this.makeRequest(`/api/v1/playbooks/execute/${executionId}/status`);
  }

  async executeNextStep(executionId: string, stepData: any) {
    return this.makeRequest(`/api/v1/playbooks/execute/${executionId}/step`, {
      method: 'POST',
      body: JSON.stringify(stepData),
    });
  }

  async approveExecution(executionId: string, approvalData: any) {
    return this.makeRequest(`/api/v1/playbooks/execute/${executionId}/approve`, {
      method: 'POST',
      body: JSON.stringify(approvalData),
    });
  }

  async getExecution(executionId: string) {
    return this.makeRequest(`/api/v1/playbooks/executions/${executionId}`);
  }

  // Analysis API methods
  async analyzeConfidence(confidenceData: any) {
    return this.makeRequest('/api/v1/analysis/confidence', {
      method: 'POST',
      body: JSON.stringify(confidenceData),
    });
  }

  async collectEvidence(evidenceRequest: any) {
    return this.makeRequest('/api/v1/analysis/evidence/collect', {
      method: 'POST',
      body: JSON.stringify(evidenceRequest),
    });
  }

  async getReasoningTrail(trailRequest: any) {
    return this.makeRequest('/api/v1/analysis/reasoning-trail', {
      method: 'POST',
      body: JSON.stringify(trailRequest),
    });
  }

  async compareAnalyses(comparisonRequest: any) {
    return this.makeRequest('/api/v1/analysis/compare', {
      method: 'POST',
      body: JSON.stringify(comparisonRequest),
    });
  }

  async getAnalysisStatistics(params: Record<string, any> = {}) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString());
      }
    });

    return this.makeRequest(`/api/v1/analysis/statistics?${searchParams.toString()}`);
  }

  async getAnalysis(analysisId: string) {
    return this.makeRequest(`/api/v1/analysis/${analysisId}`);
  }

  async deleteAnalysis(analysisId: string) {
    return this.makeRequest(`/api/v1/analysis/${analysisId}`, {
      method: 'DELETE',
    });
  }

  // Health API methods
  async getHealth() {
    return this.makeRequest('/health');
  }

  async getDetailedHealth() {
    return this.makeRequest('/health/detailed');
  }

  async getServiceInfo() {
    return this.makeRequest('/');
  }
}

// Create and export a singleton instance
export const apiClient = new ApiClient();
export default apiClient;