import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ChatInterface from './components/ChatInterface';
import IncidentDashboard from './components/IncidentDashboard';
import PlaybookViewer from './components/PlaybookViewer';
import ConfidenceMeter from './components/ConfidenceMeter';
import TimelineView from './components/TimelineView';
import EvidencePanel from './components/EvidencePanel';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import { useWebSocket } from './hooks/useWebSocket';
import { useIncidents } from './hooks/useIncidents';
import { usePlaybooks } from './hooks/usePlaybooks';
import { Incident, IncidentStatus } from './types/incident';
import { PlaybookExecution } from './types/playbook';
import { AnalysisResult, AnalysisType, ConfidenceLevel } from './types/analysis';
import './App.css';

interface AppState {
  currentView: 'chat' | 'incidents';
  selectedIncident: Incident | null;
  selectedExecution: PlaybookExecution | null;
  isConnected: boolean;
  user: {
    id: string;
    username: string;
    role: string;
  } | null;
}

function App() {
  const [appState, setAppState] = useState<AppState>({
    currentView: 'chat',
    selectedIncident: null,
    selectedExecution: null,
    isConnected: false,
    user: {
      id: 'user_001',
      username: 'SRE Engineer',
      role: 'engineer'
    }
  });

  const [notifications, setNotifications] = useState<any[]>([]);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null);
  const [isFlowComplete, setIsFlowComplete] = useState(false);

  // WebSocket connection
  const {
    socket,
    isConnected,
    sendMessage,
    subscribeToUpdates,
    unsubscribeFromUpdates
  } = useWebSocket('ws://localhost:8000/ws/chat', appState.user);

  // Data hooks
  const { 
    incidents, 
    loading: incidentsLoading, 
    createIncident, 
    updateIncident,
    refreshIncidents
  } = useIncidents();

  const {
    playbooks,
    executions,
    loading: playbooksLoading,
    executePlaybook,
    refreshPlaybooks
  } = usePlaybooks();

  // Update connection status
  useEffect(() => {
    setAppState(prev => ({ ...prev, isConnected }));
  }, [isConnected]);

  // Subscribe to real-time updates
  useEffect(() => {
    if (socket) {
      // Subscribe to incident analysis updates
      subscribeToUpdates('incident_analysis_update', (data) => {
        console.log('Analysis update received:', data);
        
        // Only process these updates if they're not from chat-based analysis
        // Chat-based analysis is handled through ai_response messages
        const isFromChatAnalysis = data.analysis_id && notifications.some(n => 
          n.type === 'ai_response' && 
          n.status === 'completed' && 
          n.full_analysis?.analysis_id === data.analysis_id
        );
        
        if (!isFromChatAnalysis) {
          if (data.status === 'started') {
            setIsAnalyzing(true);
            setCurrentAnalysisId(data.analysis_id);
          } else if (data.status === 'completed') {
            setIsAnalyzing(false);
            setCurrentAnalysisId(null);
          }
        } else {
          console.log('Skipping incident_analysis_update for chat-based analysis:', data.analysis_id);
        }
        
        if (data.status === 'completed') {
          // Always clear analyzing state when completed, regardless of source
          setIsAnalyzing(false);
          setCurrentAnalysisId(null);
          
          // Add to analysis results
          const analysisResult: AnalysisResult = {
            analysisId: data.analysis_id,
            incidentId: data.incident_id,
            analysisType: AnalysisType.COMPREHENSIVE,
            status: data.status,
            progress: 100,
            startedAt: new Date().toISOString(),
            completedAt: new Date().toISOString(),
            performedBy: 'ai-sre-bot',
            confidenceScore: data.confidence || 0,
            confidenceLevel: data.confidence >= 0.85 ? ConfidenceLevel.VERY_HIGH :
                           data.confidence >= 0.7 ? ConfidenceLevel.HIGH :
                           data.confidence >= 0.5 ? ConfidenceLevel.MEDIUM :
                           data.confidence >= 0.3 ? ConfidenceLevel.LOW : ConfidenceLevel.VERY_LOW,
            rootCause: data.root_cause,
            similarIncidents: [],
            recommendations: data.recommendations || [],
            evidenceItems: data.evidence || [],
            reasoningTrail: [],
            correlations: [],
            patterns: [],
            metrics: {
              evidenceQuality: 0.8,
              analysisDepth: 0.85,
              correlationStrength: 0.75,
              patternReliability: 0.7,
              recommendationRelevance: 0.9
            },
            limitations: [],
            assumptions: [],
            timestamp: new Date().toISOString(),
            version: '1.0',
            tags: [],
            metadata: {}
          };
          
          setAnalysisResults(prev => [analysisResult, ...prev.slice(0, 9)]);
        }
        
        // Add notification
        addNotification({
          type: 'analysis_update',
          title: 'Incident Analysis Update',
          message: data.message,
          timestamp: new Date().toISOString(),
          data: data
        });
      });

      // Subscribe to playbook execution updates
      subscribeToUpdates('playbook_execution_update', (data) => {
        console.log('Playbook update received:', data);
        
        addNotification({
          type: 'playbook_update',
          title: 'Playbook Execution Update',
          message: data.message,
          timestamp: new Date().toISOString(),
          data: data
        });
      });

      // Subscribe to AI responses
      subscribeToUpdates('ai_response', (data) => {
        console.log('AI response received:', data);
        
        addNotification({
          type: 'ai_response',
          title: 'AI SRE Agent',
          message: data.content,
          timestamp: new Date().toISOString(),
          data: data,
          // Pass full_analysis directly to the notification
          full_analysis: data.full_analysis,
          full_playbook_result: data.full_playbook_result
        });
      });

      // Subscribe to system status updates
      subscribeToUpdates('system_status', (data) => {
        console.log("System status received:", data);
        
        addNotification({
          type: "system_status",
          title: "System Status",
          message: formatSystemStatusMessage(data.status),
          timestamp: new Date().toISOString(),
          data: data,
          system_status: data.status
        });
      });

      return () => {
        unsubscribeFromUpdates('incident_analysis_update');
        unsubscribeFromUpdates('playbook_execution_update');
        unsubscribeFromUpdates('ai_response');
        unsubscribeFromUpdates('system_status');
      };
    }
  }, [socket, subscribeToUpdates, unsubscribeFromUpdates]);

  const formatSystemStatusMessage = (statusData: any) => {
    const statusEmoji = statusData.overall_status === "healthy" ? "✅" : "⚠️";
    return `System Status: ${statusEmoji} ${statusData.overall_status.toUpperCase()} | Active: ${statusData.active_incidents || 0} incidents, ${statusData.active_analyses || 0} analyses | GCP Services: ${Object.values(statusData.gcp_services || {}).filter(s => s === "healthy").length}/${Object.keys(statusData.gcp_services || {}).length} healthy`;
  };
  const notificationIds = useRef<Set<string>>(new Set());

  // const addNotification = (notification: any) => {
  //   setNotifications(prev => [
  //     { ...notification, id: Date.now() + Math.random() },
  //     ...prev.slice(0, 19) // Keep only last 20 notifications
  //   ]);
  // };
  const addNotification = (notification: any) => {
  const id = notification.id || `${notification.timestamp}_${notification.type}`;

  // ✅ Skip if already processed
  if (notificationIds.current.has(id)) return;

  // 🛑 HARD STOP: Skip if flow is complete
  if (isFlowComplete) {
    console.log("🛑 Flow is complete. Blocking notification:", notification.type);
    return;
  }

  // ✅ Mark as processed
  notificationIds.current.add(id);

  // ✅ Add to notification list
  setNotifications(prev => [
    { ...notification, id }, 
    ...prev.slice(0, 19)
  ]);
};

  const removeNotification = (id: number) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const handleViewChange = (view: AppState['currentView']) => {
    setAppState(prev => ({ ...prev, currentView: view }));
  };

  const handleIncidentSelect = (incident: Incident) => {
    setAppState(prev => ({ 
      ...prev, 
      selectedIncident: incident,
      currentView: 'incidents'
    }));
  };

  const handleExecutionSelect = (execution: PlaybookExecution) => {
    setAppState(prev => ({ 
      ...prev, 
      selectedExecution: execution,
      currentView: 'incidents'
    }));
  };

  const handleAnalyzeIncident = async (incidentId: string) => {
    if (!socket) return;
    
    setIsAnalyzing(true);
    
    sendMessage({
      type: 'incident_analysis_request',
      incident_id: incidentId,
      analysis_type: 'correlation',
      context: {
        user_id: appState.user?.id,
        timestamp: new Date().toISOString()
      }
    });
  };

  const handleAnalysisComplete = () => {
    setIsAnalyzing(false);
    setCurrentAnalysisId(null);
    setIsFlowComplete(true); // Mark flow as complete
  };

  const handleFlowReset = () => {
    setIsFlowComplete(false); // Reset flow for new analysis
  };

  const handleExecutePlaybook = async (playbookId: string, incidentId: string) => {
    if (!socket) return;
    
    sendMessage({
      type: 'playbook_execution_request',
      playbook_id: playbookId,
      incident_id: incidentId,
      execution_mode: 'automatic',
      user_context: {
        user_id: appState.user?.id,
        timestamp: new Date().toISOString()
      }
    });
  };

  const handleChatMessage = (message: string) => {
    if (!socket) return;
    
    // Set analyzing state for both analysis and playbook commands
    if (message.toLowerCase().includes('@sre-bot') && 
        (message.toLowerCase().includes('analyze') || 
         message.toLowerCase().includes('incident') ||
         message.toLowerCase().includes('execute') ||
         message.toLowerCase().includes('playbook'))) {
      setIsAnalyzing(true);
    }
    
    sendMessage({
      type: 'chat_message',
      content: message,
      user_id: appState.user?.id,
      username: appState.user?.username,
      timestamp: new Date().toISOString()
    });
  };

  const renderMainContent = () => {
    switch (appState.currentView) {
      case 'chat':
        return (
          <div className="flex flex-col h-full">
            <ChatInterface
              onSendMessage={handleChatMessage}
              isConnected={isConnected}
              user={appState.user}
              notifications={notifications}
              onAnalyzeIncident={handleAnalyzeIncident}
              onExecutePlaybook={handleExecutePlaybook}
              isAnalyzing={isAnalyzing}
              currentAnalysisId={currentAnalysisId}
              onAnalysisComplete={handleAnalysisComplete}
              isFlowComplete={isFlowComplete}
              onFlowReset={handleFlowReset}
            />
          </div>
        );
      
      case 'incidents':
        return (
          <div className="flex h-full">
            <div className="flex-1 p-6">
              <IncidentDashboard
                incidents={incidents}
                onIncidentSelect={handleIncidentSelect}
                onAnalyzeIncident={handleAnalyzeIncident}
                analysisResults={analysisResults}
                isAnalyzing={isAnalyzing}
                selectedIncident={appState.selectedIncident}
              />
            </div>
            {appState.selectedIncident && (
              <div className="w-1/3 border-l border-gray-200 bg-gray-50">
                <div className="p-4 space-y-4">
                  <TimelineView
                    incident={appState.selectedIncident}
                    className="mb-4"
                  />
                  <EvidencePanel
                    incident={appState.selectedIncident}
                    analysisResults={analysisResults.filter(a => 
                      a.incidentId === appState.selectedIncident?.incidentId
                    )}
                  />
                </div>
              </div>
            )}
          </div>
        );
      
      default:
        return <Navigate to="/chat" replace />;
    }
  };

  return (
    <Router>
      <div className="h-screen bg-gray-50 flex overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          currentView={appState.currentView}
          onViewChange={handleViewChange}
          isConnected={isConnected}
          notifications={notifications}
          onNotificationRemove={removeNotification}
        />

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <Header
            user={appState.user}
            isConnected={isConnected}
            currentView={appState.currentView}
            selectedIncident={appState.selectedIncident}
            isAnalyzing={isAnalyzing}
            notifications={notifications}
            onNotificationRemove={removeNotification}
          />

          {/* Main Content Area */}
          <main className="flex-1 overflow-hidden">
            <Routes>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route path="/chat" element={renderMainContent()} />
              <Route path="/incidents" element={renderMainContent()} />
            </Routes>
          </main>
        </div>

        {/* Connection Status Indicator */}
        <div className="fixed bottom-4 right-4 z-50">
          <div className={`flex items-center space-x-2 px-3 py-2 rounded-full text-sm font-medium ${
            isConnected 
              ? 'bg-success-100 text-success-800' 
              : 'bg-error-100 text-error-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-success-500' : 'bg-error-500'
            } ${isConnected ? 'status-pulse' : ''}`} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        {/* Analysis Progress Indicator */}
        {isAnalyzing && (
          <div className="fixed bottom-16 right-4 z-50">
            <div className="bg-primary-100 text-primary-800 px-4 py-3 rounded-lg shadow-lg border border-primary-200">
              <div className="flex items-center space-x-3">
                <div className="spinner" />
                <div>
                  <div className="font-medium">AI Analysis in Progress</div>
                  <div className="text-sm text-primary-600">
                    {currentAnalysisId ? `Analysis ID: ${currentAnalysisId}` : 'Processing...'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Router>
  );
}

export default App;