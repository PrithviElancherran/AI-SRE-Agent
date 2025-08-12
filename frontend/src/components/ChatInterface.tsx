/**
 * React TypeScript component for the main chat interface with AI SRE Agent.
 * 
 * This component provides real-time messaging, incident analysis commands,
 * playbook execution, and WebSocket communication for the AI SRE Agent frontend.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  PaperAirplaneIcon, 
  ExclamationTriangleIcon, 
  PlayIcon, 
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  InformationCircleIcon,
  ChartBarIcon,
  DocumentTextIcon,
  CogIcon
} from '@heroicons/react/24/outline';
import { format } from 'date-fns';

interface ChatMessage {
  id: string;
  type: 'user' | 'ai' | 'system' | 'notification';
  content: string;
  timestamp: string;
  user?: {
    id: string;
    username: string;
  };
  metadata?: {
    incidentId?: string;
    playbookId?: string;
    executionId?: string;
    analysisId?: string;
    confidenceScore?: number;
    status?: string;
    requiresApproval?: boolean;
    evidence?: any[];
    recommendations?: string[];
    progress?: number;
    step?: string;
    confidence?: number;
    rootCause?: string;
    similarIncidents?: any[];
    currentStep?: number;
    totalSteps?: number;
    stepDetails?: any;
    stepResult?: any;
    rootCauseFound?: boolean;
    fullAnalysis?: any;
    fullPlaybookResult?: any;
    systemStatus?: any;
    stepType?: string;
    analysisData?: any;
    isStepMessage?: boolean;
  };
  isTyping?: boolean;
  error?: boolean;
}

interface ChatInterfaceProps {
  onSendMessage: (message: string) => void;
  isConnected: boolean;
  user: {
    id: string;
    username: string;
    role: string;
  } | null;
  notifications: any[];
  onAnalyzeIncident: (incidentId: string) => void;
  onExecutePlaybook: (playbookId: string, incidentId: string) => void;
  isAnalyzing: boolean;
  currentAnalysisId: string | null;
  onAnalysisComplete?: () => void;
  isFlowComplete: boolean;
  onFlowReset: () => void;
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onSendMessage,
  isConnected,
  user,
  notifications,
  onAnalyzeIncident,
  onExecutePlaybook,
  isAnalyzing,
  currentAnalysisId,
  onAnalysisComplete,
  isFlowComplete,
  onFlowReset,
  className = ''
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const startedAnalysisIds = useRef<Set<string>>(new Set());
  const isChatSessionComplete = useRef(false);
  const handledAnalysisIds = useRef<Set<string>>(new Set());
  const [showSuggestions, setShowSuggestions] = useState(false);
  
  // Typewriter effect state
  const [typewriterStates, setTypewriterStates] = useState<{[messageId: string]: {
    displayedText: string;
    isTyping: boolean;
    fullText: string;
    currentIndex: number;
  }}>({});
  
  // Step-by-step analysis state
  const [stepByStepAnalysis, setStepByStepAnalysis] = useState<{[messageId: string]: {
    isProcessing: boolean;
    currentStep: string;
    completedSteps: string[];
    isThinking: boolean;
    similarIncidents?: any[];
    evidence?: any[];
    reasoningTrail?: any[];
    currentReasoningStep?: number;
    reasoningStep?: any;
    stepNumber?: number;
    analysisSummary?: any;
    finalResults?: any;
    aiContent?: string;
    // Playbook execution step properties
    diagnosticStep?: any;
    totalSteps?: number;
    rawData?: any;
    executionSummary?: any;
    playbookFinalResults?: any;
  }}>({});
  
  const stepTimeouts = useRef<{[messageId: string]: NodeJS.Timeout[]}>({});
  
  // Track processed notification IDs to prevent re-processing when switching tabs
  //const processedNotificationIds = useRef<Set<string>>(new Set());
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const typewriterTimeouts = useRef<{[messageId: string]: NodeJS.Timeout}>({});

  // Chat suggestions for common commands
  const suggestions = [
    {
      text: '@sre-bot analyze incident INC-2024-001',
      description: 'Analyze Payment API latency spike',
      icon: <ChartBarIcon className="w-4 h-4" />,
      category: 'analysis'
    },
    {
      text: '@sre-bot analyze incident INC-2024-002',
      description: 'Analyze Database connection timeout',
      icon: <ChartBarIcon className="w-4 h-4" />,
      category: 'analysis'
    },
    {
      text: '@sre-bot execute playbook database-timeout',
      description: 'Execute database troubleshooting playbook',
      icon: <PlayIcon className="w-4 h-4" />,
      category: 'playbook'
    },
    {
      text: '@sre-bot status',
      description: 'Get current system status',
      icon: <InformationCircleIcon className="w-4 h-4" />,
      category: 'status'
    },
    {
      text: '@sre-bot help',
      description: 'Show available commands',
      icon: <DocumentTextIcon className="w-4 h-4" />,
      category: 'help'
    }
  ];

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Typewriter effect function
  const startTypewriter = useCallback((messageId: string, fullText: string, speed: number = 30) => {
    // Clear any existing timeout for this message
    if (typewriterTimeouts.current[messageId]) {
      clearTimeout(typewriterTimeouts.current[messageId]);
    }

    // Initialize typewriter state
    setTypewriterStates(prev => ({
      ...prev,
      [messageId]: {
        displayedText: '',
        isTyping: true,
        fullText,
        currentIndex: 0
      }
    }));

    // Start typing effect
    let currentIndex = 0;
    const typeNext = () => {
      if (currentIndex < fullText.length) {
        setTypewriterStates(prev => ({
          ...prev,
          [messageId]: {
            ...prev[messageId],
            displayedText: fullText.substring(0, currentIndex + 1),
            currentIndex: currentIndex + 1
          }
        }));
        
        currentIndex++;
        typewriterTimeouts.current[messageId] = setTimeout(typeNext, speed);
      } else {
        // Typing complete
        setTypewriterStates(prev => ({
          ...prev,
          [messageId]: {
            ...prev[messageId],
            isTyping: false
          }
        }));
        
        // HARD STOP: Mark flow as complete after typewriter finishes
        setIsTyping(false); // Stop typing indicator
        
        // Notify that analysis is complete
        if (onAnalysisComplete) {
          onAnalysisComplete();
        }
        
        // Clean up timeout reference
        delete typewriterTimeouts.current[messageId];
      }
    };

    // Start typing
    typewriterTimeouts.current[messageId] = setTimeout(typeNext, speed);
  }, []);

  // Create a new message bubble for step-by-step analysis
  const createStepMessage = useCallback((content: string, stepType: string, analysis: any) => {
    const stepMessage: ChatMessage = {
      id: `step_${Date.now()}_${Math.random()}`,
      type: 'ai',
      content: '',
      timestamp: new Date().toISOString(),
      metadata: {
        stepType: stepType,
        analysisData: analysis,
        isStepMessage: true
      }
    };
    
    setMessages(prev => [...prev, stepMessage]);
    return stepMessage.id;
  }, []);

  // Step-by-step analysis processing function - creates separate messages
  const startStepByStepAnalysis = useCallback((messageId: string, fullAnalysis: any, aiContent: string) => {
    console.log('🧠 Starting step-by-step analysis with separate messages:', messageId, fullAnalysis);
    
    // Clear any existing timeouts for this message
    if (stepTimeouts.current[messageId]) {
      stepTimeouts.current[messageId].forEach(timeout => clearTimeout(timeout));
    }
    stepTimeouts.current[messageId] = [];
    
    const addTimeout = (timeout: NodeJS.Timeout) => {
      stepTimeouts.current[messageId].push(timeout);
    };
    
    // Step 1: Create message for similar incidents
    addTimeout(setTimeout(() => {
      const step1Id = createStepMessage('', 'similar_incidents', fullAnalysis.similar_incidents);
      
      // Show thinking indicator for this message
      setStepByStepAnalysis(prev => ({
        ...prev,
        [step1Id]: {
          isProcessing: true,
          currentStep: 'thinking',
          completedSteps: [],
          isThinking: true,
          similarIncidents: fullAnalysis.similar_incidents
        }
      }));
      
      // After thinking, show similar incidents
      addTimeout(setTimeout(() => {
        setStepByStepAnalysis(prev => ({
          ...prev,
          [step1Id]: {
            ...prev[step1Id],
            currentStep: 'similar_incidents',
            completedSteps: ['thinking'],
            isThinking: false
          }
        }));
      }, 3000)); // 3s thinking
    }, 1000)); // 1s delay to start
    
    // Step 2: Create message for evidence
    addTimeout(setTimeout(() => {
      const step2Id = createStepMessage('', 'evidence', fullAnalysis.evidence);
      
      // Show thinking indicator for this message
      setStepByStepAnalysis(prev => ({
        ...prev,
        [step2Id]: {
          isProcessing: true,
          currentStep: 'thinking',
          completedSteps: [],
          isThinking: true,
          evidence: fullAnalysis.evidence
        }
      }));
      
      // After thinking, show evidence
      addTimeout(setTimeout(() => {
        setStepByStepAnalysis(prev => ({
          ...prev,
          [step2Id]: {
            ...prev[step2Id],
            currentStep: 'evidence',
            completedSteps: ['thinking'],
            isThinking: false
          }
        }));
      }, 3000)); // 3s thinking
    }, 5000)); // Start after first step completes
    
    // Step 3+: Create individual messages for each reasoning trail step
    const reasoningSteps = fullAnalysis.reasoning_trail || [];
    let reasoningStartTime = 9000; // Start after evidence step completes
    
    reasoningSteps.forEach((step: any, stepIndex: number) => {
      addTimeout(setTimeout(() => {
        const stepId = createStepMessage('', 'reasoning_step', step);
        
        // Show thinking indicator for this step
        setStepByStepAnalysis(prev => ({
          ...prev,
          [stepId]: {
            isProcessing: true,
            currentStep: 'thinking',
            completedSteps: [],
            isThinking: true,
            reasoningStep: step,
            stepNumber: step.step
          }
        }));
        
        // After thinking, show the reasoning step
        addTimeout(setTimeout(() => {
          setStepByStepAnalysis(prev => ({
            ...prev,
            [stepId]: {
              ...prev[stepId],
              currentStep: 'reasoning_step',
              completedSteps: ['thinking'],
              isThinking: false
            }
          }));
        }, 1000)); // 1s thinking for each step
      }, reasoningStartTime + (stepIndex * 6000))); // 6s between each reasoning step (1s thinking + 5s display)
    });
    
    // Step N: Create message for analysis summary (after all reasoning steps)
    const totalReasoningTime = reasoningSteps.length * 6000;
    const analysisSummaryTime = 9000 + totalReasoningTime + 1000; // Start after all reasoning steps + buffer
    
    addTimeout(setTimeout(() => {
      const step4Id = createStepMessage('', 'analysis_summary', {
        incident_id: fullAnalysis.incident_id,
        analysis_id: fullAnalysis.analysis_id,
        status: fullAnalysis.status,
        confidence_score: fullAnalysis.confidence_score,
        analysis_duration_seconds: fullAnalysis.analysis_duration_seconds,
        timestamp: fullAnalysis.timestamp
      });
      
      // Show thinking indicator for this message
      setStepByStepAnalysis(prev => ({
        ...prev,
        [step4Id]: {
          isProcessing: true,
          currentStep: 'thinking',
          completedSteps: [],
          isThinking: true,
          analysisSummary: {
            incident_id: fullAnalysis.incident_id,
            analysis_id: fullAnalysis.analysis_id,
            status: fullAnalysis.status,
            confidence_score: fullAnalysis.confidence_score,
            analysis_duration_seconds: fullAnalysis.analysis_duration_seconds,
            timestamp: fullAnalysis.timestamp
          }
        }
      }));
      
      // After thinking, show analysis summary
      addTimeout(setTimeout(() => {
        setStepByStepAnalysis(prev => ({
          ...prev,
          [step4Id]: {
            ...prev[step4Id],
            currentStep: 'analysis_summary',
            completedSteps: ['thinking'],
            isThinking: false
          }
        }));
      }, 1000)); // 1s thinking
    }, analysisSummaryTime));
    
    // Step N+1: Create message for final results
    addTimeout(setTimeout(() => {
      const step5Id = createStepMessage('', 'final_results', {
        root_cause: fullAnalysis.root_cause,
        recommendations: fullAnalysis.recommendations,
        confidence_score: fullAnalysis.confidence_score
      });
      
      // Show thinking indicator for this message
      setStepByStepAnalysis(prev => ({
        ...prev,
        [step5Id]: {
          isProcessing: true,
          currentStep: 'thinking',
          completedSteps: [],
          isThinking: true,
          finalResults: {
            root_cause: fullAnalysis.root_cause,
            recommendations: fullAnalysis.recommendations,
            confidence_score: fullAnalysis.confidence_score
          }
        }
      }));
      
      // After thinking, show final results
      addTimeout(setTimeout(() => {
        setStepByStepAnalysis(prev => ({
          ...prev,
          [step5Id]: {
            ...prev[step5Id],
            currentStep: 'final_results',
            completedSteps: ['thinking'],
            isThinking: false
          }
        }));
      }, 3000)); // 3s thinking
    }, analysisSummaryTime + 6000)); // Start 6s after analysis summary
    
    // Step N+2: Create final typewriter message
    addTimeout(setTimeout(() => {
      const finalId = createStepMessage(aiContent, 'typewriter', null);
      
      // Start typewriter effect for final message
      addTimeout(setTimeout(() => {
        startTypewriter(finalId, aiContent, 25);
      }, 500));
    }, analysisSummaryTime + 12000)); // Start 6s after final results
  }, [createStepMessage, startTypewriter]);

  // Step-by-step playbook execution processing function - creates separate messages
  const startStepByStepPlaybookExecution = useCallback((messageId: string, fullPlaybookResult: any, aiContent: string) => {
    console.log('📋 Starting step-by-step playbook execution with separate messages:', messageId, fullPlaybookResult);
    
    // Clear any existing timeouts for this message
    if (stepTimeouts.current[messageId]) {
      stepTimeouts.current[messageId].forEach(timeout => clearTimeout(timeout));
    }
    stepTimeouts.current[messageId] = [];
    
    const addTimeout = (timeout: NodeJS.Timeout) => {
      stepTimeouts.current[messageId].push(timeout);
    };
    
    const stepResults = fullPlaybookResult.step_results || [];
    let currentTime = 1000; // Start delay
    
    // Step 1: Create messages for each diagnostic step
    stepResults.forEach((step: any, stepIndex: number) => {
      addTimeout(setTimeout(() => {
        const stepId = createStepMessage('', 'diagnostic_step', { step: step, stepIndex: stepIndex + 1, totalSteps: stepResults.length });
        
        // Show thinking indicator for this step
        setStepByStepAnalysis(prev => ({
          ...prev,
          [stepId]: {
            isProcessing: true,
            currentStep: 'thinking',
            completedSteps: [],
            isThinking: true,
            diagnosticStep: step,
            stepNumber: stepIndex + 1,
            totalSteps: stepResults.length
          }
        }));
        
        // After thinking, show the diagnostic step
        addTimeout(setTimeout(() => {
          setStepByStepAnalysis(prev => ({
            ...prev,
            [stepId]: {
              ...prev[stepId],
              currentStep: 'diagnostic_step',
              completedSteps: ['thinking'],
              isThinking: false
            }
          }));
        }, 2000)); // 2s thinking
      }, currentTime + (stepIndex * 5000))); // 5s between each step
    });
    
    // Update current time after all diagnostic steps
    const diagnosticStepsTime = stepResults.length * 5000;
    currentTime += diagnosticStepsTime + 2000; // Buffer time
    
    // Step 2: Create message for raw data (expandable)
    addTimeout(setTimeout(() => {
      const rawDataId = createStepMessage('', 'raw_data', fullPlaybookResult);
      
      setStepByStepAnalysis(prev => ({
        ...prev,
        [rawDataId]: {
          isProcessing: true,
          currentStep: 'raw_data',
          completedSteps: [],
          isThinking: false,
          rawData: fullPlaybookResult
        }
      }));
    }, currentTime));
    
    currentTime += 2000; // Time for raw data display
    
    // Step 3: Create message for execution summary
    addTimeout(setTimeout(() => {
      const summaryId = createStepMessage('', 'execution_summary', {
        playbook_id: fullPlaybookResult.playbook_id,
        execution_id: fullPlaybookResult.execution_id,
        status: fullPlaybookResult.status,
        confidence_score: fullPlaybookResult.confidence_score,
        execution_time_seconds: fullPlaybookResult.execution_time_seconds,
        root_cause_found: fullPlaybookResult.root_cause_found
      });
      
      // Show thinking indicator for this message
      setStepByStepAnalysis(prev => ({
        ...prev,
        [summaryId]: {
          isProcessing: true,
          currentStep: 'thinking',
          completedSteps: [],
          isThinking: true,
          executionSummary: {
            playbook_id: fullPlaybookResult.playbook_id,
            execution_id: fullPlaybookResult.execution_id,
            status: fullPlaybookResult.status,
            confidence_score: fullPlaybookResult.confidence_score,
            execution_time_seconds: fullPlaybookResult.execution_time_seconds,
            root_cause_found: fullPlaybookResult.root_cause_found
          }
        }
      }));
      
      // After thinking, show execution summary
      addTimeout(setTimeout(() => {
        setStepByStepAnalysis(prev => ({
          ...prev,
          [summaryId]: {
            ...prev[summaryId],
            currentStep: 'execution_summary',
            completedSteps: ['thinking'],
            isThinking: false
          }
        }));
      }, 2000)); // 2s thinking
    }, currentTime));
    
    currentTime += 4000; // Time for execution summary (2s thinking + 2s display)
    
    // Step 4: Create final typewriter message with completion summary FIRST
    addTimeout(setTimeout(() => {
      // Generate the complete content with completion summary first
      const completionSummary = `✅ **Playbook Execution Complete for ${fullPlaybookResult.playbook_id}**

**Confidence:** ${(fullPlaybookResult.confidence_score * 100).toFixed(1)}%

**Root Cause Found:** ${fullPlaybookResult.root_cause_found ? 'Yes' : 'No'}

`;
      
      const steps = fullPlaybookResult.step_results || [];
      let detailedContent = "";
      
      // Root Cause Analysis
      if (fullPlaybookResult.root_cause_found) {
        detailedContent += "**🔍 Root Cause Analysis**\n\n";
        let rootCause = "Database performance issues detected. ";
        let causes = [];
        
        // Analyze connection issues
        const connectionStep = steps.find((s: any) => s.evidence?.metric_name?.includes('connection'));
        if (connectionStep && !connectionStep.success) {
          causes.push(`Database connection pool near capacity (${connectionStep.actual_value} connections, threshold: ${connectionStep.expected_value})`);
        }
        
        // Analyze query performance
        const queryStep = steps.find((s: any) => s.evidence?.avg_query_time_ms);
        if (queryStep && !queryStep.success) {
          const slowQuery = queryStep.evidence.slow_queries?.[0];
          if (slowQuery) {
            causes.push(`Slow queries detected: ${slowQuery.query.substring(0, 50)}... taking ${(slowQuery.execution_time_ms/1000).toFixed(1)}s (${slowQuery.rows_affected?.toLocaleString()} rows affected)`);
          }
        }
        
        // Analyze log errors
        const logStep = steps.find((s: any) => s.evidence?.query?.includes('timeout'));
        if (logStep && (logStep.evidence.error_count > 0 || logStep.evidence.warning_count > 0)) {
          causes.push(`Connection timeouts observed in logs (${logStep.evidence.error_count} errors, ${logStep.evidence.warning_count} warnings)`);
        }
        
        if (causes.length > 0) {
          rootCause += causes.join(". ") + ".";
        } else {
          rootCause = "Multiple diagnostic checks failed but specific root cause needs further investigation.";
        }
        
        detailedContent += rootCause + "\n\n";
      }
      
      // Recommended Actions
      if (steps && steps.length > 0) {
        detailedContent += "**💡 Recommended Actions**\n\n";
        const recommendations = [];
        
        // Connection-based recommendations
        const connectionStep = steps.find((s: any) => s.evidence?.metric_name?.includes('connection'));
        if (connectionStep && !connectionStep.success) {
          recommendations.push("• Increase database connection pool size or implement connection pooling optimization");
          recommendations.push("• Monitor connection usage patterns and implement connection timeouts");
        }
        
        // Query performance recommendations
        const queryStep = steps.find((s: any) => s.evidence?.avg_query_time_ms);
        if (queryStep && !queryStep.success) {
          const slowQuery = queryStep.evidence.slow_queries?.[0];
          if (slowQuery && slowQuery.rows_affected > 10000) {
            recommendations.push("• Optimize large batch operations by processing data in smaller chunks");
            recommendations.push("• Add database indexes to improve query performance");
            recommendations.push("• Consider implementing query caching for frequently accessed data");
          }
        }
        
        // Log-based recommendations
        const logStep = steps.find((s: any) => s.evidence?.query?.includes('timeout'));
        if (logStep && logStep.evidence.warning_count > 0) {
          recommendations.push("• Implement proper connection retry logic with exponential backoff");
          recommendations.push("• Set up alerting for database connection pool exhaustion");
        }
        
        // Escalation recommendations
        const escalatedSteps = steps.filter((s: any) => s.escalation_triggered);
        if (escalatedSteps.length > 1) {
          recommendations.push("• Consider immediate database performance review and scaling");
        }
        
        if (recommendations.length === 0) {
          recommendations.push("• Review diagnostic findings and consider database performance optimization");
        }
        
        detailedContent += recommendations.join("\n");
      }
      
      // Combine completion summary with detailed content
      const fullContent = completionSummary + detailedContent;
      
      const finalId = createStepMessage(fullContent, 'typewriter', null);
      
      // Start typewriter effect for the complete message
      addTimeout(setTimeout(() => {
        startTypewriter(finalId, fullContent, 25);
      }, 500));
    }, currentTime));
  }, [createStepMessage, startTypewriter]);


  // Cleanup typewriter timeouts on unmount
  useEffect(() => {
    return () => {
      Object.values(typewriterTimeouts.current).forEach(timeout => {
        clearTimeout(timeout);
      });
      
      // Clean up step-by-step timeouts
      Object.values(stepTimeouts.current).forEach(timeoutArray => {
        timeoutArray.forEach(timeout => clearTimeout(timeout));
      });
      
      // Clear processed notification IDs on unmount
      //processedNotificationIds.current.clear();
      
    };
  }, []);

  // Convert notifications to chat messages
  useEffect(() => {
    if (notifications.length > 0) {
      const latestNotification = notifications[0];

      if (isChatSessionComplete.current) {
        console.log("🔒 Chat session is locked. Skipping processing.");
        return;
      }
      
      // HARD STOP: Prevent any processing after typewriter flow completes
      if (isFlowComplete) {
        console.log("🛑 Flow is complete. Skipping all further processing.");
        return;
      }
      // const notificationId = latestNotification.id || `${latestNotification.timestamp}_${latestNotification.type}`;
      
      // // Skip if we've already processed this notification
      // if (processedNotificationIds.current.has(notificationId)) {
      //   return;
      // }
      
      // // Mark this notification as processed
      // processedNotificationIds.current.add(notificationId);
      
      if (latestNotification.type === 'ai_response') {
        // Debug logging
        console.log('🔍 AI Response received:', latestNotification);
        console.log('🔍 Full analysis data:', latestNotification.full_analysis);
        
        const aiMessage: ChatMessage = {
          id: `ai_${Date.now()}_${Math.random()}`,
          type: 'ai',
          content: latestNotification.content || latestNotification.message,
          timestamp: latestNotification.timestamp,
          metadata: {
            status: latestNotification.status || latestNotification.data?.status,
            confidenceScore: latestNotification.confidence_score || latestNotification.data?.confidence_score,
            analysisId: latestNotification.analysis_id || latestNotification.data?.analysis_id,
            incidentId: latestNotification.incident_id || latestNotification.data?.incident_id,
            playbookId: latestNotification.playbook_id || latestNotification.data?.playbook_id,
            executionId: latestNotification.execution_id || latestNotification.data?.execution_id,
            requiresApproval: latestNotification.requires_approval || latestNotification.data?.requires_approval,
            recommendations: latestNotification.recommendations || latestNotification.data?.recommendations,
            fullAnalysis: latestNotification.full_analysis,
            fullPlaybookResult: latestNotification.full_playbook_result
          }
        };
        
        console.log('🔍 AI Message created:', aiMessage);
        console.log('🔍 Full analysis in metadata:', aiMessage.metadata?.fullAnalysis);
        
        setMessages(prev => {
          // Since we're already tracking processed notifications, we can simplify this
          // Start step-by-step analysis with separate messages (don't add the original AI message)
          if (aiMessage.metadata?.status === 'completed' && aiMessage.metadata?.fullAnalysis) {
            // const analysisId = aiMessage.metadata?.analysisId || aiMessage.metadata?.fullAnalysis?.analysis_id;
            // console.log('🚀 Starting step-by-step analysis for analysis ID:', analysisId);
            
            // // Clear analyzing state when AI analysis completes
            // if (onAnalysisComplete) {
            //   onAnalysisComplete();
            // }
            
            // setTimeout(() => {
            //   //startStepByStepAnalysis(aiMessage.id, aiMessage.metadata!.fullAnalysis, aiMessage.content);
            //   const analysisId = aiMessage.metadata?.analysisId || aiMessage.metadata?.fullAnalysis?.analysis_id;
            //       if (analysisId && !startedAnalysisIds.current.has(analysisId)) {
            //         startedAnalysisIds.current.add(analysisId);
            //         startStepByStepAnalysis(aiMessage.id, aiMessage.metadata!.fullAnalysis, aiMessage.content);
            //       }
            // }, 100); // Small delay to ensure message is rendered
            
            // // Don't add the original AI message to avoid showing the content immediately
            // return prev;
            const analysisId = aiMessage.metadata?.analysisId || aiMessage.metadata?.fullAnalysis?.analysis_id;

  // 🛑 Check if this was auto-triggered (not by a fresh user message)
  // Prevent accidental replays after tab switch
              if (handledAnalysisIds.current.has(analysisId)) {
                console.log("⛔ Already handled this analysis:", analysisId);
                return prev;
              }

              // ✅ Mark as handled
              handledAnalysisIds.current.add(analysisId);

              // 👇 Trigger the bot’s reply once only
              if (onAnalysisComplete) onAnalysisComplete();

              setTimeout(() => {
                startStepByStepAnalysis(aiMessage.id, aiMessage.metadata!.fullAnalysis, aiMessage.content);
              }, 100);

              return prev;
          } else if (aiMessage.metadata?.status === 'completed' && aiMessage.metadata?.fullPlaybookResult) {
            // Handle playbook execution step-by-step display
            const executionId = aiMessage.metadata?.executionId || aiMessage.metadata?.fullPlaybookResult?.execution_id;
            
            // Check if this playbook execution was already handled
            if (handledAnalysisIds.current.has(executionId)) {
              console.log("⛔ Already handled this playbook execution:", executionId);
              return prev;
            }
            
            // Mark as handled
            handledAnalysisIds.current.add(executionId);
            
            // Trigger completion
            if (onAnalysisComplete) onAnalysisComplete();
            
            setTimeout(() => {
              startStepByStepPlaybookExecution(aiMessage.id, aiMessage.metadata!.fullPlaybookResult, aiMessage.content);
            }, 100);
            
            return prev;
          } else if (aiMessage.metadata?.status === 'completed' && aiMessage.content) {
            // For non-analysis messages, still use typewriter
            setTimeout(() => {
              startTypewriter(aiMessage.id, aiMessage.content, 25);
            }, 100);
            
            return [...prev, aiMessage];
          }
          
          return [...prev, aiMessage];
        });
      } else if (latestNotification.type === 'incident_analysis_update') {
        const updateMessage: ChatMessage = {
          id: `analysis_${latestNotification.data?.analysis_id}_${latestNotification.data?.step || 'update'}`,
          type: 'system',
          content: formatAnalysisUpdate(latestNotification),
          timestamp: latestNotification.timestamp,
          metadata: {
            status: latestNotification.data?.status,
            analysisId: latestNotification.data?.analysis_id,
            incidentId: latestNotification.data?.incident_id,
            progress: latestNotification.data?.progress,
            step: latestNotification.data?.step,
            evidence: latestNotification.data?.evidence,
            confidence: latestNotification.data?.confidence,
            rootCause: latestNotification.data?.root_cause,
            recommendations: latestNotification.data?.recommendations,
            similarIncidents: latestNotification.data?.similar_incidents
          }
        };
        
        setMessages(prev => {
          // Always add new analysis update messages
          return [...prev, updateMessage];
        });
      } else if (latestNotification.type === 'playbook_execution_update') {
        const updateMessage: ChatMessage = {
          id: `playbook_${latestNotification.data?.execution_id}_${latestNotification.data?.current_step || 'update'}`,
          type: 'system',
          content: formatPlaybookUpdate(latestNotification),
          timestamp: latestNotification.timestamp,
          metadata: {
            status: latestNotification.data?.status,
            executionId: latestNotification.data?.execution_id,
            playbookId: latestNotification.data?.playbook_id,
            incidentId: latestNotification.data?.incident_id,
            progress: latestNotification.data?.progress,
            currentStep: latestNotification.data?.current_step,
            totalSteps: latestNotification.data?.total_steps,
            stepDetails: latestNotification.data?.step_details,
            stepResult: latestNotification.data?.step_result,
            confidence: latestNotification.data?.confidence,
            rootCauseFound: latestNotification.data?.root_cause_found,
            recommendations: latestNotification.data?.recommendations
          }
        };
        
        setMessages(prev => {
          // Always add new playbook update messages
          return [...prev, updateMessage];
        });
      } else if (latestNotification.type === 'analysis_update' || latestNotification.type === 'playbook_update') {
        const updateMessage: ChatMessage = {
          id: `update_${Date.now()}_${Math.random()}`,
          type: 'system',
          content: latestNotification.message,
          timestamp: latestNotification.timestamp,
          metadata: {
            status: latestNotification.data?.status,
            analysisId: latestNotification.data?.analysis_id,
            incidentId: latestNotification.data?.incident_id,
            playbookId: latestNotification.data?.playbook_id,
            executionId: latestNotification.data?.execution_id,
            evidence: latestNotification.data?.evidence,
            confidenceScore: latestNotification.data?.confidence
          }
        };
        
        setMessages(prev => {
          // Update existing message if it's the same analysis/execution
          const existingIndex = prev.findIndex(msg => 
            msg.metadata?.analysisId === updateMessage.metadata?.analysisId ||
            msg.metadata?.executionId === updateMessage.metadata?.executionId
          );
          
          if (existingIndex !== -1 && prev[existingIndex].type === 'system') {
            const updated = [...prev];
            updated[existingIndex] = updateMessage;
            return updated;
          } else {
            return [...prev, updateMessage];
          }
        });
      } else if (latestNotification.type === 'system_status') {
        const statusData = latestNotification.system_status;
        const statusMessage: ChatMessage = {
          id: `status_${Date.now()}_${Math.random()}`,
          type: 'ai',
          content: formatSystemStatus(statusData),
          timestamp: latestNotification.timestamp,
          metadata: {
            status: 'completed',
            systemStatus: statusData
          }
        };
        
        setMessages(prev => [...prev, statusMessage]);
      }    }
  }, [notifications, startTypewriter, startStepByStepAnalysis, isFlowComplete]);

  // Handle AI typing indicator
  useEffect(() => {
    if (isAnalyzing) {
      setIsTyping(true);
      
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      
      typingTimeoutRef.current = setTimeout(() => {
        setIsTyping(false);
      }, 30000); // Clear typing after 30 seconds
    } else {
      setIsTyping(false);
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    }

    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, [isAnalyzing]);

  // Initialize with welcome message
  useEffect(() => {
    if (messages.length === 0 && isConnected && user) {
      const welcomeMessage: ChatMessage = {
        id: 'welcome',
        type: 'ai',
        content: `🤖 Welcome ${user.username}! I'm your AI SRE Agent. I can help you with:\n\n• **Incident Analysis** - Analyze production incidents with AI\n• **Playbook Execution** - Run automated troubleshooting procedures\n• **System Monitoring** - Get real-time status and insights\n\nTry typing \`@sre-bot help\` to see all available commands, or use one of the suggestions below.`,
        timestamp: new Date().toISOString(),
        metadata: {}
      };
      
      setMessages([welcomeMessage]);
    }
  }, [isConnected, user, messages.length]);

  const handleSendMessage = useCallback(() => {
    if (!inputMessage.trim() || !isConnected) return;

    const userMessage: ChatMessage = {
      id: `user_${Date.now()}_${Math.random()}`,
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date().toISOString(),
      user: user || undefined
    };

    setMessages(prev => [...prev, userMessage]);
    onSendMessage(inputMessage.trim());
    setInputMessage('');
    setShowSuggestions(false);

    // Show typing indicator for AI response
    if (inputMessage.includes('@sre-bot') || inputMessage.includes('@ai-agent')) {
      isChatSessionComplete.current = false;
      onFlowReset(); // Reset flow completion flag for new analysis
      setIsTyping(true);
    }
  }, [inputMessage, isConnected, onSendMessage, user, onFlowReset]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage]);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setInputMessage(suggestion);
    setShowSuggestions(false);
    inputRef.current?.focus();
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInputMessage(value);
    
    // Show suggestions when user types @ symbol
    if (value.includes('@') && value.length > 0) {
      setShowSuggestions(true);
    } else {
      setShowSuggestions(false);
    }
  }, []);

  const formatTimestamp = useCallback((timestamp: string) => {
    try {
      return format(new Date(timestamp), 'HH:mm');
    } catch {
      return '';
    }
  }, []);

  const getMessageIcon = useCallback((message: ChatMessage) => {
    if (message.type === 'ai') {
      return '🤖';
    } else if (message.type === 'system') {
      if (message.metadata?.status === 'completed') {
        return <CheckCircleIcon className="w-4 h-4 text-success-500" />;
      } else if (message.metadata?.status === 'failed') {
        return <XCircleIcon className="w-4 h-4 text-error-500" />;
      } else if (message.metadata?.status === 'in_progress') {
        return <ClockIcon className="w-4 h-4 text-primary-500 animate-spin" />;
      } else {
        return <InformationCircleIcon className="w-4 h-4 text-primary-500" />;
      }
    } else {
      return null;
    }
  }, []);

  const getMessageStyle = useCallback((message: ChatMessage) => {
    const baseStyle = "max-w-xs lg:max-w-md px-4 py-2 rounded-lg shadow-sm";
    
    if (message.type === 'user') {
      return `${baseStyle} bg-primary-600 text-white ml-auto`;
    } else if (message.type === 'ai') {
      return `${baseStyle} bg-white border border-gray-200 text-gray-800`;
    } else if (message.type === 'system') {
      return `${baseStyle} bg-gray-100 border border-gray-200 text-gray-700 mx-auto text-center text-sm`;
    } else {
      return `${baseStyle} bg-blue-50 border border-blue-200 text-blue-800`;
    }
  }, []);

  // Step-by-step analysis renderer
  const renderStepByStepOrFullAnalysis = useCallback((messageId: string, metadata?: any) => {
    const stepState = stepByStepAnalysis[messageId];
    
    // If no step-by-step processing, show traditional full analysis
    if (!stepState || !stepState.isProcessing) {
      return renderFullAnalysis(metadata);
    }
    
    return (
      <div className="mt-4 space-y-4">
        {/* Show thinking indicator when thinking */}
        {stepState.isThinking && renderThinkingIndicator()}
        
        {/* Show similar incidents */}
        {(stepState.currentStep === 'similar_incidents' || stepState.completedSteps.includes('similar_incidents')) && stepState.similarIncidents && stepState.similarIncidents.length > 0 && (
          <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200 animate-fadeIn">
            <h4 className="font-medium text-yellow-800 mb-2">🔗 Similar Incidents ({stepState.similarIncidents.length})</h4>
            <div className="space-y-2">
              {stepState.similarIncidents.map((incident: any, index: number) => (
                <div key={index} className="bg-white p-3 rounded border text-sm">
                  <div className="font-medium text-yellow-700">{incident.incident_id}: {incident.title}</div>
                  <div className="text-gray-600">{incident.description}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Service: {incident.service_name} | Region: {incident.region} | 
                    Severity: {incident.severity} | Status: {incident.status}
                  </div>
                  {incident.root_cause && (
                    <div className="text-xs text-yellow-700 mt-1">
                      <strong>Root Cause:</strong> {incident.root_cause}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Show evidence */}
        {(stepState.currentStep === 'evidence' || stepState.completedSteps.includes('evidence')) && stepState.evidence && stepState.evidence.length > 0 && (
          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200 animate-fadeIn">
            <h4 className="font-medium text-blue-800 mb-2">🔬 Evidence ({stepState.evidence.length} items)</h4>
            <div className="space-y-2">
              {stepState.evidence.map((item: any, index: number) => (
                <div key={index} className="bg-white p-3 rounded border text-sm">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="font-medium text-blue-700">{item.type || 'Unknown'}</div>
                      <div className="text-gray-700">{item.description}</div>
                      <div className="text-xs text-gray-500">Source: {item.source}</div>
                    </div>
                    <div className="text-xs text-blue-600 font-medium">
                      {(item.relevance * 100).toFixed(0)}% relevance
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Show individual reasoning step */}
        {(stepState.currentStep === 'reasoning_step' || stepState.completedSteps.includes('reasoning_step')) && stepState.reasoningStep && (
          <div className="p-4 bg-purple-50 rounded-lg border border-purple-200 animate-fadeIn">
            <h4 className="font-medium text-purple-800 mb-2">🧠 Reasoning Trail - Step {stepState.stepNumber}</h4>
            <div className="bg-white p-3 rounded border text-sm">
              <div className="flex items-start gap-2">
                <div className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs font-medium min-w-0">
                  Step {stepState.reasoningStep.step}
                </div>
                <div className="flex-1">
                  <div className="font-medium text-purple-700">{stepState.reasoningStep.action}</div>
                  <div className="text-gray-600 mt-1">{stepState.reasoningStep.reasoning}</div>
                  {stepState.reasoningStep.evidence && (
                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                      <strong>Evidence:</strong>
                      <pre className="mt-1 whitespace-pre-wrap overflow-x-auto text-xs">
                        {JSON.stringify(stepState.reasoningStep.evidence, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Show analysis summary */}
        {(stepState.currentStep === 'analysis_summary' || stepState.completedSteps.includes('analysis_summary')) && stepState.analysisSummary && (
          <div className="p-4 bg-gray-50 rounded-lg border animate-fadeIn">
            <h4 className="font-medium text-gray-800 mb-2">📊 Analysis Summary</h4>
            <div className="bg-white p-3 rounded border">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><strong>Incident ID:</strong> {stepState.analysisSummary.incident_id}</div>
                <div><strong>Analysis ID:</strong> {stepState.analysisSummary.analysis_id}</div>
                <div><strong>Status:</strong> <span className="text-green-600">{stepState.analysisSummary.status}</span></div>
                <div><strong>Confidence Score:</strong> <span className="text-blue-600">{(stepState.analysisSummary.confidence_score * 100).toFixed(1)}%</span></div>
                <div><strong>Duration:</strong> {stepState.analysisSummary.analysis_duration_seconds?.toFixed(3)}s</div>
                <div><strong>Timestamp:</strong> {new Date(stepState.analysisSummary.timestamp).toLocaleString()}</div>
              </div>
            </div>
          </div>
        )}
        
        {/* Show final results */}
        {(stepState.currentStep === 'final_results' || stepState.completedSteps.includes('final_results')) && stepState.finalResults && (
          <div className="space-y-4 animate-fadeIn">
            {/* Root Cause */}
            {stepState.finalResults.root_cause && (
              <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                <h4 className="font-medium text-red-800 mb-2">🔍 Root Cause</h4>
                <p className="text-red-700">{stepState.finalResults.root_cause}</p>
              </div>
            )}
            
            {/* Recommendations */}
            {stepState.finalResults.recommendations && stepState.finalResults.recommendations.length > 0 && (
              <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                <h4 className="font-medium text-green-800 mb-2">💡 Recommendations ({stepState.finalResults.recommendations.length})</h4>
                <ul className="list-disc list-inside space-y-1 text-green-700">
                  {stepState.finalResults.recommendations.map((rec: string, index: number) => (
                    <li key={index} className="text-sm">{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
        
        {/* Show diagnostic step */}
        {(stepState.currentStep === 'diagnostic_step' || stepState.completedSteps.includes('diagnostic_step')) && stepState.diagnosticStep && (
          <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200 animate-fadeIn">
            <h4 className="font-medium text-yellow-800 mb-2">⚙️ Diagnostic Step {stepState.stepNumber} of {stepState.totalSteps}</h4>
            <div className="bg-white p-3 rounded border">
              <div className="border-l-4 pl-3 py-2" style={{borderLeftColor: stepState.diagnosticStep.success ? '#10B981' : '#EF4444'}}>
                <div className="flex justify-between items-start mb-1">
                  <div className="font-medium text-gray-700">Step {stepState.stepNumber}</div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      stepState.diagnosticStep.success 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {stepState.diagnosticStep.success ? '✅ passed' : '❌ failed'}
                    </span>
                    <span className="text-xs text-gray-500">
                      {stepState.diagnosticStep.execution_time_ms || 0}ms
                    </span>
                  </div>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  {stepState.diagnosticStep.evidence?.metric_name && (
                    <div><strong>Metric:</strong> {stepState.diagnosticStep.evidence.metric_name}</div>
                  )}
                  {stepState.diagnosticStep.evidence?.query && (
                    <div><strong>Query:</strong> {stepState.diagnosticStep.evidence.query}</div>
                  )}
                  {stepState.diagnosticStep.evidence?.latest_value !== undefined && (
                    <div><strong>Latest:</strong> {stepState.diagnosticStep.evidence.latest_value}</div>
                  )}
                  {stepState.diagnosticStep.evidence?.threshold !== undefined && (
                    <div><strong>Threshold:</strong> {stepState.diagnosticStep.evidence.threshold}</div>
                  )}
                </div>
                {stepState.diagnosticStep.escalation_triggered && (
                  <div className="mt-2 p-2 bg-red-50 rounded text-xs text-red-700">
                    <strong>⚠️ Escalation Triggered</strong>
                  </div>
                )}
                {stepState.diagnosticStep.expected_value !== undefined && stepState.diagnosticStep.actual_value !== undefined && (
                  <div className="mt-2 text-xs text-gray-600">
                    <strong>Expected:</strong> {stepState.diagnosticStep.expected_value} | <strong>Actual:</strong> {stepState.diagnosticStep.actual_value}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        
        {/* Show raw data (expandable) */}
        {(stepState.currentStep === 'raw_data' || stepState.completedSteps.includes('raw_data')) && stepState.rawData && (
          <div className="p-4 bg-gray-50 rounded-lg border animate-fadeIn">
            <details className="cursor-pointer">
              <summary className="font-medium text-gray-800 mb-2">🔍 Show Raw Data</summary>
              <div className="mt-2 bg-white p-3 rounded border">
                <pre className="text-xs text-gray-700 overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(stepState.rawData, null, 2)}
                </pre>
              </div>
            </details>
          </div>
        )}
        
        {/* Show execution summary */}
        {(stepState.currentStep === 'execution_summary' || stepState.completedSteps.includes('execution_summary')) && stepState.executionSummary && (
          <div className="p-4 bg-green-50 rounded-lg border border-green-200 animate-fadeIn">
            <h4 className="font-medium text-green-800 mb-2">📊 Execution Summary</h4>
            <div className="bg-white p-3 rounded border">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><strong>Playbook:</strong> {stepState.executionSummary.playbook_id}</div>
                <div><strong>Execution ID:</strong> {stepState.executionSummary.execution_id}</div>
                <div><strong>Status:</strong> <span className="text-green-600">{stepState.executionSummary.status}</span></div>
                <div><strong>Confidence:</strong> <span className="text-blue-600">{(stepState.executionSummary.confidence_score * 100).toFixed(1)}%</span></div>
                <div><strong>Duration:</strong> {stepState.executionSummary.execution_time_seconds?.toFixed(3)}s</div>
                <div><strong>Root Cause Found:</strong> <span className={stepState.executionSummary.root_cause_found ? "text-green-600" : "text-red-600"}>{stepState.executionSummary.root_cause_found ? "Yes" : "No"}</span></div>
              </div>
            </div>
          </div>
        )}
        
        {/* Show playbook final results */}
        {(stepState.currentStep === 'playbook_final_results' || stepState.completedSteps.includes('playbook_final_results')) && stepState.playbookFinalResults && (
          <div className="space-y-4 animate-fadeIn">
            {/* Root Cause Analysis */}
            {stepState.playbookFinalResults.root_cause_found && (
              <div className="bg-red-50 p-4 rounded border border-red-200">
                <h5 className="font-medium text-red-800 mb-2">🔍 Root Cause Analysis</h5>
                <div className="text-red-700">
                  {(() => {
                    const steps = stepState.playbookFinalResults.step_results || [];
                    let rootCause = "Database performance issues detected. ";
                    let causes = [];
                    
                    // Analyze connection issues
                    const connectionStep = steps.find((s: any) => s.evidence?.metric_name?.includes('connection'));
                    if (connectionStep && !connectionStep.success) {
                      causes.push(`Database connection pool near capacity (${connectionStep.actual_value} connections, threshold: ${connectionStep.expected_value})`);
                    }
                    
                    // Analyze query performance
                    const queryStep = steps.find((s: any) => s.evidence?.avg_query_time_ms);
                    if (queryStep && !queryStep.success) {
                      const slowQuery = queryStep.evidence.slow_queries?.[0];
                      if (slowQuery) {
                        causes.push(`Slow queries detected: ${slowQuery.query.substring(0, 50)}... taking ${(slowQuery.execution_time_ms/1000).toFixed(1)}s (${slowQuery.rows_affected?.toLocaleString()} rows affected)`);
                      }
                    }
                    
                    // Analyze log errors
                    const logStep = steps.find((s: any) => s.evidence?.query?.includes('timeout'));
                    if (logStep && (logStep.evidence.error_count > 0 || logStep.evidence.warning_count > 0)) {
                      causes.push(`Connection timeouts observed in logs (${logStep.evidence.error_count} errors, ${logStep.evidence.warning_count} warnings)`);
                    }
                    
                    if (causes.length > 0) {
                      rootCause += causes.join(". ") + ".";
                    } else {
                      rootCause = "Multiple diagnostic checks failed but specific root cause needs further investigation.";
                    }
                    
                    return rootCause;
                  })()}
                </div>
              </div>
            )}
            
            {/* Generated Recommendations */}
            {stepState.playbookFinalResults.step_results && stepState.playbookFinalResults.step_results.length > 0 && (
              <div className="bg-blue-50 p-4 rounded border border-blue-200">
                <h5 className="font-medium text-blue-800 mb-2">💡 Recommended Actions</h5>
                <div className="space-y-2">
                  {(() => {
                    const steps = stepState.playbookFinalResults.step_results || [];
                    const recommendations = [];
                    
                    // Connection-based recommendations
                    const connectionStep = steps.find((s: any) => s.evidence?.metric_name?.includes('connection'));
                    if (connectionStep && !connectionStep.success) {
                      recommendations.push("Increase database connection pool size or implement connection pooling optimization");
                      recommendations.push("Monitor connection usage patterns and implement connection timeouts");
                    }
                    
                    // Query performance recommendations
                    const queryStep = steps.find((s: any) => s.evidence?.avg_query_time_ms);
                    if (queryStep && !queryStep.success) {
                      const slowQuery = queryStep.evidence.slow_queries?.[0];
                      if (slowQuery && slowQuery.rows_affected > 10000) {
                        recommendations.push("Optimize large batch operations by processing data in smaller chunks");
                        recommendations.push("Add database indexes to improve query performance");
                        recommendations.push("Consider implementing query caching for frequently accessed data");
                      }
                    }
                    
                    // Log-based recommendations
                    const logStep = steps.find((s: any) => s.evidence?.query?.includes('timeout'));
                    if (logStep && logStep.evidence.warning_count > 0) {
                      recommendations.push("Implement proper connection retry logic with exponential backoff");
                      recommendations.push("Set up alerting for database connection pool exhaustion");
                    }
                    
                    // Escalation recommendations
                    const escalatedSteps = steps.filter((s: any) => s.escalation_triggered);
                    if (escalatedSteps.length > 1) {
                      recommendations.push("Consider immediate database performance review and scaling");
                    }
                    
                    if (recommendations.length === 0) {
                      recommendations.push("Review diagnostic findings and consider database performance optimization");
                    }
                    
                    return recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start space-x-2">
                        <span className="text-blue-600 font-bold">•</span>
                        <span className="text-blue-700 text-sm">{rec}</span>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Show typewriter at the end */}
        {(stepState.currentStep === 'typewriter' || stepState.currentStep === 'completed') && stepState.aiContent && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200 animate-fadeIn">
            <h4 className="font-medium text-blue-800 mb-2">🤖 AI Analysis Summary</h4>
            <div className="bg-white p-3 rounded border">
              {renderTypewriterContent(messageId)}
            </div>
          </div>
        )}
      </div>
    );
  }, [stepByStepAnalysis]);

  const renderMessageContent = useCallback((message: ChatMessage) => {
    // Check if this is a step message
    const isStepMessage = message.metadata?.isStepMessage;
    const stepType = message.metadata?.stepType;
    
    // Check if this message has step-by-step analysis
    const stepState = stepByStepAnalysis[message.id];
    const hasStepByStep = stepState && stepState.isProcessing;
    
    // Check if this message should use typewriter effect
    const typewriterState = typewriterStates[message.id];
    const shouldUseTypewriter = message.type === 'ai' && (message.metadata?.status === 'completed' || stepType === 'typewriter');
    
    let content = message.content;
    
    // For step messages, don't show content initially - just show step-by-step processing
    if (isStepMessage) {
      // Use typewriter text if this is a typewriter step
      if (stepType === 'typewriter' && typewriterState) {
        content = typewriterState.displayedText;
        
        // Add typing cursor if still typing
        if (typewriterState.isTyping) {
          content += '<span class="animate-pulse">|</span>';
        }
      } else {
        content = ''; // Don't show content for other step types
      }
      
      return (
        <div className="space-y-4">
          {/* Show content for typewriter step only */}
          {stepType === 'typewriter' && content && (
            <div 
              dangerouslySetInnerHTML={{ 
                __html: content
                  .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                  .replace(/\*(.*?)\*/g, '<em>$1</em>')
                  .replace(/`(.*?)`/g, '<code class="bg-gray-200 px-1 rounded text-sm">$1</code>')
                  .replace(/\n/g, '<br />')
              }}
              className="whitespace-pre-wrap"
            />
          )}
          
          {/* Show step-by-step analysis based on step type */}
          {renderStepByStepOrFullAnalysis(message.id, message.metadata)}
        </div>
      );
    }
    
    // For non-step messages, use original logic
    // Use typewriter text if available and message is still typing
    if (shouldUseTypewriter && typewriterState && !hasStepByStep) {
      content = typewriterState.displayedText;
      
      // Add typing cursor if still typing
      if (typewriterState.isTyping) {
        content += '<span class="animate-pulse">|</span>';
      }
    }

    return (
      <div className="space-y-4">
        {/* Only show initial content if NOT using step-by-step analysis */}
        {!hasStepByStep && content && (
          <div 
            dangerouslySetInnerHTML={{ 
              __html: content
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code class="bg-gray-200 px-1 rounded text-sm">$1</code>')
                .replace(/\n/g, '<br />')
            }}
            className="whitespace-pre-wrap"
          />
        )}
        
        {/* Show step-by-step analysis or full analysis */}
        {renderStepByStepOrFullAnalysis(message.id, message.metadata)}
      </div>
    );
  }, [typewriterStates, stepByStepAnalysis, renderStepByStepOrFullAnalysis]);
  
  // Typewriter content renderer
  const renderTypewriterContent = useCallback((messageId: string) => {
    const typewriterState = typewriterStates[messageId];
    
    if (!typewriterState) {
      return null;
    }
    
    let content = typewriterState.displayedText;
    
    // Add typing cursor if still typing
    if (typewriterState.isTyping) {
      content += '<span class="animate-pulse">|</span>';
    }
    
    return (
      <div 
        dangerouslySetInnerHTML={{ 
          __html: content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code class="bg-gray-200 px-1 rounded text-sm">$1</code>')
            .replace(/\n/g, '<br />')
        }}
        className="whitespace-pre-wrap"
      />
    );
  }, [typewriterStates]);
  
  // Thinking indicator component
  const renderThinkingIndicator = useCallback(() => {
    return (
      <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200 animate-pulse">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
          <span className="text-blue-700 font-medium text-sm">🤔 Thinking...</span>
        </div>
      </div>
    );
  }, []);
  
  // Full analysis renderer (moved here to fix dependency order)
  const renderFullAnalysis = useCallback((metadata?: any) => {
    console.log('🔍 renderFullAnalysis called with metadata:', metadata);
    
    const fullAnalysis = metadata?.fullAnalysis;
    const fullPlaybookResult = metadata?.fullPlaybookResult;
    
    console.log('🔍 fullAnalysis:', fullAnalysis);
    console.log('🔍 fullPlaybookResult:', fullPlaybookResult);
    
    if (!fullAnalysis && !fullPlaybookResult) {
      console.log('🔍 No full analysis or playbook result found, returning null');
      return null;
    }

    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
        <h3 className="text-lg font-semibold mb-3 text-gray-800">📊 Complete Analysis Details</h3>
        
        {/* Incident Analysis Details */}
        {fullAnalysis && (
          <div className="space-y-4">
            {/* Basic Info */}
            <div className="bg-white p-3 rounded border">
              <h4 className="font-medium text-blue-800 mb-2">🎯 Analysis Summary</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><strong>Incident ID:</strong> {fullAnalysis.incident_id}</div>
                <div><strong>Analysis ID:</strong> {fullAnalysis.analysis_id}</div>
                <div><strong>Status:</strong> <span className="text-green-600">{fullAnalysis.status}</span></div>
                <div><strong>Confidence Score:</strong> <span className="text-blue-600">{(fullAnalysis.confidence_score * 100).toFixed(1)}%</span></div>
                <div><strong>Duration:</strong> {fullAnalysis.analysis_duration_seconds?.toFixed(3)}s</div>
                <div><strong>Timestamp:</strong> {new Date(fullAnalysis.timestamp).toLocaleString()}</div>
              </div>
            </div>

            {/* Root Cause */}
            {fullAnalysis.root_cause && (
              <div className="bg-red-50 p-3 rounded border border-red-200">
                <h4 className="font-medium text-red-800 mb-2">🔍 Root Cause</h4>
                <p className="text-red-700">{fullAnalysis.root_cause}</p>
              </div>
            )}

            {/* Recommendations */}
            {fullAnalysis.recommendations && fullAnalysis.recommendations.length > 0 && (
              <div className="bg-green-50 p-3 rounded border border-green-200">
                <h4 className="font-medium text-green-800 mb-2">💡 Recommendations ({fullAnalysis.recommendations.length})</h4>
                <ul className="list-disc list-inside space-y-1 text-green-700">
                  {fullAnalysis.recommendations.map((rec: string, index: number) => (
                    <li key={index} className="text-sm">{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Evidence */}
            {fullAnalysis.evidence && fullAnalysis.evidence.length > 0 && (
              <div className="bg-blue-50 p-3 rounded border border-blue-200">
                <h4 className="font-medium text-blue-800 mb-2">🔬 Evidence ({fullAnalysis.evidence.length} items)</h4>
                <div className="space-y-2">
                  {fullAnalysis.evidence.map((item: any, index: number) => (
                    <div key={index} className="bg-white p-2 rounded border text-sm">
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="font-medium text-blue-700">{item.type || 'Unknown'}</div>
                          <div className="text-gray-700">{item.description}</div>
                          <div className="text-xs text-gray-500">Source: {item.source}</div>
                        </div>
                        <div className="text-xs text-blue-600 font-medium">
                          {(item.relevance * 100).toFixed(0)}% relevance
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Similar Incidents */}
            {fullAnalysis.similar_incidents && fullAnalysis.similar_incidents.length > 0 && (
              <div className="bg-yellow-50 p-3 rounded border border-yellow-200">
                <h4 className="font-medium text-yellow-800 mb-2">🔗 Similar Incidents ({fullAnalysis.similar_incidents.length})</h4>
                <div className="space-y-2">
                  {fullAnalysis.similar_incidents.map((incident: any, index: number) => (
                    <div key={index} className="bg-white p-2 rounded border text-sm">
                      <div className="font-medium text-yellow-700">{incident.incident_id}: {incident.title}</div>
                      <div className="text-gray-600">{incident.description}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        Service: {incident.service_name} | Region: {incident.region} | 
                        Severity: {incident.severity} | Status: {incident.status}
                      </div>
                      {incident.root_cause && (
                        <div className="text-xs text-yellow-700 mt-1">
                          <strong>Root Cause:</strong> {incident.root_cause}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Reasoning Trail */}
            {fullAnalysis.reasoning_trail && fullAnalysis.reasoning_trail.length > 0 && (
              <div className="bg-purple-50 p-3 rounded border border-purple-200">
                <h4 className="font-medium text-purple-800 mb-2">🧠 Reasoning Trail ({fullAnalysis.reasoning_trail.length} steps)</h4>
                <div className="space-y-2">
                  {fullAnalysis.reasoning_trail.map((step: any, index: number) => (
                    <div key={index} className="bg-white p-2 rounded border text-sm">
                      <div className="flex items-start gap-2">
                        <div className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs font-medium min-w-0">
                          Step {step.step}
                        </div>
                        <div className="flex-1">
                          <div className="font-medium text-purple-700">{step.action}</div>
                          <div className="text-gray-600 mt-1">{step.reasoning}</div>
                          {step.evidence && (
                            <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                              <strong>Evidence:</strong>
                              <pre className="mt-1 whitespace-pre-wrap overflow-x-auto">
                                {JSON.stringify(step.evidence, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Playbook Results Details */}
        {fullPlaybookResult && (
          <div className="bg-green-50 p-3 rounded border mt-4">
            <h4 className="font-medium text-green-800 mb-3">📋 Playbook Execution Results</h4>
            
            {/* Root Cause Analysis */}
            {fullPlaybookResult.root_cause_found && (
              <div className="bg-red-50 p-4 rounded border border-red-200 mb-3">
                <h5 className="font-medium text-red-800 mb-2">🔍 Root Cause Analysis</h5>
                <div className="text-red-700">
                  {(() => {
                    const steps = fullPlaybookResult.step_results || [];
                    let rootCause = "Database performance issues detected. ";
                    let causes = [];
                    
                    // Analyze connection issues
                    const connectionStep = steps.find((s: any) => s.evidence?.metric_name?.includes('connection'));
                    if (connectionStep && !connectionStep.success) {
                      causes.push(`Database connection pool near capacity (${connectionStep.actual_value} connections, threshold: ${connectionStep.expected_value})`);
                    }
                    
                    // Analyze query performance
                    const queryStep = steps.find((s: any) => s.evidence?.avg_query_time_ms);
                    if (queryStep && !queryStep.success) {
                      const slowQuery = queryStep.evidence.slow_queries?.[0];
                      if (slowQuery) {
                        causes.push(`Slow queries detected: ${slowQuery.query.substring(0, 50)}... taking ${(slowQuery.execution_time_ms/1000).toFixed(1)}s (${slowQuery.rows_affected?.toLocaleString()} rows affected)`);
                      }
                    }
                    
                    // Analyze log errors
                    const logStep = steps.find((s: any) => s.evidence?.query?.includes('timeout'));
                    if (logStep && (logStep.evidence.error_count > 0 || logStep.evidence.warning_count > 0)) {
                      causes.push(`Connection timeouts observed in logs (${logStep.evidence.error_count} errors, ${logStep.evidence.warning_count} warnings)`);
                    }
                    
                    if (causes.length > 0) {
                      rootCause += causes.join(". ") + ".";
                    } else {
                      rootCause = "Multiple diagnostic checks failed but specific root cause needs further investigation.";
                    }
                    
                    return rootCause;
                  })()}
                </div>
              </div>
            )}
            
            {/* Generated Recommendations */}
            {fullPlaybookResult.step_results && fullPlaybookResult.step_results.length > 0 && (
              <div className="bg-blue-50 p-4 rounded border border-blue-200 mb-3">
                <h5 className="font-medium text-blue-800 mb-2">💡 Recommended Actions</h5>
                <div className="space-y-2">
                  {(() => {
                    const steps = fullPlaybookResult.step_results || [];
                    const recommendations = [];
                    
                    // Connection-based recommendations
                    const connectionStep = steps.find((s: any) => s.evidence?.metric_name?.includes('connection'));
                    if (connectionStep && !connectionStep.success) {
                      recommendations.push("Increase database connection pool size or implement connection pooling optimization");
                      recommendations.push("Monitor connection usage patterns and implement connection timeouts");
                    }
                    
                    // Query performance recommendations
                    const queryStep = steps.find((s: any) => s.evidence?.avg_query_time_ms);
                    if (queryStep && !queryStep.success) {
                      const slowQuery = queryStep.evidence.slow_queries?.[0];
                      if (slowQuery && slowQuery.rows_affected > 10000) {
                        recommendations.push("Optimize large batch operations by processing data in smaller chunks");
                        recommendations.push("Add database indexes to improve query performance");
                        recommendations.push("Consider implementing query caching for frequently accessed data");
                      }
                    }
                    
                    // Log-based recommendations
                    const logStep = steps.find((s: any) => s.evidence?.query?.includes('timeout'));
                    if (logStep && logStep.evidence.warning_count > 0) {
                      recommendations.push("Implement proper connection retry logic with exponential backoff");
                      recommendations.push("Set up alerting for database connection pool exhaustion");
                    }
                    
                    // Escalation recommendations
                    const escalatedSteps = steps.filter((s: any) => s.escalation_triggered);
                    if (escalatedSteps.length > 1) {
                      recommendations.push("Consider immediate database performance review and scaling");
                    }
                    
                    if (recommendations.length === 0) {
                      recommendations.push("Review diagnostic findings and consider database performance optimization");
                    }
                    
                    return recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start space-x-2">
                        <span className="text-blue-600 font-bold">•</span>
                        <span className="text-blue-700 text-sm">{rec}</span>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            )}
            
            {/* Execution Summary */}
            <div className="bg-white p-3 rounded border mb-3">
              <h5 className="font-medium text-gray-800 mb-2">📊 Execution Summary</h5>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><strong>Playbook:</strong> {fullPlaybookResult.playbook_id}</div>
                <div><strong>Execution ID:</strong> {fullPlaybookResult.execution_id}</div>
                <div><strong>Status:</strong> <span className="text-green-600">{fullPlaybookResult.status}</span></div>
                <div><strong>Confidence:</strong> <span className="text-blue-600">{(fullPlaybookResult.confidence_score * 100).toFixed(1)}%</span></div>
                <div><strong>Duration:</strong> {fullPlaybookResult.execution_time_seconds?.toFixed(3)}s</div>
                <div><strong>Root Cause Found:</strong> <span className={fullPlaybookResult.root_cause_found ? "text-green-600" : "text-red-600"}>{fullPlaybookResult.root_cause_found ? "Yes" : "No"}</span></div>
              </div>
            </div>

            {/* Step Results */}
            {fullPlaybookResult.step_results && fullPlaybookResult.step_results.length > 0 && (
              <div className="bg-white p-3 rounded border mb-3">
                <h5 className="font-medium text-gray-800 mb-2">⚙️ Diagnostic Steps ({fullPlaybookResult.step_results.length})</h5>
                <div className="space-y-3">
                  {fullPlaybookResult.step_results.map((step: any, index: number) => (
                    <div key={index} className="border-l-4 pl-3 py-2" style={{borderLeftColor: step.success ? '#10B981' : '#EF4444'}}>
                      <div className="flex justify-between items-start mb-1">
                        <div className="font-medium text-gray-700">Step {index + 1}</div>
                        <div className="flex items-center space-x-2">
                          <span className={`text-xs px-2 py-1 rounded ${step.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                            {step.status?._value_ || step.status}
                          </span>
                          <span className="text-xs text-gray-500">{(step.duration_seconds * 1000).toFixed(0)}ms</span>
                        </div>
                      </div>
                      
                      {/* Step Evidence */}
                      {step.evidence && (
                        <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                          {step.evidence.metric_name && (
                            <div className="mb-1">
                              <strong>Metric:</strong> {step.evidence.metric_name}
                              <span className="ml-2 text-blue-600">Latest: {step.evidence.latest_value}</span>
                              <span className="ml-2 text-gray-600">Threshold: {step.evidence.threshold}</span>
                            </div>
                          )}
                          
                          {step.evidence.query && (
                            <div className="mb-1">
                              <strong>Query:</strong> {step.evidence.query}
                              <span className="ml-2 text-red-600">Errors: {step.evidence.error_count}</span>
                              <span className="ml-2 text-yellow-600">Warnings: {step.evidence.warning_count}</span>
                            </div>
                          )}
                          
                          {step.evidence.avg_query_time_ms && (
                            <div className="mb-1">
                              <strong>Query Performance:</strong> {step.evidence.avg_query_time_ms.toFixed(0)}ms avg
                              <span className="ml-2 text-gray-600">Threshold: {step.evidence.threshold_ms}ms</span>
                            </div>
                          )}
                          
                          {step.evidence.sample_errors && step.evidence.sample_errors.length > 0 && (
                            <div className="mt-2">
                              <strong>Sample Error:</strong>
                              <div className="text-red-700 font-mono text-xs mt-1 p-2 bg-red-50 rounded">
                                {step.evidence.sample_errors[0].message}
                              </div>
                            </div>
                          )}
                          
                          {step.evidence.slow_queries && step.evidence.slow_queries.length > 0 && (
                            <div className="mt-2">
                              <strong>Slow Query:</strong>
                              <div className="text-orange-700 font-mono text-xs mt-1 p-2 bg-orange-50 rounded">
                                {step.evidence.slow_queries[0].query}
                                <div className="text-orange-600 mt-1">
                                  Execution: {step.evidence.slow_queries[0].execution_time_ms.toFixed(0)}ms | 
                                  Rows: {step.evidence.slow_queries[0].rows_affected?.toLocaleString()}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Escalation Notice */}
                      {step.escalation_triggered && (
                        <div className="mt-2 p-2 bg-red-100 border border-red-200 rounded text-xs">
                          <span className="text-red-800 font-medium">⚠️ Escalation Triggered</span>
                          <div className="text-red-700 mt-1">
                            Expected: {step.expected_value} | Actual: {step.actual_value}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {fullPlaybookResult.actions_recommended && fullPlaybookResult.actions_recommended.length > 0 && (
              <div className="bg-white p-3 rounded border">
                <h5 className="font-medium text-gray-800 mb-2">💡 Recommended Actions</h5>
                <ul className="list-disc list-inside text-gray-700 space-y-1">
                  {fullPlaybookResult.actions_recommended.map((action: string, index: number) => (
                    <li key={index} className="text-sm">{action}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Show raw JSON as collapsible section */}
            <details className="mt-3">
              <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800">🔍 Show Raw Data</summary>
              <div className="bg-white p-3 rounded border mt-2">
                <pre className="text-xs text-gray-700 whitespace-pre-wrap overflow-x-auto">
                  {JSON.stringify(fullPlaybookResult, null, 2)}
                </pre>
              </div>
            </details>
          </div>
        )}
      </div>
    );
  }, []);
  
  
  
  const renderProgressBar = useCallback((progress?: number) => {
    if (progress === undefined) return null;
    
    return (
      <div className="mt-2">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Progress</span>
          <span>{progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-primary-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    );
  }, []);
  
  const renderStepDetails = useCallback((metadata?: any) => {
    if (!metadata) return null;
    
    const details = [];
    
    if (metadata.step) {
      details.push(
        <div key="step" className="text-xs text-gray-500">
          <strong>Step:</strong> {metadata.step}
        </div>
      );
    }
    
    if (metadata.currentStep && metadata.totalSteps) {
      details.push(
        <div key="step-count" className="text-xs text-gray-500">
          <strong>Step:</strong> {metadata.currentStep}/{metadata.totalSteps}
        </div>
      );
    }
    
    if (metadata.confidence) {
      const percentage = Math.round(metadata.confidence * 100);
      const color = metadata.confidence >= 0.8 ? 'text-success-600' : metadata.confidence >= 0.6 ? 'text-warning-600' : 'text-error-600';
      details.push(
        <div key="confidence" className="text-xs text-gray-500">
          <strong>Confidence:</strong> <span className={`font-medium ${color}`}>{percentage}%</span>
        </div>
      );
    }
    
    if (details.length === 0) return null;
    
    return (
      <div className="mt-2 space-y-1">
        {details}
      </div>
    );
  }, []);

  const renderApprovalButtons = useCallback((message: ChatMessage) => {
    if (!message.metadata?.requiresApproval) return null;

    return (
      <div className="mt-3 flex space-x-2">
        <button
          onClick={() => {
            // Handle approval
            console.log('Approved action for:', message.metadata);
          }}
          className="btn-success text-xs py-1 px-3"
        >
          ✅ Approve
        </button>
        <button
          onClick={() => {
            // Handle rejection
            console.log('Rejected action for:', message.metadata);
          }}
          className="btn-error text-xs py-1 px-3"
        >
          ❌ Reject
        </button>
      </div>
    );
  }, []);

  const formatAnalysisUpdate = useCallback((notification: any) => {
    const data = notification.data;
    let content = notification.message;
    
    // Add step information if available
    if (data?.step) {
      content = `**Step: ${data.step}**\n${content}`;
    }
    
    // Add progress bar
    if (data?.progress !== undefined) {
      const progressBar = '▓'.repeat(Math.floor(data.progress / 10)) + '░'.repeat(10 - Math.floor(data.progress / 10));
      content += `\n\n**Progress:** ${data.progress}% [${progressBar}]`;
    }
    
    // Add detailed information for completed analysis
    if (data?.status === 'completed') {
      if (data.confidence) {
        content += `\n\n**Confidence Score:** ${(data.confidence * 100).toFixed(1)}%`;
      }
      
      if (data.root_cause) {
        content += `\n\n**Root Cause:** ${data.root_cause}`;
      }
      
      if (data.recommendations && data.recommendations.length > 0) {
        content += `\n\n**Recommendations:**`;
        data.recommendations.forEach((rec: string, index: number) => {
          content += `\n• ${rec}`;
        });
      }
      
      if (data.similar_incidents && data.similar_incidents.length > 0) {
        content += `\n\n**Similar Incidents:**`;
        data.similar_incidents.forEach((incident: any) => {
          content += `\n• ${incident.id} (${(incident.similarity * 100).toFixed(1)}% similarity)`;
        });
      }
      
      if (data.evidence && data.evidence.length > 0) {
        content += `\n\n**Evidence:**`;
        data.evidence.forEach((item: any) => {
          content += `\n• ${item.description} (${item.type})`;
        });
      }
    }
    
    return content;
  }, []);


  

  const formatSystemStatus = useCallback((statusData: any) => {
    let content = "🔧 **System Status Report**\n\n";
    
    // Overall status
    const statusEmoji = statusData.overall_status === "healthy" ? "✅" : "⚠️";
    content += `**Overall Status:** ${statusEmoji} ${statusData.overall_status.toUpperCase()}\n\n`;
    
    // Active metrics
    content += `**Active Metrics:**\n`;
    content += `• Incidents: ${statusData.active_incidents || 0}\n`;
    content += `• Analyses: ${statusData.active_analyses || 0}\n`;
    content += `• Playbooks: ${statusData.active_playbooks || 0}\n`;
    content += `• Users: ${statusData.active_users || 0}\n\n`;
    
    // GCP Services
    if (statusData.gcp_services) {
      content += `**GCP Services:**\n`;
      Object.entries(statusData.gcp_services).forEach(([service, status]) => {
        const emoji = status === "healthy" ? "✅" : "⚠️";
        content += `• ${service}: ${emoji} ${status}\n`;
      });
      content += "\n";
    }
    
    // Confidence scores
    if (statusData.confidence_scores) {
      const avg = (statusData.confidence_scores.avg_last_24h * 100).toFixed(1);
      const trend = statusData.confidence_scores.trend;
      const trendEmoji = trend === "stable" ? "📊" : trend === "improving" ? "📈" : "📉";
      content += `**AI Confidence (24h):** ${avg}% ${trendEmoji} ${trend}\n`;
    }
    
    return content;
  }, []);
  const formatPlaybookUpdate = useCallback((notification: any) => {
    const data = notification.data;
    let content = notification.message;
    
    // Add step information if available
    if (data?.current_step && data?.total_steps) {
      content = `**Step ${data.current_step}/${data.total_steps}**\n${content}`;
    }
    
    // Add progress bar
    if (data?.progress !== undefined) {
      const progressBar = '▓'.repeat(Math.floor(data.progress / 10)) + '░'.repeat(10 - Math.floor(data.progress / 10));
      content += `\n\n**Progress:** ${data.progress}% [${progressBar}]`;
    }
    
    // Add step details
    if (data?.step_details) {
      const details = data.step_details;
      if (details.step_description) {
        content += `\n\n**Description:** ${details.step_description}`;
      }
      if (details.command) {
        content += `\n**Command:** \`${details.command}\``;
      }
      if (details.expected) {
        content += `\n**Expected:** ${details.expected}`;
      }
    }
    
    // Add step results
    if (data?.step_result) {
      const result = data.step_result;
      content += `\n\n**Result:**`;
      
      if (result.success !== undefined) {
        content += `\n• Success: ${result.success ? '✅' : '❌'}`;
      }
      
      if (result.actual_value !== undefined) {
        content += `\n• Actual Value: ${result.actual_value}`;
      }
      
      if (result.threshold !== undefined) {
        content += `\n• Threshold: ${result.threshold}`;
      }
      
      if (result.error_count !== undefined) {
        content += `\n• Error Count: ${result.error_count}`;
      }
      
      if (result.slow_query_identified) {
        content += `\n• Slow Query: ${result.slow_query || 'Identified'}`;
      }
    }
    
    // Add final analysis for completed playbooks
    if (data?.status === 'completed') {
      if (data.confidence) {
        content += `\n\n**Confidence Score:** ${(data.confidence * 100).toFixed(1)}%`;
      }
      
      if (data.root_cause_found) {
        content += `\n\n**Root Cause Analysis:** Complete`;
      }
      
      if (data.recommendations && data.recommendations.length > 0) {
        content += `\n\n**Recommendations:**`;
        data.recommendations.forEach((rec: string) => {
          content += `\n• ${rec}`;
        });
      }
    }
    
    return content;
  }, []);



  const renderConfidenceScore = useCallback((score?: number) => {
    if (!score) return null;

    const percentage = Math.round(score * 100);
    const color = score >= 0.8 ? 'text-success-600' : score >= 0.6 ? 'text-warning-600' : 'text-error-600';

    return (
      <div className="mt-2 text-xs text-gray-500">
        Confidence: <span className={`font-medium ${color}`}>{percentage}%</span>
      </div>
    );
  }, []);

  return (
    <div className={`flex flex-col h-full bg-white ${className}`}>
      {/* Chat Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
              <span className="text-white font-medium text-lg">🤖</span>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">AI SRE Agent</h2>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success-500' : 'bg-error-500'}`} />
                <span className="text-sm text-gray-500">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
                {isAnalyzing && (
                  <>
                    <span className="text-gray-300">•</span>
                    <span className="text-sm text-primary-600 font-medium">Analyzing...</span>
                  </>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowSuggestions(!showSuggestions)}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
              title="Show suggestions"
            >
              <DocumentTextIcon className="w-5 h-5" />
            </button>
            <button
              className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
              title="Settings"
            >
              <CogIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'} items-end space-x-2 max-w-full`}>
              {/* Message Icon/Avatar */}
              {message.type !== 'user' && (
                <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center">
                  {getMessageIcon(message)}
                </div>
              )}
              
              {/* Message Bubble */}
              <div className={`${message.type === 'user' ? 'ml-2' : 'mr-2'}`}>
                <div className={getMessageStyle(message)}>
                  {renderMessageContent(message)}
                  {renderConfidenceScore(message.metadata?.confidenceScore)}
                  {renderApprovalButtons(message)}
                </div>
                
                {/* Message Metadata */}
                <div className={`mt-1 text-xs text-gray-400 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                  {message.type === 'user' && message.user?.username && (
                    <span className="mr-2">{message.user.username}</span>
                  )}
                  <span>{formatTimestamp(message.timestamp)}</span>
                  {message.metadata?.status && (
                    <span className="ml-2 text-primary-500">• {message.metadata.status}</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="flex items-end space-x-2">
              <div className="w-8 h-8 flex items-center justify-center">
                🤖
              </div>
              <div className="bg-gray-100 rounded-lg px-4 py-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full typing-indicator"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full typing-indicator"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full typing-indicator"></div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions Panel */}
      {showSuggestions && (
        <div className="flex-shrink-0 px-6 py-3 border-t border-gray-200 bg-gray-50">
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Quick Commands</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion.text)}
                  className="flex items-center space-x-2 p-3 text-left bg-white border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors"
                >
                  <div className="flex-shrink-0 text-primary-500">
                    {suggestion.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-900 truncate">
                      {suggestion.text}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                      {suggestion.description}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="flex-shrink-0 px-6 py-4 border-t border-gray-200 bg-white">
        <div className="flex items-end space-x-3">
          <div className="flex-1">
            <div className="relative">
              <input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder={isConnected ? "Type a message or @sre-bot for commands..." : "Connecting..."}
                disabled={!isConnected}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100 disabled:cursor-not-allowed resize-none"
              />
              {inputMessage.includes('@') && (
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <span className="text-xs text-primary-500 bg-primary-50 px-2 py-1 rounded">
                    AI Command
                  </span>
                </div>
              )}
            </div>
            
            {!isConnected && (
              <div className="mt-2 flex items-center text-sm text-error-600">
                <ExclamationTriangleIcon className="w-4 h-4 mr-1" />
                Connection lost. Attempting to reconnect...
              </div>
            )}
          </div>
          
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || !isConnected}
            className="p-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </div>
        
        <div className="mt-2 text-xs text-gray-500">
          Press Enter to send • Shift+Enter for new line • Type @ for AI commands
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;