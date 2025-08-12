/**
 * Custom React hook for WebSocket connection management.
 * 
 * This hook provides WebSocket connection management, real-time message handling,
 * subscription management, and connection state tracking for the AI SRE Agent frontend.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
  [key: string]: any;
}

interface WebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  timeout?: number;
  enableHeartbeat?: boolean;
  heartbeatInterval?: number;
}

interface ConnectionState {
  isConnected: boolean;
  isConnecting: boolean;
  isReconnecting: boolean;
  reconnectAttempts: number;
  lastConnectedAt?: Date;
  lastDisconnectedAt?: Date;
  connectionId?: string;
}

interface UseWebSocketReturn {
  socket: WebSocket | null;
  isConnected: boolean;
  isConnecting: boolean;
  isReconnecting: boolean;
  connectionState: ConnectionState;
  sendMessage: (message: WebSocketMessage) => void;
  subscribeToUpdates: (eventType: string, handler: (data: any) => void) => void;
  unsubscribeFromUpdates: (eventType: string) => void;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  getConnectionStats: () => {
    totalMessages: number;
    totalReconnects: number;
    uptime: number;
    lastMessageAt?: Date;
  };
}

const DEFAULT_OPTIONS: WebSocketOptions = {
  autoConnect: true,
  reconnectAttempts: 5,
  reconnectDelay: 1000,
  timeout: 10000,
  enableHeartbeat: true,
  heartbeatInterval: 30000
};

export const useWebSocket = (
  url: string,
  user: { id: string; username: string } | null,
  options: WebSocketOptions = {}
): UseWebSocketReturn => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // WebSocket instance
  const socketRef = useRef<WebSocket | null>(null);
  
  // Connection state
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    isConnected: false,
    isConnecting: false,
    isReconnecting: false,
    reconnectAttempts: 0
  });

  // Event handlers registry
  const eventHandlersRef = useRef<Map<string, Set<(data: any) => void>>>(new Map());
  
  // Connection stats
  const statsRef = useRef({
    totalMessages: 0,
    totalReconnects: 0,
    connectedAt: null as Date | null,
    lastMessageAt: null as Date | null
  });

  // Heartbeat timer
  const heartbeatTimerRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize WebSocket connection
  const initializeConnection = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState(prev => ({
      ...prev,
      isConnecting: true,
      isReconnecting: prev.reconnectAttempts > 0
    }));

    try {
      // Create WebSocket URL with query parameters
      const wsUrl = new URL(url);
      wsUrl.searchParams.append('user_id', user?.id || '');
      wsUrl.searchParams.append('token', 'demo_token'); // In real implementation, use actual auth token
      
      // Create native WebSocket connection
      const socket = new WebSocket(wsUrl.toString());
      socketRef.current = socket;

      // Connection established
      socket.onopen = () => {
        console.log('WebSocket connected');
        
        const now = new Date();
        statsRef.current.connectedAt = now;
        
        setConnectionState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          isReconnecting: false,
          reconnectAttempts: 0,
          lastConnectedAt: now,
          connectionId: `ws_${Date.now()}`
        }));

        // Start heartbeat if enabled
        if (opts.enableHeartbeat) {
          startHeartbeat();
        }

        // Subscribe to default events
        subscribeToDefaultEvents();
      };

      // Connection error
      socket.onerror = (error) => {
        console.error('WebSocket connection error:', error);
        
        setConnectionState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false
        }));

        handleReconnection();
      };

      // Disconnection
      socket.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        
        const now = new Date();
        
        setConnectionState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
          lastDisconnectedAt: now
        }));

        stopHeartbeat();

        // Attempt reconnection if not intentional disconnect (code 1000 is normal closure)
        if (event.code !== 1000) {
          handleReconnection();
        }
      };

      // Handle incoming messages
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const eventType = data.type || 'message';
          
          console.log('WebSocket message received:', eventType, data);
          
          statsRef.current.totalMessages++;
          statsRef.current.lastMessageAt = new Date();

          // Call registered handlers for this event type
          const handlers = eventHandlersRef.current.get(eventType);
          if (handlers) {
            handlers.forEach(handler => {
              try {
                handler(data);
              } catch (error) {
                console.error(`Error in event handler for ${eventType}:`, error);
              }
            });
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

    } catch (error) {
      console.error('Failed to initialize WebSocket connection:', error);
      
      setConnectionState(prev => ({
        ...prev,
        isConnected: false,
        isConnecting: false
      }));

      handleReconnection();
    }
  }, [url, user, opts.timeout, opts.enableHeartbeat]);

  // Handle reconnection logic
  const handleReconnection = useCallback(() => {
    setConnectionState(prev => {
      if (prev.reconnectAttempts >= opts.reconnectAttempts!) {
        console.warn('Max reconnection attempts reached');
        return prev;
      }

      const delay = opts.reconnectDelay! * Math.pow(2, prev.reconnectAttempts); // Exponential backoff
      
      console.log(`Attempting reconnection in ${delay}ms (attempt ${prev.reconnectAttempts + 1}/${opts.reconnectAttempts})`);

      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }

      reconnectTimerRef.current = setTimeout(() => {
        statsRef.current.totalReconnects++;
        initializeConnection();
      }, delay);

      return {
        ...prev,
        reconnectAttempts: prev.reconnectAttempts + 1,
        isReconnecting: true
      };
    });
  }, [opts.reconnectAttempts, opts.reconnectDelay, initializeConnection]);

  // Start heartbeat mechanism
  const startHeartbeat = useCallback(() => {
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
    }

    heartbeatTimerRef.current = setInterval(() => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString(),
          userId: user?.id
        }));
      }
    }, opts.heartbeatInterval);
  }, [opts.heartbeatInterval, user?.id]);

  // Stop heartbeat mechanism
  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
  }, []);

  // Subscribe to default system events
  const subscribeToDefaultEvents = useCallback(() => {
    // Default event handlers are now handled in the onmessage callback
    // based on the message type field
  }, []);

  // Send message through WebSocket
  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      console.warn('Cannot send message: WebSocket not connected');
      return;
    }

    try {
      const messageWithMetadata = {
        ...message,
        timestamp: message.timestamp || new Date().toISOString(),
        userId: user?.id,
        username: user?.username
      };

      socketRef.current.send(JSON.stringify(messageWithMetadata));
      console.log('Message sent:', message.type, messageWithMetadata);
      
      statsRef.current.totalMessages++;
      statsRef.current.lastMessageAt = new Date();
      
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  }, [user]);

  // Subscribe to specific event updates
  const subscribeToUpdates = useCallback((eventType: string, handler: (data: any) => void) => {
    if (!eventHandlersRef.current.has(eventType)) {
      eventHandlersRef.current.set(eventType, new Set());
    }
    
    eventHandlersRef.current.get(eventType)!.add(handler);
    console.log(`Subscribed to event: ${eventType}`);

    // If connected, also subscribe on the server side
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        type: 'subscription_request',
        action: 'subscribe',
        subscription_type: eventType,
        timestamp: new Date().toISOString()
      }));
    }
  }, []);

  // Unsubscribe from specific event updates
  const unsubscribeFromUpdates = useCallback((eventType: string) => {
    const handlers = eventHandlersRef.current.get(eventType);
    if (handlers) {
      handlers.clear();
      eventHandlersRef.current.delete(eventType);
      console.log(`Unsubscribed from event: ${eventType}`);

      // If connected, also unsubscribe on the server side
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({
          type: 'subscription_request',
          action: 'unsubscribe',
          subscription_type: eventType,
          timestamp: new Date().toISOString()
        }));
      }
    }
  }, []);

  // Manual connection control
  const connect = useCallback(() => {
    if (!socketRef.current || socketRef.current.readyState === WebSocket.CLOSED) {
      initializeConnection();
    }
  }, [initializeConnection]);

  // Manual disconnection
  const disconnect = useCallback(() => {
    if (socketRef.current) {
      stopHeartbeat();
      
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }

      if (socketRef.current.readyState === WebSocket.OPEN || socketRef.current.readyState === WebSocket.CONNECTING) {
        socketRef.current.close(1000, 'Client disconnect');
      }
      socketRef.current = null;
      
      setConnectionState(prev => ({
        ...prev,
        isConnected: false,
        isConnecting: false,
        isReconnecting: false,
        reconnectAttempts: 0,
        lastDisconnectedAt: new Date()
      }));
    }
  }, [stopHeartbeat]);

  // Manual reconnection
  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(() => {
      setConnectionState(prev => ({
        ...prev,
        reconnectAttempts: 0
      }));
      initializeConnection();
    }, 100);
  }, [disconnect, initializeConnection]);

  // Get connection statistics
  const getConnectionStats = useCallback(() => {
    const now = new Date();
    const uptime = statsRef.current.connectedAt 
      ? now.getTime() - statsRef.current.connectedAt.getTime()
      : 0;

    return {
      totalMessages: statsRef.current.totalMessages,
      totalReconnects: statsRef.current.totalReconnects,
      uptime: uptime,
      lastMessageAt: statsRef.current.lastMessageAt || undefined
    };
  }, []);

  // Initialize connection on mount
  useEffect(() => {
    if (opts.autoConnect && user) {
      initializeConnection();
    }

    return () => {
      disconnect();
    };
  }, [opts.autoConnect, user, initializeConnection, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopHeartbeat();
      
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      
      if (socketRef.current && (socketRef.current.readyState === WebSocket.OPEN || socketRef.current.readyState === WebSocket.CONNECTING)) {
        socketRef.current.close();
      }
    };
  }, [stopHeartbeat]);

  return {
    socket: socketRef.current,
    isConnected: connectionState.isConnected,
    isConnecting: connectionState.isConnecting,
    isReconnecting: connectionState.isReconnecting,
    connectionState,
    sendMessage,
    subscribeToUpdates,
    unsubscribeFromUpdates,
    connect,
    disconnect,
    reconnect,
    getConnectionStats
  };
};

export default useWebSocket;