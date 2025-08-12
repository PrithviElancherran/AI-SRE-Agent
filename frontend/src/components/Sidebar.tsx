/**
 * React TypeScript component for the main application sidebar.
 * 
 * This component provides navigation, status indicators, notifications,
 * and real-time connection status for the AI SRE Agent frontend.
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  DocumentTextIcon,
  ClockIcon,
  CogIcon,
  BellIcon,
  WifiIcon,
  UserIcon,
  HomeIcon,
  XMarkIcon,
  CheckCircleIcon,
  InformationCircleIcon,
  CommandLineIcon
} from '@heroicons/react/24/outline';
import {
  ChatBubbleLeftRightIcon as ChatBubbleLeftRightIconSolid,
  ChartBarIcon as ChartBarIconSolid,
  ExclamationTriangleIcon as ExclamationTriangleIconSolid,
  DocumentTextIcon as DocumentTextIconSolid,
  ClockIcon as ClockIconSolid
} from '@heroicons/react/24/solid';
import { format } from 'date-fns';

interface SidebarProps {
  currentView: 'chat' | 'incidents';
  onViewChange: (view: 'chat' | 'incidents') => void;
  isConnected: boolean;
  notifications: any[];
  onNotificationRemove: (id: number) => void;
  className?: string;
}

interface NavigationItem {
  id: 'chat' | 'incidents';
  label: string;
  icon: React.ReactNode;
  activeIcon: React.ReactNode;
  description: string;
  badge?: number;
  color?: string;
}

const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onViewChange,
  isConnected,
  notifications,
  onNotificationRemove,
  className = ''
}) => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Navigation items configuration
  const navigationItems: NavigationItem[] = useMemo(() => [
    {
      id: 'chat',
      label: 'AI Chat',
      icon: <ChatBubbleLeftRightIcon className="w-6 h-6" />,
      activeIcon: <ChatBubbleLeftRightIconSolid className="w-6 h-6" />,
      description: 'Chat with AI SRE Agent',
      color: 'text-primary-600'
    },
    {
      id: 'incidents',
      label: 'Incidents',
      icon: <ExclamationTriangleIcon className="w-6 h-6" />,
      activeIcon: <ExclamationTriangleIconSolid className="w-6 h-6" />,
      description: 'Manage production incidents',
      badge: notifications.filter(n => n.type === 'incident' || n.type === 'analysis_update').length,
      color: 'text-orange-600'
    }
  ], [notifications]);

  // Group notifications by type
  const groupedNotifications = useMemo(() => {
    const groups = {
      ai_response: [] as any[],
      analysis_update: [] as any[],
      playbook_update: [] as any[],
      system: [] as any[],
      other: [] as any[]
    };

    notifications.forEach(notification => {
      const type = notification.type || 'other';
      if (groups[type as keyof typeof groups]) {
        groups[type as keyof typeof groups].push(notification);
      } else {
        groups.other.push(notification);
      }
    });

    return groups;
  }, [notifications]);

  // Get unread notification count
  const unreadCount = useMemo(() => {
    return notifications.filter(n => !n.read).length;
  }, [notifications]);

  // Handle navigation item click
  const handleNavClick = useCallback((viewId: typeof currentView) => {
    onViewChange(viewId);
  }, [onViewChange]);

  // Toggle notifications panel
  const toggleNotifications = useCallback(() => {
    setShowNotifications(!showNotifications);
  }, [showNotifications]);

  // Format notification timestamp
  const formatNotificationTime = useCallback((timestamp: string) => {
    try {
      return format(new Date(timestamp), 'HH:mm');
    } catch {
      return '';
    }
  }, []);

  // Get notification icon
  const getNotificationIcon = useCallback((notification: any) => {
    const iconProps = "w-4 h-4";
    
    switch (notification.type) {
      case 'ai_response':
        return <ChatBubbleLeftRightIcon className={`${iconProps} text-primary-500`} />;
      case 'analysis_update':
        return <ChartBarIcon className={`${iconProps} text-blue-500`} />;
      case 'playbook_update':
        return <CommandLineIcon className={`${iconProps} text-green-500`} />;
      case 'system':
        return <InformationCircleIcon className={`${iconProps} text-gray-500`} />;
      default:
        return <BellIcon className={`${iconProps} text-gray-500`} />;
    }
  }, []);

  // Get notification color based on priority
  const getNotificationColor = useCallback((notification: any) => {
    if (notification.data?.status === 'failed' || notification.data?.error) {
      return 'border-l-red-500 bg-red-50';
    } else if (notification.data?.status === 'completed') {
      return 'border-l-green-500 bg-green-50';
    } else if (notification.data?.status === 'in_progress') {
      return 'border-l-blue-500 bg-blue-50';
    }
    return 'border-l-gray-500 bg-gray-50';
  }, []);

  // Render navigation item
  const renderNavItem = useCallback((item: NavigationItem) => {
    const isActive = currentView === item.id;
    
    return (
      <button
        key={item.id}
        onClick={() => handleNavClick(item.id)}
        className={`w-full flex items-center space-x-3 px-3 py-3 rounded-lg transition-all duration-200 group relative ${
          isActive
            ? 'bg-primary-100 text-primary-700 shadow-sm'
            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
        }`}
        title={isCollapsed ? item.description : undefined}
      >
        <div className={`flex-shrink-0 ${isActive ? item.color : ''}`}>
          {isActive ? item.activeIcon : item.icon}
        </div>
        
        {!isCollapsed && (
          <>
            <span className="font-medium truncate">{item.label}</span>
            {item.badge !== undefined && item.badge > 0 && (
              <span className="ml-auto bg-red-500 text-white text-xs rounded-full px-2 py-0.5 min-w-[20px] text-center">
                {item.badge > 99 ? '99+' : item.badge}
              </span>
            )}
          </>
        )}

        {/* Active indicator */}
        {isActive && (
          <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-1 h-6 bg-primary-600 rounded-r" />
        )}

        {/* Tooltip for collapsed state */}
        {isCollapsed && (
          <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-sm rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 whitespace-nowrap z-50">
            {item.description}
            {item.badge !== undefined && item.badge > 0 && (
              <span className="ml-2 bg-red-500 text-white text-xs rounded-full px-1.5 py-0.5">
                {item.badge}
              </span>
            )}
          </div>
        )}
      </button>
    );
  }, [currentView, handleNavClick, isCollapsed]);

  // Render notification item
  const renderNotification = useCallback((notification: any, index: number) => {
    return (
      <div
        key={notification.id || index}
        className={`p-3 border-l-4 ${getNotificationColor(notification)} mb-2 last:mb-0`}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-2 flex-1">
            {getNotificationIcon(notification)}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {notification.title || 'Notification'}
              </p>
              <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                {notification.message}
              </p>
              <div className="flex items-center space-x-2 mt-2">
                <span className="text-xs text-gray-500">
                  {formatNotificationTime(notification.timestamp)}
                </span>
                {notification.data?.status && (
                  <span className="text-xs bg-gray-200 text-gray-700 px-2 py-0.5 rounded">
                    {notification.data.status}
                  </span>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={() => onNotificationRemove(notification.id)}
            className="ml-2 text-gray-400 hover:text-gray-600 flex-shrink-0"
          >
            <XMarkIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }, [getNotificationColor, getNotificationIcon, formatNotificationTime, onNotificationRemove]);

  // Render notifications panel
  const renderNotificationsPanel = useCallback(() => {
    if (!showNotifications) return null;

    return (
      <div className="absolute right-0 top-0 w-80 h-full bg-white border-l border-gray-200 shadow-lg z-50">
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Notifications</h3>
          <div className="flex items-center space-x-2">
            {unreadCount > 0 && (
              <span className="bg-red-500 text-white text-xs rounded-full px-2 py-1">
                {unreadCount} new
              </span>
            )}
            <button
              onClick={toggleNotifications}
              className="text-gray-400 hover:text-gray-600"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {notifications.length === 0 ? (
            <div className="text-center py-8">
              <BellIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No notifications</h3>
              <p className="mt-1 text-sm text-gray-500">
                You're all caught up! New notifications will appear here.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {/* AI Responses */}
              {groupedNotifications.ai_response.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <ChatBubbleLeftRightIcon className="w-4 h-4 mr-2 text-primary-500" />
                    AI Responses ({groupedNotifications.ai_response.length})
                  </h4>
                  {groupedNotifications.ai_response.map(renderNotification)}
                </div>
              )}

              {/* Analysis Updates */}
              {groupedNotifications.analysis_update.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <ChartBarIcon className="w-4 h-4 mr-2 text-blue-500" />
                    Analysis Updates ({groupedNotifications.analysis_update.length})
                  </h4>
                  {groupedNotifications.analysis_update.map(renderNotification)}
                </div>
              )}

              {/* Playbook Updates */}
              {groupedNotifications.playbook_update.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <CommandLineIcon className="w-4 h-4 mr-2 text-green-500" />
                    Playbook Updates ({groupedNotifications.playbook_update.length})
                  </h4>
                  {groupedNotifications.playbook_update.map(renderNotification)}
                </div>
              )}

              {/* System Notifications */}
              {groupedNotifications.system.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <InformationCircleIcon className="w-4 h-4 mr-2 text-gray-500" />
                    System ({groupedNotifications.system.length})
                  </h4>
                  {groupedNotifications.system.map(renderNotification)}
                </div>
              )}

              {/* Other Notifications */}
              {groupedNotifications.other.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <BellIcon className="w-4 h-4 mr-2 text-gray-500" />
                    Other ({groupedNotifications.other.length})
                  </h4>
                  {groupedNotifications.other.map(renderNotification)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }, [showNotifications, unreadCount, notifications, groupedNotifications, toggleNotifications, renderNotification]);

  return (
    <div className={`relative ${className}`}>
      <div className={`bg-white border-r border-gray-200 h-full flex flex-col transition-all duration-300 ${
        isCollapsed ? 'w-16' : 'w-64'
      }`}>
        {/* Header */}
        <div className="flex-shrink-0 px-4 py-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            {!isCollapsed && (
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">AI</span>
                </div>
                <div>
                  <h1 className="text-lg font-semibold text-gray-900">SRE Agent</h1>
                  <p className="text-xs text-gray-500">AI-Powered Operations</p>
                </div>
              </div>
            )}
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
              title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              <svg
                className={`w-4 h-4 transition-transform duration-200 ${isCollapsed ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            </button>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex-1 px-3 py-4 overflow-y-auto">
          <nav className="space-y-1">
            {navigationItems.map(renderNavItem)}
          </nav>
        </div>

        {/* Bottom Actions */}
        <div className="flex-shrink-0 px-3 py-4 border-t border-gray-200 space-y-3">
          {/* Connection Status */}
          <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${
            isConnected ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
          }`}>
            <div className="flex-shrink-0">
              {isConnected ? (
                <WifiIcon className="w-4 h-4" />
              ) : (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
              )}
            </div>
            {!isCollapsed && (
              <span className="text-sm font-medium">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            )}
          </div>

          {/* Notifications */}
          <button
            onClick={toggleNotifications}
            className={`w-full flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors relative ${
              showNotifications ? 'bg-primary-100 text-primary-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            }`}
            title={isCollapsed ? 'Notifications' : undefined}
          >
            <BellIcon className="w-5 h-5 flex-shrink-0" />
            {!isCollapsed && <span className="font-medium">Notifications</span>}
            {unreadCount > 0 && (
              <span className={`absolute ${isCollapsed ? '-top-1 -right-1' : 'right-2 top-1'} bg-red-500 text-white text-xs rounded-full px-1.5 py-0.5 min-w-[18px] text-center`}>
                {unreadCount > 99 ? '99+' : unreadCount}
              </span>
            )}
          </button>

          {/* Settings */}
          <button
            className="w-full flex items-center space-x-2 px-3 py-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors"
            title={isCollapsed ? 'Settings' : undefined}
          >
            <CogIcon className="w-5 h-5 flex-shrink-0" />
            {!isCollapsed && <span className="font-medium">Settings</span>}
          </button>

          {/* User Profile */}
          <button
            className="w-full flex items-center space-x-2 px-3 py-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors"
            title={isCollapsed ? 'Profile' : undefined}
          >
            <UserIcon className="w-5 h-5 flex-shrink-0" />
            {!isCollapsed && <span className="font-medium">Profile</span>}
          </button>
        </div>
      </div>

      {/* Notifications Panel */}
      {renderNotificationsPanel()}

      {/* Overlay */}
      {showNotifications && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-25 z-40"
          onClick={toggleNotifications}
        />
      )}
    </div>
  );
};

export default Sidebar;