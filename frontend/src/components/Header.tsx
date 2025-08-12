/**
 * React TypeScript component for the main application header.
 * 
 * This component provides user information, breadcrumbs, status indicators,
 * and control actions for the AI SRE Agent frontend.
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  UserIcon,
  ChevronRightIcon,
  BellIcon,
  Cog6ToothIcon,
  MagnifyingGlassIcon,
  CommandLineIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ChartBarIcon,
  DocumentTextIcon,
  HomeIcon,
  ChatBubbleLeftRightIcon,
  XMarkIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import {
  BellIcon as BellIconSolid,
  ExclamationTriangleIcon as ExclamationTriangleIconSolid
} from '@heroicons/react/24/solid';
import { format } from 'date-fns';
import { Incident } from '../types/incident';

interface HeaderProps {
  user: {
    id: string;
    username: string;
    role: string;
  } | null;
  isConnected: boolean;
  currentView: 'chat' | 'dashboard' | 'incidents' | 'playbooks' | 'timeline';
  selectedIncident?: Incident | null;
  isAnalyzing: boolean;
  notifications: any[];
  onNotificationRemove: (id: number) => void;
  className?: string;
}

interface BreadcrumbItem {
  label: string;
  icon?: React.ReactNode;
  onClick?: () => void;
}

const Header: React.FC<HeaderProps> = ({
  user,
  isConnected,
  currentView,
  selectedIncident,
  isAnalyzing,
  notifications,
  onNotificationRemove,
  className = ''
}) => {
  const [showNotificationDropdown, setShowNotificationDropdown] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(false);

  // Generate breadcrumbs based on current view
  const breadcrumbs = useMemo((): BreadcrumbItem[] => {
    const baseBreadcrumb: BreadcrumbItem = {
      label: 'AI SRE Agent',
      icon: <HomeIcon className="w-4 h-4" />
    };

    const viewConfigs = {
      chat: {
        label: 'AI Chat',
        icon: <ChatBubbleLeftRightIcon className="w-4 h-4" />
      },
      dashboard: {
        label: 'Dashboard',
        icon: <ChartBarIcon className="w-4 h-4" />
      },
      incidents: {
        label: 'Incidents',
        icon: <ExclamationTriangleIcon className="w-4 h-4" />
      },
      playbooks: {
        label: 'Playbooks',
        icon: <DocumentTextIcon className="w-4 h-4" />
      },
      timeline: {
        label: 'Timeline',
        icon: <ClockIcon className="w-4 h-4" />
      }
    };

    const breadcrumbItems = [baseBreadcrumb];

    if (currentView !== 'dashboard') {
      breadcrumbItems.push({
        label: viewConfigs[currentView].label,
        icon: viewConfigs[currentView].icon
      });
    }

    // Add incident-specific breadcrumb if selected
    if (selectedIncident && (currentView === 'incidents' || currentView === 'timeline')) {
      breadcrumbItems.push({
        label: selectedIncident.incidentId,
        icon: <ExclamationTriangleIcon className="w-4 h-4" />
      });
    }

    return breadcrumbItems;
  }, [currentView, selectedIncident]);

  // Get unread notification count
  const unreadNotificationCount = useMemo(() => {
    return notifications.filter(n => !n.read).length;
  }, [notifications]);

  // Get recent notifications for dropdown
  const recentNotifications = useMemo(() => {
    return notifications.slice(0, 5);
  }, [notifications]);

  // Get page title based on current view
  const pageTitle = useMemo(() => {
    const titles = {
      chat: 'AI Chat',
      dashboard: 'Dashboard',
      incidents: selectedIncident ? `Incident: ${selectedIncident.title}` : 'Incidents',
      playbooks: 'Playbooks',
      timeline: selectedIncident ? `Timeline: ${selectedIncident.incidentId}` : 'Timeline'
    };
    return titles[currentView];
  }, [currentView, selectedIncident]);

  // Format notification time
  const formatNotificationTime = useCallback((timestamp: string) => {
    try {
      return format(new Date(timestamp), 'HH:mm');
    } catch {
      return '';
    }
  }, []);

  // Get notification icon
  const getNotificationIcon = useCallback((notification: any) => {
    const iconClass = "w-4 h-4";
    
    switch (notification.type) {
      case 'ai_response':
        return <ChatBubbleLeftRightIcon className={`${iconClass} text-primary-500`} />;
      case 'analysis_update':
        return <ChartBarIcon className={`${iconClass} text-blue-500`} />;
      case 'playbook_update':
        return <CommandLineIcon className={`${iconClass} text-green-500`} />;
      case 'incident':
        return <ExclamationTriangleIconSolid className={`${iconClass} text-red-500`} />;
      default:
        return <BellIcon className={`${iconClass} text-gray-500`} />;
    }
  }, []);

  // Quick actions based on current view
  const quickActions = useMemo(() => {
    const actions = [];
    
    if (currentView === 'incidents') {
      actions.push({
        label: 'New Incident',
        icon: <ExclamationTriangleIcon className="w-4 h-4" />,
        onClick: () => console.log('Create new incident'),
        color: 'bg-red-600 hover:bg-red-700'
      });
    }
    
    if (currentView === 'playbooks') {
      actions.push({
        label: 'New Playbook',
        icon: <DocumentTextIcon className="w-4 h-4" />,
        onClick: () => console.log('Create new playbook'),
        color: 'bg-green-600 hover:bg-green-700'
      });
    }
    
    actions.push({
      label: 'Quick Analysis',
      icon: <ChartBarIcon className="w-4 h-4" />,
      onClick: () => console.log('Start quick analysis'),
      color: 'bg-blue-600 hover:bg-blue-700'
    });

    return actions;
  }, [currentView]);

  // Render breadcrumbs
  const renderBreadcrumbs = () => (
    <nav className="flex items-center space-x-2 text-sm">
      {breadcrumbs.map((item, index) => (
        <React.Fragment key={index}>
          {index > 0 && (
            <ChevronRightIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
          )}
          <div
            className={`flex items-center space-x-1 ${
              item.onClick
                ? 'text-primary-600 hover:text-primary-800 cursor-pointer'
                : index === breadcrumbs.length - 1
                ? 'text-gray-900 font-medium'
                : 'text-gray-500'
            }`}
            onClick={item.onClick}
          >
            {item.icon}
            <span className="truncate">{item.label}</span>
          </div>
        </React.Fragment>
      ))}
    </nav>
  );

  // Render notification dropdown
  const renderNotificationDropdown = () => {
    if (!showNotificationDropdown) return null;

    return (
      <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
        <div className="p-3 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-900">Notifications</h3>
            {unreadNotificationCount > 0 && (
              <span className="bg-red-100 text-red-800 text-xs font-medium px-2 py-1 rounded-full">
                {unreadNotificationCount} new
              </span>
            )}
          </div>
        </div>
        
        <div className="max-h-80 overflow-y-auto">
          {recentNotifications.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              <BellIcon className="mx-auto h-8 w-8 text-gray-300 mb-2" />
              <p className="text-sm">No notifications</p>
            </div>
          ) : (
            recentNotifications.map((notification, index) => (
              <div
                key={notification.id || index}
                className="p-3 border-b border-gray-100 last:border-b-0 hover:bg-gray-50"
              >
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-0.5">
                    {getNotificationIcon(notification)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {notification.title || 'Notification'}
                    </p>
                    <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                      {notification.message}
                    </p>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs text-gray-500">
                        {formatNotificationTime(notification.timestamp)}
                      </span>
                      {notification.data?.status && (
                        <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded">
                          {notification.data.status}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => onNotificationRemove(notification.id)}
                    className="flex-shrink-0 text-gray-400 hover:text-gray-600"
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
        
        {notifications.length > 5 && (
          <div className="p-3 border-t border-gray-200">
            <button className="w-full text-center text-sm text-primary-600 hover:text-primary-800 font-medium">
              View all notifications
            </button>
          </div>
        )}
      </div>
    );
  };

  // Render user menu dropdown
  const renderUserMenu = () => {
    if (!showUserMenu) return null;

    return (
      <div className="absolute right-0 top-full mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
        <div className="p-3 border-b border-gray-200">
          <p className="text-sm font-medium text-gray-900">{user?.username}</p>
          <p className="text-xs text-gray-500 capitalize">{user?.role}</p>
        </div>
        
        <div className="py-1">
          <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2">
            <UserIcon className="w-4 h-4" />
            <span>Profile</span>
          </button>
          <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2">
            <Cog6ToothIcon className="w-4 h-4" />
            <span>Settings</span>
          </button>
          <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2">
            <CommandLineIcon className="w-4 h-4" />
            <span>API Keys</span>
          </button>
        </div>
        
        <div className="border-t border-gray-200 py-1">
          <button className="w-full text-left px-3 py-2 text-sm text-red-600 hover:bg-red-50">
            Sign out
          </button>
        </div>
      </div>
    );
  };

  // Render quick actions dropdown
  const renderQuickActions = () => {
    if (!showQuickActions) return null;

    return (
      <div className="absolute right-0 top-full mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
        <div className="p-2">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide px-2 py-1">
            Quick Actions
          </h3>
          {quickActions.map((action, index) => (
            <button
              key={index}
              onClick={action.onClick}
              className={`w-full text-left px-3 py-2 text-sm text-white rounded-md mb-1 last:mb-0 flex items-center space-x-2 transition-colors ${action.color}`}
            >
              {action.icon}
              <span>{action.label}</span>
            </button>
          ))}
        </div>
      </div>
    );
  };

  return (
    <header className={`bg-white border-b border-gray-200 ${className}`}>
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left Section: Breadcrumbs and Title */}
          <div className="flex-1 min-w-0">
            {renderBreadcrumbs()}
            <div className="mt-1 flex items-center space-x-3">
              <h1 className="text-2xl font-bold text-gray-900 truncate">
                {pageTitle}
              </h1>
              
              {/* Status Indicators */}
              <div className="flex items-center space-x-2">
                {isAnalyzing && (
                  <div className="flex items-center space-x-1 bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">
                    <ArrowPathIcon className="w-3 h-3 animate-spin" />
                    <span>Analyzing</span>
                  </div>
                )}
                
                {selectedIncident && (
                  <div className="flex items-center space-x-1">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: selectedIncident.severity === 'critical' ? '#ef4444' :
                                       selectedIncident.severity === 'high' ? '#f59e0b' :
                                       selectedIncident.severity === 'medium' ? '#3b82f6' : '#22c55e'
                      }}
                    />
                    <span className="text-sm text-gray-600 capitalize">
                      {selectedIncident.severity} • {selectedIncident.status.replace('_', ' ')}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Section: Actions and User Info */}
          <div className="flex items-center space-x-4">
            {/* Search */}
            <div className="hidden md:block">
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search incidents, playbooks..."
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm w-64"
                />
              </div>
            </div>

            {/* Quick Actions */}
            <div className="relative">
              <button
                onClick={() => setShowQuickActions(!showQuickActions)}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
                title="Quick Actions"
              >
                <CommandLineIcon className="w-5 h-5" />
              </button>
              {renderQuickActions()}
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setShowNotificationDropdown(!showNotificationDropdown)}
                className="relative p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
                title="Notifications"
              >
                {unreadNotificationCount > 0 ? (
                  <BellIconSolid className="w-5 h-5 text-red-500" />
                ) : (
                  <BellIcon className="w-5 h-5" />
                )}
                {unreadNotificationCount > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 bg-red-500 text-white text-xs rounded-full px-1.5 py-0.5 min-w-[18px] text-center">
                    {unreadNotificationCount > 99 ? '99+' : unreadNotificationCount}
                  </span>
                )}
              </button>
              {renderNotificationDropdown()}
            </div>

            {/* Connection Status */}
            <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full text-sm font-medium ${
              isConnected 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              } ${isConnected ? 'animate-pulse' : ''}`} />
              <span className="hidden sm:inline">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-2 p-2 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                  <span className="text-white font-medium text-sm">
                    {user?.username?.charAt(0).toUpperCase() || 'U'}
                  </span>
                </div>
                <div className="hidden lg:block text-left">
                  <p className="text-sm font-medium text-gray-900">{user?.username}</p>
                  <p className="text-xs text-gray-500 capitalize">{user?.role}</p>
                </div>
              </button>
              {renderUserMenu()}
            </div>
          </div>
        </div>
      </div>

      {/* Close dropdowns when clicking outside */}
      {(showNotificationDropdown || showUserMenu || showQuickActions) && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => {
            setShowNotificationDropdown(false);
            setShowUserMenu(false);
            setShowQuickActions(false);
          }}
        />
      )}
    </header>
  );
};

export default Header;