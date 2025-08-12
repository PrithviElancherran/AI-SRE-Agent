/**
 * React TypeScript component for displaying AI confidence scores.
 * 
 * This component provides visual meter, color-coded levels, and detailed breakdowns
 * for analysis results in the AI SRE Agent frontend.
 */

import React, { useMemo } from 'react';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { ConfidenceLevel, CONFIDENCE_LEVEL_COLORS, CONFIDENCE_LEVEL_THRESHOLDS } from '../types/analysis';

interface ConfidenceMeterProps {
  confidenceScore: number;
  analysisType?: 'incident_analysis' | 'playbook_execution' | 'root_cause' | 'pattern_detection';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  showBreakdown?: boolean;
  showIcon?: boolean;
  className?: string;
  factors?: Array<{
    name: string;
    score: number;
    weight: number;
    description?: string;
  }>;
  qualityIndicators?: Record<string, number>;
  recommendations?: string[];
}

const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  confidenceScore,
  analysisType = 'incident_analysis',
  size = 'md',
  showLabel = true,
  showBreakdown = false,
  showIcon = true,
  className = '',
  factors = [],
  qualityIndicators = {},
  recommendations = []
}) => {
  // Determine confidence level based on score
  const confidenceLevel = useMemo((): ConfidenceLevel => {
    for (const [level, [min, max]] of Object.entries(CONFIDENCE_LEVEL_THRESHOLDS)) {
      if (confidenceScore >= min && confidenceScore <= max) {
        return level as ConfidenceLevel;
      }
    }
    return ConfidenceLevel.MEDIUM;
  }, [confidenceScore]);

  // Get display properties based on confidence level
  const displayProps = useMemo(() => {
    const color = CONFIDENCE_LEVEL_COLORS[confidenceLevel];
    const percentage = Math.round(confidenceScore * 100);
    
    const icons = {
      [ConfidenceLevel.VERY_HIGH]: <CheckCircleIcon className="w-full h-full" />,
      [ConfidenceLevel.HIGH]: <CheckCircleIcon className="w-full h-full" />,
      [ConfidenceLevel.MEDIUM]: <InformationCircleIcon className="w-full h-full" />,
      [ConfidenceLevel.LOW]: <ExclamationTriangleIcon className="w-full h-full" />,
      [ConfidenceLevel.VERY_LOW]: <XCircleIcon className="w-full h-full" />
    };

    const labels = {
      [ConfidenceLevel.VERY_HIGH]: 'Very High',
      [ConfidenceLevel.HIGH]: 'High',
      [ConfidenceLevel.MEDIUM]: 'Medium',
      [ConfidenceLevel.LOW]: 'Low',
      [ConfidenceLevel.VERY_LOW]: 'Very Low'
    };

    const descriptions = {
      [ConfidenceLevel.VERY_HIGH]: 'Highly confident in analysis results',
      [ConfidenceLevel.HIGH]: 'Good confidence in analysis results',
      [ConfidenceLevel.MEDIUM]: 'Moderate confidence, consider additional validation',
      [ConfidenceLevel.LOW]: 'Low confidence, manual review recommended',
      [ConfidenceLevel.VERY_LOW]: 'Very low confidence, results may be unreliable'
    };

    return {
      color,
      percentage,
      icon: icons[confidenceLevel],
      label: labels[confidenceLevel],
      description: descriptions[confidenceLevel]
    };
  }, [confidenceLevel, confidenceScore]);

  // Size configurations
  const sizeConfig = useMemo(() => {
    const configs = {
      sm: {
        container: 'w-16 h-4',
        icon: 'w-3 h-3',
        text: 'text-xs',
        meter: 'h-2',
        badge: 'text-xs px-1.5 py-0.5'
      },
      md: {
        container: 'w-24 h-6',
        icon: 'w-4 h-4',
        text: 'text-sm',
        meter: 'h-3',
        badge: 'text-sm px-2 py-1'
      },
      lg: {
        container: 'w-32 h-8',
        icon: 'w-5 h-5',
        text: 'text-base',
        meter: 'h-4',
        badge: 'text-base px-3 py-1.5'
      }
    };
    return configs[size];
  }, [size]);

  // Analysis type specific labels
  const analysisTypeLabels = {
    incident_analysis: 'Analysis Confidence',
    playbook_execution: 'Execution Confidence',
    root_cause: 'Root Cause Confidence',
    pattern_detection: 'Pattern Confidence'
  };

  // Render confidence meter bar
  const renderMeter = () => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className={`font-medium text-gray-700 ${sizeConfig.text}`}>
          {analysisTypeLabels[analysisType]}
        </span>
        <span className={`font-bold ${sizeConfig.text}`} style={{ color: displayProps.color }}>
          {displayProps.percentage}%
        </span>
      </div>
      
      <div className="relative">
        <div className={`w-full bg-gray-200 rounded-full ${sizeConfig.meter}`}>
          <div
            className={`rounded-full transition-all duration-500 ease-out ${sizeConfig.meter}`}
            style={{
              width: `${displayProps.percentage}%`,
              backgroundColor: displayProps.color
            }}
          />
        </div>
        
        {/* Threshold markers */}
        <div className="absolute top-0 w-full h-full">
          {[30, 50, 70, 85].map((threshold) => (
            <div
              key={threshold}
              className="absolute top-0 w-0.5 h-full bg-gray-400 opacity-50"
              style={{ left: `${threshold}%` }}
            />
          ))}
        </div>
      </div>
    </div>
  );

  // Render confidence badge
  const renderBadge = () => (
    <div className="flex items-center space-x-1">
      {showIcon && (
        <div className={`${sizeConfig.icon}`} style={{ color: displayProps.color }}>
          {displayProps.icon}
        </div>
      )}
      <span
        className={`font-medium rounded-full ${sizeConfig.badge}`}
        style={{
          backgroundColor: `${displayProps.color}20`,
          color: displayProps.color
        }}
      >
        {displayProps.percentage}%
      </span>
      {showLabel && size !== 'sm' && (
        <span className={`text-gray-600 ${sizeConfig.text}`}>
          {displayProps.label}
        </span>
      )}
    </div>
  );

  // Render detailed breakdown
  const renderBreakdown = () => {
    if (!showBreakdown) return null;

    return (
      <div className="mt-4 space-y-4">
        {/* Overall Assessment */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <ChartBarIcon className="w-4 h-4 text-gray-600" />
            <span className="font-medium text-gray-900">Assessment</span>
          </div>
          <p className="text-sm text-gray-700">{displayProps.description}</p>
        </div>

        {/* Confidence Factors */}
        {factors.length > 0 && (
          <div className="p-3 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Confidence Factors</h4>
            <div className="space-y-2">
              {factors.map((factor, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-gray-700">{factor.name}</span>
                      <span className="text-sm text-gray-600">
                        {Math.round(factor.score * 100)}% (weight: {Math.round(factor.weight * 100)}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div
                        className="bg-primary-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${factor.score * 100}%` }}
                      />
                    </div>
                    {factor.description && (
                      <p className="text-xs text-gray-500 mt-1">{factor.description}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quality Indicators */}
        {Object.keys(qualityIndicators).length > 0 && (
          <div className="p-3 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Quality Indicators</h4>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(qualityIndicators).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-sm text-gray-600 capitalize">
                    {key.replace(/_/g, ' ')}:
                  </span>
                  <span className="text-sm font-medium text-gray-900">
                    {typeof value === 'number' ? `${Math.round(value * 100)}%` : value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
            <h4 className="font-medium text-blue-900 mb-2">Recommendations</h4>
            <ul className="space-y-1">
              {recommendations.map((recommendation, index) => (
                <li key={index} className="text-sm text-blue-800 flex items-start">
                  <span className="mr-2">•</span>
                  <span>{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Confidence Scale Reference */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-3">Confidence Scale</h4>
          <div className="space-y-2">
            {Object.entries(CONFIDENCE_LEVEL_THRESHOLDS).map(([level, [min, max]]) => {
              const color = CONFIDENCE_LEVEL_COLORS[level as ConfidenceLevel];
              const isCurrentLevel = level === confidenceLevel;
              
              return (
                <div
                  key={level}
                  className={`flex items-center justify-between p-2 rounded ${
                    isCurrentLevel ? 'bg-white border-2' : 'bg-transparent'
                  }`}
                  style={isCurrentLevel ? { borderColor: color } : {}}
                >
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className={`text-sm capitalize ${isCurrentLevel ? 'font-medium' : ''}`}>
                      {level.replace('_', ' ')}
                    </span>
                  </div>
                  <span className="text-sm text-gray-600">
                    {Math.round(min * 100)}% - {Math.round(max * 100)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  // Handle invalid confidence scores
  if (confidenceScore < 0 || confidenceScore > 1 || isNaN(confidenceScore)) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <XCircleIcon className={`${sizeConfig.icon} text-error-500`} />
        <span className={`text-error-600 ${sizeConfig.text}`}>Invalid Score</span>
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      {size === 'sm' ? renderBadge() : renderMeter()}
      {showLabel && size === 'sm' && (
        <div className="mt-1 text-xs text-gray-500 text-center">
          {displayProps.label}
        </div>
      )}
      {renderBreakdown()}
    </div>
  );
};

export default ConfidenceMeter;