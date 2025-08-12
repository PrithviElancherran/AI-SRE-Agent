"""
Multi-factor confidence scoring system that calculates confidence scores for AI analysis results.

This service provides confidence scoring capabilities for the AI SRE Agent, including:
- Historical accuracy-based confidence calculation
- Evidence strength assessment
- Data quality evaluation
- Pattern matching confidence
- Similarity correlation scoring
- Multi-factor confidence aggregation
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from loguru import logger
from collections import Counter
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func, case

from models.incident import Incident, IncidentSymptom, IncidentSeverity, IncidentStatus
from models.analysis import (
    AnalysisResult, AnalysisType, EvidenceItem, EvidenceType,
    ConfidenceFactor, ConfidenceScore, ReasoningStep
)
from models.playbook import PlaybookExecution, PlaybookStepResult, ExecutionStatus
from config.database import get_database
from config.settings import get_settings
from utils.formatters import format_confidence_score, format_percentage

settings = get_settings()


@dataclass
class ConfidenceFactorWeights:
    """Weights for different confidence factors."""
    
    historical_accuracy: float = 0.25
    evidence_strength: float = 0.20
    data_quality: float = 0.15
    pattern_matching: float = 0.15
    similarity_correlation: float = 0.10
    model_confidence: float = 0.10
    validation_score: float = 0.05


class ConfidenceScorer:
    """Multi-factor confidence scoring system for AI analysis results."""
    
    def __init__(self):
        """Initialize the confidence scorer."""
        self.factor_weights = ConfidenceFactorWeights()
        self.base_confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.historical_accuracy_cache = {}
        self.evidence_quality_cache = {}
        
        # Confidence scoring models and thresholds
        self.quality_thresholds = {
            "data_completeness": 0.8,
            "evidence_relevance": 0.7,
            "pattern_strength": 0.6,
            "similarity_threshold": 0.7,
            "validation_threshold": 0.8
        }
        
        logger.info("ConfidenceScorer initialized with multi-factor scoring")
    
    async def calculate_analysis_confidence(
        self,
        incident: Incident,
        similar_incidents: List[Incident],
        root_cause: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        analysis_type: AnalysisType = AnalysisType.CORRELATION,
        model_confidence: Optional[float] = None
    ) -> ConfidenceScore:
        """
        Calculate overall confidence score for incident analysis.
        
        Args:
            incident: Current incident being analyzed
            similar_incidents: List of similar historical incidents
            root_cause: Root cause analysis results
            evidence_items: Supporting evidence collected
            analysis_type: Type of analysis performed
            model_confidence: ML model confidence score
            
        Returns:
            ConfidenceScore with detailed factor breakdown
        """
        logger.info(f"Calculating confidence for incident {incident.incident_id}")
        
        try:
            confidence_factors = []
            
            # Factor 1: Historical Accuracy
            historical_factor = await self._calculate_historical_accuracy_factor(
                incident, similar_incidents, root_cause, analysis_type
            )
            confidence_factors.append(historical_factor)
            
            # Factor 2: Evidence Strength
            evidence_factor = await self._calculate_evidence_strength_factor(
                evidence_items, incident, root_cause
            )
            confidence_factors.append(evidence_factor)
            
            # Factor 3: Data Quality
            data_quality_factor = await self._calculate_data_quality_factor(
                incident, evidence_items, similar_incidents
            )
            confidence_factors.append(data_quality_factor)
            
            # Factor 4: Pattern Matching
            pattern_factor = await self._calculate_pattern_matching_factor(
                incident, similar_incidents, root_cause
            )
            confidence_factors.append(pattern_factor)
            
            # Factor 5: Similarity Correlation
            similarity_factor = await self._calculate_similarity_correlation_factor(
                incident, similar_incidents
            )
            confidence_factors.append(similarity_factor)
            
            # Factor 6: Model Confidence
            model_factor = await self._calculate_model_confidence_factor(
                model_confidence, analysis_type, incident
            )
            confidence_factors.append(model_factor)
            
            # Factor 7: Validation Score
            validation_factor = await self._calculate_validation_factor(
                incident, root_cause, evidence_items
            )
            confidence_factors.append(validation_factor)
            
            # Calculate overall confidence score
            confidence_score = ConfidenceScore(
                overall_score=0.0,
                factors=confidence_factors,
                calculation_method="weighted_multi_factor",
                quality_indicators=await self._calculate_quality_indicators(
                    incident, evidence_items, similar_incidents
                )
            )
            
            # Recalculate with proper weights
            confidence_score.recalculate_score()
            
            # Apply confidence adjustments
            confidence_score = await self._apply_confidence_adjustments(
                confidence_score, incident, analysis_type
            )
            
            # Set reliability assessment
            confidence_score.reliability_assessment = self._assess_reliability(confidence_score)
            
            logger.info(f"Calculated confidence score: {confidence_score.overall_score:.3f}")
            return confidence_score
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            
            # Return default low confidence score
            return ConfidenceScore(
                overall_score=0.3,
                factors=[
                    ConfidenceFactor(
                        factor_name="error_fallback",
                        factor_type="system_error",
                        weight=1.0,
                        score=0.3,
                        contribution=0.3,
                        explanation=f"Error in confidence calculation: {str(e)}"
                    )
                ],
                reliability_assessment="low"
            )
    
    async def calculate_playbook_confidence(
        self,
        playbook_execution: PlaybookExecution,
        step_results: List[PlaybookStepResult],
        incident: Incident
    ) -> ConfidenceScore:
        """
        Calculate confidence score for playbook execution results.
        
        Args:
            playbook_execution: Playbook execution details
            step_results: Results of individual playbook steps
            incident: Incident being diagnosed
            
        Returns:
            ConfidenceScore for playbook execution
        """
        logger.info(f"Calculating playbook confidence for execution {playbook_execution.execution_id}")
        
        try:
            confidence_factors = []
            
            # Factor 1: Step Success Rate
            success_factor = await self._calculate_step_success_factor(step_results)
            confidence_factors.append(success_factor)
            
            # Factor 2: Evidence Collection Quality
            evidence_factor = await self._calculate_playbook_evidence_factor(step_results)
            confidence_factors.append(evidence_factor)
            
            # Factor 3: Threshold Compliance
            threshold_factor = await self._calculate_threshold_compliance_factor(step_results)
            confidence_factors.append(threshold_factor)
            
            # Factor 4: Execution Consistency
            consistency_factor = await self._calculate_execution_consistency_factor(
                step_results, playbook_execution
            )
            confidence_factors.append(consistency_factor)
            
            # Factor 5: Root Cause Identification
            root_cause_factor = await self._calculate_root_cause_identification_factor(
                playbook_execution, step_results
            )
            confidence_factors.append(root_cause_factor)
            
            # Calculate overall confidence
            confidence_score = ConfidenceScore(
                overall_score=0.0,
                factors=confidence_factors,
                calculation_method="playbook_weighted_average",
                quality_indicators={
                    "steps_executed": len(step_results),
                    "success_rate": len([r for r in step_results if r.is_successful()]) / max(1, len(step_results)),
                    "escalations_triggered": len([r for r in step_results if r.escalation_triggered]),
                    "evidence_collected": len([r for r in step_results if r.evidence])
                }
            )
            
            confidence_score.recalculate_score()
            confidence_score.reliability_assessment = self._assess_reliability(confidence_score)
            
            return confidence_score
            
        except Exception as e:
            logger.error(f"Error calculating playbook confidence: {e}")
            return ConfidenceScore(overall_score=0.4, factors=[], reliability_assessment="low")
    
    async def _calculate_historical_accuracy_factor(
        self,
        incident: Incident,
        similar_incidents: List[Incident],
        root_cause: Dict[str, Any],
        analysis_type: AnalysisType
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on historical accuracy."""
        try:
            if not similar_incidents:
                return ConfidenceFactor(
                    factor_name="historical_accuracy",
                    factor_type="historical_data",
                    weight=self.factor_weights.historical_accuracy,
                    score=0.5,
                    contribution=0.5 * self.factor_weights.historical_accuracy,
                    explanation="No historical incidents available for comparison"
                )
            
            # Calculate accuracy based on similar incidents
            primary_cause = root_cause.get('primary_cause', '').lower()
            matching_causes = 0
            total_similar = len(similar_incidents)
            
            for similar_incident in similar_incidents:
                if similar_incident.root_cause:
                    similar_cause = similar_incident.root_cause.lower()
                    # Enhanced keyword matching for cause similarity
                    cause_words = set(word for word in primary_cause.split() if len(word) > 3)
                    similar_words = set(word for word in similar_cause.split() if len(word) > 3)
                    
                    # Calculate word overlap
                    word_overlap = len(cause_words.intersection(similar_words))
                    total_words = len(cause_words.union(similar_words))
                    
                    if total_words > 0:
                        similarity_ratio = word_overlap / total_words
                        if similarity_ratio > 0.3:  # At least 30% word overlap
                            matching_causes += similarity_ratio
            
            # Calculate accuracy score with enhanced precision
            if total_similar > 0:
                accuracy_score = matching_causes / total_similar
                
                # Root cause specific confidence boosts
                if 'database' in primary_cause or 'connection' in primary_cause:
                    accuracy_score += 0.1  # Database issues are well-documented
                elif 'cache' in primary_cause or 'redis' in primary_cause:
                    accuracy_score += 0.08  # Cache issues are common and well-understood
                elif 'latency' in primary_cause or 'performance' in primary_cause:
                    accuracy_score += 0.06  # Performance issues are frequent
                
                # Boost confidence if we have more historical data
                data_boost = min(0.2, total_similar * 0.02)  # Up to 20% boost for 10+ incidents
                accuracy_score = min(1.0, accuracy_score + data_boost)
            else:
                accuracy_score = 0.5
            
            # Get historical performance for this analysis type
            historical_performance = await self._get_historical_analysis_performance(analysis_type)
            
            # Combine with historical performance
            final_score = (accuracy_score * 0.7) + (historical_performance * 0.3)
            
            return ConfidenceFactor(
                factor_name="historical_accuracy",
                factor_type="historical_data",
                weight=self.factor_weights.historical_accuracy,
                score=final_score,
                contribution=final_score * self.factor_weights.historical_accuracy,
                explanation=f"Based on {matching_causes}/{total_similar} matching historical root causes"
            )
            
        except Exception as e:
            logger.error(f"Error calculating historical accuracy factor: {e}")
            return ConfidenceFactor(
                factor_name="historical_accuracy",
                factor_type="historical_data",
                weight=self.factor_weights.historical_accuracy,
                score=0.3,
                contribution=0.3 * self.factor_weights.historical_accuracy,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_evidence_strength_factor(
        self,
        evidence_items: List[EvidenceItem],
        incident: Incident,
        root_cause: Dict[str, Any]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on evidence strength."""
        try:
            if not evidence_items:
                return ConfidenceFactor(
                    factor_name="evidence_strength",
                    factor_type="evidence_quality",
                    weight=self.factor_weights.evidence_strength,
                    score=0.2,
                    contribution=0.2 * self.factor_weights.evidence_strength,
                    explanation="No evidence items collected"
                )
            
            # Calculate evidence quality metrics
            total_relevance = sum(item.relevance_score for item in evidence_items)
            avg_relevance = total_relevance / len(evidence_items)
            
            # Evidence diversity score (different types of evidence)
            evidence_types = set(item.evidence_type for item in evidence_items)
            diversity_score = min(1.0, len(evidence_types) / 4)  # Optimal: 4 different types
            
            # Evidence recency score
            recent_evidence = [
                item for item in evidence_items 
                if (datetime.utcnow() - item.timestamp).total_seconds() < 3600  # Within 1 hour
            ]
            recency_score = len(recent_evidence) / len(evidence_items)
            
            # Evidence completeness (presence of key evidence types)
            key_evidence_types = {
                EvidenceType.GCP_MONITORING,
                EvidenceType.GCP_LOGGING,
                EvidenceType.HISTORICAL_CORRELATION
            }
            present_key_types = evidence_types.intersection(key_evidence_types)
            completeness_score = len(present_key_types) / len(key_evidence_types)
            
            # Combine evidence strength factors
            evidence_score = (
                avg_relevance * 0.4 +
                diversity_score * 0.25 +
                recency_score * 0.2 +
                completeness_score * 0.15
            )
            
            return ConfidenceFactor(
                factor_name="evidence_strength",
                factor_type="evidence_quality",
                weight=self.factor_weights.evidence_strength,
                score=evidence_score,
                contribution=evidence_score * self.factor_weights.evidence_strength,
                explanation=f"Based on {len(evidence_items)} evidence items with {avg_relevance:.2f} avg relevance"
            )
            
        except Exception as e:
            logger.error(f"Error calculating evidence strength factor: {e}")
            return ConfidenceFactor(
                factor_name="evidence_strength",
                factor_type="evidence_quality",
                weight=self.factor_weights.evidence_strength,
                score=0.3,
                contribution=0.3 * self.factor_weights.evidence_strength,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_data_quality_factor(
        self,
        incident: Incident,
        evidence_items: List[EvidenceItem],
        similar_incidents: List[Incident]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on data quality."""
        try:
            quality_scores = []
            quality_explanations = []
            
            # Incident data completeness
            incident_completeness = 0.0
            if incident.title:
                incident_completeness += 0.2
            if incident.description:
                incident_completeness += 0.2
            if incident.symptoms:
                incident_completeness += 0.2
            if incident.incident_symptoms:
                incident_completeness += 0.2
            if incident.service_name and incident.region:
                incident_completeness += 0.2
            
            quality_scores.append(incident_completeness)
            quality_explanations.append(f"Incident data {incident_completeness*100:.0f}% complete")
            
            # Evidence data quality
            evidence_quality = 0.8  # Default good quality
            if evidence_items:
                evidence_with_data = [item for item in evidence_items if item.data]
                evidence_quality = len(evidence_with_data) / len(evidence_items)
            else:
                evidence_quality = 0.3
            
            quality_scores.append(evidence_quality)
            quality_explanations.append(f"Evidence quality: {evidence_quality*100:.0f}%")
            
            # Historical data availability
            historical_quality = min(1.0, len(similar_incidents) / 5)  # Optimal: 5+ similar incidents
            quality_scores.append(historical_quality)
            quality_explanations.append(f"Historical data: {len(similar_incidents)} similar incidents")
            
            # Symptom data quality
            symptom_quality = 0.5
            if incident.incident_symptoms:
                detailed_symptoms = [
                    s for s in incident.incident_symptoms 
                    if s.metric_data and s.severity_score > 0
                ]
                symptom_quality = len(detailed_symptoms) / len(incident.incident_symptoms)
                quality_explanations.append(f"Symptom data: {len(detailed_symptoms)}/{len(incident.incident_symptoms)} detailed")
            else:
                quality_explanations.append("No detailed symptom data available")
            
            quality_scores.append(symptom_quality)
            
            # Calculate overall data quality
            overall_quality = sum(quality_scores) / len(quality_scores)
            
            return ConfidenceFactor(
                factor_name="data_quality",
                factor_type="data_completeness",
                weight=self.factor_weights.data_quality,
                score=overall_quality,
                contribution=overall_quality * self.factor_weights.data_quality,
                explanation="; ".join(quality_explanations)
            )
            
        except Exception as e:
            logger.error(f"Error calculating data quality factor: {e}")
            return ConfidenceFactor(
                factor_name="data_quality",
                factor_type="data_completeness",
                weight=self.factor_weights.data_quality,
                score=0.4,
                contribution=0.4 * self.factor_weights.data_quality,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_pattern_matching_factor(
        self,
        incident: Incident,
        similar_incidents: List[Incident],
        root_cause: Dict[str, Any]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on pattern matching."""
        try:
            if not similar_incidents:
                return ConfidenceFactor(
                    factor_name="pattern_matching",
                    factor_type="pattern_analysis",
                    weight=self.factor_weights.pattern_matching,
                    score=0.3,
                    contribution=0.3 * self.factor_weights.pattern_matching,
                    explanation="No similar incidents for pattern analysis"
                )
            
            pattern_scores = []
            pattern_explanations = []
            
            # Service pattern matching
            current_service = incident.service_name
            similar_services = [inc.service_name for inc in similar_incidents]
            service_matches = similar_services.count(current_service)
            service_pattern_score = service_matches / len(similar_incidents)
            pattern_scores.append(service_pattern_score)
            pattern_explanations.append(f"{service_matches}/{len(similar_incidents)} service matches")
            
            # Severity pattern matching
            current_severity = incident.severity
            similar_severities = [inc.severity for inc in similar_incidents]
            severity_matches = similar_severities.count(current_severity)
            severity_pattern_score = severity_matches / len(similar_incidents)
            pattern_scores.append(severity_pattern_score)
            pattern_explanations.append(f"{severity_matches}/{len(similar_incidents)} severity matches")
            
            # Region pattern matching
            current_region = incident.region
            similar_regions = [inc.region for inc in similar_incidents]
            region_matches = similar_regions.count(current_region)
            region_pattern_score = region_matches / len(similar_incidents)
            pattern_scores.append(region_pattern_score)
            pattern_explanations.append(f"{region_matches}/{len(similar_incidents)} region matches")
            
            # Symptom pattern matching
            current_symptoms = set(incident.symptoms)
            symptom_overlaps = []
            for similar_incident in similar_incidents:
                similar_symptoms = set(similar_incident.symptoms)
                if current_symptoms and similar_symptoms:
                    overlap = len(current_symptoms.intersection(similar_symptoms))
                    total = len(current_symptoms.union(similar_symptoms))
                    symptom_overlaps.append(overlap / total if total > 0 else 0)
            
            symptom_pattern_score = sum(symptom_overlaps) / len(symptom_overlaps) if symptom_overlaps else 0
            pattern_scores.append(symptom_pattern_score)
            pattern_explanations.append(f"Avg symptom overlap: {symptom_pattern_score:.2f}")
            
            # Root cause pattern matching
            primary_cause = root_cause.get('primary_cause', '').lower()
            cause_matches = 0
            for similar_incident in similar_incidents:
                if similar_incident.root_cause:
                    similar_cause = similar_incident.root_cause.lower()
                    # Check for keyword matches
                    if any(word in similar_cause for word in primary_cause.split() if len(word) > 3):
                        cause_matches += 1
            
            cause_pattern_score = cause_matches / len(similar_incidents)
            pattern_scores.append(cause_pattern_score)
            pattern_explanations.append(f"{cause_matches}/{len(similar_incidents)} root cause matches")
            
            # Calculate overall pattern strength
            overall_pattern_score = sum(pattern_scores) / len(pattern_scores)
            
            return ConfidenceFactor(
                factor_name="pattern_matching",
                factor_type="pattern_analysis",
                weight=self.factor_weights.pattern_matching,
                score=overall_pattern_score,
                contribution=overall_pattern_score * self.factor_weights.pattern_matching,
                explanation="; ".join(pattern_explanations)
            )
            
        except Exception as e:
            logger.error(f"Error calculating pattern matching factor: {e}")
            return ConfidenceFactor(
                factor_name="pattern_matching",
                factor_type="pattern_analysis",
                weight=self.factor_weights.pattern_matching,
                score=0.3,
                contribution=0.3 * self.factor_weights.pattern_matching,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_similarity_correlation_factor(
        self,
        incident: Incident,
        similar_incidents: List[Incident]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on similarity correlation strength."""
        try:
            if not similar_incidents:
                return ConfidenceFactor(
                    factor_name="similarity_correlation",
                    factor_type="correlation_strength",
                    weight=self.factor_weights.similarity_correlation,
                    score=0.2,
                    contribution=0.2 * self.factor_weights.similarity_correlation,
                    explanation="No similar incidents found for correlation"
                )
            
            # Simulate similarity scores (in real implementation, these would come from vector search)
            # For demo, generate realistic similarity scores based on incident attributes
            similarity_scores = []
            
            for similar_incident in similar_incidents:
                # Calculate synthetic similarity based on attributes
                score = 0.0
                
                # Service similarity
                if incident.service_name == similar_incident.service_name:
                    score += 0.3
                
                # Severity similarity
                severity_weights = {
                    IncidentSeverity.CRITICAL: 4,
                    IncidentSeverity.HIGH: 3,
                    IncidentSeverity.MEDIUM: 2,
                    IncidentSeverity.LOW: 1
                }
                current_weight = severity_weights.get(incident.severity, 2)
                similar_weight = severity_weights.get(similar_incident.severity, 2)
                severity_sim = 1 - abs(current_weight - similar_weight) / 4
                score += severity_sim * 0.2
                
                # Symptom similarity
                if incident.symptoms and similar_incident.symptoms:
                    current_symptoms = set(incident.symptoms)
                    similar_symptoms = set(similar_incident.symptoms)
                    if current_symptoms and similar_symptoms:
                        overlap = len(current_symptoms.intersection(similar_symptoms))
                        union = len(current_symptoms.union(similar_symptoms))
                        symptom_sim = overlap / union if union > 0 else 0
                        score += symptom_sim * 0.3
                
                # Region similarity
                if incident.region == similar_incident.region:
                    score += 0.2
                
                similarity_scores.append(min(1.0, score))
            
            # Calculate correlation metrics
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            high_similarity_count = len([s for s in similarity_scores if s > 0.7])
            correlation_strength = high_similarity_count / len(similarity_scores)
            
            # Calculate final correlation factor
            correlation_score = (avg_similarity * 0.6) + (correlation_strength * 0.4)
            
            return ConfidenceFactor(
                factor_name="similarity_correlation",
                factor_type="correlation_strength",
                weight=self.factor_weights.similarity_correlation,
                score=correlation_score,
                contribution=correlation_score * self.factor_weights.similarity_correlation,
                explanation=f"Avg similarity: {avg_similarity:.2f}, {high_similarity_count}/{len(similarity_scores)} high similarity"
            )
            
        except Exception as e:
            logger.error(f"Error calculating similarity correlation factor: {e}")
            return ConfidenceFactor(
                factor_name="similarity_correlation",
                factor_type="correlation_strength",
                weight=self.factor_weights.similarity_correlation,
                score=0.3,
                contribution=0.3 * self.factor_weights.similarity_correlation,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_model_confidence_factor(
        self,
        model_confidence: Optional[float],
        analysis_type: AnalysisType,
        incident: Incident
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on ML model confidence."""
        try:
            if model_confidence is not None:
                # Use provided model confidence
                score = model_confidence
                explanation = f"ML model confidence: {model_confidence:.2f}"
            else:
                # Estimate model confidence based on analysis type and incident characteristics
                base_confidence = {
                    AnalysisType.CORRELATION: 0.8,
                    AnalysisType.PLAYBOOK: 0.85,
                    AnalysisType.ML_PREDICTION: 0.75,
                    AnalysisType.PATTERN_ANALYSIS: 0.7,
                    AnalysisType.ANOMALY_DETECTION: 0.65,
                    AnalysisType.ROOT_CAUSE_ANALYSIS: 0.8
                }.get(analysis_type, 0.7)
                
                # Adjust based on incident characteristics
                if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
                    base_confidence += 0.05  # More confident with severe incidents
                
                if len(incident.symptoms) > 3:
                    base_confidence += 0.05  # More confident with more symptoms
                
                if incident.incident_symptoms:
                    base_confidence += 0.1  # More confident with detailed symptoms
                    
                    # Adjust based on specific symptom types and their severity
                    for symptom in incident.incident_symptoms:
                        symptom_confidence_boost = 0.0
                        
                        # Higher confidence for well-understood symptom types
                        if symptom.symptom_type.value in ['connection_timeout', 'cache_performance']:
                            symptom_confidence_boost += 0.08 * symptom.severity_score
                        elif symptom.symptom_type.value in ['latency', 'query_performance']:
                            symptom_confidence_boost += 0.06 * symptom.severity_score
                        elif symptom.symptom_type.value in ['memory_usage', 'cpu_usage']:
                            symptom_confidence_boost += 0.04 * symptom.severity_score
                        else:
                            symptom_confidence_boost += 0.02 * symptom.severity_score
                        
                        base_confidence += symptom_confidence_boost
                    
                    # Service-specific confidence adjustments
                    if incident.service_name == 'payment-api':
                        base_confidence += 0.03  # Payment API incidents are well-documented
                    elif incident.service_name == 'ecommerce-api':
                        base_confidence += 0.02  # E-commerce API incidents are common
                
                score = min(1.0, base_confidence)
                explanation = f"Estimated model confidence for {analysis_type.value}: {score:.2f}"
            
            return ConfidenceFactor(
                factor_name="model_confidence",
                factor_type="ml_model",
                weight=self.factor_weights.model_confidence,
                score=score,
                contribution=score * self.factor_weights.model_confidence,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error calculating model confidence factor: {e}")
            return ConfidenceFactor(
                factor_name="model_confidence",
                factor_type="ml_model",
                weight=self.factor_weights.model_confidence,
                score=0.6,
                contribution=0.6 * self.factor_weights.model_confidence,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_validation_factor(
        self,
        incident: Incident,
        root_cause: Dict[str, Any],
        evidence_items: List[EvidenceItem]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on validation checks."""
        try:
            validation_scores = []
            validation_explanations = []
            
            # Logical consistency validation
            primary_cause = root_cause.get('primary_cause', '').lower()
            
            # Check if root cause is consistent with symptoms
            symptom_consistency = 0.8  # Default good consistency
            if incident.symptoms:
                relevant_symptoms = 0
                for symptom in incident.symptoms:
                    symptom_lower = symptom.lower()
                    # Check for logical connections
                    if ('cache' in primary_cause and 'cache' in symptom_lower) or \
                       ('database' in primary_cause and ('database' in symptom_lower or 'connection' in symptom_lower)) or \
                       ('latency' in primary_cause and 'latency' in symptom_lower) or \
                       ('memory' in primary_cause and 'memory' in symptom_lower):
                        relevant_symptoms += 1
                
                if incident.symptoms:
                    symptom_consistency = relevant_symptoms / len(incident.symptoms)
            
            validation_scores.append(symptom_consistency)
            validation_explanations.append(f"Symptom consistency: {symptom_consistency:.2f}")
            
            # Evidence consistency validation
            evidence_consistency = 0.7  # Default
            if evidence_items:
                relevant_evidence = 0
                for evidence in evidence_items:
                    if evidence.evidence_type in [
                        EvidenceType.GCP_MONITORING,
                        EvidenceType.GCP_LOGGING,
                        EvidenceType.HISTORICAL_CORRELATION
                    ] and evidence.relevance_score > 0.6:
                        relevant_evidence += 1
                
                evidence_consistency = relevant_evidence / len(evidence_items)
            
            validation_scores.append(evidence_consistency)
            validation_explanations.append(f"Evidence consistency: {evidence_consistency:.2f}")
            
            # Temporal validation
            temporal_consistency = 0.8  # Default good temporal consistency
            if incident.incident_symptoms:
                # Check if symptom detection times are reasonable
                symptom_times = [s.detected_at for s in incident.incident_symptoms]
                if len(symptom_times) > 1:
                    time_spans = [
                        abs((t2 - t1).total_seconds()) 
                        for t1, t2 in zip(symptom_times[:-1], symptom_times[1:])
                    ]
                    # Reasonable if symptoms detected within reasonable timeframe
                    reasonable_spans = [span for span in time_spans if span < 3600]  # Within 1 hour
                    temporal_consistency = len(reasonable_spans) / len(time_spans) if time_spans else 0.8
            
            validation_scores.append(temporal_consistency)
            validation_explanations.append(f"Temporal consistency: {temporal_consistency:.2f}")
            
            # Calculate overall validation score
            overall_validation = sum(validation_scores) / len(validation_scores)
            
            return ConfidenceFactor(
                factor_name="validation_score",
                factor_type="logical_validation",
                weight=self.factor_weights.validation_score,
                score=overall_validation,
                contribution=overall_validation * self.factor_weights.validation_score,
                explanation="; ".join(validation_explanations)
            )
            
        except Exception as e:
            logger.error(f"Error calculating validation factor: {e}")
            return ConfidenceFactor(
                factor_name="validation_score",
                factor_type="logical_validation",
                weight=self.factor_weights.validation_score,
                score=0.5,
                contribution=0.5 * self.factor_weights.validation_score,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_step_success_factor(
        self,
        step_results: List[PlaybookStepResult]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on playbook step success rate."""
        try:
            if not step_results:
                return ConfidenceFactor(
                    factor_name="step_success_rate",
                    factor_type="execution_quality",
                    weight=0.3,
                    score=0.0,
                    contribution=0.0,
                    explanation="No step results available"
                )
            
            successful_steps = [r for r in step_results if r.is_successful()]
            success_rate = len(successful_steps) / len(step_results)
            
            # Boost confidence for high success rates
            confidence_score = success_rate
            if success_rate > 0.8:
                confidence_score = min(1.0, success_rate + 0.1)
            
            return ConfidenceFactor(
                factor_name="step_success_rate",
                factor_type="execution_quality",
                weight=0.3,
                score=confidence_score,
                contribution=confidence_score * 0.3,
                explanation=f"Step success rate: {len(successful_steps)}/{len(step_results)} ({success_rate*100:.0f}%)"
            )
            
        except Exception as e:
            logger.error(f"Error calculating step success factor: {e}")
            return ConfidenceFactor(
                factor_name="step_success_rate",
                factor_type="execution_quality",
                weight=0.3,
                score=0.3,
                contribution=0.09,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_playbook_evidence_factor(
        self,
        step_results: List[PlaybookStepResult]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on evidence collected during playbook execution."""
        try:
            if not step_results:
                return ConfidenceFactor(
                    factor_name="evidence_collection",
                    factor_type="evidence_quality",
                    weight=0.25,
                    score=0.0,
                    contribution=0.0,
                    explanation="No step results for evidence analysis"
                )
            
            evidence_steps = [r for r in step_results if r.evidence]
            evidence_rate = len(evidence_steps) / len(step_results)
            
            # Quality of evidence collected
            evidence_quality_scores = []
            for result in evidence_steps:
                if result.evidence:
                    # Simple quality assessment based on evidence content
                    evidence_content = result.evidence
                    quality_score = 0.5  # Base score
                    
                    # Boost for rich evidence
                    if isinstance(evidence_content, dict):
                        if len(evidence_content) > 3:
                            quality_score += 0.2
                        if any(key in evidence_content for key in ['gcp_dashboard_url', 'metric_value', 'error_count']):
                            quality_score += 0.2
                        if 'data_points' in evidence_content:
                            quality_score += 0.1
                    
                    evidence_quality_scores.append(min(1.0, quality_score))
            
            avg_evidence_quality = sum(evidence_quality_scores) / len(evidence_quality_scores) if evidence_quality_scores else 0.5
            
            # Combine evidence rate and quality
            overall_score = (evidence_rate * 0.6) + (avg_evidence_quality * 0.4)
            
            return ConfidenceFactor(
                factor_name="evidence_collection",
                factor_type="evidence_quality",
                weight=0.25,
                score=overall_score,
                contribution=overall_score * 0.25,
                explanation=f"Evidence collected in {len(evidence_steps)}/{len(step_results)} steps, avg quality: {avg_evidence_quality:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating playbook evidence factor: {e}")
            return ConfidenceFactor(
                factor_name="evidence_collection",
                factor_type="evidence_quality",
                weight=0.25,
                score=0.3,
                contribution=0.075,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_threshold_compliance_factor(
        self,
        step_results: List[PlaybookStepResult]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on threshold compliance."""
        try:
            threshold_results = [r for r in step_results if r.threshold_met is not None]
            
            if not threshold_results:
                return ConfidenceFactor(
                    factor_name="threshold_compliance",
                    factor_type="diagnostic_accuracy",
                    weight=0.2,
                    score=0.5,
                    contribution=0.1,
                    explanation="No threshold data available"
                )
            
            compliant_steps = [r for r in threshold_results if r.threshold_met]
            compliance_rate = len(compliant_steps) / len(threshold_results)
            
            # Adjust score based on compliance patterns
            confidence_score = compliance_rate
            
            # Penalty for mixed results (indicates uncertainty)
            if 0.3 < compliance_rate < 0.7:
                confidence_score *= 0.8  # Reduce confidence for mixed results
            
            return ConfidenceFactor(
                factor_name="threshold_compliance",
                factor_type="diagnostic_accuracy",
                weight=0.2,
                score=confidence_score,
                contribution=confidence_score * 0.2,
                explanation=f"Threshold compliance: {len(compliant_steps)}/{len(threshold_results)} steps"
            )
            
        except Exception as e:
            logger.error(f"Error calculating threshold compliance factor: {e}")
            return ConfidenceFactor(
                factor_name="threshold_compliance",
                factor_type="diagnostic_accuracy",
                weight=0.2,
                score=0.4,
                contribution=0.08,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_execution_consistency_factor(
        self,
        step_results: List[PlaybookStepResult],
        execution: PlaybookExecution
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on execution consistency."""
        try:
            if not step_results:
                return ConfidenceFactor(
                    factor_name="execution_consistency",
                    factor_type="process_quality",
                    weight=0.15,
                    score=0.0,
                    contribution=0.0,
                    explanation="No step results for consistency analysis"
                )
            
            consistency_scores = []
            
            # Timing consistency (steps completed in reasonable time)
            timed_steps = [r for r in step_results if r.duration_seconds is not None]
            if timed_steps:
                durations = [r.duration_seconds for r in timed_steps]
                avg_duration = sum(durations) / len(durations)
                # Consistent if most steps are within 2x average duration
                consistent_timings = [d for d in durations if d <= avg_duration * 2]
                timing_consistency = len(consistent_timings) / len(durations)
                consistency_scores.append(timing_consistency)
            
            # Error consistency (similar types of errors)
            error_steps = [r for r in step_results if r.error_message]
            if error_steps:
                # Simple error categorization
                error_categories = set()
                for result in error_steps:
                    error_msg = result.error_message.lower()
                    if 'timeout' in error_msg:
                        error_categories.add('timeout')
                    elif 'connection' in error_msg:
                        error_categories.add('connection')
                    elif 'data' in error_msg:
                        error_categories.add('data')
                    else:
                        error_categories.add('other')
                
                # More consistent if errors are of same type
                error_consistency = 1 - (len(error_categories) - 1) / max(1, len(error_steps))
                consistency_scores.append(error_consistency)
            else:
                consistency_scores.append(1.0)  # No errors is perfectly consistent
            
            # Status consistency (avoid flip-flopping between success/failure)
            status_changes = 0
            for i in range(1, len(step_results)):
                prev_success = step_results[i-1].success
                curr_success = step_results[i].success
                if prev_success is not None and curr_success is not None and prev_success != curr_success:
                    status_changes += 1
            
            status_consistency = 1 - (status_changes / max(1, len(step_results) - 1))
            consistency_scores.append(status_consistency)
            
            # Calculate overall consistency
            overall_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
            
            return ConfidenceFactor(
                factor_name="execution_consistency",
                factor_type="process_quality",
                weight=0.15,
                score=overall_consistency,
                contribution=overall_consistency * 0.15,
                explanation=f"Execution consistency across {len(step_results)} steps: {overall_consistency:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating execution consistency factor: {e}")
            return ConfidenceFactor(
                factor_name="execution_consistency",
                factor_type="process_quality",
                weight=0.15,
                score=0.5,
                contribution=0.075,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _calculate_root_cause_identification_factor(
        self,
        execution: PlaybookExecution,
        step_results: List[PlaybookStepResult]
    ) -> ConfidenceFactor:
        """Calculate confidence factor based on root cause identification."""
        try:
            base_score = 0.5
            
            if execution.root_cause_found:
                base_score = 0.8
                
                # Boost confidence if multiple steps point to same conclusion
                escalation_steps = [r for r in step_results if r.escalation_triggered]
                if len(escalation_steps) > 1:
                    base_score += 0.1
                
                # Boost if strong evidence was collected
                high_confidence_steps = [
                    r for r in step_results 
                    if r.threshold_met is False and r.actual_value and r.expected_value
                ]
                if high_confidence_steps:
                    base_score += 0.1
            
            final_score = min(1.0, base_score)
            
            return ConfidenceFactor(
                factor_name="root_cause_identification",
                factor_type="diagnostic_success",
                weight=0.1,
                score=final_score,
                contribution=final_score * 0.1,
                explanation=f"Root cause found: {execution.root_cause_found}, confidence: {final_score:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating root cause identification factor: {e}")
            return ConfidenceFactor(
                factor_name="root_cause_identification",
                factor_type="diagnostic_success",
                weight=0.1,
                score=0.3,
                contribution=0.03,
                explanation=f"Error in calculation: {str(e)}"
            )
    
    async def _get_historical_analysis_performance(
        self,
        analysis_type: AnalysisType
    ) -> float:
        """Get historical performance for specific analysis type."""
        try:
            # In a real implementation, this would query historical analysis results
            # For demo, return realistic performance scores
            performance_scores = {
                AnalysisType.CORRELATION: 0.85,
                AnalysisType.PLAYBOOK: 0.88,
                AnalysisType.ML_PREDICTION: 0.78,
                AnalysisType.PATTERN_ANALYSIS: 0.75,
                AnalysisType.ANOMALY_DETECTION: 0.72,
                AnalysisType.ROOT_CAUSE_ANALYSIS: 0.82
            }
            
            return performance_scores.get(analysis_type, 0.75)
            
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return 0.7
    
    async def _calculate_quality_indicators(
        self,
        incident: Incident,
        evidence_items: List[EvidenceItem],
        similar_incidents: List[Incident]
    ) -> Dict[str, float]:
        """Calculate quality indicators for confidence assessment."""
        try:
            indicators = {}
            
            # Data completeness indicator
            completeness_factors = [
                1.0 if incident.title else 0.0,
                1.0 if incident.description else 0.0,
                1.0 if incident.symptoms else 0.0,
                1.0 if incident.incident_symptoms else 0.0,
                1.0 if len(evidence_items) > 2 else len(evidence_items) / 3
            ]
            indicators["data_completeness"] = sum(completeness_factors) / len(completeness_factors)
            
            # Evidence relevance indicator
            if evidence_items:
                indicators["evidence_relevance"] = sum(item.relevance_score for item in evidence_items) / len(evidence_items)
            else:
                indicators["evidence_relevance"] = 0.0
            
            # Historical coverage indicator
            indicators["historical_coverage"] = min(1.0, len(similar_incidents) / 5)
            
            # Analysis depth indicator
            analysis_depth = 0.0
            if incident.incident_symptoms:
                analysis_depth += 0.4
            if evidence_items:
                analysis_depth += 0.3
            if similar_incidents:
                analysis_depth += 0.3
            
            indicators["analysis_depth"] = analysis_depth
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating quality indicators: {e}")
            return {"error": 1.0}
    
    async def _apply_confidence_adjustments(
        self,
        confidence_score: ConfidenceScore,
        incident: Incident,
        analysis_type: AnalysisType
    ) -> ConfidenceScore:
        """Apply confidence adjustments based on context."""
        try:
            original_score = confidence_score.overall_score
            adjusted_score = original_score
            adjustments = []
            
            # Severity-based adjustments
            if incident.severity == IncidentSeverity.CRITICAL:
                adjusted_score *= 1.05  # Slight boost for critical incidents
                adjustments.append("Critical severity boost: +5%")
            elif incident.severity == IncidentSeverity.LOW:
                adjusted_score *= 0.95  # Slight penalty for low severity
                adjustments.append("Low severity adjustment: -5%")
            
            # Analysis type adjustments
            type_multipliers = {
                AnalysisType.CORRELATION: 1.0,
                AnalysisType.PLAYBOOK: 1.02,  # Slight boost for playbook-driven analysis
                AnalysisType.ML_PREDICTION: 0.98,  # Slight penalty for ML prediction uncertainty
                AnalysisType.PATTERN_ANALYSIS: 0.99,
                AnalysisType.ANOMALY_DETECTION: 0.97,
                AnalysisType.ROOT_CAUSE_ANALYSIS: 1.01
            }
            
            type_multiplier = type_multipliers.get(analysis_type, 1.0)
            if type_multiplier != 1.0:
                adjusted_score *= type_multiplier
                adjustments.append(f"{analysis_type.value} type adjustment: {(type_multiplier-1)*100:+.0f}%")
            
            # Time-based adjustments (newer incidents might have better data)
            incident_age_hours = (datetime.utcnow() - incident.timestamp).total_seconds() / 3600
            if incident_age_hours < 24:
                adjusted_score *= 1.02  # Recent incident boost
                adjustments.append("Recent incident boost: +2%")
            elif incident_age_hours > 720:  # 30 days
                adjusted_score *= 0.98  # Old incident penalty
                adjustments.append("Old incident adjustment: -2%")
            
            # Apply bounds
            adjusted_score = max(0.0, min(1.0, adjusted_score))
            
            # Update confidence score
            confidence_score.overall_score = adjusted_score
            
            # Add adjustment explanation
            if adjustments:
                confidence_score.quality_indicators["adjustment_count"] = float(len(adjustments))
                confidence_score.quality_indicators["original_score"] = original_score
                confidence_score.quality_indicators["adjustment_delta"] = adjusted_score - original_score
            
            return confidence_score
            
        except Exception as e:
            logger.error(f"Error applying confidence adjustments: {e}")
            return confidence_score
    
    def _assess_reliability(self, confidence_score: ConfidenceScore) -> str:
        """Assess overall reliability of the confidence score."""
        try:
            score = confidence_score.overall_score
            factors_count = len(confidence_score.factors)
            
            # Base assessment on score
            if score >= 0.9:
                base_reliability = "very_high"
            elif score >= 0.8:
                base_reliability = "high"
            elif score >= 0.7:
                base_reliability = "medium"
            elif score >= 0.5:
                base_reliability = "low"
            else:
                base_reliability = "very_low"
            
            # Adjust based on number of factors
            if factors_count < 3:
                # Downgrade if not enough factors
                downgrades = {"very_high": "high", "high": "medium", "medium": "low", "low": "very_low"}
                base_reliability = downgrades.get(base_reliability, "very_low")
            
            # Check for quality indicators
            quality_indicators = confidence_score.quality_indicators
            if quality_indicators:
                data_completeness = quality_indicators.get("data_completeness", 0.5)
                evidence_relevance = quality_indicators.get("evidence_relevance", 0.5)
                
                if data_completeness < 0.5 or evidence_relevance < 0.5:
                    # Downgrade for poor data quality
                    downgrades = {"very_high": "high", "high": "medium", "medium": "low", "low": "very_low"}
                    base_reliability = downgrades.get(base_reliability, "very_low")
            
            return base_reliability
            
        except Exception as e:
            logger.error(f"Error assessing reliability: {e}")
            return "unknown"
    
    async def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get confidence scoring statistics and performance metrics."""
        try:
            # In a real implementation, this would query historical confidence scores
            # For demo, return sample statistics
            
            stats = {
                "total_analyses": 1247,
                "avg_confidence_score": 0.82,
                "confidence_distribution": {
                    "very_high": 0.15,
                    "high": 0.35,
                    "medium": 0.30,
                    "low": 0.15,
                    "very_low": 0.05
                },
                "factor_contributions": {
                    "historical_accuracy": 0.25,
                    "evidence_strength": 0.20,
                    "data_quality": 0.15,
                    "pattern_matching": 0.15,
                    "similarity_correlation": 0.10,
                    "model_confidence": 0.10,
                    "validation_score": 0.05
                },
                "accuracy_by_type": {
                    "correlation": 0.87,
                    "playbook": 0.91,
                    "ml_prediction": 0.78,
                    "pattern_analysis": 0.74,
                    "anomaly_detection": 0.69,
                    "root_cause_analysis": 0.84
                },
                "reliability_distribution": {
                    "very_high": 0.12,
                    "high": 0.38,
                    "medium": 0.32,
                    "low": 0.14,
                    "very_low": 0.04
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting confidence statistics: {e}")
            return {"error": str(e)}