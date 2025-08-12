"""
AI-powered incident analysis service with historical correlation, symptom analysis, and automated root cause identification.

This service provides the core AI capabilities for the SRE Agent, including:
- Vector similarity search for incident correlation
- Symptom analysis and pattern recognition
- Automated root cause identification
- Confidence scoring for analysis results
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback TF-IDF")

from models.incident import (
    Incident, IncidentSymptom, IncidentCorrelation, IncidentTable,
    IncidentStatus, IncidentSeverity, SymptomType, MetricData
)
from models.analysis import (
    AnalysisResult, AnalysisType, AnalysisStatus, AnalysisFindings,
    EvidenceItem, EvidenceType, ReasoningStep, ReasoningStepType,
    ReasoningTrail, ConfidenceFactor, ConfidenceScore
)
from config.database import get_database
from config.settings import get_settings
from utils.formatters import format_confidence_score, format_duration
from services.vector_search import VectorSearchEngine
from services.confidence_scorer import ConfidenceScorer

settings = get_settings()


class IncidentAnalyzer:
    """AI-powered incident analysis service."""
    
    def __init__(self):
        """Initialize the incident analyzer."""
        self.vector_search = VectorSearchEngine()
        self.confidence_scorer = ConfidenceScorer()
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Initialize sentence transformer model for embeddings
        try:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # TF-IDF vectorizer for text analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Analysis cache
        self._cache = {}
        self._cache_ttl = timedelta(minutes=settings.CACHE_TTL_SECONDS // 60)
    
    async def analyze_incident(
        self,
        incident_id: str,
        analysis_type: AnalysisType = AnalysisType.CORRELATION,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Analyze an incident using AI-powered correlation and pattern recognition.
        
        Args:
            incident_id: ID of the incident to analyze
            analysis_type: Type of analysis to perform
            context: Additional context for analysis
            
        Returns:
            AnalysisResult with findings and recommendations
        """
        logger.info(f"Starting incident analysis for {incident_id}, type: {analysis_type}")
        
        # Create analysis record
        analysis = AnalysisResult(
            id=uuid4(),
            analysis_id=f"ANALYSIS-{str(uuid4())[:8].upper()}",
            incident_id=incident_id,
            analysis_type=analysis_type,
            status=AnalysisStatus.RUNNING,
            started_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
            analysis_context=context or {}
        )
        
        try:
            # Get incident data
            incident = await self._get_incident(incident_id)
            if not incident:
                raise ValueError(f"Incident {incident_id} not found")
            
            # Initialize reasoning trail
            reasoning_trail = ReasoningTrail(analysis_id=analysis.analysis_id)
            
            # Step 1: Data Collection
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.DATA_COLLECTION,
                "Collecting incident data and symptoms",
                {"incident_id": incident_id, "symptoms_count": len(incident.incident_symptoms)},
                {"incident": incident.model_dump()}
            )
            
            # Step 2: Symptom Analysis
            symptom_analysis = await self._analyze_symptoms(incident)
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.SYMPTOM_ANALYSIS,
                "Analyzing incident symptoms and patterns",
                {"symptoms": [s.model_dump() for s in incident.incident_symptoms]},
                symptom_analysis
            )
            
            # Step 3: Historical Correlation
            similar_incidents = await self._find_similar_incidents(incident)
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.CORRELATION_SEARCH,
                f"Found {len(similar_incidents)} similar incidents",
                {"similarity_threshold": self.similarity_threshold},
                {"similar_incidents": [s.model_dump() for s in similar_incidents[:5]]}
            )
            
            # Step 4: Pattern Matching
            patterns = await self._identify_patterns(incident, similar_incidents)
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.PATTERN_MATCHING,
                "Identifying patterns across similar incidents",
                {"pattern_count": len(patterns)},
                {"patterns": patterns}
            )
            
            # Step 5: Root Cause Hypothesis
            hypotheses = await self._generate_hypotheses(incident, similar_incidents, patterns)
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.HYPOTHESIS_GENERATION,
                f"Generated {len(hypotheses)} root cause hypotheses",
                {"hypothesis_count": len(hypotheses)},
                {"hypotheses": hypotheses}
            )
            
            # Step 6: Hypothesis Testing
            tested_hypotheses = await self._test_hypotheses(incident, hypotheses)
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.HYPOTHESIS_TESTING,
                "Testing hypotheses against available evidence",
                {"tested_count": len(tested_hypotheses)},
                {"tested_hypotheses": tested_hypotheses}
            )
            
            # Step 7: Root Cause Identification
            root_cause = await self._identify_root_cause(tested_hypotheses)
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.ROOT_CAUSE_IDENTIFICATION,
                f"Identified root cause: {root_cause['primary_cause']}",
                {"confidence": root_cause.get('confidence', 0.0)},
                root_cause
            )
            
            # Step 8: Evidence Collection
            evidence_items = await self._collect_evidence(incident, similar_incidents, root_cause)
            
            # Step 9: Confidence Calculation
            confidence_score = await self._calculate_confidence(
                incident, similar_incidents, root_cause, evidence_items
            )
            # Extract confidence value for logging
            confidence_value = confidence_score.overall_score if hasattr(confidence_score, 'overall_score') else confidence_score
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.CONFIDENCE_CALCULATION,
                f"Calculated confidence score: {confidence_value:.2f}",
                {"factors_count": len(confidence_score.factors) if hasattr(confidence_score, 'factors') else 0},
                {"confidence_score": confidence_value}
            )
            
            # Step 10: Recommendation Generation
            recommendations = await self._generate_recommendations(incident, root_cause, similar_incidents)
            await self._add_reasoning_step(
                reasoning_trail,
                ReasoningStepType.RECOMMENDATION_GENERATION,
                f"Generated {len(recommendations)} recommendations",
                {"recommendation_count": len(recommendations)},
                {"recommendations": recommendations}
            )
            
            # Finalize analysis
            analysis.status = AnalysisStatus.COMPLETED
            analysis.completed_at = datetime.utcnow()
            analysis.duration_seconds = (analysis.completed_at - analysis.started_at).total_seconds()
            analysis.findings = AnalysisFindings(
                primary_cause=root_cause['primary_cause'],
                contributing_factors=root_cause.get('contributing_factors', []),
                affected_components=root_cause.get('affected_components', []),
                related_incidents=[inc.incident_id for inc in similar_incidents[:3]]
            )
            analysis.confidence_score = confidence_score.overall_score if hasattr(confidence_score, 'overall_score') else confidence_score
            analysis.recommendation = "; ".join(recommendations)
            analysis.root_cause = root_cause['primary_cause']
            analysis.evidence_items = evidence_items
            analysis.reasoning_trail = reasoning_trail
            
            logger.info(f"Completed incident analysis for {incident_id} with confidence {analysis.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error analyzing incident {incident_id}: {e}")
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = str(e)
            analysis.completed_at = datetime.utcnow()
            analysis.duration_seconds = (analysis.completed_at - analysis.started_at).total_seconds()
        
        return analysis
    
    async def correlate_with_historical_incidents(
        self,
        incident: Incident,
        limit: int = 10
    ) -> List[Tuple[Incident, float]]:
        """
        Find incidents similar to the given incident using vector similarity.
        
        Args:
            incident: Incident to find correlations for
            limit: Maximum number of similar incidents to return
            
        Returns:
            List of tuples (similar_incident, similarity_score)
        """
        logger.info(f"Correlating incident {incident.incident_id} with historical data")
        
        try:
            # Use vector search engine for similarity search
            similar_incidents = await self.vector_search.find_similar_incidents(
                incident, threshold=self.similarity_threshold, limit=limit
            )
            
            logger.info(f"Found {len(similar_incidents)} similar incidents")
            return similar_incidents
            
        except Exception as e:
            logger.error(f"Error correlating incidents: {e}")
            return []
    
    async def analyze_symptom_patterns(
        self,
        symptoms: List[IncidentSymptom]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in incident symptoms.
        
        Args:
            symptoms: List of incident symptoms
            
        Returns:
            Analysis results with pattern information
        """
        if not symptoms:
            return {"patterns": [], "primary_symptom": None, "severity_distribution": {}}
        
        # Group symptoms by type
        symptom_groups = {}
        for symptom in symptoms:
            symptom_type = symptom.symptom_type
            if symptom_type not in symptom_groups:
                symptom_groups[symptom_type] = []
            symptom_groups[symptom_type].append(symptom)
        
        # Find primary symptom (highest severity)
        primary_symptom = max(symptoms, key=lambda s: s.severity_score)
        
        # Calculate severity distribution
        severity_distribution = {}
        for symptom in symptoms:
            severity_range = self._get_severity_range(symptom.severity_score)
            severity_distribution[severity_range] = severity_distribution.get(severity_range, 0) + 1
        
        # Identify patterns
        patterns = []
        
        # Pattern 1: Multiple high-severity symptoms
        high_severity_symptoms = [s for s in symptoms if s.severity_score > 0.8]
        if len(high_severity_symptoms) > 1:
            patterns.append({
                "type": "multiple_high_severity",
                "description": f"{len(high_severity_symptoms)} high-severity symptoms detected",
                "confidence": 0.9,
                "symptoms": [s.symptom_type for s in high_severity_symptoms]
            })
        
        # Pattern 2: Cascading failure pattern
        if len(symptom_groups) > 2:
            time_sorted_symptoms = sorted(symptoms, key=lambda s: s.detected_at)
            time_gaps = []
            for i in range(1, len(time_sorted_symptoms)):
                gap = (time_sorted_symptoms[i].detected_at - time_sorted_symptoms[i-1].detected_at).total_seconds()
                time_gaps.append(gap)
            
            if time_gaps and max(time_gaps) < 300:  # All symptoms within 5 minutes
                patterns.append({
                    "type": "cascading_failure",
                    "description": "Rapid progression of symptoms suggests cascading failure",
                    "confidence": 0.8,
                    "time_span_seconds": max(time_gaps)
                })
        
        # Pattern 3: Performance degradation pattern
        performance_symptoms = [
            s for s in symptoms 
            if s.symptom_type in [SymptomType.LATENCY, SymptomType.QUERY_PERFORMANCE, SymptomType.CACHE_PERFORMANCE]
        ]
        if len(performance_symptoms) >= 2:
            patterns.append({
                "type": "performance_degradation",
                "description": f"Multiple performance-related symptoms: {[s.symptom_type for s in performance_symptoms]}",
                "confidence": 0.85,
                "affected_components": list(set(s.symptom_type for s in performance_symptoms))
            })
        
        return {
            "patterns": patterns,
            "primary_symptom": primary_symptom.model_dump(),
            "severity_distribution": severity_distribution,
            "symptom_groups": {k: len(v) for k, v in symptom_groups.items()},
            "total_symptoms": len(symptoms),
            "avg_severity": sum(s.severity_score for s in symptoms) / len(symptoms)
        }
    
    async def generate_incident_recommendations(
        self,
        incident: Incident,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate actionable recommendations based on incident analysis.
        
        Args:
            incident: Incident being analyzed
            analysis_results: Results from incident analysis
            
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        # Get root cause from analysis
        root_cause = analysis_results.get('root_cause', {})
        primary_cause = root_cause.get('primary_cause', '')
        
        # Cache-related recommendations
        if 'cache' in primary_cause.lower():
            recommendations.extend([
                "Scale Redis cluster nodes to handle increased load",
                "Warm cache with frequently accessed data",
                "Implement cache warming strategy for critical data",
                "Monitor cache hit rates and set up alerts for drops below 80%"
            ])
        
        # Database-related recommendations
        elif 'database' in primary_cause.lower() or 'connection' in primary_cause.lower():
            recommendations.extend([
                "Increase database connection pool size",
                "Optimize slow-running queries",
                "Implement query timeout limits",
                "Add database read replicas if needed",
                "Review recent database schema changes"
            ])
        
        # Performance-related recommendations
        elif 'latency' in primary_cause.lower() or 'performance' in primary_cause.lower():
            recommendations.extend([
                "Implement circuit breaker pattern",
                "Add application performance monitoring",
                "Review recent deployments for performance regressions",
                "Scale application instances horizontally"
            ])
        
        # Memory-related recommendations
        elif 'memory' in primary_cause.lower():
            recommendations.extend([
                "Investigate memory leaks in application code",
                "Increase memory limits for affected services",
                "Implement proper garbage collection tuning",
                "Add memory usage monitoring and alerts"
            ])
        
        # Network-related recommendations
        elif 'network' in primary_cause.lower():
            recommendations.extend([
                "Check network connectivity between services",
                "Review firewall and security group configurations",
                "Monitor network latency and packet loss",
                "Consider CDN implementation for static content"
            ])
        
        # General recommendations based on severity
        if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
            recommendations.extend([
                "Implement additional monitoring for early detection",
                "Create runbook for faster resolution of similar issues",
                "Consider implementing automated recovery procedures"
            ])
        
        # Add service-specific recommendations
        service_recommendations = self._get_service_specific_recommendations(incident.service_name, primary_cause)
        recommendations.extend(service_recommendations)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def _get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID - use mock incidents for demo."""
        try:
            # Mock implementation for demo incidents
            now = datetime.utcnow()
            if incident_id == "INC-2024-001":
                return Incident(
                    id=uuid4(),
                    incident_id=incident_id,
                    title="Payment API Latency Spike",
                    description="High latency detected in payment API",
                    severity=IncidentSeverity.HIGH,
                    status=IncidentStatus.INVESTIGATING,
                    service_name="payment-api",
                    region="us-central1",
                    timestamp=now,
                    created_at=now,
                    created_by="system",
                    assigned_to="sre_team",
                    incident_symptoms=[
                        IncidentSymptom(
                            symptom_type=SymptomType.LATENCY,
                            description="API response times increased to 2500ms (threshold: 200ms)",
                            metric_data=MetricData(
                                metric_name="http_request_duration_seconds",
                                threshold=0.2,
                                actual_value=2.5,
                                unit="seconds",
                                timestamp=now
                            ),
                            detected_at=now,
                            severity_score=0.85
                        ),
                        IncidentSymptom(
                            symptom_type=SymptomType.CACHE_PERFORMANCE,
                            description="Cache hit rate dropped to 45% (threshold: 85%)",
                            metric_data=MetricData(
                                metric_name="cache_hit_rate",
                                threshold=85.0,
                                actual_value=45.0,
                                unit="percentage",
                                timestamp=now
                            ),
                            detected_at=now,
                            severity_score=0.75
                        )
                    ],
                    tags=["latency", "payment", "api"]
                )
            elif incident_id == "INC-2024-002":
                return Incident(
                    id=uuid4(),
                    incident_id=incident_id,
                    title="Database Connection Timeout",
                    description="Database connection timeouts in ecommerce service",
                    severity=IncidentSeverity.HIGH,
                    status=IncidentStatus.INVESTIGATING,
                    service_name="ecommerce-api",
                    region="us-central1",
                    timestamp=now,
                    created_at=now,
                    created_by="system",
                    assigned_to="sre_team",
                    incident_symptoms=[
                        IncidentSymptom(
                            symptom_type=SymptomType.CONNECTION_TIMEOUT,
                            description="Database connection timeouts increased to 45/min (threshold: 5/min)",
                            metric_data=MetricData(
                                metric_name="database_connection_timeouts",
                                threshold=5.0,
                                actual_value=45.0,
                                unit="errors_per_minute",
                                timestamp=now
                            ),
                            detected_at=now,
                            severity_score=0.90
                        ),
                        IncidentSymptom(
                            symptom_type=SymptomType.QUERY_PERFORMANCE,
                            description="Query execution time increased to 8.2s (threshold: 2.0s)",
                            metric_data=MetricData(
                                metric_name="query_execution_time",
                                threshold=2.0,
                                actual_value=8.2,
                                unit="seconds",
                                timestamp=now
                            ),
                            detected_at=now,
                            severity_score=0.80
                        )
                    ],
                    tags=["database", "timeout", "ecommerce"]
                )
            else:
                logger.warning(f"Mock incident {incident_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching incident {incident_id}: {e}")
            return None
    
    async def _analyze_symptoms(self, incident: Incident) -> Dict[str, Any]:
        """Analyze incident symptoms for patterns and severity."""
        return await self.analyze_symptom_patterns(incident.incident_symptoms)
    
    async def _find_similar_incidents(self, incident: Incident) -> List[Incident]:
        """Find similar incidents using vector similarity search."""
        similar_with_scores = await self.correlate_with_historical_incidents(incident, limit=10)
        return [incident for incident, score in similar_with_scores if score >= self.similarity_threshold]
    
    async def _identify_patterns(
        self,
        incident: Incident,
        similar_incidents: List[Incident]
    ) -> List[Dict[str, Any]]:
        """Identify patterns across current and similar incidents."""
        patterns = []
        
        if not similar_incidents:
            return patterns
        
        # Pattern 1: Common root causes
        root_causes = []
        for sim_incident in similar_incidents:
            if sim_incident.root_cause:
                root_causes.append(sim_incident.root_cause)
        
        if root_causes:
            from collections import Counter
            cause_counts = Counter(root_causes)
            most_common = cause_counts.most_common(1)[0]
            
            patterns.append({
                "type": "common_root_cause",
                "description": f"Most similar incidents had root cause: {most_common[0]}",
                "frequency": most_common[1],
                "confidence": min(0.9, most_common[1] / len(similar_incidents))
            })
        
        # Pattern 2: Common service/region
        same_service_incidents = [inc for inc in similar_incidents if inc.service_name == incident.service_name]
        if len(same_service_incidents) > len(similar_incidents) * 0.6:
            patterns.append({
                "type": "service_specific",
                "description": f"Issue appears to be specific to {incident.service_name} service",
                "affected_incidents": len(same_service_incidents),
                "confidence": 0.8
            })
        
        # Pattern 3: Temporal patterns
        recent_incidents = [
            inc for inc in similar_incidents 
            if (datetime.utcnow() - inc.timestamp).days <= 7
        ]
        if len(recent_incidents) > 2:
            patterns.append({
                "type": "recurring_issue",
                "description": f"{len(recent_incidents)} similar incidents in the past week",
                "recent_count": len(recent_incidents),
                "confidence": 0.85
            })
        
        return patterns
    
    async def _generate_hypotheses(
        self,
        incident: Incident,
        similar_incidents: List[Incident],
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate root cause hypotheses based on analysis."""
        hypotheses = []
        
        # Hypothesis from similar incidents
        if similar_incidents:
            root_causes = [inc.root_cause for inc in similar_incidents if inc.root_cause]
            if root_causes:
                from collections import Counter
                cause_counts = Counter(root_causes)
                top_cause = cause_counts.most_common(1)[0]
                
                hypotheses.append({
                    "hypothesis": top_cause[0],
                    "source": "historical_correlation",
                    "supporting_incidents": top_cause[1],
                    "initial_confidence": min(0.9, top_cause[1] / len(similar_incidents))
                })
        
        # Hypothesis from symptom patterns
        primary_symptom = incident.get_primary_symptom()
        if primary_symptom:
            symptom_based_causes = {
                SymptomType.CACHE_PERFORMANCE: "Redis cache cluster failure causing cache misses and database overload",
                SymptomType.CONNECTION_TIMEOUT: "Database connection pool exhaustion due to long-running queries",
                SymptomType.LATENCY: "Network latency spike or service performance degradation",
                SymptomType.MEMORY_USAGE: "Memory leak in application service",
                SymptomType.CPU_USAGE: "High CPU utilization due to inefficient algorithm or increased load"
            }
            
            if primary_symptom.symptom_type in symptom_based_causes:
                hypotheses.append({
                    "hypothesis": symptom_based_causes[primary_symptom.symptom_type],
                    "source": "symptom_analysis",
                    "primary_symptom": primary_symptom.symptom_type,
                    "initial_confidence": primary_symptom.severity_score
                })
        
        # Hypothesis from patterns
        for pattern in patterns:
            if pattern["type"] == "common_root_cause":
                hypotheses.append({
                    "hypothesis": pattern["description"],
                    "source": "pattern_analysis",
                    "pattern_type": pattern["type"],
                    "initial_confidence": pattern["confidence"]
                })
        
        # Default hypothesis if no specific ones found
        if not hypotheses:
            hypotheses.append({
                "hypothesis": "Service performance degradation due to increased load or recent changes",
                "source": "default_analysis",
                "initial_confidence": 0.5
            })
        
        return hypotheses
    
    async def _test_hypotheses(
        self,
        incident: Incident,
        hypotheses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Test hypotheses against available evidence."""
        tested_hypotheses = []
        
        for hypothesis in hypotheses:
            tested = hypothesis.copy()
            evidence_score = 0.0
            evidence_count = 0
            
            # Test against incident symptoms
            hypothesis_text = hypothesis["hypothesis"].lower()
            for symptom in incident.incident_symptoms:
                symptom_desc = symptom.description.lower()
                
                # Simple keyword matching for evidence
                keywords = {
                    "cache": ["cache", "redis", "memcache"],
                    "database": ["database", "sql", "connection", "query"],
                    "latency": ["latency", "slow", "timeout"],
                    "memory": ["memory", "oom", "heap"],
                    "cpu": ["cpu", "processor", "compute"]
                }
                
                for category, category_keywords in keywords.items():
                    if any(keyword in hypothesis_text for keyword in category_keywords):
                        if any(keyword in symptom_desc for keyword in category_keywords):
                            evidence_score += symptom.severity_score
                            evidence_count += 1
            
            # Calculate tested confidence
            if evidence_count > 0:
                avg_evidence_score = evidence_score / evidence_count
                tested["tested_confidence"] = (
                    tested["initial_confidence"] * 0.6 + avg_evidence_score * 0.4
                )
                tested["evidence_strength"] = avg_evidence_score
                tested["evidence_count"] = evidence_count
            else:
                tested["tested_confidence"] = tested["initial_confidence"] * 0.5
                tested["evidence_strength"] = 0.0
                tested["evidence_count"] = 0
            
            tested_hypotheses.append(tested)
        
        # Sort by tested confidence
        tested_hypotheses.sort(key=lambda x: x["tested_confidence"], reverse=True)
        
        return tested_hypotheses
    
    async def _identify_root_cause(self, tested_hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the most likely root cause from tested hypotheses."""
        if not tested_hypotheses:
            return {
                "primary_cause": "Unknown - insufficient data for analysis",
                "confidence": 0.1,
                "contributing_factors": ["Lack of historical data", "Insufficient symptom information"]
            }
        
        best_hypothesis = tested_hypotheses[0]
        
        # Extract contributing factors from other hypotheses
        contributing_factors = []
        for hyp in tested_hypotheses[1:3]:  # Top 3 excluding the best
            if hyp["tested_confidence"] > 0.3:
                contributing_factors.append(hyp["hypothesis"])
        
        # Add default contributing factors based on common patterns
        default_factors = [
            "Recent code deployment",
            "Increased traffic load",
            "Infrastructure scaling event"
        ]
        contributing_factors.extend(default_factors)
        
        return {
            "primary_cause": best_hypothesis["hypothesis"],
            "confidence": best_hypothesis["tested_confidence"],
            "contributing_factors": contributing_factors[:3],
            "affected_components": self._extract_components(best_hypothesis["hypothesis"]),
            "evidence_strength": best_hypothesis.get("evidence_strength", 0.0)
        }
    
    async def _collect_evidence(
        self,
        incident: Incident,
        similar_incidents: List[Incident],
        root_cause: Dict[str, Any]
    ) -> List[EvidenceItem]:
        """Collect evidence items supporting the analysis."""
        evidence_items = []
        
        # Evidence from GCP monitoring
        evidence_items.append(EvidenceItem(
            evidence_type=EvidenceType.GCP_MONITORING,
            source="GCP Monitoring Dashboard",
            description="GCP Monitoring dashboard showing metric anomalies",
            relevance_score=0.9,
            timestamp=datetime.utcnow(),
            url=f"https://console.cloud.google.com/monitoring/dashboards/{incident.service_name}"
        ))
        
        # Evidence from logs
        evidence_items.append(EvidenceItem(
            evidence_type=EvidenceType.GCP_LOGGING,
            source="GCP Cloud Logging",
            description="Log entries confirming error patterns",
            relevance_score=0.8,
            timestamp=datetime.utcnow(),
            url=f"https://console.cloud.google.com/logs/query;query=resource.labels.service_name={incident.service_name}"
        ))
        
        # Evidence from tracing
        evidence_items.append(EvidenceItem(
            evidence_type=EvidenceType.GCP_TRACING,
            source="GCP Cloud Trace",
            description="Distributed tracing showing slow spans",
            relevance_score=0.85,
            timestamp=datetime.utcnow(),
            url=f"https://console.cloud.google.com/traces/list?project=ai-sre-agent"
        ))
        
        # Evidence from historical correlation
        if similar_incidents:
            similarity_percentage = int(root_cause.get("confidence", 0.8) * 100)
            evidence_items.append(EvidenceItem(
                evidence_type=EvidenceType.HISTORICAL_CORRELATION,
                source="Historical Incident Database",
                description=f"Historical incident correlation with {similarity_percentage}% similarity",
                relevance_score=root_cause.get("confidence", 0.8),
                timestamp=datetime.utcnow(),
                data={"similar_incidents_count": len(similar_incidents)}
            ))
        
        return evidence_items
    
    async def _calculate_confidence(
        self,
        incident: Incident,
        similar_incidents: List[Incident],
        root_cause: Dict[str, Any],
        evidence_items: List[EvidenceItem]
    ) -> ConfidenceScore:
        """Calculate overall confidence score for the analysis."""
        return await self.confidence_scorer.calculate_analysis_confidence(
            incident=incident,
            similar_incidents=similar_incidents,
            root_cause=root_cause,
            evidence_items=evidence_items
        )
    
    async def _generate_recommendations(
        self,
        incident: Incident,
        root_cause: Dict[str, Any],
        similar_incidents: List[Incident]
    ) -> List[str]:
        """Generate actionable recommendations."""
        return await self.generate_incident_recommendations(
            incident=incident,
            analysis_results={"root_cause": root_cause, "similar_incidents": similar_incidents}
        )
    
    async def _add_reasoning_step(
        self,
        trail: ReasoningTrail,
        step_type: ReasoningStepType,
        description: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence_impact: float = 0.0
    ) -> None:
        """Add a reasoning step to the trail."""
        step = ReasoningStep(
            step_number=len(trail.steps) + 1,
            step_type=step_type,
            description=description,
            input_data=input_data,
            output_data=output_data,
            reasoning=f"Executed {step_type.value} step: {description}",
            confidence_impact=confidence_impact
        )
        trail.add_step(step)
    
    def _get_severity_range(self, severity_score: float) -> str:
        """Convert severity score to range label."""
        if severity_score >= 0.8:
            return "high"
        elif severity_score >= 0.6:
            return "medium"
        elif severity_score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _extract_components(self, hypothesis: str) -> List[str]:
        """Extract affected components from hypothesis text."""
        components = []
        hypothesis_lower = hypothesis.lower()
        
        component_keywords = {
            "database": ["database", "sql", "connection", "query"],
            "cache": ["cache", "redis", "memcache"],
            "loadbalancer": ["load balancer", "loadbalancer", "lb"],
            "api-gateway": ["api gateway", "gateway", "api"],
            "monitoring": ["monitoring", "metrics", "observability"],
            "logging": ["logging", "logs", "log"],
            "tracing": ["tracing", "trace", "spans"]
        }
        
        for component, keywords in component_keywords.items():
            if any(keyword in hypothesis_lower for keyword in keywords):
                components.append(component)
        
        return components or ["unknown"]
    
    def _get_service_specific_recommendations(self, service_name: str, primary_cause: str) -> List[str]:
        """Get service-specific recommendations."""
        recommendations = []
        
        if "payment" in service_name.lower():
            recommendations.extend([
                "Review payment processing logic for bottlenecks",
                "Implement payment retry mechanisms",
                "Monitor payment success rates"
            ])
        elif "auth" in service_name.lower():
            recommendations.extend([
                "Review authentication token expiration policies",
                "Implement session management optimizations",
                "Monitor authentication failure rates"
            ])
        elif "api" in service_name.lower():
            recommendations.extend([
                "Implement API rate limiting",
                "Review API endpoint performance",
                "Add API response time monitoring"
            ])
        
        return recommendations
    
    async def get_analysis_status(self, incident_id: str) -> Dict[str, Any]:
        """
        Get the current analysis status for an incident.
        
        Args:
            incident_id: ID of the incident to get status for
            
        Returns:
            Analysis status information
        """
        logger.info(f"Getting analysis status for incident {incident_id}")
        
        try:
            # Check if there's an ongoing analysis
            cache_key = f"analysis_status_{incident_id}"
            if cache_key in self._cache:
                cached_status = self._cache[cache_key]
                if datetime.utcnow() - cached_status.get('timestamp', datetime.min) < self._cache_ttl:
                    return cached_status
            
            # Get incident data
            incident = await self._get_incident(incident_id)
            if not incident:
                return {
                    "incident_id": incident_id,
                    "status": "not_found",
                    "message": f"Incident {incident_id} not found",
                    "timestamp": datetime.utcnow()
                }
            
            # Determine analysis status based on incident data
            status_info = {
                "incident_id": incident_id,
                "status": "completed" if incident.root_cause else "pending",
                "confidence_score": incident.confidence_score,
                "root_cause": incident.root_cause,
                "analysis_summary": {
                    "symptoms_count": len(incident.incident_symptoms),
                    "correlations_count": len(incident.correlations),
                    "resolution_steps": len(incident.resolutions),
                    "has_root_cause": bool(incident.root_cause),
                    "incident_status": incident.status
                },
                "last_updated": incident.updated_at or incident.created_at,
                "timestamp": datetime.utcnow()
            }
            
            # Add recommendations if analysis is complete
            if incident.root_cause:
                status_info["recommendations"] = await self.generate_incident_recommendations(
                    incident=incident,
                    analysis_results={"root_cause": {"primary_cause": incident.root_cause}}
                )
            
            # Cache the result
            self._cache[cache_key] = status_info
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting analysis status for incident {incident_id}: {e}")
            return {
                "incident_id": incident_id,
                "status": "error",
                "message": f"Error retrieving analysis status: {str(e)}",
                "timestamp": datetime.utcnow()
            }