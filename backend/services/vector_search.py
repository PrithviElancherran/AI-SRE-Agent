"""
Vector similarity search engine for incident correlation using sentence transformers and cosine similarity.

This service provides vector-based similarity search capabilities for the AI SRE Agent, including:
- Incident embedding generation using sentence transformers
- Cosine similarity computation for incident correlation
- Efficient vector storage and retrieval
- Batch processing for large datasets
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from loguru import logger
import pickle
import os
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback TF-IDF")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func

from models.incident import Incident, IncidentTable, IncidentSeverity, IncidentStatus
from models.analysis import EvidenceItem, EvidenceType
from config.database import get_database
from config.settings import get_settings
from services.synthetic_data_loader import synthetic_data_loader
from utils.formatters import format_confidence_score

settings = get_settings()


class VectorSearchEngine:
    """Vector similarity search engine for incident correlation."""
    
    def __init__(self):
        """Initialize the vector search engine."""
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.incident_embeddings = {}
        self.embedding_cache = {}
        self.cache_file = Path(settings.SYNTHETIC_DATA_PATH) / "embedding_cache.pkl"
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Load or create embedding cache
        self._load_embedding_cache()
        
        logger.info(f"VectorSearchEngine initialized with model: {settings.EMBEDDING_MODEL}")
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the sentence transformer model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info(f"Loaded sentence transformer model: {settings.EMBEDDING_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.embedding_model = None
        
        # Fallback to TF-IDF if sentence transformers not available
        if not self.embedding_model:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            logger.info("Using TF-IDF vectorizer as fallback")
    
    def _load_embedding_cache(self) -> None:
        """Load embedding cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            else:
                self.embedding_cache = {}
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_embedding_cache(self) -> None:
        """Save embedding cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    async def find_similar_incidents(
        self,
        target_incident: Incident,
        threshold: Optional[float] = None,
        limit: int = 10,
        exclude_same_incident: bool = True
    ) -> List[Tuple[Incident, float]]:
        """
        Find incidents similar to the target incident using vector similarity.
        
        Args:
            target_incident: Incident to find similarities for
            threshold: Minimum similarity threshold (default: settings.SIMILARITY_THRESHOLD)
            limit: Maximum number of similar incidents to return
            exclude_same_incident: Whether to exclude the target incident from results
            
        Returns:
            List of tuples (similar_incident, similarity_score) sorted by similarity
        """
        logger.info(f"Finding similar incidents for {target_incident.incident_id}")
        
        threshold = threshold or self.similarity_threshold
        
        try:
            # Generate embedding for target incident
            target_embedding = await self._generate_incident_embedding(target_incident)
            if target_embedding is None:
                logger.error("Failed to generate embedding for target incident")
                return []
            
            # Get all historical incidents
            historical_incidents = await self._get_historical_incidents(
                exclude_incident_id=target_incident.incident_id if exclude_same_incident else None
            )
            
            if not historical_incidents:
                logger.warning("No historical incidents found for comparison")
                return []
            
            # Generate embeddings for historical incidents
            historical_embeddings = []
            incident_mapping = []
            
            for incident in historical_incidents:
                embedding = await self._generate_incident_embedding(incident)
                if embedding is not None:
                    historical_embeddings.append(embedding)
                    incident_mapping.append(incident)
            
            if not historical_embeddings:
                logger.warning("No valid embeddings generated for historical incidents")
                return []
            
            # Calculate similarities
            similarities = await self._calculate_similarities(
                target_embedding, historical_embeddings
            )
            
            # Create results with similarity scores
            results = []
            for i, similarity_score in enumerate(similarities):
                if similarity_score >= threshold:
                    results.append((incident_mapping[i], float(similarity_score)))
            
            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:limit]
            
            logger.info(f"Found {len(results)} similar incidents above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar incidents: {e}")
            return []
    
    async def build_incident_index(
        self,
        incidents: Optional[List[Incident]] = None,
        force_rebuild: bool = False
    ) -> bool:
        """
        Build or rebuild the incident embedding index.
        
        Args:
            incidents: List of incidents to index (default: all incidents from DB)
            force_rebuild: Whether to force rebuild even if cache exists
            
        Returns:
            True if index was built successfully
        """
        logger.info("Building incident embedding index")
        
        try:
            if incidents is None:
                incidents = await self._get_all_incidents()
            
            if not incidents:
                logger.warning("No incidents found to index")
                return False
            
            # Clear existing cache if force rebuild
            if force_rebuild:
                self.embedding_cache.clear()
            
            # Generate embeddings for all incidents
            embeddings_generated = 0
            for incident in incidents:
                cache_key = self._get_cache_key(incident)
                
                if cache_key not in self.embedding_cache or force_rebuild:
                    embedding = await self._generate_incident_embedding(incident, use_cache=False)
                    if embedding is not None:
                        self.embedding_cache[cache_key] = embedding.tolist()
                        embeddings_generated += 1
            
            # Save cache to disk
            self._save_embedding_cache()
            
            logger.info(f"Built index for {len(incidents)} incidents, generated {embeddings_generated} new embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error building incident index: {e}")
            return False
    
    async def search_similar_by_text(
        self,
        query_text: str,
        incident_types: Optional[List[str]] = None,
        severity_levels: Optional[List[IncidentSeverity]] = None,
        limit: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[Incident, float]]:
        """
        Search for incidents similar to the given text query.
        
        Args:
            query_text: Text description to search for
            incident_types: Filter by incident types
            severity_levels: Filter by severity levels
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (incident, similarity_score)
        """
        logger.info(f"Searching incidents by text: {query_text[:100]}...")
        
        threshold = threshold or self.similarity_threshold
        
        try:
            # Generate embedding for query text
            query_embedding = await self._generate_text_embedding(query_text)
            if query_embedding is None:
                logger.error("Failed to generate embedding for query text")
                return []
            
            # Get filtered incidents
            incidents = await self._get_filtered_incidents(
                incident_types=incident_types,
                severity_levels=severity_levels
            )
            
            if not incidents:
                logger.warning("No incidents found matching filters")
                return []
            
            # Generate embeddings and calculate similarities
            similarities = []
            for incident in incidents:
                incident_embedding = await self._generate_incident_embedding(incident)
                if incident_embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        incident_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= threshold:
                        similarities.append((incident, float(similarity)))
            
            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = similarities[:limit]
            
            logger.info(f"Found {len(results)} incidents matching text query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []
    
    async def analyze_incident_clusters(
        self,
        incidents: Optional[List[Incident]] = None,
        cluster_threshold: float = 0.8,
        min_cluster_size: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Analyze incident clusters to identify patterns.
        
        Args:
            incidents: List of incidents to analyze
            cluster_threshold: Similarity threshold for clustering
            min_cluster_size: Minimum incidents per cluster
            
        Returns:
            List of cluster information dictionaries
        """
        logger.info("Analyzing incident clusters")
        
        try:
            if incidents is None:
                incidents = await self._get_recent_incidents(days=30)
            
            if len(incidents) < min_cluster_size:
                logger.warning(f"Not enough incidents ({len(incidents)}) for clustering")
                return []
            
            # Generate embeddings for all incidents
            embeddings = []
            incident_mapping = []
            
            for incident in incidents:
                embedding = await self._generate_incident_embedding(incident)
                if embedding is not None:
                    embeddings.append(embedding)
                    incident_mapping.append(incident)
            
            if len(embeddings) < min_cluster_size:
                logger.warning("Not enough valid embeddings for clustering")
                return []
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Simple clustering based on similarity threshold
            clusters = []
            used_indices = set()
            
            for i in range(len(embeddings)):
                if i in used_indices:
                    continue
                
                # Find all incidents similar to this one
                cluster_indices = [i]
                for j in range(i + 1, len(embeddings)):
                    if j not in used_indices and similarity_matrix[i][j] >= cluster_threshold:
                        cluster_indices.append(j)
                        used_indices.add(j)
                
                # Create cluster if it meets minimum size
                if len(cluster_indices) >= min_cluster_size:
                    cluster_incidents = [incident_mapping[idx] for idx in cluster_indices]
                    cluster_info = await self._analyze_cluster(cluster_incidents, cluster_indices, similarity_matrix)
                    clusters.append(cluster_info)
                    
                    for idx in cluster_indices:
                        used_indices.add(idx)
            
            logger.info(f"Found {len(clusters)} incident clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error analyzing incident clusters: {e}")
            return []
    
    async def _generate_incident_embedding(
        self,
        incident: Incident,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """Generate embedding vector for an incident."""
        cache_key = self._get_cache_key(incident)
        
        # Check cache first
        if use_cache and cache_key in self.embedding_cache:
            return np.array(self.embedding_cache[cache_key])
        
        try:
            # Create text representation of incident
            incident_text = self._incident_to_text(incident)
            
            # Generate embedding
            embedding = await self._generate_text_embedding(incident_text)
            
            # Cache the embedding
            if embedding is not None and use_cache:
                self.embedding_cache[cache_key] = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for incident {incident.incident_id}: {e}")
            return None
    
    async def _generate_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding vector for text."""
        try:
            if self.embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use sentence transformer
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding
            
            elif self.tfidf_vectorizer:
                # Use TF-IDF as fallback
                # Need to fit on some corpus first
                if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                    # Get some sample texts to fit the vectorizer
                    sample_texts = await self._get_sample_incident_texts()
                    if sample_texts:
                        self.tfidf_vectorizer.fit(sample_texts)
                    else:
                        # Fallback corpus
                        sample_texts = [
                            "database connection timeout error",
                            "high cpu usage performance issue",
                            "memory leak out of memory",
                            "network latency spike slow response",
                            "cache miss redis performance",
                            "query timeout slow database"
                        ]
                        self.tfidf_vectorizer.fit(sample_texts)
                
                # Generate TF-IDF vector
                embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
                return embedding
            
            else:
                logger.error("No embedding method available")
                return None
                
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    async def _calculate_similarities(
        self,
        target_embedding: np.ndarray,
        embeddings: List[np.ndarray]
    ) -> List[float]:
        """Calculate cosine similarities between target and multiple embeddings."""
        try:
            if not embeddings:
                return []
            
            # Stack embeddings into matrix
            embedding_matrix = np.vstack(embeddings)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                target_embedding.reshape(1, -1),
                embedding_matrix
            )[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return []
    
    def _incident_to_text(self, incident: Incident) -> str:
        """Convert incident to text representation for embedding."""
        text_parts = []
        
        # Add title and description
        if incident.title:
            text_parts.append(incident.title)
        if incident.description:
            text_parts.append(incident.description)
        
        # Add symptoms
        if incident.symptoms:
            text_parts.extend(incident.symptoms)
        
        # Add service and region context
        text_parts.append(f"service: {incident.service_name}")
        text_parts.append(f"region: {incident.region}")
        text_parts.append(f"severity: {incident.severity}")
        
        # Add root cause if available
        if incident.root_cause:
            text_parts.append(f"root cause: {incident.root_cause}")
        
        # Add tags
        if incident.tags:
            text_parts.extend(incident.tags)
        
        # Add incident symptoms details
        for symptom in incident.incident_symptoms:
            text_parts.append(symptom.description)
            text_parts.append(f"symptom type: {symptom.symptom_type}")
        
        return " ".join(text_parts)
    
    def _get_cache_key(self, incident: Incident) -> str:
        """Generate cache key for incident."""
        # Use incident ID and a hash of key content for cache key
        content = f"{incident.incident_id}_{incident.title}_{len(incident.symptoms)}_{incident.service_name}"
        return f"incident_{hash(content)}"
    
    async def _get_historical_incidents(
        self,
        exclude_incident_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Incident]:
        """Get historical incidents from synthetic data loader."""
        try:
            # First try to get from synthetic data loader
            if synthetic_data_loader.is_loaded():
                all_incidents = synthetic_data_loader.get_loaded_incidents_models()
                # Filter for resolved incidents and exclude current incident
                historical_incidents = [
                    incident for incident in all_incidents
                    if incident.status == IncidentStatus.RESOLVED 
                    and incident.incident_id != exclude_incident_id
                ]
                
                # Sort by timestamp and limit results
                historical_incidents.sort(key=lambda x: x.timestamp, reverse=True)
                result = historical_incidents[:limit]
                
                logger.debug(f"Retrieved {len(result)} historical incidents from synthetic data")
                logger.debug(f"All incidents count: {len(all_incidents)}")
                logger.debug(f"Filtered resolved incidents count: {len(historical_incidents)}")
                if len(all_incidents) > 0:
                    logger.debug(f"First incident status: {all_incidents[0].status}")
                return result
            
            # Fallback to database query if synthetic data not loaded
            async with get_database() as session:
                query = select(IncidentTable).where(
                    IncidentTable.status == IncidentStatus.RESOLVED
                ).order_by(desc(IncidentTable.timestamp)).limit(limit)
                
                if exclude_incident_id:
                    query = query.where(IncidentTable.incident_id != exclude_incident_id)
                
                result = await session.execute(query)
                incident_records = result.scalars().all()
                
                incidents = [Incident.model_validate(record) for record in incident_records]
                logger.debug(f"Retrieved {len(incidents)} historical incidents from database")
                return incidents
                
        except Exception as e:
            logger.error(f"Error fetching historical incidents: {e}")
            return []
    
    async def _get_all_incidents(self) -> List[Incident]:
        """Get all incidents from database."""
        async with get_database() as session:
            try:
                result = await session.execute(
                    select(IncidentTable).order_by(desc(IncidentTable.timestamp))
                )
                incident_records = result.scalars().all()
                
                incidents = [Incident.model_validate(record) for record in incident_records]
                logger.debug(f"Retrieved {len(incidents)} total incidents")
                return incidents
                
            except Exception as e:
                logger.error(f"Error fetching all incidents: {e}")
                return []
    
    async def _get_filtered_incidents(
        self,
        incident_types: Optional[List[str]] = None,
        severity_levels: Optional[List[IncidentSeverity]] = None,
        limit: int = 1000
    ) -> List[Incident]:
        """Get filtered incidents from database."""
        async with get_database() as session:
            try:
                query = select(IncidentTable).order_by(desc(IncidentTable.timestamp)).limit(limit)
                
                if severity_levels:
                    query = query.where(IncidentTable.severity.in_([s.value for s in severity_levels]))
                
                # Note: incident_types would need to be added to the database schema
                # For now, we'll filter by service_name as a proxy
                if incident_types:
                    query = query.where(IncidentTable.service_name.in_(incident_types))
                
                result = await session.execute(query)
                incident_records = result.scalars().all()
                
                incidents = [Incident.model_validate(record) for record in incident_records]
                logger.debug(f"Retrieved {len(incidents)} filtered incidents")
                return incidents
                
            except Exception as e:
                logger.error(f"Error fetching filtered incidents: {e}")
                return []
    
    async def _get_recent_incidents(self, days: int = 30) -> List[Incident]:
        """Get recent incidents from database."""
        async with get_database() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                result = await session.execute(
                    select(IncidentTable).where(
                        IncidentTable.timestamp >= cutoff_date
                    ).order_by(desc(IncidentTable.timestamp))
                )
                incident_records = result.scalars().all()
                
                incidents = [Incident.model_validate(record) for record in incident_records]
                logger.debug(f"Retrieved {len(incidents)} recent incidents from last {days} days")
                return incidents
                
            except Exception as e:
                logger.error(f"Error fetching recent incidents: {e}")
                return []
    
    async def _get_sample_incident_texts(self, limit: int = 100) -> List[str]:
        """Get sample incident texts for TF-IDF fitting."""
        try:
            incidents = await self._get_recent_incidents(days=90)
            if not incidents:
                incidents = await self._get_historical_incidents(limit=limit)
            
            texts = [self._incident_to_text(incident) for incident in incidents[:limit]]
            return [text for text in texts if text.strip()]
            
        except Exception as e:
            logger.error(f"Error getting sample texts: {e}")
            return []
    
    async def _analyze_cluster(
        self,
        cluster_incidents: List[Incident],
        cluster_indices: List[int],
        similarity_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze a cluster of similar incidents."""
        try:
            # Calculate cluster statistics
            avg_similarity = np.mean([
                similarity_matrix[i][j] 
                for i in cluster_indices 
                for j in cluster_indices 
                if i != j
            ]) if len(cluster_indices) > 1 else 0.0
            
            # Analyze common patterns
            services = [inc.service_name for inc in cluster_incidents]
            severities = [inc.severity for inc in cluster_incidents]
            root_causes = [inc.root_cause for inc in cluster_incidents if inc.root_cause]
            
            from collections import Counter
            service_counts = Counter(services)
            severity_counts = Counter(severities)
            root_cause_counts = Counter(root_causes) if root_causes else Counter()
            
            # Time analysis
            timestamps = [inc.timestamp for inc in cluster_incidents]
            time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else timedelta(0)
            
            # MTTR analysis
            resolved_incidents = [inc for inc in cluster_incidents if inc.mttr_minutes]
            avg_mttr = np.mean([inc.mttr_minutes for inc in resolved_incidents]) if resolved_incidents else None
            
            cluster_info = {
                "cluster_id": str(uuid4())[:8],
                "incident_count": len(cluster_incidents),
                "avg_similarity": float(avg_similarity),
                "incident_ids": [inc.incident_id for inc in cluster_incidents],
                "common_services": dict(service_counts.most_common(3)),
                "severity_distribution": dict(severity_counts),
                "common_root_causes": dict(root_cause_counts.most_common(3)),
                "time_span_days": time_span.days,
                "avg_mttr_minutes": avg_mttr,
                "pattern_strength": self._calculate_pattern_strength(cluster_incidents),
                "recommended_actions": self._generate_cluster_recommendations(cluster_incidents)
            }
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Error analyzing cluster: {e}")
            return {
                "cluster_id": str(uuid4())[:8],
                "incident_count": len(cluster_incidents),
                "error": str(e)
            }
    
    def _calculate_pattern_strength(self, incidents: List[Incident]) -> float:
        """Calculate how strong the pattern is in a cluster."""
        try:
            if len(incidents) < 2:
                return 0.0
            
            # Factors for pattern strength
            factors = []
            
            # Service consistency
            services = [inc.service_name for inc in incidents]
            service_consistency = len(set(services)) / len(services)
            factors.append(1 - service_consistency)  # Higher consistency = stronger pattern
            
            # Root cause consistency
            root_causes = [inc.root_cause for inc in incidents if inc.root_cause]
            if root_causes:
                cause_consistency = len(set(root_causes)) / len(root_causes)
                factors.append(1 - cause_consistency)
            
            # Severity consistency
            severities = [inc.severity for inc in incidents]
            severity_consistency = len(set(severities)) / len(severities)
            factors.append(1 - severity_consistency)
            
            # Frequency (more incidents = stronger pattern)
            frequency_score = min(1.0, len(incidents) / 10)  # Cap at 10 incidents
            factors.append(frequency_score)
            
            return sum(factors) / len(factors)
            
        except Exception as e:
            logger.error(f"Error calculating pattern strength: {e}")
            return 0.0
    
    def _generate_cluster_recommendations(self, incidents: List[Incident]) -> List[str]:
        """Generate recommendations based on incident cluster analysis."""
        recommendations = []
        
        try:
            # Analyze common root causes
            root_causes = [inc.root_cause for inc in incidents if inc.root_cause]
            if root_causes:
                from collections import Counter
                common_causes = Counter(root_causes).most_common(1)
                if common_causes:
                    most_common_cause = common_causes[0][0]
                    recommendations.append(f"Investigate recurring issue: {most_common_cause}")
            
            # Service-specific recommendations
            services = [inc.service_name for inc in incidents]
            service_counts = Counter(services)
            
            for service, count in service_counts.most_common(2):
                if count > 1:
                    recommendations.append(f"Review {service} service for systemic issues ({count} incidents)")
            
            # Severity-based recommendations
            critical_incidents = [inc for inc in incidents if inc.severity == IncidentSeverity.CRITICAL]
            if len(critical_incidents) > len(incidents) * 0.5:
                recommendations.append("High number of critical incidents - consider infrastructure review")
            
            # MTTR-based recommendations
            high_mttr_incidents = [inc for inc in incidents if inc.mttr_minutes and inc.mttr_minutes > 60]
            if len(high_mttr_incidents) > len(incidents) * 0.3:
                recommendations.append("Many incidents with high MTTR - improve debugging procedures")
            
            # Temporal recommendations
            timestamps = [inc.timestamp for inc in incidents]
            if len(timestamps) > 1:
                time_span = max(timestamps) - min(timestamps)
                if time_span.days < 7:
                    recommendations.append("Incidents clustered in time - investigate recent changes")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating cluster recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache and index."""
        try:
            stats = {
                "cache_size": len(self.embedding_cache),
                "model_type": "sentence_transformer" if self.embedding_model else "tfidf",
                "model_name": settings.EMBEDDING_MODEL if self.embedding_model else "TF-IDF",
                "cache_file_exists": self.cache_file.exists(),
                "similarity_threshold": self.similarity_threshold
            }
            
            if self.cache_file.exists():
                stats["cache_file_size_mb"] = self.cache_file.stat().st_size / (1024 * 1024)
            
            # Get some sample embeddings to check dimensionality
            if self.embedding_cache:
                sample_embedding = list(self.embedding_cache.values())[0]
                stats["embedding_dimension"] = len(sample_embedding)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {"error": str(e)}
    
    async def clear_cache(self) -> bool:
        """Clear the embedding cache."""
        try:
            self.embedding_cache.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()
            logger.info("Embedding cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False