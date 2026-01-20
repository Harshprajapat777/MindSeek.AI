"""
Expert Search Service - FAISS-based semantic search with reranking.

This module provides a multi-layer precision search system for finding expert profiles:
1. Initial FAISS retrieval (fast approximate search)
2. Cross-encoder reranking (precise semantic matching)
3. Precision scoring layers (skill matching, keyword boosting, etc.)

Usage:
    from utils.expert_search import ExpertSearchService

    search_service = ExpertSearchService()
    results = search_service.search("machine learning expert with Python")
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import re

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Search configuration parameters."""
    # Paths
    data_dir: Path = BASE_DIR / 'searchengine' / 'data'
    embeddings_file: Path = None
    metadata_file: Path = None
    faiss_index_file: Path = None

    # Model names
    embedding_model: str = 'all-MiniLM-L6-v2'
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    # Search parameters
    initial_retrieval_k: int = 100  # Number of candidates from FAISS
    rerank_top_k: int = 50          # Number to rerank with cross-encoder
    final_results_k: int = 20       # Final number of results to return

    # Precision layer weights
    semantic_weight: float = 0.4    # Weight for semantic similarity
    rerank_weight: float = 0.35     # Weight for cross-encoder score
    keyword_weight: float = 0.15    # Weight for keyword matching
    skill_weight: float = 0.10      # Weight for skill matching

    # Thresholds
    min_similarity_threshold: float = 0.3
    min_rerank_threshold: float = 0.1

    def __post_init__(self):
        self.embeddings_file = self.data_dir / 'expert_embeddings.pkl'
        self.metadata_file = self.data_dir / 'expert_metadata.pkl'
        self.faiss_index_file = self.data_dir / 'expert_faiss.index'


# Global config instance
CONFIG = SearchConfig()


# =============================================================================
# MODEL LOADER (Singleton Pattern)
# =============================================================================

class ModelLoader:
    """Singleton class for loading and caching ML models."""

    _instance = None
    _embedding_model = None
    _reranker_model = None
    _faiss_index = None
    _embeddings_data = None
    _metadata = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embedding_model(self) -> SentenceTransformer:
        """Load and cache the embedding model."""
        if self._embedding_model is None:
            print(f"Loading embedding model: {CONFIG.embedding_model}")
            self._embedding_model = SentenceTransformer(CONFIG.embedding_model)
        return self._embedding_model

    def get_reranker_model(self) -> CrossEncoder:
        """Load and cache the reranker model."""
        if self._reranker_model is None:
            print(f"Loading reranker model: {CONFIG.reranker_model}")
            self._reranker_model = CrossEncoder(CONFIG.reranker_model)
        return self._reranker_model

    def get_faiss_index(self) -> faiss.Index:
        """Load and cache the FAISS index."""
        if self._faiss_index is None:
            if not CONFIG.faiss_index_file.exists():
                raise FileNotFoundError(
                    f"FAISS index not found: {CONFIG.faiss_index_file}\n"
                    "Please run build_faiss.py first."
                )
            print(f"Loading FAISS index from: {CONFIG.faiss_index_file}")
            self._faiss_index = faiss.read_index(str(CONFIG.faiss_index_file))
            print(f"FAISS index loaded. Total vectors: {self._faiss_index.ntotal}")
        return self._faiss_index

    def get_embeddings_data(self) -> Dict[str, Any]:
        """Load and cache embeddings data."""
        if self._embeddings_data is None:
            if not CONFIG.embeddings_file.exists():
                raise FileNotFoundError(
                    f"Embeddings file not found: {CONFIG.embeddings_file}\n"
                    "Please run generate_embeddings.py first."
                )
            with open(CONFIG.embeddings_file, 'rb') as f:
                self._embeddings_data = pickle.load(f)
        return self._embeddings_data

    def get_metadata(self) -> List[Dict[str, Any]]:
        """Load and cache metadata."""
        if self._metadata is None:
            if not CONFIG.metadata_file.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {CONFIG.metadata_file}\n"
                    "Please run generate_embeddings.py first."
                )
            with open(CONFIG.metadata_file, 'rb') as f:
                self._metadata = pickle.load(f)
        return self._metadata


# Global model loader instance
MODEL_LOADER = ModelLoader()


# =============================================================================
# PRECISION LAYERS
# =============================================================================

class PrecisionLayer:
    """Base class for precision scoring layers."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def score(self, query: str, expert: Dict[str, Any]) -> float:
        """Calculate score for this layer. Override in subclasses."""
        raise NotImplementedError


class KeywordMatchLayer(PrecisionLayer):
    """Precision layer for keyword matching."""

    def __init__(self, weight: float = CONFIG.keyword_weight):
        super().__init__(weight)

    def score(self, query: str, expert: Dict[str, Any]) -> float:
        """Score based on keyword overlap between query and expert profile."""
        # Extract query keywords (lowercase, remove common words)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                      'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                      'very', 'just', 'expert', 'experience', 'looking', 'find',
                      'search', 'need', 'want', 'who', 'what', 'where', 'when'}

        # Tokenize query
        query_tokens = set(
            token.lower() for token in re.findall(r'\b\w+\b', query)
            if token.lower() not in stop_words and len(token) > 2
        )

        if not query_tokens:
            return 0.0

        # Combine expert fields for matching
        expert_text = ' '.join([
            str(expert.get('title', '') or ''),
            str(expert.get('skills', '') or ''),
            str(expert.get('biography', '') or ''),
            str(expert.get('position', '') or ''),
            str(expert.get('companies', '') or ''),
            str(expert.get('sector', '') or ''),
        ]).lower()

        # Count matches
        matches = sum(1 for token in query_tokens if token in expert_text)

        # Return normalized score
        return matches / len(query_tokens)


class SkillMatchLayer(PrecisionLayer):
    """Precision layer for skill matching."""

    def __init__(self, weight: float = CONFIG.skill_weight):
        super().__init__(weight)

        # Common skill synonyms and related terms
        self.skill_synonyms = {
            'ml': ['machine learning', 'ml', 'deep learning', 'ai'],
            'python': ['python', 'py', 'pytorch', 'tensorflow', 'keras'],
            'data science': ['data science', 'data scientist', 'analytics', 'data analysis'],
            'nlp': ['nlp', 'natural language processing', 'text mining', 'language model'],
            'cv': ['computer vision', 'cv', 'image processing', 'object detection'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker'],
            'frontend': ['react', 'vue', 'angular', 'javascript', 'typescript', 'frontend'],
            'backend': ['backend', 'api', 'rest', 'graphql', 'microservices'],
            'database': ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'database'],
        }

    def _expand_query_skills(self, query: str) -> set:
        """Expand query with related skill terms."""
        query_lower = query.lower()
        expanded_skills = set()

        for skill_group, terms in self.skill_synonyms.items():
            if any(term in query_lower for term in terms):
                expanded_skills.update(terms)

        return expanded_skills

    def score(self, query: str, expert: Dict[str, Any]) -> float:
        """Score based on skill matching."""
        skills_text = str(expert.get('skills', '') or '').lower()

        if not skills_text:
            return 0.0

        # Get expanded query skills
        query_skills = self._expand_query_skills(query)

        if not query_skills:
            return 0.0

        # Count matches
        matches = sum(1 for skill in query_skills if skill in skills_text)

        return min(1.0, matches / len(query_skills))


class TitleMatchLayer(PrecisionLayer):
    """Precision layer for title/role matching."""

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def score(self, query: str, expert: Dict[str, Any]) -> float:
        """Score based on title relevance."""
        title = str(expert.get('title', '') or '').lower()
        query_lower = query.lower()

        if not title:
            return 0.0

        # Check for direct matches
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        title_words = set(re.findall(r'\b\w+\b', title))

        # Calculate Jaccard similarity
        if not query_words or not title_words:
            return 0.0

        intersection = query_words & title_words
        union = query_words | title_words

        return len(intersection) / len(union) if union else 0.0


# =============================================================================
# SEARCH RESULT
# =============================================================================

@dataclass
class SearchResult:
    """Represents a single search result."""
    expert: Dict[str, Any]
    semantic_score: float
    rerank_score: float
    precision_scores: Dict[str, float]
    final_score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'rank': self.rank,
            'uid': self.expert.get('uid', ''),
            'name': f"{self.expert.get('first_name', '')} {self.expert.get('surname', '')}".strip(),
            'title': self.expert.get('title', ''),
            'skills': self.expert.get('skills', ''),
            'biography': self.expert.get('biography', ''),
            'location': f"{self.expert.get('city', '')}, {self.expert.get('country', '')}".strip(', '),
            'email': self.expert.get('email', ''),
            'companies': self.expert.get('companies', ''),
            'sector': self.expert.get('sector', ''),
            'external_link': self.expert.get('external_link', ''),
            'scores': {
                'semantic': round(self.semantic_score, 4),
                'rerank': round(self.rerank_score, 4),
                'precision': {k: round(v, 4) for k, v in self.precision_scores.items()},
                'final': round(self.final_score, 4),
            }
        }


# =============================================================================
# EXPERT SEARCH SERVICE
# =============================================================================

class ExpertSearchService:
    """
    Main search service implementing multi-layer precision search.

    Search Pipeline:
    1. Query Embedding: Convert query text to embedding vector
    2. FAISS Retrieval: Fast approximate search to get top-k candidates
    3. Cross-Encoder Reranking: Precise semantic scoring of candidates
    4. Precision Layers: Apply additional scoring (keywords, skills, etc.)
    5. Final Ranking: Weighted combination of all scores
    """

    def __init__(self, config: SearchConfig = None):
        """Initialize the search service."""
        self.config = config or CONFIG

        # Initialize precision layers
        self.precision_layers = [
            KeywordMatchLayer(weight=self.config.keyword_weight),
            SkillMatchLayer(weight=self.config.skill_weight),
            TitleMatchLayer(weight=0.05),
        ]

        # Lazy loading - models loaded on first search
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure all models and data are loaded."""
        if not self._initialized:
            # Trigger model loading
            MODEL_LOADER.get_embedding_model()
            MODEL_LOADER.get_faiss_index()
            MODEL_LOADER.get_metadata()
            self._initialized = True

    def _encode_query(self, query: str) -> np.ndarray:
        """Convert query text to embedding vector."""
        model = MODEL_LOADER.get_embedding_model()
        embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32).reshape(1, -1)

    def _faiss_search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform FAISS similarity search.

        Returns:
            Tuple of (distances/scores, indices)
        """
        index = MODEL_LOADER.get_faiss_index()

        # Ensure k doesn't exceed total vectors
        k = min(k, index.ntotal)

        # Search FAISS index
        distances, indices = index.search(query_embedding, k)

        return distances[0], indices[0]

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Tuple[int, float, Dict[str, Any]]]
    ) -> List[Tuple[int, float, float, Dict[str, Any]]]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Original query text
            candidates: List of (index, semantic_score, expert_data) tuples

        Returns:
            List of (index, semantic_score, rerank_score, expert_data) tuples
        """
        if not candidates:
            return []

        reranker = MODEL_LOADER.get_reranker_model()

        # Prepare query-document pairs for cross-encoder
        pairs = []
        for idx, sem_score, expert in candidates:
            # Create document text from expert profile
            doc_text = ' '.join([
                str(expert.get('title', '') or ''),
                str(expert.get('biography', '') or ''),
                str(expert.get('skills', '') or ''),
                str(expert.get('position', '') or ''),
            ])[:512]  # Truncate for cross-encoder
            pairs.append([query, doc_text])

        # Get reranker scores
        rerank_scores = reranker.predict(pairs)

        # Normalize rerank scores to 0-1 range using sigmoid
        rerank_scores = 1 / (1 + np.exp(-rerank_scores))

        # Combine with original data
        reranked = []
        for i, (idx, sem_score, expert) in enumerate(candidates):
            reranked.append((idx, sem_score, float(rerank_scores[i]), expert))

        return reranked

    def _apply_precision_layers(
        self,
        query: str,
        expert: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply all precision layers and return scores."""
        scores = {}
        for layer in self.precision_layers:
            layer_name = layer.__class__.__name__.replace('Layer', '').lower()
            scores[layer_name] = layer.score(query, expert)
        return scores

    def _calculate_final_score(
        self,
        semantic_score: float,
        rerank_score: float,
        precision_scores: Dict[str, float]
    ) -> float:
        """Calculate weighted final score."""
        # Base scores
        final = (
            semantic_score * self.config.semantic_weight +
            rerank_score * self.config.rerank_weight
        )

        # Add precision layer scores
        for layer in self.precision_layers:
            layer_name = layer.__class__.__name__.replace('Layer', '').lower()
            if layer_name in precision_scores:
                final += precision_scores[layer_name] * layer.weight

        return final

    def search(
        self,
        query: str,
        top_k: int = None,
        include_scores: bool = True,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Perform expert search with multi-layer precision.

        Args:
            query: Search query text
            top_k: Number of results to return (default: config.final_results_k)
            include_scores: Whether to include detailed scores
            filters: Optional filters (e.g., {'country': 'USA', 'sector': 'Technology'})

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        self._ensure_initialized()

        if not query or not query.strip():
            return []

        top_k = top_k or self.config.final_results_k
        metadata = MODEL_LOADER.get_metadata()

        # Step 1: Generate query embedding
        query_embedding = self._encode_query(query)

        # Step 2: FAISS retrieval
        scores, indices = self._faiss_search(
            query_embedding,
            self.config.initial_retrieval_k
        )

        # Step 3: Filter and prepare candidates
        candidates = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0 or idx >= len(metadata):
                continue

            if score < self.config.min_similarity_threshold:
                continue

            expert = metadata[idx]

            # Apply filters if provided
            if filters:
                skip = False
                for key, value in filters.items():
                    expert_value = str(expert.get(key, '')).lower()
                    if value.lower() not in expert_value:
                        skip = True
                        break
                if skip:
                    continue

            candidates.append((idx, float(score), expert))

        # Limit candidates for reranking
        candidates = candidates[:self.config.rerank_top_k]

        if not candidates:
            return []

        # Step 4: Cross-encoder reranking
        reranked_candidates = self._rerank_candidates(query, candidates)

        # Step 5: Apply precision layers and calculate final scores
        results = []
        for idx, sem_score, rerank_score, expert in reranked_candidates:
            # Skip if rerank score is too low
            if rerank_score < self.config.min_rerank_threshold:
                continue

            # Get precision layer scores
            precision_scores = self._apply_precision_layers(query, expert)

            # Calculate final score
            final_score = self._calculate_final_score(
                sem_score, rerank_score, precision_scores
            )

            results.append(SearchResult(
                expert=expert,
                semantic_score=sem_score,
                rerank_score=rerank_score,
                precision_scores=precision_scores,
                final_score=final_score,
                rank=0  # Will be set after sorting
            ))

        # Step 6: Sort by final score and assign ranks
        results.sort(key=lambda x: x.final_score, reverse=True)

        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1

        return results[:top_k]

    def search_simple(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Simple search returning dictionary results.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of expert dictionaries with scores
        """
        results = self.search(query, top_k=top_k)
        return [r.to_dict() for r in results]

    def search_fast(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Fast search without reranking (FAISS only).

        Useful for:
        - Autocomplete suggestions
        - Quick previews
        - High-volume requests

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of expert dictionaries with semantic scores only
        """
        self._ensure_initialized()

        if not query or not query.strip():
            return []

        metadata = MODEL_LOADER.get_metadata()

        # Generate query embedding
        query_embedding = self._encode_query(query)

        # FAISS search
        scores, indices = self._faiss_search(query_embedding, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            if idx < 0 or idx >= len(metadata):
                continue

            if score < self.config.min_similarity_threshold:
                continue

            expert = metadata[idx]
            results.append({
                'rank': rank,
                'uid': expert.get('uid', ''),
                'name': f"{expert.get('first_name', '')} {expert.get('surname', '')}".strip(),
                'title': expert.get('title', ''),
                'skills': expert.get('skills', ''),
                'location': f"{expert.get('city', '')}, {expert.get('country', '')}".strip(', '),
                'score': round(float(score), 4),
            })

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global search service instance
_search_service = None


def get_search_service() -> ExpertSearchService:
    """Get or create the global search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = ExpertSearchService()
    return _search_service


def search_experts(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Convenience function for expert search.

    Args:
        query: Search query text
        top_k: Number of results to return

    Returns:
        List of expert dictionaries with scores
    """
    service = get_search_service()
    return service.search_simple(query, top_k=top_k)


def search_experts_fast(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Convenience function for fast expert search (no reranking).

    Args:
        query: Search query text
        top_k: Number of results to return

    Returns:
        List of expert dictionaries with semantic scores
    """
    service = get_search_service()
    return service.search_fast(query, top_k=top_k)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for testing search."""
    import json

    print("=" * 60)
    print("Expert Search Service - Test Interface")
    print("=" * 60)

    # Initialize service
    print("\nInitializing search service...")
    service = ExpertSearchService()

    # Test queries
    test_queries = [
        "machine learning expert with Python experience",
        "data scientist with NLP skills",
        "cloud architect AWS Azure",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        results = service.search_simple(query, top_k=5)

        for result in results:
            print(f"\n#{result['rank']} - {result['name']}")
            print(f"   Title: {result['title']}")
            print(f"   Location: {result['location']}")
            print(f"   Skills: {result['skills'][:100]}..." if result['skills'] else "   Skills: N/A")
            print(f"   Scores: Final={result['scores']['final']:.3f}, "
                  f"Semantic={result['scores']['semantic']:.3f}, "
                  f"Rerank={result['scores']['rerank']:.3f}")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Search Mode (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break

            if not query:
                continue

            results = service.search_simple(query, top_k=10)

            if not results:
                print("No results found.")
                continue

            print(f"\nFound {len(results)} results:\n")
            for result in results:
                print(f"#{result['rank']} {result['name']} - {result['title']}")
                print(f"   Score: {result['scores']['final']:.3f} | {result['location']}")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


if __name__ == '__main__':
    main()
