"""
Expert Search Service - LLM-Powered Dynamic Search with FAISS.

This module provides intelligent expert search using:
1. FAISS for fast candidate retrieval
2. OpenAI LLM for query understanding and expert evaluation

Usage:
    from utils.expert_search import ExpertSearchService

    service = ExpertSearchService()
    results = service.search("I need someone who can build ML pipelines for healthcare")
"""

import os
import sys
import pickle
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load environment variables
load_dotenv(BASE_DIR / 'searchengine' / '.env')
load_dotenv(BASE_DIR / '.env')

import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Search configuration."""
    data_dir: Path = BASE_DIR / 'searchengine' / 'data'
    embedding_model: str = 'all-MiniLM-L6-v2'
    openai_model: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    faiss_candidates: int = int(os.getenv('FAISS_CANDIDATES', 100))
    top_k: int = int(os.getenv('SEARCH_TOP_K', 20))

    def __post_init__(self):
        self.embeddings_file = self.data_dir / 'expert_embeddings.pkl'
        self.metadata_file = self.data_dir / 'expert_metadata.pkl'
        self.faiss_index_file = self.data_dir / 'expert_faiss.index'


CONFIG = SearchConfig()


# =============================================================================
# MODEL LOADER (Singleton)
# =============================================================================

class ModelLoader:
    """Singleton for loading and caching models."""

    _instance = None
    _embedding_model = None
    _faiss_index = None
    _metadata = None
    _openai_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            print(f"Loading embedding model: {CONFIG.embedding_model}")
            self._embedding_model = SentenceTransformer(CONFIG.embedding_model)
        return self._embedding_model

    def get_faiss_index(self) -> faiss.Index:
        if self._faiss_index is None:
            if not CONFIG.faiss_index_file.exists():
                raise FileNotFoundError(f"FAISS index not found: {CONFIG.faiss_index_file}")
            print(f"Loading FAISS index...")
            self._faiss_index = faiss.read_index(str(CONFIG.faiss_index_file))
        return self._faiss_index

    def get_metadata(self) -> List[Dict[str, Any]]:
        if self._metadata is None:
            if not CONFIG.metadata_file.exists():
                raise FileNotFoundError(f"Metadata not found: {CONFIG.metadata_file}")
            with open(CONFIG.metadata_file, 'rb') as f:
                self._metadata = pickle.load(f)
        return self._metadata

    def get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client


MODELS = ModelLoader()


# =============================================================================
# LLM PROMPTS
# =============================================================================

QUERY_UNDERSTANDING_PROMPT = """You are an expert search assistant. Analyze the user's search query and extract key requirements.

User Query: {query}

Extract and return a JSON object with:
{{
    "interpreted_need": "Brief description of what the user is looking for",
    "required_skills": ["list", "of", "key", "skills"],
    "preferred_experience": "type of experience needed",
    "industry_focus": "relevant industry/sector if mentioned",
    "seniority_level": "junior/mid/senior/any",
    "keywords": ["additional", "search", "keywords"],
    "search_query": "optimized search query for semantic search"
}}

Return ONLY valid JSON, no other text."""


EXPERT_RANKING_PROMPT = """You are an expert matching assistant. Evaluate how well each expert matches the user's requirements.

USER REQUIREMENTS:
{requirements}

CANDIDATE EXPERTS:
{experts}

For each expert, provide a relevance score (0-100) and brief explanation.

Return a JSON array with format:
[
    {{
        "uid": "expert_uid",
        "score": 85,
        "match_reasons": ["reason1", "reason2"],
        "gaps": ["missing skill or experience if any"]
    }}
]

Rank by how well the expert matches the specific requirements. Be strict - only high scores for strong matches.
Return ONLY valid JSON array, no other text."""


EXPERT_SUMMARY_PROMPT = """Based on the search results, provide a brief summary.

User was looking for: {query}

Top matches found:
{experts}

Provide a 2-3 sentence summary of the search results and key findings. Be concise."""


# =============================================================================
# EXPERT SEARCH SERVICE
# =============================================================================

class ExpertSearchService:
    """
    LLM-Powered Expert Search Service.

    Pipeline:
    1. LLM understands the query and extracts requirements
    2. FAISS retrieves initial candidates using semantic similarity
    3. LLM evaluates and ranks candidates based on requirements
    4. Returns ranked experts with explanations
    """

    def __init__(self, config: SearchConfig = None):
        self.config = config or CONFIG
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load all models."""
        if not self._initialized:
            MODELS.get_embedding_model()
            MODELS.get_faiss_index()
            MODELS.get_metadata()
            MODELS.get_openai_client()
            self._initialized = True

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Call OpenAI LLM."""
        client = MODELS.get_openai_client()

        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns only valid JSON when asked."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=2000
        )

        return response.choices[0].message.content.strip()

    def _understand_query(self, query: str) -> Dict[str, Any]:
        """Use LLM to understand and parse the query."""
        prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query)

        try:
            response = self._call_llm(prompt)
            # Clean response - remove markdown code blocks if present
            response = response.replace('```json', '').replace('```', '').strip()
            return json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Query understanding error: {e}")
            # Fallback to basic parsing
            return {
                "interpreted_need": query,
                "required_skills": [],
                "preferred_experience": "",
                "industry_focus": "",
                "seniority_level": "any",
                "keywords": query.split(),
                "search_query": query
            }

    def _faiss_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve candidates using FAISS."""
        model = MODELS.get_embedding_model()
        index = MODELS.get_faiss_index()
        metadata = MODELS.get_metadata()

        # Encode query
        query_embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)

        # Search
        k = min(k, index.ntotal)
        scores, indices = index.search(query_embedding, k)

        # Get candidates
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(metadata):
                expert = metadata[idx].copy()
                expert['faiss_score'] = float(score)
                candidates.append(expert)

        return candidates

    def _format_expert_for_llm(self, expert: Dict[str, Any]) -> str:
        """Format expert data for LLM evaluation."""
        return f"""
UID: {expert.get('uid', 'N/A')}
Name: {expert.get('first_name', '')} {expert.get('surname', '')}
Title: {expert.get('title', 'N/A')}
Skills: {expert.get('skills', 'N/A')}
Position: {expert.get('position', 'N/A')}
Companies: {expert.get('companies', 'N/A')}
Sector: {expert.get('sector', 'N/A')}
Location: {expert.get('city', '')}, {expert.get('country', '')}
Bio: {expert.get('biography', 'N/A')[:300]}...
"""

    def _rank_with_llm(
        self,
        requirements: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to rank candidates based on requirements."""

        # Format experts for prompt
        experts_text = "\n---\n".join([
            self._format_expert_for_llm(e) for e in candidates[:30]  # Limit for token size
        ])

        requirements_text = json.dumps(requirements, indent=2)

        prompt = EXPERT_RANKING_PROMPT.format(
            requirements=requirements_text,
            experts=experts_text
        )

        try:
            response = self._call_llm(prompt, temperature=0.2)
            response = response.replace('```json', '').replace('```', '').strip()
            rankings = json.loads(response)

            # Create lookup for rankings
            ranking_map = {r['uid']: r for r in rankings}

            # Merge rankings with candidate data
            ranked_results = []
            for expert in candidates:
                uid = expert.get('uid', '')
                if uid in ranking_map:
                    expert['llm_score'] = ranking_map[uid].get('score', 50)
                    expert['match_reasons'] = ranking_map[uid].get('match_reasons', [])
                    expert['gaps'] = ranking_map[uid].get('gaps', [])
                else:
                    expert['llm_score'] = 50
                    expert['match_reasons'] = []
                    expert['gaps'] = []

                # Combined score (LLM weighted higher)
                expert['final_score'] = (
                    expert['llm_score'] * 0.7 +
                    expert.get('faiss_score', 0) * 100 * 0.3
                )
                ranked_results.append(expert)

            # Sort by final score
            ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
            return ranked_results

        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM ranking error: {e}")
            # Fallback to FAISS scores
            for expert in candidates:
                expert['llm_score'] = 50
                expert['final_score'] = expert.get('faiss_score', 0) * 100
                expert['match_reasons'] = []
                expert['gaps'] = []

            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            return candidates

    def search(
        self,
        query: str,
        top_k: int = None,
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Perform intelligent expert search.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            include_analysis: Include LLM analysis in response

        Returns:
            Dict with query_analysis, results, and summary
        """
        self._ensure_initialized()

        if not query or not query.strip():
            return {
                'success': False,
                'error': 'Query is required',
                'results': []
            }

        top_k = top_k or self.config.top_k

        # Step 1: Understand the query
        print(f"Understanding query: {query}")
        query_analysis = self._understand_query(query)

        # Step 2: Get FAISS candidates using optimized search query
        search_query = query_analysis.get('search_query', query)
        print(f"Searching with: {search_query}")
        candidates = self._faiss_search(search_query, self.config.faiss_candidates)

        if not candidates:
            return {
                'success': True,
                'query': query,
                'query_analysis': query_analysis if include_analysis else None,
                'total_results': 0,
                'results': [],
                'summary': 'No matching experts found.'
            }

        # Step 3: Rank with LLM
        print(f"Ranking {len(candidates)} candidates with LLM...")
        ranked_results = self._rank_with_llm(query_analysis, candidates)

        # Step 4: Format final results
        final_results = []
        for i, expert in enumerate(ranked_results[:top_k]):
            final_results.append({
                'rank': i + 1,
                'uid': expert.get('uid', ''),
                'name': f"{expert.get('first_name', '')} {expert.get('surname', '')}".strip(),
                'title': expert.get('title', ''),
                'skills': expert.get('skills', ''),
                'biography': expert.get('biography', ''),
                'location': f"{expert.get('city', '')}, {expert.get('country', '')}".strip(', '),
                'email': expert.get('email', ''),
                'companies': expert.get('companies', ''),
                'sector': expert.get('sector', ''),
                'external_link': expert.get('external_link', ''),
                'scores': {
                    'final': round(expert.get('final_score', 0), 2),
                    'llm': expert.get('llm_score', 0),
                    'semantic': round(expert.get('faiss_score', 0) * 100, 2)
                },
                'match_reasons': expert.get('match_reasons', []),
                'gaps': expert.get('gaps', [])
            })

        return {
            'success': True,
            'query': query,
            'query_analysis': query_analysis if include_analysis else None,
            'total_results': len(final_results),
            'results': final_results
        }

    def search_fast(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Fast search without LLM (FAISS only).
        Use for autocomplete or quick searches.
        """
        self._ensure_initialized()

        if not query or not query.strip():
            return {'success': False, 'error': 'Query required', 'results': []}

        candidates = self._faiss_search(query, top_k)

        results = []
        for i, expert in enumerate(candidates):
            results.append({
                'rank': i + 1,
                'uid': expert.get('uid', ''),
                'name': f"{expert.get('first_name', '')} {expert.get('surname', '')}".strip(),
                'title': expert.get('title', ''),
                'skills': expert.get('skills', ''),
                'location': f"{expert.get('city', '')}, {expert.get('country', '')}".strip(', '),
                'score': round(expert.get('faiss_score', 0) * 100, 2)
            })

        return {
            'success': True,
            'query': query,
            'total_results': len(results),
            'results': results
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_search_service = None

def get_search_service() -> ExpertSearchService:
    """Get or create global search service."""
    global _search_service
    if _search_service is None:
        _search_service = ExpertSearchService()
    return _search_service


def search_experts(query: str, top_k: int = 20) -> Dict[str, Any]:
    """Search experts with LLM-powered ranking."""
    return get_search_service().search(query, top_k=top_k)


def search_experts_fast(query: str, top_k: int = 20) -> Dict[str, Any]:
    """Fast search without LLM."""
    return get_search_service().search_fast(query, top_k=top_k)


# =============================================================================
# CLI TEST
# =============================================================================

def main():
    """Test the search service."""
    print("=" * 60)
    print("Expert Search Service - LLM Powered")
    print("=" * 60)

    service = ExpertSearchService()

    # Test query
    query = "I need a machine learning expert who has experience with NLP and Python"
    print(f"\nQuery: {query}\n")

    results = service.search(query, top_k=5)

    if results['success']:
        print(f"Query Analysis: {json.dumps(results.get('query_analysis', {}), indent=2)}\n")
        print(f"Found {results['total_results']} results:\n")

        for r in results['results']:
            print(f"#{r['rank']} - {r['name']}")
            print(f"   Title: {r['title']}")
            print(f"   Score: {r['scores']['final']} (LLM: {r['scores']['llm']}, Semantic: {r['scores']['semantic']})")
            print(f"   Match: {', '.join(r['match_reasons'][:2])}")
            print()
    else:
        print(f"Error: {results.get('error')}")


if __name__ == '__main__':
    main()
