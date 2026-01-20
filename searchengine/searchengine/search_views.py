"""
Search Views - Handles search-related page rendering.

This module contains views for:
- Search results page
- Expert profile page (future)
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import traceback

from .utils.expert_search import (
    get_search_service,
    search_experts,
    search_experts_fast,
)


def search_results_page(request):
    """
    Render the search results page.

    The actual search is performed via JavaScript calling the API.
    This view just renders the template with the query parameter.
    """
    query = request.GET.get('q', '')

    context = {
        'query': query,
    }

    return render(request, 'search_results.html', context)


@csrf_exempt
@require_http_methods(["POST", "GET"])
def search_api(request):
    """
    LLM-Powered Expert Search API.

    GET: /api/search?q=<query>&top_k=<number>
    POST: /api/search with JSON body {"query": "...", "top_k": 20}

    Returns JSON with:
    - query_analysis: LLM's understanding of the query
    - results: Ranked experts with match reasons
    """
    try:
        # Parse request
        if request.method == 'POST':
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                data = {}
            query = data.get('query', '')
            top_k = data.get('top_k', 20)
        else:
            query = request.GET.get('q', '') or request.GET.get('query', '')
            top_k = int(request.GET.get('top_k', 20))

        # Validate
        if not query or not query.strip():
            return JsonResponse({
                'success': False,
                'error': 'Query parameter is required',
                'results': []
            }, status=400)

        top_k = max(1, min(100, top_k))

        # Perform LLM-powered search
        results = search_experts(query, top_k=top_k)

        return JsonResponse(results)

    except ValueError as e:
        # OpenAI key not configured
        return JsonResponse({
            'success': False,
            'error': str(e),
            'results': []
        }, status=503)

    except FileNotFoundError as e:
        return JsonResponse({
            'success': False,
            'error': f'Search index not initialized: {str(e)}',
            'results': []
        }, status=503)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Search failed: {str(e)}',
            'results': []
        }, status=500)


@csrf_exempt
@require_http_methods(["POST", "GET"])
def search_fast_api(request):
    """
    Fast Search API (FAISS only, no LLM).

    Use for:
    - Autocomplete
    - Quick previews
    - When LLM cost is a concern

    GET: /api/search/fast?q=<query>&top_k=<number>
    POST: /api/search/fast with JSON body {"query": "...", "top_k": 20}
    """
    try:
        if request.method == 'POST':
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                data = {}
            query = data.get('query', '')
            top_k = data.get('top_k', 20)
        else:
            query = request.GET.get('q', '') or request.GET.get('query', '')
            top_k = int(request.GET.get('top_k', 20))

        if not query or not query.strip():
            return JsonResponse({
                'success': False,
                'error': 'Query parameter is required',
                'results': []
            }, status=400)

        top_k = max(1, min(100, top_k))

        results = search_experts_fast(query, top_k=top_k)

        return JsonResponse(results)

    except FileNotFoundError as e:
        return JsonResponse({
            'success': False,
            'error': f'Search index not initialized: {str(e)}',
            'results': []
        }, status=503)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Search failed: {str(e)}',
            'results': []
        }, status=500)


@require_http_methods(["GET"])
def search_health(request):
    """
    Health check for search service.

    GET: /api/search/health
    """
    import os

    try:
        service = get_search_service()

        from .utils.expert_search import MODELS, CONFIG

        # Check FAISS index
        index = MODELS.get_faiss_index()
        metadata = MODELS.get_metadata()

        # Check OpenAI key
        openai_configured = bool(os.getenv('OPENAI_API_KEY'))

        return JsonResponse({
            'status': 'healthy',
            'index_size': index.ntotal,
            'metadata_count': len(metadata),
            'openai_configured': openai_configured,
            'model': CONFIG.openai_model
        })

    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)
