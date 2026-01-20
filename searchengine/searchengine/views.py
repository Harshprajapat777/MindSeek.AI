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


def home(request):
    return render(request, "Home.html")


@csrf_exempt
@require_http_methods(["POST", "GET"])
def search_api(request):
    """
    Main search API endpoint with full reranking.

    GET: /api/search?q=<query>&top_k=<number>&filters=<json>
    POST: /api/search with JSON body {"query": "...", "top_k": 20, "filters": {...}}

    Returns:
        JSON response with search results
    """
    try:
        # Parse request parameters
        if request.method == 'POST':
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                data = {}
            query = data.get('query', '')
            top_k = data.get('top_k', 20)
            filters = data.get('filters', None)
        else:  # GET
            query = request.GET.get('q', '') or request.GET.get('query', '')
            top_k = int(request.GET.get('top_k', 20))
            filters_str = request.GET.get('filters', '')
            filters = json.loads(filters_str) if filters_str else None

        # Validate query
        if not query or not query.strip():
            return JsonResponse({
                'success': False,
                'error': 'Query parameter is required',
                'results': []
            }, status=400)

        # Limit top_k to reasonable range
        top_k = max(1, min(100, top_k))

        # Perform search
        service = get_search_service()
        results = service.search(query, top_k=top_k, filters=filters)

        # Convert to dict for JSON response
        results_data = [r.to_dict() for r in results]

        return JsonResponse({
            'success': True,
            'query': query,
            'total_results': len(results_data),
            'results': results_data
        })

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
    Fast search API endpoint (FAISS only, no reranking).

    Useful for:
    - Autocomplete suggestions
    - Quick previews
    - High-volume requests

    GET: /api/search/fast?q=<query>&top_k=<number>
    POST: /api/search/fast with JSON body {"query": "...", "top_k": 20}

    Returns:
        JSON response with search results (semantic scores only)
    """
    try:
        # Parse request parameters
        if request.method == 'POST':
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                data = {}
            query = data.get('query', '')
            top_k = data.get('top_k', 20)
        else:  # GET
            query = request.GET.get('q', '') or request.GET.get('query', '')
            top_k = int(request.GET.get('top_k', 20))

        # Validate query
        if not query or not query.strip():
            return JsonResponse({
                'success': False,
                'error': 'Query parameter is required',
                'results': []
            }, status=400)

        # Limit top_k
        top_k = max(1, min(100, top_k))

        # Perform fast search
        results = search_experts_fast(query, top_k=top_k)

        return JsonResponse({
            'success': True,
            'query': query,
            'total_results': len(results),
            'results': results
        })

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
    Health check endpoint for search service.

    GET: /api/search/health

    Returns:
        JSON response with service status
    """
    try:
        service = get_search_service()
        service._ensure_initialized()

        from .utils.expert_search import MODEL_LOADER
        index = MODEL_LOADER.get_faiss_index()
        metadata = MODEL_LOADER.get_metadata()

        return JsonResponse({
            'status': 'healthy',
            'index_size': index.ntotal,
            'metadata_count': len(metadata),
            'models_loaded': True
        })

    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)
