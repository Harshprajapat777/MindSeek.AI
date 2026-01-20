"""
Main Views - Handles core page rendering.

This module contains views for:
- Home page
"""

from django.shortcuts import render


def home(request):
    """Render the home page."""
    return render(request, "Home.html")
