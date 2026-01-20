"""
URL configuration for searchengine project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
"""

from django.contrib import admin
from django.urls import path
from . import views
from . import search_views

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),

    # Main Pages
    path("", views.home, name="home"),
    path("search/", search_views.search_results_page, name="search_results"),

    # Search API endpoints
    path("api/search/", search_views.search_api, name="search_api"),
    path("api/search/fast/", search_views.search_fast_api, name="search_fast_api"),
    path("api/search/health/", search_views.search_health, name="search_health"),
]
