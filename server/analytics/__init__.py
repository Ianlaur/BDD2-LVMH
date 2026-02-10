"""
ML Analytics Layer - Post-Extraction Intelligence

This module provides advanced analytics on extracted concepts:
1. Client Clustering & Similarity
2. Predictive Analytics (purchase likelihood, churn, CLV)
3. Recommendation Engine
"""

from .clustering import ClientClusterer
from .predictions import PredictiveAnalytics
from .recommendations import RecommendationEngine

__all__ = [
    'ClientClusterer',
    'PredictiveAnalytics', 
    'RecommendationEngine'
]
