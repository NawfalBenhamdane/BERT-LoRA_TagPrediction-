"""
RAG+BERT Code Classification System
A retrieval-augmented system for multi-label classification of coding problems.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import Config
from .models import ContrastiveEncoder, CrossEncoder
from .predictor import RetrievalAugmentedPredictor

__all__ = [
    "Config",
    "ContrastiveEncoder",
    "CrossEncoder",
    "RetrievalAugmentedPredictor",
]
