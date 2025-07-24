"""
Belgian Document Deidentification System

A waterproof deidentification system specifically designed for Belgian healthcare 
documents containing sensitive patient data in Dutch.
"""

__version__ = "1.0.0"
__author__ = "Manus AI"
__email__ = "manus@example.com"

from .core.pipeline import DeidentificationPipeline
from .core.batch_processor import BatchProcessor
from .core.config import Config

__all__ = [
    "DeidentificationPipeline",
    "BatchProcessor", 
    "Config",
]

