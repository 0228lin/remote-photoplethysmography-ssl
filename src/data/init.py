"""
Data handling and preprocessing for remote photoplethysmography.

This module provides dataset classes, preprocessing utilities,
and data loading functionality for rPPG research.
"""

from .datasets import RPPGDataset, UBFCDataset, PUREDataset
from .preprocessing import FacePreprocessor, SignalPreprocessor

__all__ = [
    "RPPGDataset", "UBFCDataset", "PUREDataset",
    "FacePreprocessor", "SignalPreprocessor"
]
