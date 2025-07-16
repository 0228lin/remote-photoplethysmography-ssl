"""
Model architectures for remote photoplethysmography.

This module contains neural network architectures for rPPG signal extraction,
including the PhysNet backbone and frequency contrast learning modules.
"""

from .physnet import PhysNet
from .frequency_contrast import FrequencyContrast, FrequencyRankLoss

__all__ = ["PhysNet", "FrequencyContrast", "FrequencyRankLoss"]
