## **src/__init__.py**
```python
"""
Remote Photoplethysmography with Self-Supervised Learning
========================================================

A PyTorch implementation of self-supervised learning for remote photoplethysmography.
Developed for healthcare AI research - demonstration purposes only.

Modules:
    models: Neural network architectures
    training: Training utilities and distributed training
    data: Dataset handling and preprocessing
    utils: Utility functions for signal processing
"""

__version__ = "1.0.0"
__author__ = "Healthcare AI Research Team"

from . import models
from . import training
from . import data
from . import utils

__all__ = ["models", "training", "data", "utils"]
