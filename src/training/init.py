"""
Training utilities for distributed self-supervised learning.

This module provides training loops, loss functions, and distributed
training utilities for remote photoplethysmography models.
"""

from .distributed_utils import (
    init_distributed_mode, cleanup, is_main_process, 
    get_rank, get_world_size
)
from .losses import ContrastLoss, RankLoss, SNRLoss, EMDLoss
from .trainer import SelfSupervisedTrainer

__all__ = [
    "init_distributed_mode", "cleanup", "is_main_process",
    "get_rank", "get_world_size", "ContrastLoss", "RankLoss", 
    "SNRLoss", "EMDLoss", "SelfSupervisedTrainer"
]
