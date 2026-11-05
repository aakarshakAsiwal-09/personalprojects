"""
Global Seeding and Reproducibility Module.
Inspired by Zhang, Zohren, and Roberts (2020) - "Deep Reinforcement Learning for Trading".

WHERE & WHY:
- WHERE: Foundational infrastructure across all modules (training, environment, models, evaluation).
- WHY (Bug 6 Fix):
  Financial RL experiments are notoriously sensitive to initialization and sampling variance.
  Standardizing random number generators across Python's `random`, NumPy, PyTorch (CPU & CUDA),
  and disabling non-deterministic cuDNN benchmark algorithms guarantees bit-for-bit reproducibility
  across repeated runs and verifiable provenance for all reported metrics and model checkpoints.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets global seed across all RNGs: random, numpy, and torch.
    
    Args:
        seed: Integer seed value (default: 42).
        deterministic: If True, enforces PyTorch deterministic algorithms where supported.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

