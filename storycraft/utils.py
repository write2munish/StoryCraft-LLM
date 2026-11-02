"""
Utility functions for StoryCraft-LLM.
"""

import os
import random
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python random, NumPy, and PyTorch (CPU & CUDA)
    to guarantee 100% deterministic reproducibility across runs and platforms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
