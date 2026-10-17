"""Reproducibility and device management utilities."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for Python, NumPy, and PyTorch for deterministic results.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Select compute device based on configuration and hardware availability.

    Args:
        preference: 'auto', 'cuda', or 'cpu'.

    Returns:
        torch.device object.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preference == "cpu":
        return torch.device("cpu")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
