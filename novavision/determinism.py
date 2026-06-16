"""Global seeding for reproducible runs."""

from __future__ import annotations

import os
import random


def set_determinism(seed: int = 0) -> None:
    """Seed python, numpy, and torch, and ask torch for deterministic kernels.

    Bit-exactness still depends on the device and dtype, so the run manifest
    records both; this pins everything that is in our control.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
