"""Small tools for tensor manipulation/creation."""

import torch

from scitorch._globals import DEVICE as device

def _create_tensor(val):
    return torch.as_tensor(val, device=device, dtype=torch.float64)