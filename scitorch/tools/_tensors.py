"""Small tools for tensor manipulation/creation."""

import torch

from scitorch._device import DEVICE as device

def T(val):
    return torch.as_tensor(val, device=device, dtype=torch.float64)