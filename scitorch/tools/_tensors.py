"""Small tools for tensor manipulation/creation."""

import torch
from sympy.core.compatibility import exec_

from scitorch._device import DEVICE

def _create_tensor(val):
    return torch.as_tensor(val, device=DEVICE, dtype=torch.float64)

def _get_local_variables():
    _locals = {}
    exec_('from sympy.abc import *', _locals)
    return _locals


