"""Test cases for the Tools module."""

import torch
import sympy

from scitorch.tools._tensors import _create_tensor
from scitorch._device import DEVICE


class TestTools(object):
    def test_create_tensor(self):
        assert torch.equal(_create_tensor(34), torch.tensor(34, device=DEVICE, dtype=torch.float64))