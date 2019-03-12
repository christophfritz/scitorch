import torch

from scitorch.tools._tensors import _create_tensor
from sympy import sympify


class Tensor:

    def __init__(self, val, dim=None):
        self.val = _create_tensor(val)

        # Sympify also allows numbers as input. Should that be restricted to strings only?
        if dim is not None:
            self.dim = sympify(dim)
        else:
            pass
