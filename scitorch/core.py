import torch

from scitorch.tools._tensors import _create_tensor
from sympy import sympify


class Tensor(object):
    """
    A Tensor class which holds the value as a PyTorch Tensor and the dimensionality of the (physical) object as a
    Sympy class.
    """

    def __init__(self, val, dim=None):
        self.val = _create_tensor(val)

        # Sympify also allows numbers as input. Should that be restricted to strings only?
        if dim is not None:
            self.dim = sympify(dim)
        else:
            pass

    def __eq__(self, other):
        """Overrides the default implementation."""
        if isinstance(other, self.__class__):
            try:
                return torch.equal(self.val, other.val) and self.dim == other.dim
            except AttributeError:
                return torch.equal(self.val, other.val)
        else:
            return NotImplemented

    def __add__(self, other):
        """Overrides the default implementation."""
        if isinstance(other, self.__class__):
            try:
                if self.dim == other.dim:
                    return Tensor(self.val + other.val, self.dim)
                else:
                    raise NotImplementedError(f'Same dimensionality needed for addition of two Tensors.')
            except AttributeError:
                if hasattr(self, 'dim'):
                    return Tensor(self.val + other.val, self.dim)
                elif hasattr(other, 'dim'):
                    return Tensor(self.val + other.val, other.dim)
                else:
                    return Tensor(self.val + other.val)
        else:
            return Tensor(self.val + other)

    def __sub__(self, other):
        """Overrides the default implementation."""
        if isinstance(other, self.__class__):
            try:
                if self.dim == other.dim:
                    return Tensor(self.val - other.val, self.dim)
                else:
                    raise NotImplementedError(f'Same dimensionality needed for subtraction of two Tensors.')
            except AttributeError:
                if hasattr(self, 'dim'):
                    return Tensor(self.val - other.val, self.dim)
                elif hasattr(other, 'dim'):
                    return Tensor(self.val - other.val, other.dim)
                else:
                    return Tensor(self.val - other.val)
        else:
            return Tensor(self.val - other)

    def __mul__(self, other):
        """Overrides the default implementation."""
        if isinstance(other, self.__class__):
            try:
                return Tensor(self.val * other.val, self.dim * other.dim)
            except AttributeError:
                return Tensor(self.val * other.val)
        else:
            return Tensor(self.val * other)

    def __truediv__(self, other):
        """Overrides the default implementation."""
        if isinstance(other, self.__class__):
            try:
                return Tensor(self.val / other.val, self.dim / other.dim)
            except AttributeError:
                return Tensor(self.val / other.val)
        else:
            return Tensor(self.val / other)
