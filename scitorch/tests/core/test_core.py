"""Test cases for the SciTorch core."""
import torch

from scitorch.core import Tensor as T
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from sympy import sympify


class TestCore(object):

    def test_Tensor_empty(self):
        with raises(TypeError):
            T()

    def test_Tensor_val(self):
        tensor = T([1,2,3,4,5])
        assert torch.equal(tensor.val, _create_tensor([1,2,3,4,5]))

    def test_Tensor_wrong_value(self):
        with raises(TypeError):
            T('kg')

    def test_Tensor_dim(self):
        tensor = T(3, 'm/s^2')
        assert tensor.dim == sympify('m/s^2')


