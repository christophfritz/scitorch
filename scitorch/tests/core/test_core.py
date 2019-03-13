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

    def test_Tensor(self):
        tensor_list = T([1,2,3,4,5])
        assert torch.equal(tensor_list.val, _create_tensor([1, 2, 3, 4, 5]))

        tensor_scalar = T(3)
        assert torch.equal(tensor_scalar.val, _create_tensor(3))

        tensor = T(3, 'm/s^2')
        assert tensor.dim == sympify('m/s^2')
        assert torch.equal(tensor.val, _create_tensor(3))

    def test_Tensor_wrong_value(self):
        with raises(TypeError):
            T('kg')

    def test_Tensor_equal(self):
        assert T(3) == T(3)
        assert T([1,2,3,4,5]) == T([1,2,3,4,5])
        assert T(3, 'kg') == T(3, 'kg')
        assert T([1,2,3,4,5], 'kg') == T([1,2,3,4,5], 'kg')

    def test_Tensor_addition(self):
        mass = T(20, 'kg')
        double_mass = mass + mass
        assert double_mass == T(40, 'kg')

        double_mass = mass + 20
        assert double_mass == T(40, 'kg')

        double_mass = mass + T(20)
        assert double_mass == T(40, 'kg')

        double_mass = T(20) + mass
        assert double_mass == T(40, 'kg')

        acceleration = T(10, 'm/s^2')
        with raises(NotImplementedError):
            mass + acceleration

        assert T(20) + T(20) == T(40)

    def test_Tensor_subtraction(self):
        mass = T(40, 'kg')
        sub_mass = mass - mass
        assert sub_mass == T(0, 'kg')

        sub_mass = mass - 20
        assert sub_mass == T(20, 'kg')

        sub_mass = mass - T(20)
        assert sub_mass == T(20, 'kg')

        sub_mass = T(20) - mass
        assert sub_mass == T(-20, 'kg')

        acceleration = T(10, 'm/s^2')
        with raises(NotImplementedError):
            mass - acceleration

        assert T(40) - T(20) == T(20)

    def test_Tensor_multiplication(self):
        mass = T(20, 'kg')
        acceleration = T(10, 'm/s^2')
        force = mass * acceleration
        assert force == T(200, '(kg*m)/s^2')

        triple_mass = mass * 3.0
        assert triple_mass == T(60, 'kg')

        triple_mass = mass * T(3.0)
        assert triple_mass == T(60, 'kg')

        triple_mass = T(3.0) * mass
        assert triple_mass == T(60, 'kg')

        assert T(3) * T(10) == T(30)

    def test_Tensor_division(self):
        acceleration = T(50, 'm/s^2')
        force = T(500, '(kg*m)/s^2')
        mass = force / acceleration
        assert mass == T(10, 'kg')

        half_mass = mass / 2.0
        assert half_mass == T(5, 'kg')

        half_mass = mass / T(2.0)
        assert half_mass == T(5, 'kg')

        half_mass = T(2.0) / mass
        assert half_mass == T(0.2, 'kg')

        assert T(50) / T(2) == T(25)



