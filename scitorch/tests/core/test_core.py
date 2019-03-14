"""Test cases for the SciTorch core."""
import torch

from scitorch.core import Tensor
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from sympy import sympify


class TestCore(object):

    def test_Tensor_empty(self):
        with raises(TypeError):
            Tensor()

    def test_Tensor(self):
        tensor_list = Tensor([1,2,3,4,5])
        assert torch.equal(tensor_list.val, _create_tensor([1, 2, 3, 4, 5]))

        tensor_scalar = Tensor(3)
        assert torch.equal(tensor_scalar.val, _create_tensor(3))

        tensor = Tensor(3, 'm/s^2')
        assert tensor.dim == sympify('m/s^2')
        assert torch.equal(tensor.val, _create_tensor(3))

    def test_Tensor_wrong_value(self):
        with raises(TypeError):
            Tensor('kg')

    def test_Tensor_equal(self):
        assert Tensor(3) == Tensor(3)
        assert Tensor([1,2,3,4,5]) == Tensor([1,2,3,4,5])
        assert Tensor(3, 'kg') == Tensor(3, 'kg')
        assert Tensor([1,2,3,4,5], 'kg') == Tensor([1,2,3,4,5], 'kg')

    def test_Tensor_addition(self):
        mass = Tensor(20, 'kg')
        double_mass = mass + mass
        assert double_mass == Tensor(40, 'kg')

        double_mass = mass + 20
        assert double_mass == Tensor(40, 'kg')

        double_mass = mass + Tensor(20)
        assert double_mass == Tensor(40, 'kg')

        double_mass = Tensor(20) + mass
        assert double_mass == Tensor(40, 'kg')

        acceleration = Tensor(10, 'm/s^2')
        with raises(NotImplementedError):
            mass + acceleration

        assert Tensor(20) + Tensor(20) == Tensor(40)

    def test_Tensor_subtraction(self):
        mass = Tensor(40, 'kg')
        sub_mass = mass - mass
        assert sub_mass == Tensor(0, 'kg')

        sub_mass = mass - 20
        assert sub_mass == Tensor(20, 'kg')

        sub_mass = mass - Tensor(20)
        assert sub_mass == Tensor(20, 'kg')

        sub_mass = Tensor(20) - mass
        assert sub_mass == Tensor(-20, 'kg')

        acceleration = Tensor(10, 'm/s^2')
        with raises(NotImplementedError):
            mass - acceleration

        assert Tensor(40) - Tensor(20) == Tensor(20)

    def test_Tensor_multiplication(self):
        mass = Tensor(20, 'kg')
        acceleration = Tensor(10, 'm/s^2')
        force = mass * acceleration
        assert force == Tensor(200, '(kg*m)/s^2')

        triple_mass = mass * 3.0
        assert triple_mass == Tensor(60, 'kg')

        triple_mass = mass * Tensor(3.0)
        assert triple_mass == Tensor(60, 'kg')

        triple_mass = Tensor(3.0) * mass
        assert triple_mass == Tensor(60, 'kg')

        assert Tensor(3) * Tensor(10) == Tensor(30)

    def test_Tensor_division(self):
        acceleration = Tensor(50, 'm/s^2')
        force = Tensor(500, '(kg*m)/s^2')
        mass = force / acceleration
        assert mass == Tensor(10, 'kg')

        half_mass = mass / 2.0
        assert half_mass == Tensor(5, 'kg')

        half_mass = mass / Tensor(2.0)
        assert half_mass == Tensor(5, 'kg')

        half_mass = Tensor(2.0) / mass
        assert half_mass == Tensor(0.2, 'kg')

        assert Tensor(50) / Tensor(2) == Tensor(25)



