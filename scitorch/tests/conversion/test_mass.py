import torch

from pytest import raises
from scitorch.constants import constants
from scitorch.conversion.mass import *


class TestToKilogram(object):
    def test_to_kilogram_default_values(self):
        kilogram = to_kilogram()
        assert torch.equal(kilogram, T(0))

    def test_to_kilogram_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_kilogram('Kg', 0)

    def test_to_kilogram_wrong_arguments_list(self):
        with raises(TypeError):
            to_kilogram('Kg', [0, 0])

    def test_to_kilogram_tensor(self):
        kilogram_tensor = T([0, 0])
        kilogram = to_kilogram(kilogram_tensor)
        assert torch.equal(kilogram, T([0, 0]))

    def test_to_kilogram_wrong_uni_scalar(self):
        with raises(NotImplementedError):
            to_kilogram(0, 'l')

    def test_to_kilogram_wrong_unit_list(self):
        with raises(NotImplementedError):
            to_kilogram([0, 0], 'l')

    def test_to_kilogram_with_dimension(self):
        kilogram = to_kilogram([0, 1], 'g', dim=True)
        assert isinstance(kilogram, dict)
        assert torch.equal(kilogram['val'], T([0, constants.milli])) and kilogram['dim'] == 'Kg'

    def test_to_kilogram_from_kilogram_scalar(self):
        kilogram = to_kilogram(0, 'Kg')
        assert torch.equal(kilogram, T(0))

    def test_to_kilogram_from_kilogram_list(self):
        kilogram = to_kilogram([0, 50], 'Kg')
        assert torch.equal(kilogram, T([0, 50]))

    def test_to_kilogram_from_milligram_scalar(self):
        kilogram = to_kilogram(1, 'mg')
        assert torch.equal(kilogram, T(constants.micro))

    def test_to_kilogram_from_milligram_list(self):
        kilogram = to_kilogram([0, 1], 'mg')
        assert torch.equal(kilogram, T([0, constants.micro]))

    def test_to_kilogram_from_gram_scalar(self):
        kilogram = to_kilogram(1, 'g')
        assert torch.equal(kilogram, T(constants.milli))

    def test_to_kilogram_from_gram_list(self):
        kilogram = to_kilogram([0, 1], 'g')
        assert torch.equal(kilogram, T([0, constants.milli]))

    def test_to_kilogram_from_tonne_scalar(self):
        kilogram = to_kilogram(1, 't')
        assert torch.equal(kilogram, T(constants.kilo))

    def test_to_kilogram_from_tonne_list(self):
        kilogram = to_kilogram([0, 1], 't')
        assert torch.equal(kilogram, T([0, constants.kilo]))