import torch

from pytest import raises
from scitorch.conversion.mass import *
from scitorch.tools._tensors import _create_tensor


class TestToKilogram(object):
    def test_to_kilogram_default_values(self):
        kilogram = to_kilogram()
        assert torch.equal(kilogram, _create_tensor(0))

    def test_to_kilogram_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_kilogram('kg', 0)

    def test_to_kilogram_wrong_arguments_list(self):
        with raises(TypeError):
            to_kilogram('kg', [0, 0])

    def test_to_kilogram_tensor(self):
        kilogram_tensor = _create_tensor([0, 0])
        kilogram = to_kilogram(kilogram_tensor)
        assert torch.equal(kilogram, _create_tensor([0, 0]))

    def test_to_kilogram_wrong_uni_scalar(self):
        with raises(NotImplementedError):
            to_kilogram(0, 'l')

    def test_to_kilogram_wrong_unit_list(self):
        with raises(NotImplementedError):
            to_kilogram([0, 0], 'l')

    def test_to_kilogram_with_dimension(self):
        kilogram = to_kilogram([0, 1], 'g', dim=True)
        assert isinstance(kilogram, dict)
        assert torch.equal(kilogram['val'], _create_tensor([0, constants.milli])) and kilogram['dim'] == 'kg'

    def test_to_kilogram_from_kilogram_scalar(self):
        kilogram = to_kilogram(0, 'kg')
        assert torch.equal(kilogram, _create_tensor(0))

    def test_to_kilogram_from_kilogram_list(self):
        kilogram = to_kilogram([0, 50], 'kg')
        assert torch.equal(kilogram, _create_tensor([0, 50]))

    def test_to_kilogram_from_milligram_scalar(self):
        kilogram = to_kilogram(1, 'mg')
        assert torch.equal(kilogram, _create_tensor(constants.micro))

    def test_to_kilogram_from_milligram_list(self):
        kilogram = to_kilogram([0, 1], 'mg')
        assert torch.equal(kilogram, _create_tensor([0, constants.micro]))

    def test_to_kilogram_from_gram_scalar(self):
        kilogram = to_kilogram(1, 'g')
        assert torch.equal(kilogram, _create_tensor(constants.milli))

    def test_to_kilogram_from_gram_list(self):
        kilogram = to_kilogram([0, 1], 'g')
        assert torch.equal(kilogram, _create_tensor([0, constants.milli]))

    def test_to_kilogram_from_tonne_scalar(self):
        kilogram = to_kilogram(1, 't')
        assert torch.equal(kilogram, _create_tensor(constants.kilo))

    def test_to_kilogram_from_tonne_list(self):
        kilogram = to_kilogram([0, 1], 't')
        assert torch.equal(kilogram, _create_tensor([0, constants.kilo]))