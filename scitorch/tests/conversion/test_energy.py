import torch

from pytest import raises
from scitorch.conversion.energy import *
from scitorch.tools._tensors import _create_tensor


class TestToJoule(object):
    def test_to_joule_default_values(self):
        joule = to_joule()
        assert torch.equal(joule, _create_tensor(0))

    def test_to_joule_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_joule('J', 0)

    def test_to_joule_wrong_arguments_list(self):
        with raises(TypeError):
            to_joule('J', [0, 0])

    def test_to_joule_tensor(self):
        joule_tensor = _create_tensor([0, 0])
        joule = to_joule(joule_tensor)
        assert torch.equal(joule, _create_tensor([0, 0]))

    def test_to_joule_wrong_unit_scalar(self):
        with raises(NotImplementedError):
            to_joule(0, 'Kcal')

    def test_to_joule_wrong_unit_list(self):
        with raises(NotImplementedError):
            to_joule([0, 0], 'l')

    def test_to_joule_with_dimension(self):
        joule = to_joule([0, 1], 'Wh', dim=True)
        assert isinstance(joule, dict)
        assert torch.equal(joule['val'], _create_tensor([0, 3600])) and joule['dim'] == 'J'

    def test_to_joule_from_joule_scalar(self):
        joule = to_joule(0, 'J')
        assert torch.equal(joule, _create_tensor(0))

    def test_to_joule_from_joule_list(self):
        joule = to_joule([0, 50], 'J')
        assert torch.equal(joule, _create_tensor([0, 50]))

    def test_to_joule_from_kilojoule_scalar(self):
        joule = to_joule(1, 'KJ')
        assert torch.equal(joule, _create_tensor(1000))

    def test_to_joule_from_kilojoule_list(self):
        joule = to_joule([0, 1], 'KJ')
        assert torch.equal(joule, _create_tensor([0, 1000]))

    def test_to_joule_from_watt_h_scalar(self):
        joule = to_joule(1, 'Wh')
        assert torch.equal(joule, _create_tensor(3.6 * constants.kilo))

    def test_to_joule_from_watt_h_list(self):
        joule = to_joule([0, 1], 'Wh')
        assert torch.equal(joule, _create_tensor([0, 3.6 * constants.kilo]))

    def test_to_joule_from_kilowatt_h_scalar(self):
        joule = to_joule(1, 'KWh')
        assert torch.equal(joule, _create_tensor(3.6 * constants.mega))

    def test_to_joule_from_kilowatt_h_list(self):
        joule = to_joule([0, 1], 'KWh')
        assert torch.equal(joule, _create_tensor([0, 3.6 * constants.mega]))

    def test_to_joule_from_eV_scalar(self):
        joule = to_joule(1, 'eV')
        assert torch.equal(joule, _create_tensor(constants.eV.get('val')))

    def test_to_joule_from_fahrenheit_list(self):
        joule = to_joule([0, 1], 'eV')
        assert torch.equal(joule, _create_tensor([0, constants.eV.get('val')]))