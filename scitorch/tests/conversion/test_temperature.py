import torch

from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.temperature import to_kelvin, to_celsius, to_fahrenheit


class TestToKelvin(object):
    def test_to_kelvin_default_values(self):
        kelvin = to_kelvin()
        assert torch.all(torch.eq(kelvin, _create_tensor(0)))

    def test_to_kelvin_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_kelvin('f', 0)

    def test_to_kelvin_wrong_arguments_list(self):
        with raises(TypeError):
            to_kelvin('f', [0, 0])

    def test_to_kelvin_tensor(self):
        kelvin_tensor = torch.tensor([0, 0])
        kelvin = to_kelvin(kelvin_tensor)
        assert torch.all(torch.eq(kelvin, _create_tensor([0,0])))

    def test_to_kelvin_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_kelvin(0, 'l')

    def test_to_kelvin_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_kelvin([0, 0], 'l')

    def test_to_kelvin_with_dimension(self):
        kelvin = to_kelvin([0, -273.15], 'c', dim=True)
        assert isinstance(kelvin, dict)
        assert torch.all(torch.eq(kelvin['val'], _create_tensor([273.15, 0]))) and kelvin['dim'] == 'K'

    def test_to_kelvin_from_kelvin_scalar(self):
        kelvin = to_kelvin(0, 'k')
        assert torch.all(torch.eq(kelvin, _create_tensor(0)))

    def test_to_kelvin_from_kelvin_list(self):
        kelvin = to_kelvin([0, 273.15], 'k')
        assert torch.all(torch.eq(kelvin, _create_tensor([0, 273.15])))

    def test_to_kelvin_from_celsius_scalar(self):
        kelvin = to_kelvin(0, 'c')
        assert torch.all(torch.eq(kelvin, _create_tensor(273.15)))

    def test_to_kelvin_from_celsius_list(self):
        kelvin = to_kelvin([0, -273.15], 'c')
        assert torch.all(torch.eq(kelvin, _create_tensor([273.15, 0])))

    def test_to_kelvin_from_fahrenheit_scalar(self):
        kelvin = to_kelvin(32, 'f')
        assert torch.all(torch.eq(kelvin, _create_tensor(273.15)))

    def test_to_kelvin_from_fahrenheit_list(self):
        kelvin = to_kelvin([32, 5], 'f')
        assert torch.all(torch.eq(kelvin, _create_tensor([273.15, 258.15])))


class TestToCelsius(object):

    def test_to_celsius_default_values(self):
        celsius = to_celsius()
        assert torch.all(torch.eq(celsius, _create_tensor(0)))

    def test_to_celsius_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_celsius('k', 0)

    def test_to_celsius_wrong_arguments_list(self):
        with raises(TypeError):
            to_celsius('k', [0, 0])

    def test_to_celsius_with_dimension(self):
        celsius = to_celsius([32, 5], 'f', dim=True)
        assert isinstance(celsius, dict)
        assert torch.all(torch.eq(celsius['val'], _create_tensor([0, -15]))) and celsius['dim'] == 'C'

    def test_to_celsius_from_celsius_scalar(self):
        celsius = to_celsius(0, 'c')
        assert torch.all(torch.eq(celsius, _create_tensor(0)))

    def test_to_celsius_from_celsius_list(self):
        celsius = to_celsius([0, -273.15], 'c')
        assert torch.all(torch.eq(celsius, _create_tensor([0, -273.15])))

    def test_to_celsius_from_kelvin_scalar(self):
        celsius = to_celsius(0, 'k')
        assert torch.all(torch.eq(celsius, _create_tensor(-273.15)))

    def test_to_celsius_from_kelvin_list(self):
        celsius = to_celsius([0, 273.15], 'k')
        assert torch.all(torch.eq(celsius, _create_tensor([-273.15, 0])))

    def test_to_celsius_from_fahrenheit_scalar(self):
        celsius = to_celsius(32, 'f')
        assert torch.all(torch.eq(celsius, _create_tensor(0)))

    def test_to_celsius_from_fahrenheit_list(self):
        celsius = to_celsius([32, 5], 'f')
        assert torch.all(torch.eq(celsius, _create_tensor([0, -15])))

class TestToFahrenheit(object):

    def test_to_fahrenheit_default_values(self):
        fahrenheit = to_fahrenheit()
        assert torch.all(torch.eq(fahrenheit, _create_tensor(0)))

    def test_to_fahrenheit_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_fahrenheit('c', 0)

    def test_to_fahrenheit_wrong_arguments_list(self):
        with raises(TypeError):
            to_fahrenheit('c', [0, 0])

    def test_to_fahrenheit_with_dimension(self):
        fahrenheit = to_fahrenheit([273.15, 258.15], 'k', dim=True)
        assert isinstance(fahrenheit, dict)
        assert torch.all(torch.eq(fahrenheit['val'], _create_tensor([32, 5]))) and fahrenheit['dim'] == 'F'

    def test_to_fahrenheit_from_fahrenheit_scalar(self):
        fahrenheit = to_fahrenheit(32, 'f')
        assert torch.all(torch.eq(fahrenheit, _create_tensor(32)))

    def test_to_fahrenheit_from_fahrenheit_list(self):
        fahrenheit = to_fahrenheit([32, 0], 'f')
        assert torch.all(torch.eq(fahrenheit, _create_tensor([32, 0])))

    def test_to_fahrenheit_from_celsius_scalar(self):
        fahrenheit = to_fahrenheit(0, 'c')
        assert torch.all(torch.eq(fahrenheit, _create_tensor(32)))

    def test_to_fahrenheit_from_celsius_list(self):
        fahrenheit = to_fahrenheit([0, -15], 'c')
        assert torch.all(torch.eq(fahrenheit, _create_tensor([32, 5])))

    def test_to_fahrenheit_from_kelvin_scalar(self):
        fahrenheit = to_fahrenheit(273.15, 'k')
        assert torch.all(torch.eq(fahrenheit, _create_tensor(32)))

    def test_to_fahrenheit_from_kelvin_list(self):
        fahrenheit = to_fahrenheit([273.15, 258.15], 'k')
        assert torch.all(torch.eq(fahrenheit, _create_tensor([32, 5])))


