from pytest import raises
from scitorch.conversion.temperature import to_kelvin, to_celsius, to_fahrenheit


class TestToKelvin(object):
    def test_to_kelvin_default_values_scalar(self):
        _, kelvin = to_kelvin()
        assert kelvin == 0

    def test_to_kelvin_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_kelvin('f', 0)

    def test_to_kelvin_wrong_arguments_list(self):
        with raises(TypeError):
            to_kelvin('f', [0, 0])

    def test_to_kelvin_from_kelvin_scalar(self):
        _, kelvin = to_kelvin(0, 'k')
        assert kelvin == 0

    def test_to_kelvin_from_kelvin_list(self):
        _, kelvin = to_kelvin([0, 273.15], 'k')
        assert kelvin == [0, 273.15]

    def test_to_kelvin_from_celsius_scalar(self):
        _, kelvin = to_kelvin(0, 'c')
        assert kelvin == 273.15

    def test_to_kelvin_from_celsius_list(self):
        _, kelvin = to_kelvin([0, -273.15], 'c')
        assert kelvin == [273.15, 0]

    def test_to_kelvin_from_fahrenheit_scalar(self):
        _, kelvin = to_kelvin(32, 'f')
        assert kelvin == 273.15

    def test_to_kelvin_from_fahrenheit_list(self):
        _, kelvin = to_kelvin([32, 5], 'f')
        assert kelvin == [273.15, 258.15]


class TestToCelsius(object):

    def test_to_celsius_default_values_scalar(self):
        celsius = to_celsius()
        assert celsius == 0

    def test_to_celsius_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_celsius('k', 0)

    def test_to_celsius_wrong_arguments_list(self):
        with raises(TypeError):
            to_celsius('k', [0, 0])

    def test_to_celsius_from_celsius_scalar(self):
        celsius = to_celsius(0, 'c')
        assert celsius == 0

    def test_to_celsius_from_celsius_list(self):
        celsius = to_celsius([0, -273.15], 'c')
        assert celsius == [0, -273.15]

    def test_to_celsius_from_kelvin_scalar(self):
        celsius = to_celsius(0, 'k')
        assert celsius == -273.15

    def test_to_celsius_from_kelvin_list(self):
        celsius = to_celsius([0, 273.15], 'k')
        assert celsius == [-273.15, 0]

    def test_to_celsius_from_fahrenheit_scalar(self):
        celsius = to_celsius(32, 'f')
        assert celsius == 0

    def test_to_celsius_from_fahrenheit_list(self):
        celsius = to_celsius([32, 5], 'f')
        assert celsius == [0, -15]

class TestToFahrenheit(object):

    def test_to_fahrenheit_default_values_scalar(self):
        fahrenheit = to_fahrenheit()
        assert fahrenheit == 0

    def test_to_fahrenheit_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_fahrenheit('c', 0)

    def test_to_fahrenheit_wrong_arguments_list(self):
        with raises(TypeError):
            to_fahrenheit('c', [0, 0])

    def test_to_fahrenheit_from_fahrenheit_scalar(self):
        fahrenheit = to_fahrenheit(32, 'f')
        assert fahrenheit == 32

    def test_to_fahrenheit_from_fahrenheit_list(self):
        fahrenheit = to_fahrenheit([32, 0], 'f')
        assert fahrenheit == [32, 0]

    def test_to_fahrenheit_from_celsius_scalar(self):
        fahrenheit = to_fahrenheit(0, 'c')
        assert fahrenheit == 32

    def test_to_fahrenheit_from_celsius_list(self):
        fahrenheit = to_fahrenheit([0, -15], 'c')
        assert fahrenheit == [32, 5]

    def test_to_fahrenheit_from_kelvin_scalar(self):
        fahrenheit = to_fahrenheit(273.15, 'k')
        assert fahrenheit == 32

    def test_to_fahrenheit_from_kelvin_list(self):
        fahrenheit = to_fahrenheit([273.15, 258.15], 'k')
        assert fahrenheit == [32, 5]


