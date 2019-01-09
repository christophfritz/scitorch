import torch
from scitorch._globals import DEVICE as device

def _create_tensor(val):
    return torch.tensor(val, device=device, dtype=torch.float64)

def to_kelvin(val=0.0, scale='k'):
    """Converts a value from Celsius/Fahrenheit to Kelvin.

    :arg

    val -- (int) value
    scale -- (char) scale

    :returns

    kelvin -- (Tensor and float/list) torch.tensor with converted values and Python number or list
    """
    temp = _create_tensor(val)
    if scale == 'k':
        kelvin = temp
    if scale == 'c':
        kelvin = temp + 273.15
    if scale == 'f':
        kelvin = (temp - 32) * 5/9 + 273.15

    return kelvin, kelvin.tolist()

def to_celsius(val=0.0, scale='c'):
    """Converts a value from Kelvin/Fahrenheit to Celsius.

    :arg

    val -- (int) value
    scale -- (char) scale

    :returns

    celsius -- (float/list) Python number or list
    """
    temp, _ = to_kelvin(val, scale)
    if scale =='c':
        celsius = _create_tensor(val)
    else:
        celsius = temp - 273.15

    return celsius.tolist()

def to_fahrenheit(val=0.0, scale='f'):
    """Converts a value from Kelvin/Celsius to Fahrenheit.

    :arg

    val -- (int) value
    scale -- (char) scale

    :returns

    fahrenheit -- (float/list) Python number or list
    """

    temp, _ = to_kelvin(val, scale)
    if scale == 'f':
        fahrenheit = _create_tensor(val)
    else:
        fahrenheit = (temp - 273.15) * 9/5 + 32

    return fahrenheit.tolist()