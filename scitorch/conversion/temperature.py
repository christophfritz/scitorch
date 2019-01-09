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

    temp -- (Tensor) value in Kelvin scale
    """
    temp = _create_tensor(val)
    if scale == 'k':
        temp = temp
    if scale == 'c':
        temp = temp + 273.15
    if scale == 'f':
        temp = (temp - 32) * 5/9 + 273.15

    return temp

def to_celsius(val=0.0, scale='c'):
    """Converts a value from Kelvin/Fahrenheit to Celsius.

    :arg

    val -- (int) value
    scale -- (char) scale

    :returns

    temp -- (Tensor) value in Celsius scale
    """
    if scale =='c':
        temp = _create_tensor(val)
    else:
        temp = to_kelvin(val, scale)
        temp = temp - 273.15

    return temp

def to_fahrenheit(val=0.0, scale='f'):
    """Converts a value from Kelvin/Celsius to Fahrenheit.

    :arg

    val -- (int) value
    scale -- (char) scale

    :returns

    temp -- (Tensor) value in Fahrenheit scale
    """

    if scale == 'f':
        temp = _create_tensor(val)
    else:
        temp = to_kelvin(val, scale)
        temp = (temp - 273.15) * 9/5 + 32

    return temp