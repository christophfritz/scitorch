import torch
from scitorch._globals import DEVICE as device

def _create_tensor(val):
    return torch.as_tensor(val, device=device, dtype=torch.float64)

def to_kelvin(val=0.0, scale='k'):
    """Converts a value from Celsius/Fahrenheit to Kelvin.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) scale that the value is in

    Returns:
    --------

    temp -- (Tensor) value in Kelvin scale

    Example:
    --------

    >>> fahrenheit = 32
    >>> temperature.to_kelvin(fahrenheit, 'f')
    tensor(273.1500, dtype=torch.float64)

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

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) scale that the value is in

    Returns:
    --------

    temp -- (Tensor) value in Kelvin scale

    Example:
    --------

    >>> kelvin = 0
    >>> temperature.to_celsius(kelvin, 'k')
    tensor(-273.1500, dtype=torch.float64)

    """

    if scale =='c':
        temp = _create_tensor(val)
    else:
        temp = to_kelvin(val, scale)
        temp = temp - 273.15

    return temp

def to_fahrenheit(val=0.0, scale='f'):
    """Converts a value from Kelvin/Celsius to Fahrenheit.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) scale that the value is in

    Returns:
    --------

    temp -- (Tensor) value in Kelvin scale

    Example:
    --------

    >>> celsius = 0
    >>> temperature.to_fahrenheit(celsius, 'c')
    tensor(32., dtype=torch.float64)

    """

    if scale == 'f':
        temp = _create_tensor(val)
    else:
        temp = to_kelvin(val, scale)
        temp = (temp - 273.15) * 9/5 + 32

    return temp