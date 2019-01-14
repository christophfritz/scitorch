"""Conversion of different temperature scales (Kelvin, Fahrenheit and Celsius)."""

from scitorch.conversion._standard_units import kelvin
from scitorch.tools._tensors import _create_tensor

def to_kelvin(val=0.0, scale='k'):
    """Converts a value from Celsius/Fahrenheit to Kelvin.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

    Returns:
    --------

    temp -- (Tensor) value in Kelvin scale

    Example:
    --------

    >>> fahrenheit = 32
    >>> temperature.to_kelvin(fahrenheit, 'f')
    tensor(273.1500, dtype=torch.float64)

    """
    return kelvin(val, scale)

def to_celsius(val=0.0, scale='c'):
    """Converts a value from Kelvin/Fahrenheit to Celsius.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

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
        temp = kelvin(val, scale)
        temp = temp - 273.15

    return temp

def to_fahrenheit(val=0.0, scale='f'):
    """Converts a value from Kelvin/Celsius to Fahrenheit.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

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
        temp = kelvin(val, scale)
        temp = (temp - 273.15) * 9/5 + 32

    return temp