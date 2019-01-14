"""Converts given units into standard units first (e.g. Kelvin, Meter, Bytes, etc.)"""

from scitorch.tools._tensors import _create_tensor

def kelvin(val=0.0, scale='k'):
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
    temp = _create_tensor(val)
    if scale == 'k':
        temp = temp
    elif scale == 'c':
        temp = temp + 273.15
    elif scale == 'f':
        temp = (temp - 32) * 5/9 + 273.15
    else:
        raise NotImplementedError(f'{scale} is not supported. See documentation for available scales.')

    return temp