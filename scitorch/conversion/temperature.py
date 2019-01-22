"""Conversion of different temperature scales (Kelvin, Fahrenheit and Celsius)."""

from scitorch.tools._tensors import _create_tensor

def to_kelvin(val=0.0, scale='k', dim=False):
    """Converts a value from Celsius/Fahrenheit to Kelvin.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

    Returns:
    --------

    temp -- (Tensor) value in Kelvin scale

    or

    {'value' : temp, 'dim' : 'K'} -- (dict) dictionary of value and dimension

    Example:
    --------

    >>> from scitorch.conversion import temperature
    >>> fahrenheit = 32
    >>> temperature.to_kelvin(fahrenheit, 'f')
    tensor(273.1500, dtype=torch.float64)
    >>> fahrenheit = [32, 5]
    >>> temperature.to_kelvin(fahrenheit, 'f')
    tensor([273.1500, 258.1500], dtype=torch.float64)
    >>> temperature.to_kelvin(fahrenheit, 'f', dim=True)
    {'val': tensor([273.1500, 258.1500], dtype=torch.float64), 'dim': 'K'}

    """

    temp = _create_tensor(val)
    if scale == 'k':
        temp = temp
    elif scale == 'c':
        temp = temp + 273.15
    elif scale == 'f':
        temp = (temp - 32) * 5 / 9 + 273.15
    else:
        raise NotImplementedError(f'{scale} is not supported. See documentation for available scales.')

    if dim == False:
        return temp
    else:
        return dict(val=temp, dim='K')

def to_celsius(val=0.0, scale='c', dim=False):
    """Converts a value from Kelvin/Fahrenheit to Celsius.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

    Returns:
    --------

    temp -- (Tensor) value in Celsius scale

    or

    {'value' : temp, 'dim' : 'C'} -- (dict) dictionary of value and dimension

    Example:
    --------

    >>> kelvin = 0
    >>> temperature.to_celsius(kelvin, 'k')
    tensor(-273.1500, dtype=torch.float64)
    >>> kelvin = [0, 273.15]
    >>> temperature.to_celsius(kelvin, 'k')
    tensor([-273.1500,    0.0000], dtype=torch.float64)
    >>> temperature.to_celsius(kelvin, 'k', dim=True)
    {'val': tensor([-273.1500,    0.0000], dtype=torch.float64), 'dim': 'C'}

    """

    if scale =='c':
        temp = _create_tensor(val)
    else:
        temp = to_kelvin(val, scale)
        temp = temp - 273.15

    if dim == False:
        return temp
    else:
        return dict(val=temp, dim='C')

def to_fahrenheit(val=0.0, scale='f', dim=False):
    """Converts a value from Kelvin/Celsius to Fahrenheit.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

    Returns:
    --------

    temp -- (Tensor) value in Fahrenheit scale

    or

    {'value' : temp, 'dim' : 'F'} -- (dict) dictionary of value and dimension

    Example:
    --------

    >>> celsius = 0
    >>> temperature.to_fahrenheit(celsius, 'c')
    tensor(32., dtype=torch.float64)
    >>> celsius = [0, -15]
    >>> temperature.to_fahrenheit(celsius, 'c')
    tensor([32.,  5.], dtype=torch.float64)
    >>> temperature.to_fahrenheit(celsius, 'c', dim=True)
    {'val': tensor([32.,  5.], dtype=torch.float64), 'dim': 'F'}

    """

    if scale == 'f':
        temp = _create_tensor(val)
    else:
        temp = to_kelvin(val, scale)
        temp = (temp - 273.15) * 9/5 + 32

    if dim == False:
        return temp
    else:
        return dict(val=temp, dim='F')
