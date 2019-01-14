"""Converts given units into standard units first (e.g. Kelvin, Meter, Bytes, etc.)"""

from scitorch.tools._tensors import _create_tensor
from scitorch.constants import constants

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

def bytes(val=0.0, unit='B'):
    """Converts a value from any byte or bit format to bytes.

     Parameters:
        -----------

        val -- (int) value
        scale -- (char) new unit

        Returns:
        --------

        temp -- (Tensor) value in Kelvin scale

        Example:
        --------


        """

    # ds := digital storage
    ds = _create_tensor(val)
    if 'b' not in unit:
        if unit == 'B':
            ds = ds
        elif unit == 'KB':
            ds = ds * constants.kilo
        elif unit == 'MB':
            ds = ds * constants.mega
        elif unit == 'GB':
            ds = ds * constants.giga
        elif unit == 'TB':
            ds = ds * constants.tera
        elif unit == 'PB':
            ds = ds * constants.peta
        elif unit == 'KiB':
            ds = ds * constants.kibi
        elif unit == 'MiB':
            ds = ds * constants.mebi
        elif unit == 'GiB':
            ds = ds * constants.gibi
        elif unit == 'TiB':
            ds = ds * constants.tebi
        elif unit == 'PiB':
            ds = ds * constants.pebi
        else:
            raise NotImplementedError(f'{unit} is not supported. See documentation for available units.')
    else:
        if unit == 'b':
            pass
        else:
            raise NotImplementedError(f'{unit} is not supported. See documentation for available units.')

    return ds