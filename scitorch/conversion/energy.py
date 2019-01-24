"""Conversion of different energy types (e.g. eV, Joule,..)."""

from scitorch.tools._tensors import T
from scitorch.constants import constants

def to_joule(val=0.0, unit='J', dim=False):
    """Converts a value from any energy unit to Kelvin.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

    Returns:
    --------

    temp -- (Tensor) value in Joule

    or

    {'value' : temp, 'dim' : 'J'} -- (dict) dictionary of value and dimension

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

    energy = T(val)
    if unit == 'J':
        energy = energy
    elif unit == 'KJ':
        energy = energy * 1000
    elif unit == 'Wh':
        energy = energy * 3600
    elif unit == 'KWh':
        energy = energy * 3.6 * constants.mega
    elif unit == 'eV':
        energy = energy * constants.eV.get('val')
    else:
        raise NotImplementedError(f'{unit} is not supported. See documentation for available units.')

    if dim == False:
        return energy
    else:
        return dict(val=energy, dim='J')
