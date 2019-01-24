"""Conversion of different mass types (e.g. mg, Kg, tonne ..)."""

from scitorch.tools._tensors import T
from scitorch.constants import constants

def to_kilogram(val=0.0, unit='Kg', dim=False):
    """Converts a value from any mass unit to Kilogram.

    Parameters:
    -----------

    val -- (int) value
    scale -- (char) new scale

    Returns:
    --------

    temp -- (Tensor) value in Joule

    or

    {'value' : temp, 'dim' : 'Kg'} -- (dict) dictionary of value and dimension

    Example:
    --------

    >>> milligram = 3000
    >>> from scitorch.conversion import mass
    >>> milligram = 3000
    >>> mass.to_kilogram(milligram, 'mg')
    tensor(0.0030, dtype=torch.float64)
    >>> tonne = 3
    >>> mass.to_kilogram(tonne, 't')
    tensor(3000., dtype=torch.float64)
    >>> mass.to_kilogram(tonne, 't', dim=True)
    {'val': tensor(3000., dtype=torch.float64), 'dim': 'Kg'}

    """

    mass = T(val)
    if unit == 'Kg':
        mass = mass
    elif unit == 'g':
        mass = mass / constants.kilo
    elif unit == 'mg':
        mass = mass / constants.mega
    elif unit == 't':
        mass = mass * constants.kilo
    else:
        raise NotImplementedError(f'{unit} is not supported. See documentation for available units.')

    if dim == False:
        return mass
    else:
        return dict(val=mass, dim='Kg')
