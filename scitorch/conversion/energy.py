"""Conversion of different energy types (e.g. eV, Joule,..)."""

from scitorch.tools._tensors import _create_tensor
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

    >>> watthours = 300
    >>> energy.to_joule(watthours, 'Wh')
    tensor(1080000., dtype=torch.float64)
    >>> electronvolt = [1, 2]
    >>> energy.to_joule(electronvolt, 'eV')
    tensor([1.6022e-19, 3.2044e-19], dtype=torch.float64)
    >>> energy.to_joule(electronvolt, 'eV', dim=True)
    {'val': tensor([1.6022e-19, 3.2044e-19], dtype=torch.float64), 'dim': 'J'}

    """

    energy = _create_tensor(val)
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
