"""Conversion of different digital storage types (Bytes, Bits)."""

from scitorch.tools._tensors import _create_tensor
from scitorch.constants import constants


def to_byte(val=0.0, unit='B'):
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
    if unit == 'B':
        return ds
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
    else:
        raise NotImplementedError(f'{unit} is not supported. See documentation for available units.')

    return ds