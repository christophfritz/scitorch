"""Conversion of different digital storage types (Bytes, Bits)."""

from scitorch.tools._tensors import _create_tensor
from scitorch.constants import constants


def to_bytes(val=0.0, unit='B'):
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
            ds = ds / 8
        elif unit == 'Kbit':
            ds = ds * 125
        elif unit == 'Mbit':
            ds = ds * 125 * constants.kilo
        elif unit == 'Gbit':
            ds = ds * 125 * constants.mega
        elif unit == 'Tbit':
            ds = ds * 125 * constants.giga
        elif unit == 'Pbit':
            ds = ds * 125 * constants.tera
        elif unit == 'Kib':
            ds = ds * constants.kibi / 8
        elif unit == 'Mib':
            ds = ds * constants.mebi / 8
        elif unit == 'Gib':
            ds = ds * constants.gibi / 8
        elif unit == 'Tib':
            ds = ds * constants.tebi / 8
        elif unit == 'Pib':
            ds = ds * constants.pebi / 8
        else:
            raise NotImplementedError(f'{unit} is not supported. See documentation for available units.')

    return ds

def to_kilobytes(val=0.0, unit='KB'):
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
    if unit == 'KB':
        ds = _create_tensor(val)
    else:
        ds = to_bytes(val, unit)
        ds = ds / constants.kilo

    return ds