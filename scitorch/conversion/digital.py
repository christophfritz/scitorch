"""Conversion of different digital storage types (Bytes, Bits)."""

from scitorch.tools._tensors import T
from scitorch.constants import constants


def to_bytes(val=0.0, unit='B', dim=False):
    """
    Converts a value from any byte or bit format to bytes.

    Parameters:
    -----------

    val : float
        Value(s) of units to be converted to Bytes.

    unit : char
        Specifies as a string the original unit from which the units will be converted to Bytes.

    Returns:
    --------

    temp -- (Tensor) value in bytes

    Example:
    --------

    >>> megabytes = 200
    >>> digital.to_bytes(megabytes, 'MB')
    tensor(2.0000e+08, dtype=torch.float64)
    >>> kibibits = 1024
    >>> digital.to_bytes(kibibits, 'Kib')
    tensor(131072., dtype=torch.float64)
    >>> digital.to_bytes(kibibits, 'Kib', dim=True)
    {'val': tensor(131072., dtype=torch.float64), 'dim': 'B'}

    """

    # ds := digital storage
    ds = T(val)
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

    if dim == False:
        return ds
    else:
        return dict(val=ds, dim='B')

# def to_kilobytes(val=0.0, unit='KB'):
#     """Converts a value from any byte or bit format to kilobytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> bytes = 500
#         >>> digital.to_kilobytes(bytes, 'B')
#         tensor(0.5000, dtype=torch.float64)
#
#         >>> mebibits = 1024
#         >>> digital.to_kilobytes(mebibits, 'Mib')
#         tensor(134217.7280, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'KB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.kilo
#
#     return ds
#
#
# def to_megabytes(val=0.0, unit='MB'):
#     """Converts a value from any byte or bit format to megabytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> bytes = 500
#         >>> digital.to_megabytes(bytes, 'B')
#         tensor(0.0005, dtype=torch.float64)
#         >>> bits = 256
#         >>> digital.to_megabytes(bits, 'b')
#         tensor(3.2000e-05, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'MB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.mega
#
#     return ds
#
# def to_gigabytes(val=0.0, unit='GB'):
#     """Converts a value from any byte or bit format to gigabytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> terabyte = 2
#         >>> digital.to_gigabytes(terabyte, 'TB')
#         tensor(2000., dtype=torch.float64)
#         >>> tebibyte = 2
#         >>> digital.to_gigabytes(tebibyte, 'TiB')
#         tensor(2199.0233, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'GB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.giga
#
#     return ds
#
# def to_terabytes(val=0.0, unit='TB'):
#     """Converts a value from any byte or bit format to terabytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> gigabit = 200
#         >>> digital.to_terabytes(gigabit, 'Gbit')
#         tensor(0.0250, dtype=torch.float64)
#         >>> megabyte = 3000
#         >>> digital.to_terabytes(megabyte, 'MB')
#         tensor(0.0030, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'TB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.tera
#
#     return ds
#
# def to_petabytes(val=0.0, unit='PB'):
#     """Converts a value from any byte or bit format to petabytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> gigabit = 200
#         >>> digital.to_terabytes(gigabit, 'Gbit')
#         tensor(0.0250, dtype=torch.float64)
#         >>> megabyte = 3000
#         >>> digital.to_terabytes(megabyte, 'MB')
#         tensor(0.0030, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'PB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.peta
#
#     return ds
#
# def to_kibibytes(val=0.0, unit='KiB'):
#     """Converts a value from any byte or bit format to kibibytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> bytes = 1024
#         >>> digital.to_kibibytes(bytes, 'B')
#         tensor(1., dtype=torch.float64)
#         >>> kibibits = 1
#         >>> digital.to_kibibytes(kibibits, 'Kib')
#         tensor(0.1250, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'KiB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.kibi
#
#     return ds
#
# def to_mebibytes(val=0.0, unit='MiB'):
#     """Converts a value from any byte or bit format to mebibytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> kilobytes = 1024
#         >>> digital.to_mebibytes(kilobytes, 'KB')
#         tensor(0.9766, dtype=torch.float64)
#         >>> tebibits = 1
#         >>> digital.to_mebibytes(tebibits, 'Tib')
#         tensor(131072., dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'MiB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.mebi
#
#     return ds
#
# def to_gibibytes(val=0.0, unit='GiB'):
#     """Converts a value from any byte or bit format to gibibytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> terabytes = 2
#         >>> digital.to_gibibytes(terabytes, 'TB')
#         tensor(1862.6451, dtype=torch.float64)
#         >>> mebibit = 234
#         >>> digital.to_gibibytes(mebibit, 'Mib')
#         tensor(0.0286, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'GiB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.gibi
#
#     return ds
#
# def to_tebibytes(val=0.0, unit='TiB'):
#     """Converts a value from any byte or bit format to tebibytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> bytes = 9
#         >>> digital.to_tebibytes(bytes, 'B')
#         tensor(8.1855e-12, dtype=torch.float64)
#         >>> megabit = 300
#         >>> digital.to_tebibytes(megabit, 'Mbit')
#         tensor(3.4106e-05, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'TiB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.tebi
#
#     return ds
#
# def to_pebibytes(val=0.0, unit='PiB'):
#     """Converts a value from any byte or bit format to pebibytes.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> petabytes = 4
#         >>> digital.to_pebibytes(petabytes, 'PB')
#         tensor(3.5527, dtype=torch.float64)
#         >>> bits = 1024
#         >>> digital.to_pebibytes(bits, 'b')
#         tensor(1.1369e-13, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'PiB':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / constants.pebi
#
#     return ds

def to_bits(val=0.0, unit='b', dim=False):
    """Converts a value from any byte or bit format to bits.

     Parameters:
        -----------

        val -- (int) value
        scale -- (char) new unit

        Returns:
        --------

        temp -- (Tensor) value in Kelvin scale

        Example:
        --------

        >>> bytes = 4
        >>> digital.to_bits(bytes, 'B')
        tensor(32., dtype=torch.float64)
        >>> kilobits = 32
        >>> digital.to_bits(kilobits, 'Kbit')
        tensor(32000., dtype=torch.float64)
        >>> digital.to_bits(kilobits, 'Kbit', dim=True)
        {'val': tensor(32000., dtype=torch.float64), 'dim': 'b'}

        """

    # ds := digital storage

    if unit == 'b':
        ds = T(val)
    else:
        ds = to_bytes(val, unit)
        ds = ds * 8

    if dim == False:
        return ds
    else:
        return dict(val=ds, dim='b')

# def to_kilobits(val=0.0, unit='Kbit'):
#     """Converts a value from any byte or bit format to kilobits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> gigabytes = 500
#         >>> digital.to_kilobits(gigabytes, 'GB')
#         tensor(4.0000e+09, dtype=torch.float64)
#         >>> mebibits = 32
#         >>> digital.to_kilobits(mebibits, 'Mib')
#         tensor(33554.4320, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Kbit':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds * 8 * constants.milli
#
#     return ds
#
# def to_megabits(val=0.0, unit='Mbit'):
#     """Converts a value from any byte or bit format to megabits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> petabytes = 4
#         >>> digital.to_megabits(petabytes, 'PB')
#         tensor(3.2000e+10, dtype=torch.float64)
#         >>> bits = 1024
#         >>> digital.to_megabits(bits, 'b')
#         tensor(0.0010, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Mbit':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds * 8 * constants.micro
#
#     return ds
#
# def to_gigabits(val=0.0, unit='Gbit'):
#     """Converts a value from any byte or bit format to gigabits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> kilobytes = 400
#         >>> digital.to_gigabits(kilobytes, 'KB')
#         tensor(0.0032, dtype=torch.float64)
#         >>> pebibit = 400
#         >>> digital.to_gigabits(pebibit, 'Pib')
#         tensor(4.5036e+08, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Gbit':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds * 8 * constants.nano
#
#     return ds
#
# def to_terabits(val=0.0, unit='Tbit'):
#     """Converts a value from any byte or bit format to terabits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> bytes = 6900
#         >>> digital.to_terabits(bytes, 'B')
#         tensor(5.5200e-08, dtype=torch.float64)
#         >>> terabits = 45
#         >>> digital.to_terabits(terabits, 'Tbit')
#         tensor(45., dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Tbit':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds * 8 * constants.pico
#
#     return ds
#
# def to_petabits(val=0.0, unit='Pbit'):
#     """Converts a value from any byte or bit format to petabits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> mebibytes = 465
#         >>> digital.to_petabits(mebibytes, 'MiB')
#         tensor(3.9007e-06, dtype=torch.float64)
#         >>> tebibits = 34
#         >>> digital.to_petabits(tebibits, 'Tib')
#         tensor(0.0374, dtype=torch.float64)
#
#
#         """
#
#     # ds := digital storage
#     if unit == 'Pbit':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds * 8 * constants.femto
#
#     return ds
#
# def to_kibibits(val=0.0, unit='Kib'):
#     """Converts a value from any byte or bit format to kibibits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> gibibyte = 23
#         >>> digital.to_kibibits(gibibyte, 'GiB')
#         tensor(1.9294e+08, dtype=torch.float64)
#         >>> gigabit = 56
#         >>> digital.to_kibibits(gigabit, 'Gbit')
#         tensor(54687500., dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Kib':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / (constants.kibi / 8)
#
#     return ds
#
# def to_mebibits(val=0.0, unit='Mib'):
#     """Converts a value from any byte or bit format to mebibits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> tebibyte = 5
#         >>> digital.to_mebibits(tebibyte, 'TiB')
#         tensor(41943040., dtype=torch.float64)
#         >>> bit = 45
#         >>> digital.to_mebibits(bit, 'b')
#         tensor(4.2915e-05, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Mib':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / (constants.mebi / 8)
#
#     return ds
#
# def to_gibibits(val=0.0, unit='Gib'):
#     """Converts a value from any byte or bit format to gibibits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> kilobyte = 45
#         >>> digital.to_gibibits(kilobyte, 'KB')
#         tensor(0.0003, dtype=torch.float64)
#         >>> mebibit = 235
#         >>> digital.to_gibibits(mebibit, 'Mib')
#         tensor(0.2295, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Gib':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / (constants.gibi / 8)
#
#     return ds
#
# def to_tebibits(val=0.0, unit='Tib'):
#     """Converts a value from any byte or bit format to tebibits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> byte = 5
#         >>> digital.to_tebibits(byte, 'B')
#         tensor(3.6380e-11, dtype=torch.float64)
#         >>> bit = 23
#         >>> digital.to_tebibits(bit, 'b')
#         tensor(2.0918e-11, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Tib':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / (constants.tebi / 8)
#
#     return ds
#
# def to_pebibits(val=0.0, unit='Pib'):
#     """Converts a value from any byte or bit format to pebibits.
#
#      Parameters:
#         -----------
#
#         val -- (int) value
#         scale -- (char) new unit
#
#         Returns:
#         --------
#
#         temp -- (Tensor) value in Kelvin scale
#
#         Example:
#         --------
#
#         >>> petabyte = 15
#         >>> digital.to_pebibits(petabyte, 'PB')
#         tensor(106.5814, dtype=torch.float64)
#         >>> megabit = 1
#         >>> digital.to_pebibits(megabit, 'Mbit')
#         tensor(8.8818e-10, dtype=torch.float64)
#
#         """
#
#     # ds := digital storage
#     if unit == 'Pib':
#         ds = T(val)
#     else:
#         ds = to_bytes(val, unit)
#         ds = ds / (constants.pebi / 8)
#
#    return ds