"""Conversion of different digital storage types (Bytes, Bits)."""

from scitorch.conversion._standard_units import bytes

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

    return bytes(val, unit)
