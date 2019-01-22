"""
    Tests for the conversion.digital module

    As to_bytes() is the basis of all the other functions most of the things are tested there. Therefore,
    mainly the default values and the conversion from bytes to the new unit are tested in the other functions.
"""

import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital import *
from scitorch.constants import constants

"""Unit Tests"""

# Test function with general parameters
class TestToBytes(object):
    def test_to_bytes_default_values_scalar(self):
        bytes = to_bytes()
        assert torch.all(torch.eq(bytes, _create_tensor(0)))

    def test_to_bytes_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_bytes('B', 1)

    def test_to_bytes_wrong_arguments_list(self):
        with raises(TypeError):
            to_bytes('B', [0, 1])

    def test_to_bytes_tensor(self):
        byte_tensor = torch.tensor([0, 1])
        bytes = to_bytes(byte_tensor)
        assert torch.all(torch.eq(bytes, _create_tensor([0, 1])))

    def test_to_bytes_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_bytes(1, 'kb')

    def test_to_bytes_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_bytes([0, 1], 'kb')

    def test_to_bytes_with_dimension(self):
        bytes = to_bytes([0, 1], 'GB', dim=True)
        assert isinstance(bytes, dict)
        assert torch.all(torch.eq(bytes['val'], _create_tensor([0, constants.giga]))) and bytes['dim'] == 'B'


# Test function with values that are already byte units (e.g. Kilobyte, Megabyte)
class TestToBytesFromByteUnits(object):
    def test_to_bytes_from_byte_scalar(self):
        bytes = to_bytes(1, 'B')
        assert torch.all(torch.eq(bytes, _create_tensor(1)))

    def test_to_bytes_from_byte_list(self):
        bytes = to_bytes([0, 1], 'B')
        assert torch.all(torch.eq(bytes, _create_tensor([0, 1])))

    def test_to_bytes_from_kilobyte_scalar(self):
        bytes = to_bytes(1, 'KB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.kilo)))

    def test_to_bytes_from_kilobyte_list(self):
        bytes = to_bytes([0, 1], 'KB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.kilo])))

    def test_to_bytes_from_megabyte_scalar(self):
        bytes = to_bytes(1, 'MB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.mega)))

    def test_to_bytes_from_megabyte_list(self):
        bytes = to_bytes([0, 1], 'MB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.mega])))

    def test_to_bytes_from_gigabyte_scalar(self):
        bytes = to_bytes(1, 'GB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.giga)))

    def test_to_bytes_from_gigaobyte_list(self):
        bytes = to_bytes([0, 1], 'GB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.giga])))

    def test_to_bytes_from_terabyte_scalar(self):
        bytes = to_bytes(1, 'TB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.tera)))

    def test_to_bytes_from_terabyte_list(self):
        bytes = to_bytes([0, 1], 'TB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.tera])))

    def test_to_bytes_from_petabyte_scalar(self):
        bytes = to_bytes(1, 'PB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.peta)))

    def test_to_bytes_from_petabyte_list(self):
        bytes = to_bytes([0, 1], 'PB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.peta])))

    def test_to_bytes_from_kibibyte_scalar(self):
        bytes = to_bytes(1, 'KiB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.kibi)))

    def test_to_bytes_from_kibibyte_list(self):
        bytes = to_bytes([0, 1], 'KiB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.kibi])))

    def test_to_bytes_from_mebibyte_scalar(self):
        bytes = to_bytes(1, 'MiB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.mebi)))

    def test_to_bytes_from_mebibyte_list(self):
        bytes = to_bytes([0, 1], 'MiB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.mebi])))

    def test_to_bytes_from_gibibyte_scalar(self):
        bytes = to_bytes(1, 'GiB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.gibi)))

    def test_to_bytes_from_gibibyte_list(self):
        bytes = to_bytes([0, 1], 'GiB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.gibi])))

    def test_to_bytes_from_tebibyte_scalar(self):
        bytes = to_bytes(1, 'TiB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.tebi)))

    def test_to_bytes_from_tebibyte_list(self):
        bytes = to_bytes([0, 1], 'TiB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.tebi])))

    def test_to_bytes_from_pebibyte_scalar(self):
        bytes = to_bytes(1, 'PiB')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.pebi)))

    def test_to_bytes_from_pebibyte_list(self):
        bytes = to_bytes([0, 1], 'PiB')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.pebi])))


# Test function with values that are already bit units (e.g. Kibibit, Megabit)
class TestToBytesFromBitUnits(object):
    def test_to_bytes_from_bit_scalar(self):
        bytes = to_bytes(8, 'b')
        assert torch.all(torch.eq(bytes, _create_tensor(1)))

    def test_to_bytes_from_bit_list(self):
        bytes = to_bytes([0, 8], 'b')
        assert torch.all(torch.eq(bytes, _create_tensor([0, 1])))

    def test_to_bytes_from_kilobit_scalar(self):
        bytes = to_bytes(8, 'Kbit')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.kilo)))

    def test_to_bytes_from_kilobit_list(self):
        bytes = to_bytes([0, 8], 'Kbit')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.kilo])))

    def test_to_bytes_from_megabit_scalar(self):
        bytes = to_bytes(8, 'Mbit')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.mega)))

    def test_to_bytes_from_megabit_list(self):
        bytes = to_bytes([0, 8], 'Mbit')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.mega])))

    def test_to_bytes_from_gigabit_scalar(self):
        bytes = to_bytes(8, 'Gbit')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.giga)))

    def test_to_bytes_from_gigabit_list(self):
        bytes = to_bytes([0, 8], 'Gbit')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.giga])))

    def test_to_bytes_from_terabit_scalar(self):
        bytes = to_bytes(8, 'Tbit')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.tera)))

    def test_to_bytes_from_terabit_list(self):
        bytes = to_bytes([0, 8], 'Tbit')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.tera])))

    def test_to_bytes_from_petabit_scalar(self):
        bytes = to_bytes(8, 'Pbit')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.peta)))

    def test_to_bytes_from_petabit_list(self):
        bytes = to_bytes([0, 8], 'Pbit')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.peta])))

    def test_to_bytes_from_kibibit_scalar(self):
        bytes = to_bytes(1, 'Kib')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.kibi / 8)))

    def test_to_bytes_from_kibibit_list(self):
        bytes = to_bytes([0, 1], 'Kib')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.kibi / 8])))

    def test_to_bytes_from_mebibit_scalar(self):
        bytes = to_bytes(1, 'Mib')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.mebi / 8)))

    def test_to_bytes_from_mebibit_list(self):
        bytes = to_bytes([0, 1], 'Mib')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.mebi / 8])))

    def test_to_bytes_from_gibibit_scalar(self):
        bytes = to_bytes(1, 'Gib')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.gibi / 8)))

    def test_to_bytes_from_gibibit_list(self):
        bytes = to_bytes([0, 1], 'Gib')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.gibi / 8])))

    def test_to_bytes_from_tebibit_scalar(self):
        bytes = to_bytes(1, 'Tib')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.tebi / 8)))

    def test_to_bytes_from_tebibit_list(self):
        bytes = to_bytes([0, 1], 'Tib')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.tebi / 8])))

    def test_to_bytes_from_pebibyte_scalar(self):
        bytes = to_bytes(1, 'Pib')
        assert torch.all(torch.eq(bytes, _create_tensor(constants.pebi / 8)))

    def test_to_bytes_from_pebibyte_list(self):
        bytes = to_bytes([0, 1], 'Pib')
        assert torch.all(torch.eq(bytes, _create_tensor([0, constants.pebi / 8])))

#
# class TestToKilobytes(object):
#     def test_to_kilobytes_default_values_scalar(self):
#         kilobytes = to_kilobytes()
#         assert torch.all(torch.eq(kilobytes, _create_tensor(0)))
#
#     def test_to_kilobytes_from_byte_scalar(self):
#         kilobytes = to_kilobytes(1, 'B')
#         assert torch.all(torch.eq(kilobytes, _create_tensor(constants.milli)))
#
#     def test_to_kilobytes_from_byte_list(self):
#         kilobytes = to_kilobytes([0, 1], 'B')
#         assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.milli])))
#
#     def test_to_kilobytes_from_kilobyte_scalar(self):
#         kilobytes = to_kilobytes(1, 'KB')
#         assert torch.all(torch.eq(kilobytes, _create_tensor(1)))
#
#     def test_to_kilobytes_from_kilobyte_list(self):
#         kilobytes = to_kilobytes([0, 1], 'KB')
#         assert torch.all(torch.eq(kilobytes, _create_tensor([0, 1])))
#
#
# class TestToMegabytes(object):
#     def test_to_megabytes_default_values_scalar(self):
#         megabytes = to_megabytes()
#         assert torch.all(torch.eq(megabytes, _create_tensor(0)))
#
#     def test_to_megabytes_from_byte_scalar(self):
#         megabytes = to_megabytes(1, 'B')
#         assert torch.all(torch.eq(megabytes, _create_tensor(constants.micro)))
#
#     def test_to_megabytes_from_byte_list(self):
#         megabytes = to_megabytes([0, 1], 'B')
#         assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.micro])))
#
#     def test_to_megabytes_from_megabyte_scalar(self):
#         megabytes = to_megabytes(1, 'MB')
#         assert torch.all(torch.eq(megabytes, _create_tensor(1)))
#
#     def test_to_megabytes_from_megabyte_list(self):
#         megabytes = to_megabytes([0, 1], 'MB')
#         assert torch.all(torch.eq(megabytes, _create_tensor([0, 1])))
#
#
# class TestToGigabytes(object):
#     def test_to_gigabytes_default_values_scalar(self):
#         gigabytes = to_gigabytes()
#         assert torch.all(torch.eq(gigabytes, _create_tensor(0)))
#
#     def test_to_gigabytes_from_byte_scalar(self):
#         gigabytes = to_gigabytes(1, 'B')
#         assert torch.all(torch.eq(gigabytes, _create_tensor(constants.nano)))
#
#     def test_to_gigabytes_from_byte_list(self):
#         gigabytes = to_gigabytes([0, 1], 'B')
#         assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.nano])))
#
#     def test_to_gigabytes_from_gigabyte_scalar(self):
#         gigabytes = to_gigabytes(1, 'GB')
#         assert torch.all(torch.eq(gigabytes, _create_tensor(1)))
#
#     def test_to_gigabytes_from_gigabyte_list(self):
#         gigabytes = to_gigabytes([0, 1], 'GB')
#         assert torch.all(torch.eq(gigabytes, _create_tensor([0, 1])))
#
#
# class TestToTerabytes(object):
#     def test_to_terabytes_default_values_scalar(self):
#         terabytes = to_terabytes()
#         assert torch.all(torch.eq(terabytes, _create_tensor(0)))
#
#     def test_to_terabytes_from_byte_scalar(self):
#         terabytes = to_terabytes(1, 'B')
#         assert torch.all(torch.eq(terabytes, _create_tensor(constants.pico)))
#
#     def test_to_terabytes_from_byte_list(self):
#         terabytes = to_terabytes([0, 1], 'B')
#         assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.pico])))
#
#     def test_to_terabytes_from_terabyte_scalar(self):
#         terabytes = to_terabytes(1, 'TB')
#         assert torch.all(torch.eq(terabytes, _create_tensor(1)))
#
#     def test_to_terabytes_from_terabyte_list(self):
#         terabytes = to_terabytes([0, 1], 'TB')
#         assert torch.all(torch.eq(terabytes, _create_tensor([0, 1])))
#
#
# class TestToPetabytes(object):
#     def test_to_petabytes_default_values_scalar(self):
#         petabytes = to_petabytes()
#         assert torch.all(torch.eq(petabytes, _create_tensor(0)))
#
#     def test_to_petabytes_from_byte_scalar(self):
#         petabytes = to_petabytes(1, 'B')
#         assert torch.all(torch.eq(petabytes, _create_tensor(constants.femto)))
#
#     def test_to_petabytes_from_byte_list(self):
#         petabytes = to_petabytes([0, 1], 'B')
#         assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.femto])))
#
#     def test_to_petabytes_from_petabyte_scalar(self):
#         petabytes = to_petabytes(1, 'PB')
#         assert torch.all(torch.eq(petabytes, _create_tensor(1)))
#
#     def test_to_petabytes_from_petabyte_list(self):
#         petabytes = to_petabytes([0, 1], 'PB')
#         assert torch.all(torch.eq(petabytes, _create_tensor([0, 1])))
#
#
# class TestToKibibyte(object):
#     def test_to_kibibytes_default_values_scalar(self):
#         kibibytes = to_kibibytes()
#         assert torch.all(torch.eq(kibibytes, _create_tensor(0)))
#
#     def test_to_kibibytes_from_kibibytes_scalar(self):
#         kibibytes = to_kibibytes(1, 'KiB')
#         assert torch.all(torch.eq(kibibytes, _create_tensor(1)))
#
#     def test_to_kibibytes_from_kibibytes_list(self):
#         kibibytes = to_kibibytes([0, 1], 'KiB')
#         assert torch.all(torch.eq(kibibytes, _create_tensor([0, 1])))
#
#     def test_to_kibibytes_from_byte_scalar(self):
#         kibibytes = to_kibibytes(1, 'B')
#         assert torch.all(torch.eq(kibibytes, _create_tensor(1 / constants.kibi)))
#
#     def test_to_kibibytes_from_byte_list(self):
#         kibibytes = to_kibibytes([0, 1], 'B')
#         assert torch.all(torch.eq(kibibytes, _create_tensor([0, 1 / constants.kibi])))
#
#
# class TestToMebibyte(object):
#     def test_to_mebibytes_default_values_scalar(self):
#         mebibytes = to_mebibytes()
#         assert torch.all(torch.eq(mebibytes, _create_tensor(0)))
#
#     def test_to_mebibytes_from_mebibytes_scalar(self):
#         mebibytes = to_mebibytes(1, 'MiB')
#         assert torch.all(torch.eq(mebibytes, _create_tensor(1)))
#
#     def test_to_mebibytes_from_mebibytes_list(self):
#         mebibytes = to_mebibytes([0, 1], 'MiB')
#         assert torch.all(torch.eq(mebibytes, _create_tensor([0, 1])))
#
#     def test_to_mebibytes_from_byte_scalar(self):
#         mebibytes = to_mebibytes(1, 'B')
#         assert torch.all(torch.eq(mebibytes, _create_tensor(1 / constants.mebi)))
#
#     def test_to_mebibytes_from_byte_list(self):
#         mebibytes = to_mebibytes([0, 1], 'B')
#         assert torch.all(torch.eq(mebibytes, _create_tensor([0, 1 / constants.mebi])))
#
#
# class TestToGibibyte(object):
#     def test_to_gibibytes_default_values_scalar(self):
#         gibibytes = to_gibibytes()
#         assert torch.all(torch.eq(gibibytes, _create_tensor(0)))
#
#     def test_to_gibibytes_from_gibibytes_scalar(self):
#         gibibytes = to_gibibytes(1, 'GiB')
#         assert torch.all(torch.eq(gibibytes, _create_tensor(1)))
#
#     def test_to_gibibytes_from_gibibytes_list(self):
#         gibibytes = to_gibibytes([0, 1], 'GiB')
#         assert torch.all(torch.eq(gibibytes, _create_tensor([0, 1])))
#
#     def test_to_gibibytes_from_byte_scalar(self):
#         gibibytes = to_gibibytes(1, 'B')
#         assert torch.all(torch.eq(gibibytes, _create_tensor(1 / constants.gibi)))
#
#     def test_to_gibibytes_from_byte_list(self):
#         gibibytes = to_gibibytes([0, 1], 'B')
#         assert torch.all(torch.eq(gibibytes, _create_tensor([0, 1 / constants.gibi])))
#
#
# class TestToTebibyte(object):
#     def test_to_tebibytes_default_values_scalar(self):
#         tebibytes = to_tebibytes()
#         assert torch.all(torch.eq(tebibytes, _create_tensor(0)))
#
#     def test_to_tebibytes_from_tebibytes_scalar(self):
#         tebibytes = to_tebibytes(1, 'TiB')
#         assert torch.all(torch.eq(tebibytes, _create_tensor(1)))
#
#     def test_to_tebibytes_from_tebibytes_list(self):
#         tebibytes = to_tebibytes([0, 1], 'TiB')
#         assert torch.all(torch.eq(tebibytes, _create_tensor([0, 1])))
#
#     def test_to_tebibytes_from_byte_scalar(self):
#         tebibytes = to_tebibytes(1, 'B')
#         assert torch.all(torch.eq(tebibytes, _create_tensor(1 / constants.tebi)))
#
#     def test_to_tebibytes_from_byte_list(self):
#         tebibytes = to_tebibytes([0, 1], 'B')
#         assert torch.all(torch.eq(tebibytes, _create_tensor([0, 1 / constants.tebi])))
#
#
# class TestToPebibyte(object):
#     def test_to_pebibytes_default_values_scalar(self):
#         pebibytes = to_pebibytes()
#         assert torch.all(torch.eq(pebibytes, _create_tensor(0)))
#
#     def test_to_pebibytes_from_pebibytes_scalar(self):
#         pebibytes = to_pebibytes(1, 'PiB')
#         assert torch.all(torch.eq(pebibytes, _create_tensor(1)))
#
#     def test_to_pebibytes_from_pebibytes_list(self):
#         pebibytes = to_pebibytes([0, 1], 'PiB')
#         assert torch.all(torch.eq(pebibytes, _create_tensor([0, 1])))
#
#     def test_to_pebibytes_from_byte_scalar(self):
#         pebibytes = to_pebibytes(1, 'B')
#         assert torch.all(torch.eq(pebibytes, _create_tensor(1 / constants.pebi)))
#
#     def test_to_pebibytes_from_byte_list(self):
#         pebibytes = to_pebibytes([0, 1], 'B')
#         assert torch.all(torch.eq(pebibytes, _create_tensor([0, 1 / constants.pebi])))


class TestToBit(object):
    def test_to_bits_default_values_scalar(self):
        bits = to_bits()
        assert torch.all(torch.eq(bits, _create_tensor(0)))

    def test_to_bits_with_dimension(self):
        bits = to_bits([0, 1], 'B', dim=True)
        assert isinstance(bits, dict)
        assert torch.all(torch.eq(bits['val'], _create_tensor([0, 8]))) and bits['dim'] == 'b'

    def test_to_bits_from_bits_scalar(self):
        bits = to_bits(1, 'b')
        assert torch.all(torch.eq(bits, _create_tensor(1)))

    def test_to_bits_from_bits_list(self):
        bits = to_bits([0, 1], 'b')
        assert torch.all(torch.eq(bits, _create_tensor([0, 1])))

    def test_to_bits_from_byte_scalar(self):
        bits = to_bits(1, 'B')
        assert torch.all(torch.eq(bits, _create_tensor(8)))

    def test_to_bits_from_byte_list(self):
        bits = to_bits([0, 1], 'B')
        assert torch.all(torch.eq(bits, _create_tensor([0, 8])))


# class TestToKilobit(object):
#     def test_to_kilobits_default_values_scalar(self):
#         kilobits = to_kilobits()
#         assert torch.all(torch.eq(kilobits, _create_tensor(0)))
#
#     def test_to_kilobits_from_kilobits_scalar(self):
#         kilobits = to_kilobits(1, 'Kbit')
#         assert torch.all(torch.eq(kilobits, _create_tensor(1)))
#
#     def test_to_kilobits_from_kilobits_list(self):
#         kilobits = to_kilobits([0, 1], 'Kbit')
#         assert torch.all(torch.eq(kilobits, _create_tensor([0, 1])))
#
#     def test_to_kilobits_from_byte_scalar(self):
#         kilobits = to_kilobits(1, 'B')
#         assert torch.all(torch.eq(kilobits, _create_tensor(8 * constants.milli)))
#
#     def test_to_kilobits_from_byte_list(self):
#         kilobits = to_kilobits([0, 1], 'B')
#         assert torch.all(torch.eq(kilobits, _create_tensor([0, 8 * constants.milli])))
#
#
# class TestToMegabit(object):
#     def test_to_megabits_default_values_scalar(self):
#         megabits = to_megabits()
#         assert torch.all(torch.eq(megabits, _create_tensor(0)))
#
#     def test_to_megabits_from_megabits_scalar(self):
#         megabits = to_megabits(1, 'Mbit')
#         assert torch.all(torch.eq(megabits, _create_tensor(1)))
#
#     def test_to_megabits_from_megabits_list(self):
#         megabits = to_megabits([0, 1], 'Mbit')
#         assert torch.all(torch.eq(megabits, _create_tensor([0, 1])))
#
#     def test_to_megabits_from_byte_scalar(self):
#         megabits = to_megabits(1, 'B')
#         assert torch.all(torch.eq(megabits, _create_tensor(8 * constants.micro)))
#
#     def test_to_megabits_from_byte_list(self):
#         megabits = to_megabits([0, 1], 'B')
#         assert torch.all(torch.eq(megabits, _create_tensor([0, 8 * constants.micro])))
#
#
# class TestToGigabit(object):
#     def test_to_gigabits_default_values_scalar(self):
#         gigabits = to_gigabits()
#         assert torch.all(torch.eq(gigabits, _create_tensor(0)))
#
#     def test_to_gigabits_from_gigabits_scalar(self):
#         gigabits = to_gigabits(1, 'Gbit')
#         assert torch.all(torch.eq(gigabits, _create_tensor(1)))
#
#     def test_to_gigabits_from_gigabits_list(self):
#         gigabits = to_gigabits([0, 1], 'Gbit')
#         assert torch.all(torch.eq(gigabits, _create_tensor([0, 1])))
#
#     def test_to_gigabits_from_byte_scalar(self):
#         gigabits = to_gigabits(1, 'B')
#         assert torch.all(torch.eq(gigabits, _create_tensor(8 * constants.nano)))
#
#     def test_to_gigabits_from_byte_list(self):
#         gigabits = to_gigabits([0, 1], 'B')
#         assert torch.all(torch.eq(gigabits, _create_tensor([0, 8 * constants.nano])))
#
#
# class TestToTerabit(object):
#     def test_to_terabits_default_values_scalar(self):
#         terabits = to_terabits()
#         assert torch.all(torch.eq(terabits, _create_tensor(0)))
#
#     def test_to_terabits_from_terabits_scalar(self):
#         terabits = to_terabits(1, 'Tbit')
#         assert torch.all(torch.eq(terabits, _create_tensor(1)))
#
#     def test_to_terabits_from_terabits_list(self):
#         terabits = to_terabits([0, 1], 'Tbit')
#         assert torch.all(torch.eq(terabits, _create_tensor([0, 1])))
#
#     def test_to_terabits_from_byte_scalar(self):
#         terabits = to_terabits(1, 'B')
#         assert torch.all(torch.eq(terabits, _create_tensor(8 * constants.pico)))
#
#     def test_to_terabits_from_byte_list(self):
#         terabits = to_terabits([0, 1], 'B')
#         assert torch.all(torch.eq(terabits, _create_tensor([0, 8 * constants.pico])))
#
#
# class TestToPetabit(object):
#     def test_to_petabits_default_values_scalar(self):
#         petabits = to_petabits()
#         assert torch.all(torch.eq(petabits, _create_tensor(0)))
#
#     def test_to_petabits_from_petabits_scalar(self):
#         petabits = to_petabits(1, 'Pbit')
#         assert torch.all(torch.eq(petabits, _create_tensor(1)))
#
#     def test_to_petabits_from_petabits_list(self):
#         petabits = to_petabits([0, 1], 'Pbit')
#         assert torch.all(torch.eq(petabits, _create_tensor([0, 1])))
#
#     def test_to_petabits_from_byte_scalar(self):
#         petabits = to_petabits(1, 'B')
#         assert torch.all(torch.eq(petabits, _create_tensor(8 * constants.femto)))
#
#     def test_to_petabits_from_byte_list(self):
#         petabits = to_petabits([0, 1], 'B')
#         assert torch.all(torch.eq(petabits, _create_tensor([0, 8 * constants.femto])))
#
#
# class TestToKibibit(object):
#     def test_to_kibibits_default_values_scalar(self):
#         kibibits = to_kibibits()
#         assert torch.all(torch.eq(kibibits, _create_tensor(0)))
#
#     def test_to_kibibits_from_kibibits_scalar(self):
#         kibibits = to_kibibits(1, 'Kib')
#         assert torch.all(torch.eq(kibibits, _create_tensor(1)))
#
#     def test_to_kibibits_from_kibibits_list(self):
#         kibibits = to_kibibits([0, 1], 'Kib')
#         assert torch.all(torch.eq(kibibits, _create_tensor([0, 1])))
#
#     def test_to_kibibits_from_byte_scalar(self):
#         kibibits = to_kibibits(1, 'B')
#         assert torch.all(torch.eq(kibibits, _create_tensor(1 / (constants.kibi / 8))))
#
#     def test_to_kibibits_from_byte_list(self):
#         kibibits = to_kibibits([0, 1], 'B')
#         assert torch.all(torch.eq(kibibits, _create_tensor([0, 1 / (constants.kibi / 8)])))
#
#
# class TestToMebibit(object):
#     def test_to_mebibits_default_values_scalar(self):
#         mebibits = to_mebibits()
#         assert torch.all(torch.eq(mebibits, _create_tensor(0)))
#
#     def test_to_mebibits_from_mebibits_scalar(self):
#         mebibits = to_mebibits(1, 'Mib')
#         assert torch.all(torch.eq(mebibits, _create_tensor(1)))
#
#     def test_to_mebibits_from_mebibits_list(self):
#         mebibits = to_mebibits([0, 1], 'Mib')
#         assert torch.all(torch.eq(mebibits, _create_tensor([0, 1])))
#
#     def test_to_mebibits_from_byte_scalar(self):
#         mebibits = to_mebibits(1, 'B')
#         assert torch.all(torch.eq(mebibits, _create_tensor(1 / (constants.mebi / 8))))
#
#     def test_to_mebibits_from_byte_list(self):
#         mebibits = to_mebibits([0, 1], 'B')
#         assert torch.all(torch.eq(mebibits, _create_tensor([0, 1 / (constants.mebi / 8)])))
#
#
# class TestToGibibit(object):
#     def test_to_gibibits_default_values_scalar(self):
#         gibibits = to_gibibits()
#         assert torch.all(torch.eq(gibibits, _create_tensor(0)))
#
#     def test_to_gibibits_from_gibibits_scalar(self):
#         gibibits = to_gibibits(1, 'Gib')
#         assert torch.all(torch.eq(gibibits, _create_tensor(1)))
#
#     def test_to_gibibits_from_gibibits_list(self):
#         gibibits = to_gibibits([0, 1], 'Gib')
#         assert torch.all(torch.eq(gibibits, _create_tensor([0, 1])))
#
#     def test_to_gibibits_from_byte_scalar(self):
#         gibibits = to_gibibits(1, 'B')
#         assert torch.all(torch.eq(gibibits, _create_tensor(1 / (constants.gibi / 8))))
#
#     def test_to_gibibits_from_byte_list(self):
#         gibibits = to_gibibits([0, 1], 'B')
#         assert torch.all(torch.eq(gibibits, _create_tensor([0, 1 / (constants.gibi / 8)])))
#
#
# class TestToTebibit(object):
#     def test_to_tebibits_default_values_scalar(self):
#         tebibits = to_tebibits()
#         assert torch.all(torch.eq(tebibits, _create_tensor(0)))
#
#     def test_to_tebibits_from_tebibits_scalar(self):
#         tebibits = to_tebibits(1, 'Tib')
#         assert torch.all(torch.eq(tebibits, _create_tensor(1)))
#
#     def test_to_tebibits_from_tebibits_list(self):
#         tebibits = to_tebibits([0, 1], 'Tib')
#         assert torch.all(torch.eq(tebibits, _create_tensor([0, 1])))
#
#     def test_to_tebibits_from_byte_scalar(self):
#         tebibits = to_tebibits(1, 'B')
#         assert torch.all(torch.eq(tebibits, _create_tensor(1 / (constants.tebi / 8))))
#
#     def test_to_tebibits_from_byte_list(self):
#         tebibits = to_tebibits([0, 1], 'B')
#         assert torch.all(torch.eq(tebibits, _create_tensor([0, 1 / (constants.tebi / 8)])))
#
#
# class TestToPebibit(object):
#     def test_to_pebibits_default_values_scalar(self):
#         pebibits = to_pebibits()
#         assert torch.all(torch.eq(pebibits, _create_tensor(0)))
#
#     def test_to_pebibits_from_pebibits_scalar(self):
#         pebibits = to_pebibits(1, 'Pib')
#         assert torch.all(torch.eq(pebibits, _create_tensor(1)))
#
#     def test_to_pebibits_from_pebibits_list(self):
#         pebibits = to_pebibits([0, 1], 'Pib')
#         assert torch.all(torch.eq(pebibits, _create_tensor([0, 1])))
#
#     def test_to_pebibits_from_byte_scalar(self):
#         pebibits = to_pebibits(1, 'B')
#         assert torch.all(torch.eq(pebibits, _create_tensor(1 / (constants.pebi / 8))))
#
#     def test_to_pebibits_from_byte_list(self):
#         pebibits = to_pebibits([0, 1], 'B')
#         assert torch.all(torch.eq(pebibits, _create_tensor([0, 1 / (constants.pebi / 8)])))
#
# """Functional Tests"""
#
# class TestDigitalFunctional(object):
#     """Imagine buying two hard drives. One hard drive has 500 GB and is 40$, the other one has 500 GiB and is 42$.
#     What is the storage (in GB) per dollar for both of them?"""
#
#     hd1 = to_gigabytes(500, 'GB')
#     hd2 = to_gigabytes(500, 'GiB')
#
#     hd1_price = 40/hd1
#     hd2_price = 42/hd2
#
#     assert round(hd1_price.item(), 4) == 0.08
#     assert round(hd2_price.item(), 4) == 0.0782

