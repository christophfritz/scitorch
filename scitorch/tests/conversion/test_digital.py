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


class TestToKilobytes(object):
    def test_to_kilobytes_default_values_scalar(self):
        kilobytes = to_kilobytes()
        assert torch.all(torch.eq(kilobytes, _create_tensor(0)))

    def test_to_kilobytes_from_byte_scalar(self):
        kilobytes = to_kilobytes(1, 'B')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.milli)))

    def test_to_kilobytes_from_byte_list(self):
        kilobytes = to_kilobytes([0, 1], 'B')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.milli])))

    def test_to_kilobytes_from_kilobyte_scalar(self):
        kilobytes = to_kilobytes(1, 'KB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(1)))

    def test_to_kilobytes_from_kilobyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'KB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, 1])))


class TestToMegabytes(object):
    def test_to_megabytes_default_values_scalar(self):
        megabytes = to_megabytes()
        assert torch.all(torch.eq(megabytes, _create_tensor(0)))

    def test_to_megabytes_from_byte_scalar(self):
        megabytes = to_megabytes(1, 'B')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.micro)))

    def test_to_megabytes_from_byte_list(self):
        megabytes = to_megabytes([0, 1], 'B')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.micro])))

    def test_to_megabytes_from_megabyte_scalar(self):
        megabytes = to_megabytes(1, 'MB')
        assert torch.all(torch.eq(megabytes, _create_tensor(1)))

    def test_to_megabytes_from_megabyte_list(self):
        megabytes = to_megabytes([0, 1], 'MB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, 1])))


class TestToGigabytes(object):
    def test_to_gigabytes_default_values_scalar(self):
        gigabytes = to_gigabytes()
        assert torch.all(torch.eq(gigabytes, _create_tensor(0)))

    def test_to_gigabytes_from_byte_scalar(self):
        gigabytes = to_gigabytes(1, 'B')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.nano)))

    def test_to_gigabytes_from_byte_list(self):
        gigabytes = to_gigabytes([0, 1], 'B')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.nano])))

    def test_to_gigabytes_from_gigabyte_scalar(self):
        gigabytes = to_gigabytes(1, 'GB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(1)))

    def test_to_gigabytes_from_gigabyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'GB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, 1])))
        
    
class TestToTerabytes(object):
    def test_to_terabytes_default_values_scalar(self):
        terabytes = to_terabytes()
        assert torch.all(torch.eq(terabytes, _create_tensor(0)))

    def test_to_terabytes_from_byte_scalar(self):
        terabytes = to_terabytes(1, 'B')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.pico)))

    def test_to_terabytes_from_byte_list(self):
        terabytes = to_terabytes([0, 1], 'B')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.pico])))

    def test_to_terabytes_from_terabyte_scalar(self):
        terabytes = to_terabytes(1, 'TB')
        assert torch.all(torch.eq(terabytes, _create_tensor(1)))

    def test_to_terabytes_from_terabyte_list(self):
        terabytes = to_terabytes([0, 1], 'TB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, 1])))


class TestToPetabytes(object):
    def test_to_petabytes_default_values_scalar(self):
        petabytes = to_petabytes()
        assert torch.all(torch.eq(petabytes, _create_tensor(0)))

    def test_to_petabytes_from_byte_scalar(self):
        petabytes = to_petabytes(1, 'B')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.femto)))

    def test_to_petabytes_from_byte_list(self):
        petabytes = to_petabytes([0, 1], 'B')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.femto])))

    def test_to_petabytes_from_petabyte_scalar(self):
        petabytes = to_petabytes(1, 'PB')
        assert torch.all(torch.eq(petabytes, _create_tensor(1)))

    def test_to_petabytes_from_petabyte_list(self):
        petabytes = to_petabytes([0, 1], 'PB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, 1])))


class TestToKibibyte(object):
    def test_to_kibibytes_default_values_scalar(self):
        kibibytes = to_kibibytes()
        assert torch.all(torch.eq(kibibytes, _create_tensor(0)))

    def test_to_kibibytes_from_kibibytes_scalar(self):
        kibibytes = to_kibibytes(1, 'KiB')
        assert torch.all(torch.eq(kibibytes, _create_tensor(1)))

    def test_to_kibibytes_from_kibibytes_list(self):
        kibibytes = to_kibibytes([0, 1], 'KiB')
        assert torch.all(torch.eq(kibibytes, _create_tensor([0, 1])))

    def test_to_kibibytes_from_byte_scalar(self):
        kibibytes = to_kibibytes(1, 'B')
        assert torch.all(torch.eq(kibibytes, _create_tensor(1 / constants.kibi)))

    def test_to_kibibytes_from_byte_list(self):
        kibibytes = to_kibibytes([0, 1], 'B')
        assert torch.all(torch.eq(kibibytes, _create_tensor([0, 1 / constants.kibi])))


class TestToMebibyte(object):
    def test_to_mebibytes_default_values_scalar(self):
        mebibytes = to_mebibytes()
        assert torch.all(torch.eq(mebibytes, _create_tensor(0)))

    def test_to_mebibytes_from_mebibytes_scalar(self):
        mebibytes = to_mebibytes(1, 'MiB')
        assert torch.all(torch.eq(mebibytes, _create_tensor(1)))

    def test_to_mebibytes_from_mebibytes_list(self):
        mebibytes = to_mebibytes([0, 1], 'MiB')
        assert torch.all(torch.eq(mebibytes, _create_tensor([0, 1])))

    def test_to_mebibytes_from_byte_scalar(self):
        mebibytes = to_mebibytes(1, 'B')
        assert torch.all(torch.eq(mebibytes, _create_tensor(1 / constants.mebi)))

    def test_to_mebibytes_from_byte_list(self):
        mebibytes = to_mebibytes([0, 1], 'B')
        assert torch.all(torch.eq(mebibytes, _create_tensor([0, 1 / constants.mebi])))

