import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital_storage import to_kilobytes
from scitorch.constants import constants


"""Tests for to_kilobytes"""

# Test function with general parameters
class TestToKilobytes(object):
    def test_to_kilobytes_default_values_scalar(self):
        kilobytes = to_kilobytes()
        assert torch.all(torch.eq(kilobytes, _create_tensor(0)))

    def test_to_kilobytes_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_kilobytes('KB', 1)

    def test_to_kilobytes_wrong_arguments_list(self):
        with raises(TypeError):
            to_kilobytes('B', [0, 1])

    def test_to_kilobytes_tensor(self):
        kilobyte_tensor = torch.tensor([0, 1])
        kilobytes = to_kilobytes(kilobyte_tensor)
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, 1])))

    def test_to_kilobytes_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_kilobytes(1, 'kb')

    def test_to_kilobytes_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_kilobytes([0, 1], 'kb')


# Test function with values that are already byte units (e.g. Kilobyte, Megabyte)
class TestToKilobytesFromByteUnits(object):
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

    def test_to_kilobytes_from_megabyte_scalar(self):
        kilobytes = to_kilobytes(1, 'MB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.kilo)))

    def test_to_kilobytes_from_megabyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'MB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.kilo])))

    def test_to_kilobytes_from_gigabyte_scalar(self):
        kilobytes = to_kilobytes(1, 'GB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.mega)))

    def test_to_kilobytes_from_gigaobyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'GB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.mega])))

    def test_to_kilobytes_from_terabyte_scalar(self):
        kilobytes = to_kilobytes(1, 'TB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.giga)))

    def test_to_kilobytes_from_terabyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'TB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.giga])))

    def test_to_kilobytes_from_petabyte_scalar(self):
        kilobytes = to_kilobytes(1, 'PB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.tera)))

    def test_to_kilobytes_from_petabyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'PB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.tera])))

    def test_to_kilobytes_from_kibibyte_scalar(self):
        kilobytes = to_kilobytes(1, 'KiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.kibi / constants.kilo)))

    def test_to_kilobytes_from_kibibyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'KiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.kibi / constants.kilo])))

    def test_to_kilobytes_from_mebibyte_scalar(self):
        kilobytes = to_kilobytes(1, 'MiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.mebi / constants.kilo)))

    def test_to_kilobytes_from_mebibyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'MiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.mebi / constants.kilo])))

    def test_to_kilobytes_from_gibibyte_scalar(self):
        kilobytes = to_kilobytes(1, 'GiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.gibi / constants.kilo)))

    def test_to_kilobytes_from_gibibyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'GiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.gibi / constants.kilo])))

    def test_to_kilobytes_from_tebibyte_scalar(self):
        kilobytes = to_kilobytes(1, 'TiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.tebi / constants.kilo)))

    def test_to_kilobytes_from_tebibyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'TiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.tebi / constants.kilo])))

    def test_to_kilobytes_from_pebibyte_scalar(self):
        kilobytes = to_kilobytes(1, 'PiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.pebi / constants.kilo)))

    def test_to_kilobytes_from_pebibyte_list(self):
        kilobytes = to_kilobytes([0, 1], 'PiB')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.pebi / constants.kilo])))


# Test function with values that are already bit units (e.g. Kibibit, Megabit)
class TestToKilobytesFromBitUnits(object):
    def test_to_kilobytes_from_bit_scalar(self):
        kilobytes = to_kilobytes(8, 'b')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.milli)))

    def test_to_kilobytes_from_bit_list(self):
        kilobytes = to_kilobytes([0, 8], 'b')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.milli])))

    def test_to_kilobytes_from_kilobit_scalar(self):
        kilobytes = to_kilobytes(8, 'Kbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor(1)))

    def test_to_kilobytes_from_kilobit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Kbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, 1])))

    def test_to_kilobytes_from_megabit_scalar(self):
        kilobytes = to_kilobytes(8, 'Mbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.kilo)))

    def test_to_kilobytes_from_megabit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Mbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.kilo])))

    def test_to_kilobytes_from_gigabit_scalar(self):
        kilobytes = to_kilobytes(8, 'Gbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.mega)))

    def test_to_kilobytes_from_gigabit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Gbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.mega])))

    def test_to_kilobytes_from_terabit_scalar(self):
        kilobytes = to_kilobytes(8, 'Tbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.giga)))

    def test_to_kilobytes_from_terabit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Tbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.giga])))

    def test_to_kilobytes_from_petabit_scalar(self):
        kilobytes = to_kilobytes(8, 'Pbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.tera)))

    def test_to_kilobytes_from_petabit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Pbit')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.tera])))

    def test_to_kilobytes_from_kibibit_scalar(self):
        kilobytes = to_kilobytes(8, 'Kib')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.kibi / constants.kilo)))

    def test_to_kilobytes_from_kibibit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Kib')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.kibi / constants.kilo])))

    def test_to_kilobytes_from_mebibit_scalar(self):
        kilobytes = to_kilobytes(8, 'Mib')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.mebi / constants.kilo)))

    def test_to_kilobytes_from_mebibit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Mib')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.mebi / constants.kilo])))

    def test_to_kilobytes_from_gibibit_scalar(self):
        kilobytes = to_kilobytes(8, 'Gib')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.gibi / constants.kilo)))

    def test_to_kilobytes_from_gibibit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Gib')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.gibi / constants.kilo])))

    def test_to_kilobytes_from_tebibit_scalar(self):
        kilobytes = to_kilobytes(8, 'Tib')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.tebi / constants.kilo)))

    def test_to_kilobytes_from_tebibit_list(self):
        kilobytes = to_kilobytes([0, 8], 'Tib')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.tebi / constants.kilo])))

    def test_to_kilobytes_from_pebibyte_scalar(self):
        kilobytes = to_kilobytes(8, 'Pib')
        assert torch.all(torch.eq(kilobytes, _create_tensor(constants.pebi / constants.kilo)))

    def test_to_kilobytes_from_pebibyte_list(self):
        kilobytes = to_kilobytes([0, 8], 'Pib')
        assert torch.all(torch.eq(kilobytes, _create_tensor([0, constants.pebi / constants.kilo])))
