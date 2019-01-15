import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital import to_terabytes
from scitorch.constants import constants


"""Tests for to_terabytes"""

# Test function with general parameters
class TestToTerabytes(object):
    def test_to_terabytes_default_values_scalar(self):
        terabytes = to_terabytes()
        assert torch.all(torch.eq(terabytes, _create_tensor(0)))

    def test_to_terabytes_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_terabytes('MB', 1)

    def test_to_terabytes_wrong_arguments_list(self):
        with raises(TypeError):
            to_terabytes('MB', [0, 1])

    def test_to_terabytes_tensor(self):
        terabyte_tensor = torch.tensor([0, 1])
        terabytes = to_terabytes(terabyte_tensor)
        assert torch.all(torch.eq(terabytes, _create_tensor([0, 1])))

    def test_to_terabytes_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_terabytes(1, 'kb')

    def test_to_terabytes_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_terabytes([0, 1], 'kb')


# Test function with values that are already byte units (e.g. Kilobyte, Megabyte)
class TestToTerabytesFromByteUnits(object):
    def test_to_terabytes_from_byte_scalar(self):
        terabytes = to_terabytes(1, 'B')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.pico)))

    def test_to_terabytes_from_byte_list(self):
        terabytes = to_terabytes([0, 1], 'B')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.pico])))

    def test_to_terabytes_from_kilobyte_scalar(self):
        terabytes = to_terabytes(1, 'KB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.nano)))

    def test_to_terabytes_from_kilobyte_list(self):
        terabytes = to_terabytes([0, 1], 'KB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.nano])))

    def test_to_terabytes_from_megabyte_scalar(self):
        terabytes = to_terabytes(1, 'MB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.micro)))

    def test_to_terabytes_from_megabyte_list(self):
        terabytes = to_terabytes([0, 1], 'MB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.micro])))

    def test_to_terabytes_from_gigabyte_scalar(self):
        terabytes = to_terabytes(1, 'GB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.milli)))

    def test_to_terabytes_from_gigaobyte_list(self):
        terabytes = to_terabytes([0, 1], 'GB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.milli])))

    def test_to_terabytes_from_terabyte_scalar(self):
        terabytes = to_terabytes(1, 'TB')
        assert torch.all(torch.eq(terabytes, _create_tensor(1)))

    def test_to_terabytes_from_terabyte_list(self):
        terabytes = to_terabytes([0, 1], 'TB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, 1])))

    def test_to_terabytes_from_petabyte_scalar(self):
        terabytes = to_terabytes(1, 'PB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.kilo)))

    def test_to_terabytes_from_petabyte_list(self):
        terabytes = to_terabytes([0, 1], 'PB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.kilo])))

    def test_to_terabytes_from_kibibyte_scalar(self):
        terabytes = to_terabytes(1, 'KiB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.kibi / constants.tera)))

    def test_to_terabytes_from_kibibyte_list(self):
        terabytes = to_terabytes([0, 1], 'KiB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.kibi / constants.tera])))

    def test_to_terabytes_from_mebibyte_scalar(self):
        terabytes = to_terabytes(1, 'MiB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.mebi / constants.tera)))

    def test_to_terabytes_from_mebibyte_list(self):
        terabytes = to_terabytes([0, 1], 'MiB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.mebi / constants.tera])))

    def test_to_terabytes_from_gibibyte_scalar(self):
        terabytes = to_terabytes(1, 'GiB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.gibi / constants.tera)))

    def test_to_terabytes_from_gibibyte_list(self):
        terabytes = to_terabytes([0, 1], 'GiB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.gibi / constants.tera])))

    def test_to_terabytes_from_tebibyte_scalar(self):
        terabytes = to_terabytes(1, 'TiB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.tebi / constants.tera)))

    def test_to_terabytes_from_tebibyte_list(self):
        terabytes = to_terabytes([0, 1], 'TiB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.tebi / constants.tera])))

    def test_to_terabytes_from_pebibyte_scalar(self):
        terabytes = to_terabytes(1, 'PiB')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.pebi / constants.tera)))

    def test_to_terabytes_from_pebibyte_list(self):
        terabytes = to_terabytes([0, 1], 'PiB')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.pebi / constants.tera])))


# Test function with values that are already bit units (e.g. Kibibit, Megabit)
class TestMegabytesFromBitUnits(object):
    def test_to_terabytes_from_bit_scalar(self):
        terabytes = to_terabytes(8, 'b')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.pico)))

    def test_to_terabytes_from_bit_list(self):
        terabytes = to_terabytes([0, 8], 'b')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.pico])))

    def test_to_terabytes_from_kilobit_scalar(self):
        terabytes = to_terabytes(8, 'Kbit')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.nano)))

    def test_to_terabytes_from_kilobit_list(self):
        terabytes = to_terabytes([0, 8], 'Kbit')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.nano])))

    def test_to_terabytes_from_megabit_scalar(self):
        terabytes = to_terabytes(8, 'Mbit')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.micro)))

    def test_to_terabytes_from_megabit_list(self):
        terabytes = to_terabytes([0, 8], 'Mbit')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.micro])))

    def test_to_terabytes_from_gigabit_scalar(self):
        terabytes = to_terabytes(8, 'Gbit')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.milli)))

    def test_to_terabytes_from_gigabit_list(self):
        terabytes = to_terabytes([0, 8], 'Gbit')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.milli])))

    def test_to_terabytes_from_terabit_scalar(self):
        terabytes = to_terabytes(8, 'Tbit')
        assert torch.all(torch.eq(terabytes, _create_tensor(1)))

    def test_to_terabytes_from_terabit_list(self):
        terabytes = to_terabytes([0, 8], 'Tbit')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, 1])))

    def test_to_terabytes_from_petabit_scalar(self):
        terabytes = to_terabytes(8, 'Pbit')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.kilo)))

    def test_to_terabytes_from_petabit_list(self):
        terabytes = to_terabytes([0, 8], 'Pbit')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.kilo])))

    def test_to_terabytes_from_kibibit_scalar(self):
        terabytes = to_terabytes(8, 'Kib')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.kibi / constants.tera)))

    def test_to_terabytes_from_kibibit_list(self):
        terabytes = to_terabytes([0, 8], 'Kib')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.kibi / constants.tera])))

    def test_to_terabytes_from_mebibit_scalar(self):
        terabytes = to_terabytes(8, 'Mib')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.mebi / constants.tera)))

    def test_to_terabytes_from_mebibit_list(self):
        terabytes = to_terabytes([0, 8], 'Mib')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.mebi / constants.tera])))

    def test_to_terabytes_from_gibibit_scalar(self):
        terabytes = to_terabytes(8, 'Gib')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.gibi / constants.tera)))

    def test_to_terabytes_from_gibibit_list(self):
        terabytes = to_terabytes([0, 8], 'Gib')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.gibi / constants.tera])))

    def test_to_terabytes_from_tebibit_scalar(self):
        terabytes = to_terabytes(8, 'Tib')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.tebi / constants.tera)))

    def test_to_terabytes_from_tebibit_list(self):
        terabytes = to_terabytes([0, 8], 'Tib')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.tebi / constants.tera])))

    def test_to_terabytes_from_pebibyte_scalar(self):
        terabytes = to_terabytes(8, 'Pib')
        assert torch.all(torch.eq(terabytes, _create_tensor(constants.pebi / constants.tera)))

    def test_to_terabytes_from_pebibyte_list(self):
        terabytes = to_terabytes([0, 8], 'Pib')
        assert torch.all(torch.eq(terabytes, _create_tensor([0, constants.pebi / constants.tera])))
