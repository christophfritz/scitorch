import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital import to_megabytes
from scitorch.constants import constants


"""Tests for to_megabytes"""

# Test function with general parameters
class TestToMegabytes(object):
    def test_to_megabytes_default_values_scalar(self):
        megabytes = to_megabytes()
        assert torch.all(torch.eq(megabytes, _create_tensor(0)))

    def test_to_megabytes_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_megabytes('MB', 1)

    def test_to_megabytes_wrong_arguments_list(self):
        with raises(TypeError):
            to_megabytes('MB', [0, 1])

    def test_to_megabytes_tensor(self):
        megabyte_tensor = torch.tensor([0, 1])
        megabytes = to_megabytes(megabyte_tensor)
        assert torch.all(torch.eq(megabytes, _create_tensor([0, 1])))

    def test_to_megabytes_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_megabytes(1, 'kb')

    def test_to_megabytes_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_megabytes([0, 1], 'kb')


# Test function with values that are already byte units (e.g. Kilobyte, Megabyte)
class TestToMegabytesFromByteUnits(object):
    def test_to_megabytes_from_byte_scalar(self):
        megabytes = to_megabytes(1, 'B')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.micro)))

    def test_to_megabytes_from_byte_list(self):
        megabytes = to_megabytes([0, 1], 'B')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.micro])))

    def test_to_megabytes_from_kilobyte_scalar(self):
        megabytes = to_megabytes(1, 'KB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.milli)))

    def test_to_megabytes_from_kilobyte_list(self):
        megabytes = to_megabytes([0, 1], 'KB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.milli])))

    def test_to_megabytes_from_megabyte_scalar(self):
        megabytes = to_megabytes(1, 'MB')
        assert torch.all(torch.eq(megabytes, _create_tensor(1)))

    def test_to_megabytes_from_megabyte_list(self):
        megabytes = to_megabytes([0, 1], 'MB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, 1])))

    def test_to_megabytes_from_gigabyte_scalar(self):
        megabytes = to_megabytes(1, 'GB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.kilo)))

    def test_to_megabytes_from_gigaobyte_list(self):
        megabytes = to_megabytes([0, 1], 'GB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.kilo])))

    def test_to_megabytes_from_terabyte_scalar(self):
        megabytes = to_megabytes(1, 'TB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.mega)))

    def test_to_megabytes_from_terabyte_list(self):
        megabytes = to_megabytes([0, 1], 'TB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.mega])))

    def test_to_megabytes_from_petabyte_scalar(self):
        megabytes = to_megabytes(1, 'PB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.giga)))

    def test_to_megabytes_from_petabyte_list(self):
        megabytes = to_megabytes([0, 1], 'PB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.giga])))

    def test_to_megabytes_from_kibibyte_scalar(self):
        megabytes = to_megabytes(1, 'KiB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.kibi / constants.mega)))

    def test_to_megabytes_from_kibibyte_list(self):
        megabytes = to_megabytes([0, 1], 'KiB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.kibi / constants.mega])))

    def test_to_megabytes_from_mebibyte_scalar(self):
        megabytes = to_megabytes(1, 'MiB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.mebi / constants.mega)))

    def test_to_megabytes_from_mebibyte_list(self):
        megabytes = to_megabytes([0, 1], 'MiB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.mebi / constants.mega])))

    def test_to_megabytes_from_gibibyte_scalar(self):
        megabytes = to_megabytes(1, 'GiB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.gibi / constants.mega)))

    def test_to_megabytes_from_gibibyte_list(self):
        megabytes = to_megabytes([0, 1], 'GiB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.gibi / constants.mega])))

    def test_to_megabytes_from_tebibyte_scalar(self):
        megabytes = to_megabytes(1, 'TiB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.tebi / constants.mega)))

    def test_to_megabytes_from_tebibyte_list(self):
        megabytes = to_megabytes([0, 1], 'TiB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.tebi / constants.mega])))

    def test_to_megabytes_from_pebibyte_scalar(self):
        megabytes = to_megabytes(1, 'PiB')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.pebi / constants.mega)))

    def test_to_megabytes_from_pebibyte_list(self):
        megabytes = to_megabytes([0, 1], 'PiB')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.pebi / constants.mega])))


# Test function with values that are already bit units (e.g. Kibibit, Megabit)
class TestMegabytesFromBitUnits(object):
    def test_to_megabytes_from_bit_scalar(self):
        megabytes = to_megabytes(8, 'b')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.micro)))

    def test_to_megabytes_from_bit_list(self):
        megabytes = to_megabytes([0, 8], 'b')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.micro])))

    def test_to_megabytes_from_kilobit_scalar(self):
        megabytes = to_megabytes(8, 'Kbit')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.milli)))

    def test_to_megabytes_from_kilobit_list(self):
        megabytes = to_megabytes([0, 8], 'Kbit')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.milli])))

    def test_to_megabytes_from_megabit_scalar(self):
        megabytes = to_megabytes(8, 'Mbit')
        assert torch.all(torch.eq(megabytes, _create_tensor(1)))

    def test_to_megabytes_from_megabit_list(self):
        megabytes = to_megabytes([0, 8], 'Mbit')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, 1])))

    def test_to_megabytes_from_gigabit_scalar(self):
        megabytes = to_megabytes(8, 'Gbit')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.kilo)))

    def test_to_megabytes_from_gigabit_list(self):
        megabytes = to_megabytes([0, 8], 'Gbit')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.kilo])))

    def test_to_megabytes_from_terabit_scalar(self):
        megabytes = to_megabytes(8, 'Tbit')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.mega)))

    def test_to_megabytes_from_terabit_list(self):
        megabytes = to_megabytes([0, 8], 'Tbit')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.mega])))

    def test_to_megabytes_from_petabit_scalar(self):
        megabytes = to_megabytes(8, 'Pbit')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.giga)))

    def test_to_megabytes_from_petabit_list(self):
        megabytes = to_megabytes([0, 8], 'Pbit')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.giga])))

    def test_to_megabytes_from_kibibit_scalar(self):
        megabytes = to_megabytes(8, 'Kib')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.kibi / constants.mega)))

    def test_to_megabytes_from_kibibit_list(self):
        megabytes = to_megabytes([0, 8], 'Kib')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.kibi / constants.mega])))

    def test_to_megabytes_from_mebibit_scalar(self):
        megabytes = to_megabytes(8, 'Mib')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.mebi / constants.mega)))

    def test_to_megabytes_from_mebibit_list(self):
        megabytes = to_megabytes([0, 8], 'Mib')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.mebi / constants.mega])))

    def test_to_megabytes_from_gibibit_scalar(self):
        megabytes = to_megabytes(8, 'Gib')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.gibi / constants.mega)))

    def test_to_megabytes_from_gibibit_list(self):
        megabytes = to_megabytes([0, 8], 'Gib')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.gibi / constants.mega])))

    def test_to_megabytes_from_tebibit_scalar(self):
        megabytes = to_megabytes(8, 'Tib')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.tebi / constants.mega)))

    def test_to_megabytes_from_tebibit_list(self):
        megabytes = to_megabytes([0, 8], 'Tib')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.tebi / constants.mega])))

    def test_to_megabytes_from_pebibyte_scalar(self):
        megabytes = to_megabytes(8, 'Pib')
        assert torch.all(torch.eq(megabytes, _create_tensor(constants.pebi / constants.mega)))

    def test_to_megabytes_from_pebibyte_list(self):
        megabytes = to_megabytes([0, 8], 'Pib')
        assert torch.all(torch.eq(megabytes, _create_tensor([0, constants.pebi / constants.mega])))
