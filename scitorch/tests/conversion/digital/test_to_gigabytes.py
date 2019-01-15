import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital import to_gigabytes
from scitorch.constants import constants


"""Tests for to_gigabytes"""

# Test function with general parameters
class TestToGigabytes(object):
    def test_to_gigabytes_default_values_scalar(self):
        gigabytes = to_gigabytes()
        assert torch.all(torch.eq(gigabytes, _create_tensor(0)))

    def test_to_gigabytes_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_gigabytes('MB', 1)

    def test_to_gigabytes_wrong_arguments_list(self):
        with raises(TypeError):
            to_gigabytes('MB', [0, 1])

    def test_to_gigabytes_tensor(self):
        gigabyte_tensor = torch.tensor([0, 1])
        gigabytes = to_gigabytes(gigabyte_tensor)
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, 1])))

    def test_to_gigabytes_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_gigabytes(1, 'kb')

    def test_to_gigabytes_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_gigabytes([0, 1], 'kb')


# Test function with values that are already byte units (e.g. Kilobyte, Megabyte)
class TestToGigabytesFromByteUnits(object):
    def test_to_gigabytes_from_byte_scalar(self):
        gigabytes = to_gigabytes(1, 'B')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.nano)))

    def test_to_gigabytes_from_byte_list(self):
        gigabytes = to_gigabytes([0, 1], 'B')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.nano])))

    def test_to_gigabytes_from_kilobyte_scalar(self):
        gigabytes = to_gigabytes(1, 'KB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.micro)))

    def test_to_gigabytes_from_kilobyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'KB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.micro])))

    def test_to_gigabytes_from_megabyte_scalar(self):
        gigabytes = to_gigabytes(1, 'MB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.milli)))

    def test_to_gigabytes_from_megabyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'MB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.milli])))

    def test_to_gigabytes_from_gigabyte_scalar(self):
        gigabytes = to_gigabytes(1, 'GB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(1)))

    def test_to_gigabytes_from_gigaobyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'GB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, 1])))

    def test_to_gigabytes_from_terabyte_scalar(self):
        gigabytes = to_gigabytes(1, 'TB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.kilo)))

    def test_to_gigabytes_from_terabyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'TB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.kilo])))

    def test_to_gigabytes_from_petabyte_scalar(self):
        gigabytes = to_gigabytes(1, 'PB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.mega)))

    def test_to_gigabytes_from_petabyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'PB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.mega])))

    def test_to_gigabytes_from_kibibyte_scalar(self):
        gigabytes = to_gigabytes(1, 'KiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.kibi / constants.giga)))

    def test_to_gigabytes_from_kibibyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'KiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.kibi / constants.giga])))

    def test_to_gigabytes_from_mebibyte_scalar(self):
        gigabytes = to_gigabytes(1, 'MiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.mebi / constants.giga)))

    def test_to_gigabytes_from_mebibyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'MiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.mebi / constants.giga])))

    def test_to_gigabytes_from_gibibyte_scalar(self):
        gigabytes = to_gigabytes(1, 'GiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.gibi / constants.giga)))

    def test_to_gigabytes_from_gibibyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'GiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.gibi / constants.giga])))

    def test_to_gigabytes_from_tebibyte_scalar(self):
        gigabytes = to_gigabytes(1, 'TiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.tebi / constants.giga)))

    def test_to_gigabytes_from_tebibyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'TiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.tebi / constants.giga])))

    def test_to_gigabytes_from_pebibyte_scalar(self):
        gigabytes = to_gigabytes(1, 'PiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.pebi / constants.giga)))

    def test_to_gigabytes_from_pebibyte_list(self):
        gigabytes = to_gigabytes([0, 1], 'PiB')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.pebi / constants.giga])))


# Test function with values that are already bit units (e.g. Kibibit, Megabit)
class TestMegabytesFromBitUnits(object):
    def test_to_gigabytes_from_bit_scalar(self):
        gigabytes = to_gigabytes(8, 'b')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.nano)))

    def test_to_gigabytes_from_bit_list(self):
        gigabytes = to_gigabytes([0, 8], 'b')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.nano])))

    def test_to_gigabytes_from_kilobit_scalar(self):
        gigabytes = to_gigabytes(8, 'Kbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.micro)))

    def test_to_gigabytes_from_kilobit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Kbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.micro])))

    def test_to_gigabytes_from_megabit_scalar(self):
        gigabytes = to_gigabytes(8, 'Mbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.milli)))

    def test_to_gigabytes_from_megabit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Mbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.milli])))

    def test_to_gigabytes_from_gigabit_scalar(self):
        gigabytes = to_gigabytes(8, 'Gbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor(1)))

    def test_to_gigabytes_from_gigabit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Gbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, 1])))

    def test_to_gigabytes_from_terabit_scalar(self):
        gigabytes = to_gigabytes(8, 'Tbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.kilo)))

    def test_to_gigabytes_from_terabit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Tbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.kilo])))

    def test_to_gigabytes_from_petabit_scalar(self):
        gigabytes = to_gigabytes(8, 'Pbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.mega)))

    def test_to_gigabytes_from_petabit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Pbit')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.mega])))

    def test_to_gigabytes_from_kibibit_scalar(self):
        gigabytes = to_gigabytes(8, 'Kib')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.kibi / constants.giga)))

    def test_to_gigabytes_from_kibibit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Kib')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.kibi / constants.giga])))

    def test_to_gigabytes_from_mebibit_scalar(self):
        gigabytes = to_gigabytes(8, 'Mib')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.mebi / constants.giga)))

    def test_to_gigabytes_from_mebibit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Mib')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.mebi / constants.giga])))

    def test_to_gigabytes_from_gibibit_scalar(self):
        gigabytes = to_gigabytes(8, 'Gib')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.gibi / constants.giga)))

    def test_to_gigabytes_from_gibibit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Gib')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.gibi / constants.giga])))

    def test_to_gigabytes_from_tebibit_scalar(self):
        gigabytes = to_gigabytes(8, 'Tib')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.tebi / constants.giga)))

    def test_to_gigabytes_from_tebibit_list(self):
        gigabytes = to_gigabytes([0, 8], 'Tib')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.tebi / constants.giga])))

    def test_to_gigabytes_from_pebibyte_scalar(self):
        gigabytes = to_gigabytes(8, 'Pib')
        assert torch.all(torch.eq(gigabytes, _create_tensor(constants.pebi / constants.giga)))

    def test_to_gigabytes_from_pebibyte_list(self):
        gigabytes = to_gigabytes([0, 8], 'Pib')
        assert torch.all(torch.eq(gigabytes, _create_tensor([0, constants.pebi / constants.giga])))
