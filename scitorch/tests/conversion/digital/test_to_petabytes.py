import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital import to_petabytes
from scitorch.constants import constants


"""Tests for to_petabytes"""

# Test function with general parameters
class TestToPetabytes(object):
    def test_to_petabytes_default_values_scalar(self):
        petabytes = to_petabytes()
        assert torch.all(torch.eq(petabytes, _create_tensor(0)))

    def test_to_petabytes_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_petabytes('MB', 1)

    def test_to_petabytes_wrong_arguments_list(self):
        with raises(TypeError):
            to_petabytes('MB', [0, 1])

    def test_to_petabytes_tensor(self):
        petabyte_tensor = torch.tensor([0, 1])
        petabytes = to_petabytes(petabyte_tensor)
        assert torch.all(torch.eq(petabytes, _create_tensor([0, 1])))

    def test_to_petabytes_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_petabytes(1, 'kb')

    def test_to_petabytes_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_petabytes([0, 1], 'kb')


# Test function with values that are already byte units (e.g. Kilobyte, Megabyte)
class TestToPetabytesFromByteUnits(object):
    def test_to_petabytes_from_byte_scalar(self):
        petabytes = to_petabytes(1, 'B')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.femto)))

    def test_to_petabytes_from_byte_list(self):
        petabytes = to_petabytes([0, 1], 'B')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.femto])))

    def test_to_petabytes_from_kilobyte_scalar(self):
        petabytes = to_petabytes(1, 'KB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.pico)))

    def test_to_petabytes_from_kilobyte_list(self):
        petabytes = to_petabytes([0, 1], 'KB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.pico])))

    def test_to_petabytes_from_megabyte_scalar(self):
        petabytes = to_petabytes(1, 'MB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.nano)))

    def test_to_petabytes_from_megabyte_list(self):
        petabytes = to_petabytes([0, 1], 'MB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.nano])))

    def test_to_petabytes_from_gigabyte_scalar(self):
        petabytes = to_petabytes(1, 'GB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.micro)))

    def test_to_petabytes_from_gigaobyte_list(self):
        petabytes = to_petabytes([0, 1], 'GB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.micro])))

    def test_to_petabytes_from_terabyte_scalar(self):
        petabytes = to_petabytes(1, 'TB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.milli)))

    def test_to_petabytes_from_terabyte_list(self):
        petabytes = to_petabytes([0, 1], 'TB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.milli])))

    def test_to_petabytes_from_petabyte_scalar(self):
        petabytes = to_petabytes(1, 'PB')
        assert torch.all(torch.eq(petabytes, _create_tensor(1)))

    def test_to_petabytes_from_petabyte_list(self):
        petabytes = to_petabytes([0, 1], 'PB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, 1])))

    def test_to_petabytes_from_kibibyte_scalar(self):
        petabytes = to_petabytes(1, 'KiB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.kibi / constants.peta)))

    def test_to_petabytes_from_kibibyte_list(self):
        petabytes = to_petabytes([0, 1], 'KiB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.kibi / constants.peta])))

    def test_to_petabytes_from_mebibyte_scalar(self):
        petabytes = to_petabytes(1, 'MiB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.mebi / constants.peta)))

    def test_to_petabytes_from_mebibyte_list(self):
        petabytes = to_petabytes([0, 1], 'MiB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.mebi / constants.peta])))

    def test_to_petabytes_from_gibibyte_scalar(self):
        petabytes = to_petabytes(1, 'GiB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.gibi / constants.peta)))

    def test_to_petabytes_from_gibibyte_list(self):
        petabytes = to_petabytes([0, 1], 'GiB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.gibi / constants.peta])))

    def test_to_petabytes_from_tebibyte_scalar(self):
        petabytes = to_petabytes(1, 'TiB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.tebi / constants.peta)))

    def test_to_petabytes_from_tebibyte_list(self):
        petabytes = to_petabytes([0, 1], 'TiB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.tebi / constants.peta])))

    def test_to_petabytes_from_pebibyte_scalar(self):
        petabytes = to_petabytes(1, 'PiB')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.pebi / constants.peta)))

    def test_to_petabytes_from_pebibyte_list(self):
        petabytes = to_petabytes([0, 1], 'PiB')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.pebi / constants.peta])))


# Test function with values that are already bit units (e.g. Kibibit, Megabit)
class TestMegabytesFromBitUnits(object):
    def test_to_petabytes_from_bit_scalar(self):
        petabytes = to_petabytes(8, 'b')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.femto)))

    def test_to_petabytes_from_bit_list(self):
        petabytes = to_petabytes([0, 8], 'b')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.femto])))

    def test_to_petabytes_from_kilobit_scalar(self):
        petabytes = to_petabytes(8, 'Kbit')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.pico)))

    def test_to_petabytes_from_kilobit_list(self):
        petabytes = to_petabytes([0, 8], 'Kbit')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.pico])))

    def test_to_petabytes_from_megabit_scalar(self):
        petabytes = to_petabytes(8, 'Mbit')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.nano)))

    def test_to_petabytes_from_megabit_list(self):
        petabytes = to_petabytes([0, 8], 'Mbit')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.nano])))

    def test_to_petabytes_from_gigabit_scalar(self):
        petabytes = to_petabytes(8, 'Gbit')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.micro)))

    def test_to_petabytes_from_gigabit_list(self):
        petabytes = to_petabytes([0, 8], 'Gbit')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.micro])))

    def test_to_petabytes_from_terabit_scalar(self):
        petabytes = to_petabytes(8, 'Tbit')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.milli)))

    def test_to_petabytes_from_terabit_list(self):
        petabytes = to_petabytes([0, 8], 'Tbit')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.milli])))

    def test_to_petabytes_from_petabit_scalar(self):
        petabytes = to_petabytes(8, 'Pbit')
        assert torch.all(torch.eq(petabytes, _create_tensor(1)))

    def test_to_petabytes_from_petabit_list(self):
        petabytes = to_petabytes([0, 8], 'Pbit')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, 1])))

    def test_to_petabytes_from_kibibit_scalar(self):
        petabytes = to_petabytes(8, 'Kib')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.kibi / constants.peta)))

    def test_to_petabytes_from_kibibit_list(self):
        petabytes = to_petabytes([0, 8], 'Kib')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.kibi / constants.peta])))

    def test_to_petabytes_from_mebibit_scalar(self):
        petabytes = to_petabytes(8, 'Mib')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.mebi / constants.peta)))

    def test_to_petabytes_from_mebibit_list(self):
        petabytes = to_petabytes([0, 8], 'Mib')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.mebi / constants.peta])))

    def test_to_petabytes_from_gibibit_scalar(self):
        petabytes = to_petabytes(8, 'Gib')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.gibi / constants.peta)))

    def test_to_petabytes_from_gibibit_list(self):
        petabytes = to_petabytes([0, 8], 'Gib')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.gibi / constants.peta])))

    def test_to_petabytes_from_tebibit_scalar(self):
        petabytes = to_petabytes(8, 'Tib')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.tebi / constants.peta)))

    def test_to_petabytes_from_tebibit_list(self):
        petabytes = to_petabytes([0, 8], 'Tib')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.tebi / constants.peta])))

    def test_to_petabytes_from_pebibyte_scalar(self):
        petabytes = to_petabytes(8, 'Pib')
        assert torch.all(torch.eq(petabytes, _create_tensor(constants.pebi / constants.peta)))

    def test_to_petabytes_from_pebibyte_list(self):
        petabytes = to_petabytes([0, 8], 'Pib')
        assert torch.all(torch.eq(petabytes, _create_tensor([0, constants.pebi / constants.peta])))
