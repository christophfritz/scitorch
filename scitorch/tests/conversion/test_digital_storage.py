import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital_storage import to_byte

class TestToByte(object):
    def test_to_byte_default_values_scalar(self):
        byte = to_byte()
        assert torch.all(torch.eq(byte, _create_tensor(0)))

    def test_to_byte_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_byte('B', 1)

    def test_to_byte_wrong_arguments_scalar(self):
        with raises(TypeError):
            to_byte('B', [0, 1])

    def test_to_byte_tensor(self):
        byte_tensor = torch.tensor([0, 1])
        byte = to_byte(byte_tensor)
        assert torch.all(torch.eq(byte, _create_tensor([0, 1])))

    def test_to_byte_wrong_scale_scalar(self):
        with raises(NotImplementedError):
            to_byte(1, 'l')

    def test_to_byte_wrong_scale_list(self):
        with raises(NotImplementedError):
            to_byte([0, 1], 'l')

    def test_to_byte_from_byte_scalar(self):
        byte = to_byte(1, 'B')
        assert torch.all(torch.eq(byte, _create_tensor(1)))

    def test_to_byte_from_byte_list(self):
        byte = to_byte([0, 1], 'B')
        assert torch.all(torch.eq(byte, _create_tensor([0, 1])))

    def test_to_byte_from_kilobyte_scalar(self):
        byte = to_byte(1, 'KB')
        assert torch.all(torch.eq(byte, _create_tensor(1e3)))

    def test_to_byte_from_kilobyte_list(self):
        byte = to_byte([0, 1], 'KB')
        assert torch.all(torch.eq(byte, _create_tensor([0, 1e3])))

    def test_to_byte_from_megabyte_scalar(self):
        byte = to_byte(1, 'MB')
        assert torch.all(torch.eq(byte, _create_tensor(1e6)))

    def test_to_byte_from_megabyte_list(self):
        byte = to_byte([0, 1], 'MB')
        assert torch.all(torch.eq(byte, _create_tensor([0, 1e6])))

    def test_to_byte_from_gigabyte_scalar(self):
        byte = to_byte(1, 'GB')
        assert torch.all(torch.eq(byte, _create_tensor(1e9)))

    def test_to_byte_from_gigaobyte_list(self):
        byte = to_byte([0, 1], 'GB')
        assert torch.all(torch.eq(byte, _create_tensor([0, 1e9])))

    def test_to_byte_from_terabyte_scalar(self):
        byte = to_byte(1, 'TB')
        assert torch.all(torch.eq(byte, _create_tensor(1e12)))

    def test_to_byte_from_terabyte_list(self):
        byte = to_byte([0, 1], 'TB')
        assert torch.all(torch.eq(byte, _create_tensor([0, 1e12])))

    def test_to_byte_from_petabyte_scalar(self):
        byte = to_byte(1, 'PB')
        assert torch.all(torch.eq(byte, _create_tensor(1e15)))

    def test_to_byte_from_petabyte_list(self):
        byte = to_byte([0, 1], 'PB')
        assert torch.all(torch.eq(byte, _create_tensor([0, 1e15])))