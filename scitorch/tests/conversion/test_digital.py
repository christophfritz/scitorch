"""
    Tests for the conversion.digital module

    As to_bytes() is the basis of all the other functions most of the things are tested there. Therefore,
    mainly the default values and the conversion from bytes to the new unit are tested in the other functions.
"""

import torch
from pytest import raises
from scitorch.tools._tensors import _create_tensor
from scitorch.conversion.digital import to_kibibytes
from scitorch.constants import constants


class TestToKibibyte(object):
    def test_to_kibibytes_default_values_scalar(self):
        kibibytes = to_kibibytes()
        assert torch.all(torch.eq(kibibytes, _create_tensor(0)))

    def test_to_petabytes_from_petabyte_scalar(self):
        kibibytes = to_kibibytes(1, 'KiB')
        assert torch.all(torch.eq(kibibytes, _create_tensor(1)))

    def test_to_petabytes_from_petabyte_list(self):
        kibibytes = to_kibibytes([0, 1], 'KiB')
        assert torch.all(torch.eq(kibibytes, _create_tensor([0, 1])))

    def test_to_kibibytes_from_byte_scalar(self):
        kibibytes = to_kibibytes(1, 'B')
        assert torch.all(torch.eq(kibibytes, _create_tensor(1 / constants.kibi)))

    def test_to_kibibytes_from_byte_list(self):
        kibibytes = to_kibibytes([0, 1], 'B')
        assert torch.all(torch.eq(kibibytes, _create_tensor([0, 1 / constants.kibi])))

