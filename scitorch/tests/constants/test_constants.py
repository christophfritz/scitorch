from scitorch.constants import constants


class TestPrefixes(object):
    def test_yocto(self):
        assert constants.yocto == 1e-24
