from scitorch.constants import constants


class TestPrefixesSI(object):
    def test_yocto(self):
        assert constants.yocto == 1e-24

    def test_zepto(self):
        assert constants.zepto == 1e-21

    def test_atto(self):
        assert constants.atto == 1e-18

    def test_femto(self):
        assert constants.femto == 1e-15

    def test_pico(self):
        assert constants.pico == 1e-12

    def test_nano(self):
        assert constants.nano == 1e-9

    def test_micro(self):
        assert constants.micro == 1e-6

    def test_milli(self):
        assert constants.milli == 1e-3

    def test_kilo(self):
        assert constants.kilo == 1e3

    def test_mega(self):
        assert constants.mega == 1e6

    def test_giga(self):
        assert constants.giga == 1e9

    def test_tera(self):
        assert constants.tera == 1e12

    def test_peta(self):
        assert constants.peta == 1e15

    def test_exa(self):
        assert constants.exa == 1e18

    def test_zetta(self):
        assert constants.zetta == 1e21

    def test_yotta(self):
        assert constants.yotta == 1e24


class TestPrefixesBinary(object):
    def test_kibi(self):
        assert constants.kibi == 2 ** 10

    def test_mebi(self):
        assert constants.mebi == 2 ** 20

    def test_gibi(self):
        assert constants.gibi == 2 ** 30

    def test_tebi(self):
        assert constants.tebi == 2 ** 40

    def test_pebi(self):
        assert constants.pebi == 2 ** 50

    def test_exbi(self):
        assert constants.exbi == 2 ** 60

    def test_zebi(self):
        assert constants.zebi == 2 ** 70

    def test_yobi(self):
        assert constants.yobi == 2 ** 80
