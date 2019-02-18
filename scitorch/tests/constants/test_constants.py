from scitorch.constants import constants
from math import pi, sqrt
from pytest import approx


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


class TestPhysicalConstantsScalar(object):
    def test_speed_of_light_val(self):
        assert constants.speed_of_light.get('val') == 299792458
        assert constants.c.get('val') == 299792458

    def test_speed_of_light_dim(self):
        assert constants.speed_of_light.get('dim') == 'm*s^(-1)'
        assert constants.c.get('dim') == 'm*s^(-1)'

    def test_magnetic_constant_val(self):
        assert constants.magnetic_constant.get('val') == 4 * pi * 1e-7
        assert constants.mu_0.get('val') == 4 * pi * 1e-7

    def test_magnetic_constant_dim(self):
        assert constants.magnetic_constant.get('dim') == 'N*A^(-2)'
        assert constants.mu_0.get('dim') == 'N*A^(-2)'

    def test_gravitational_constant_val(self):
        assert constants.gravitational_constant.get('val') == 6.67408 * 1e-11
        assert constants.G.get('val') == 6.67408 * 1e-11

    def test_gravitational_constant_dim(self):
        assert constants.gravitational_constant.get('dim') == 'm^(3)*kg^(-1)*s^(-2)'
        assert constants.G.get('dim') == 'm^(3)*kg^(-1)*s^(-2)'

    def test_planck_constant_val(self):
        assert constants.planck_constant.get('val') == 6.626070040 * 1e-34
        assert constants.h.get('val') == 6.626070040 * 1e-34

    def test_planck_constant_dim(self):
        assert constants.planck_constant.get('dim') == 'J*s'
        assert constants.h.get('dim') == 'J*s'

    def test_gas_constant_val(self):
        assert constants.gas_constant.get('val') == 8.3144598
        assert constants.R.get('val') == 8.3144598

    def test_gas_constant_dim(self):
        assert constants.gas_constant.get('dim') == 'J*mol^(-1)*K^(-1)'
        assert constants.R.get('dim') == 'J*mol^(-1)*K^(-1)'

    def test_avogadro_constant_val(self):
        assert constants.avogadro_constant.get('val') == 6.022140857 * 1e23
        assert constants.N_A.get('val') == 6.022140857 * 1e23

    def test_avogadro_constant_dim(self):
        assert constants.avogadro_constant.get('dim') == 'mol^(-1)'
        assert constants.N_A.get('dim') == 'mol^(-1)'

    def test_elementary_charge_val(self):
        assert constants.elementary_charge.get('val') == 1.6021766208 * 1e-19
        assert constants.e.get('val') == 1.6021766208 * 1e-19

    def test_elementary_charge_dim(self):
        assert constants.elementary_charge.get('dim') == 'C'
        assert constants.e.get('dim') == 'C'

    def test_electron_mass_val(self):
        assert constants.electron_mass.get('val') == 9.10938356 * 1e-31
        assert constants.m_e.get('val') == 9.10938356 * 1e-31

    def test_electron_mass_dim(self):
        assert constants.electron_mass.get('dim') == 'kg'
        assert constants.m_e.get('dim') == 'kg'


class TestPhysicalConstantsDerived(object):
    def test_electronvolt_val(self):
        assert constants.electronvolt.get('val') == constants.elementary_charge.get('val')
        assert constants.eV.get('val') == constants.elementary_charge.get('val')
        assert constants.electronvolt.get('val') == approx(1.6021766208 * 1e-19)
        assert constants.eV.get('val') == approx(1.6021766208 * 1e-19)

    def test_electronvolt_dim(self):
        assert constants.electronvolt.get('dim') == 'J'
        assert constants.eV.get('dim') == 'J'

    def test_electric_constant_val(self):
        assert constants.electric_constant.get('val') == \
               1 / (constants.mu_0.get('val') * constants.c.get('val') * constants.c.get('val'))
        assert constants.epsilon_0.get('val') == \
               1 / (constants.mu_0.get('val') * constants.c.get('val') * constants.c.get('val'))
        assert constants.epsilon_0.get('val') == approx(8.854187817 * 1e-12)
        assert constants.electric_constant.get('val') == approx(8.854187817 * 1e-12)

    def test_electric_constant_dim(self):
        assert constants.electric_constant.get('dim') == 'F*m^(-1)'
        assert constants.epsilon_0.get('dim') == 'F*m^(-1)'

    def test_hbar_val(self):
        assert constants.hbar.get('val') == constants.h.get('val') / (2*pi)
        assert constants.hbar.get('val') == approx(1.054571800 * 1e-34)

    def test_hbar_dim(self):
        assert constants.hbar.get('dim') == 'J*s'

    def test_planck_mass_val(self):
        assert constants.planck_mass.get('val') == \
               sqrt((constants.hbar.get('val') * constants.c.get('val'))/constants.G.get('val'))
        assert constants.m_p.get('val') == \
               sqrt((constants.hbar.get('val') * constants.c.get('val'))/constants.G.get('val'))
        assert constants.planck_mass.get('val') == approx(2.176470 * 1e-8)
        assert constants.m_p.get('val') == approx(2.176470 * 1e-8)

    def test_planck_mass_dim(self):
        assert constants.planck_mass.get('dim') == 'kg'
        assert constants.m_p.get('dim') == 'kg'
        
    def test_planck_temperature_val(self):
        assert constants.planck_temperature.get('val') == \
               sqrt((constants.hbar.get('val') * constants.c.get('val') ** 5)/constants.G.get('val')) / constants.k_B.get('val')
        assert constants.T_p.get('val') == \
               sqrt((constants.hbar.get('val') * constants.c.get('val') ** 5) / constants.G.get('val')) / constants.k_B.get('val')
        assert constants.planck_temperature.get('val') == approx(1.416808 * 1e32)
        assert constants.T_p.get('val') == approx(1.416808 * 1e32)

    def test_planck_temperature_dim(self):
        assert constants.planck_temperature.get('dim') == 'K'
        assert constants.T_p.get('dim') == 'K'
        
    def test_planck_length_val(self):
        assert constants.planck_length.get('val') == \
               constants.hbar.get('val') / (constants.m_p.get('val')*constants.c.get('val'))
        assert constants.l_p.get('val') == \
               constants.hbar.get('val') / (constants.m_p.get('val')*constants.c.get('val'))
        assert constants.planck_length.get('val') == approx(1.616229 * 1e-35)
        assert constants.l_p.get('val') == approx(1.616229 * 1e-35)
        
    def test_planck_length_dim(self):
        assert constants.planck_length.get('dim') == 'm'
        assert constants.l_p.get('dim') == 'm'

    def test_planck_time_val(self):
        assert constants.planck_time.get('val') == constants.l_p.get('val') / constants.c.get('val')
        assert constants.t_p.get('val') == constants.l_p.get('val') / constants.c.get('val')
        assert constants.planck_time.get('val') == approx(5.39116 * 1e-44)
        assert constants.t_p.get('val') == approx(5.39116 * 1e-44)

    def test_planck_time_dim(self):
        assert constants.planck_time.get('dim') == 's'
        assert constants.t_p.get('dim') == 's'

    def test_boltzmann_constant_val(self):
        assert constants.boltzmann_constant.get('val') == constants.R.get('val') / constants.N_A.get('val')
        assert constants.k_B.get('val') == constants.R.get('val') / constants.N_A.get('val')
        assert constants.boltzmann_constant.get('val') == approx(1.38064852 * 1e-23)
        assert constants.k_B.get('val') == approx(1.38064852 * 1e-23)

    def test_boltzmann_constant_dim(self):
        assert constants.boltzmann_constant.get('dim') == 'J*K^(-1)'
        assert constants.k_B.get('dim') == 'J*K^(-1)'

    def test_magnetic_flux_quantum_val(self):
        assert constants.magnetic_flux_quantum.get('val') == \
               constants.h.get('val') / (2 * constants.elementary_charge.get('val'))
        assert constants.phi_0.get('val') == \
               constants.h.get('val') / (2 * constants.elementary_charge.get('val'))
        assert constants.magnetic_flux_quantum.get('val') == approx(2.067833831 * 1e-15)
        assert constants.phi_0.get('val') == approx(2.067833831 * 1e-15)

    def test_magnetic_flux_quantum_dim(self):
        assert constants.magnetic_flux_quantum.get('dim') == 'Wb'
        assert constants.phi_0.get('dim') == 'Wb'
        
    def test_conductance_quantum_val(self):
        assert constants.conductance_quantum.get('val') == (2 * constants.e.get('val')**2) / constants.h.get('val')
        assert constants.G_0.get('val') == (2 * constants.e.get('val')**2) / constants.h.get('val')
        assert constants.conductance_quantum.get('val') == approx(7.7480917310 * 1e-5)
        assert constants.G_0.get('val') == approx(7.7480917310 * 1e-5)

    def test_conductance_quantum_dim(self):
        assert constants.conductance_quantum.get('dim') == 'S'
        assert constants.G_0.get('dim') == 'S'

    def test_josephson_constant_val(self):
        assert constants.josephson_constant.get('val') == (2 * constants.e.get('val')) / constants.h.get('val')
        assert constants.K_J.get('val') == (2 * constants.e.get('val')) / constants.h.get('val')
        assert constants.josephson_constant.get('val') == approx(483597.8525 * constants.giga)
        assert constants.K_J.get('val') == approx(483597.8525 * constants.giga)

    def test_josephson_constant_dim(self):
        assert constants.josephson_constant.get('dim') == 'Hz*V^(-1)'
        assert constants.K_J.get('dim') == 'Hz*V^(-1)'
        
    def test_von_klitzing_constant_val(self):
        assert constants.von_klitzing_constant.get('val') == constants.h.get('val') / (constants.e.get('val')**2)
        assert constants.R_K.get('val') == constants.h.get('val') / (constants.e.get('val')**2)
        assert constants.von_klitzing_constant.get('val') == approx(25812.8074555)
        assert constants.R_K.get('val') == approx(25812.8074555)

    def test_von_klitzing_constant_dim(self):
        assert constants.von_klitzing_constant.get('dim') == 'Omega'
        assert constants.R_K.get('dim') == 'Omega'
        
    def test_bohr_magneton_val(self):
        assert constants.bohr_magneton.get('val') == \
               (constants.e.get('val') * constants.hbar.get('val')) / (2 * constants.m_e.get('val'))
        assert constants.mu_B.get('val') == \
               (constants.e.get('val') * constants.hbar.get('val')) / (2 * constants.m_e.get('val'))
        assert constants.bohr_magneton.get('val') == approx(927.4009994 * 1e-26)
        assert constants.mu_B.get('val') == approx(927.4009994 * 1e-26)

    def test_bohr_magneton_dim(self):
        assert constants.bohr_magneton.get('dim') == 'J*T^(-1)'
        assert constants.mu_B.get('dim') == 'J*T^(-1)'
        
    def test_nuclear_magneton_val(self):
        assert constants.nuclear_magneton.get('val') == \
               (constants.e.get('val') * constants.hbar.get('val')) / (2 * constants.m_p.get('val'))
        assert constants.mu_N.get('val')== \
               (constants.e.get('val') * constants.hbar.get('val')) / (2 * constants.m_p.get('val'))
        assert constants.nuclear_magneton.get('val') == approx(5.050783699 * 1e-27)
        assert constants.mu_N.get('val') == approx(5.050783699 * 1e-27)

    def test_nuclear_magneton_dim(self):
        assert constants.nuclear_magneton.get('dim') == 'J*T^(-1)'
        assert constants.mu_N.get('dim') == 'J*T^(-1)'
