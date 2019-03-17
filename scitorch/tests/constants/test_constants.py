"""Test cases for the Constants module."""

import torch

from scitorch.constants import constants
from math import pi, sqrt
from pytest import approx
from scitorch.tools._tensors import _get_local_variables
from sympy import sympify


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
        assert constants.speed_of_light.val == 299792458
        assert constants.c.val == 299792458

    def test_speed_of_light_dim(self):
        assert constants.speed_of_light.dim == sympify('m*s^(-1)', locals=_get_local_variables())
        assert constants.c.dim == sympify('m*s^(-1)', locals=_get_local_variables())

    def test_magnetic_constant_val(self):
        assert constants.magnetic_constant.val == 4 * pi * 1e-7
        assert constants.mu_0.val == 4 * pi * 1e-7

    def test_magnetic_constant_dim(self):
        assert constants.magnetic_constant.dim == sympify('N*A^(-2)', locals=_get_local_variables())
        assert constants.mu_0.dim == sympify('N*A^(-2)', locals=_get_local_variables())

    def test_gravitational_constant_val(self):
        assert constants.gravitational_constant.val == 6.67408 * 1e-11
        assert constants.G.val == 6.67408 * 1e-11

    def test_gravitational_constant_dim(self):
        assert constants.gravitational_constant.dim == sympify('m^(3)*kg^(-1)*s^(-2)', locals=_get_local_variables())
        assert constants.G.dim == sympify('m^(3)*kg^(-1)*s^(-2)', locals=_get_local_variables())

    def test_planck_constant_val(self):
        assert constants.planck_constant.val == 6.626070040 * 1e-34
        assert constants.h.val == 6.626070040 * 1e-34

    def test_planck_constant_dim(self):
        assert constants.planck_constant.dim == sympify('J*s', locals=_get_local_variables())
        assert constants.h.dim == sympify('J*s', locals=_get_local_variables())

    def test_gas_constant_val(self):
        assert constants.gas_constant.val == 8.3144598
        assert constants.R.val == 8.3144598

    def test_gas_constant_dim(self):
        assert constants.gas_constant.dim == sympify('J*mol^(-1)*K^(-1)', locals=_get_local_variables())
        assert constants.R.dim == sympify('J*mol^(-1)*K^(-1)', locals=_get_local_variables())

    def test_avogadro_constant_val(self):
        assert constants.avogadro_constant.val == 6.022140857 * 1e23
        assert constants.N_A.val == 6.022140857 * 1e23

    def test_avogadro_constant_dim(self):
        assert constants.avogadro_constant.dim == sympify('mol^(-1)', locals=_get_local_variables())
        assert constants.N_A.dim == sympify('mol^(-1)', locals=_get_local_variables())

    def test_elementary_charge_val(self):
        assert constants.elementary_charge.val == 1.6021766208 * 1e-19
        assert constants.e.val == 1.6021766208 * 1e-19

    def test_elementary_charge_dim(self):
        assert constants.elementary_charge.dim == sympify('C', locals=_get_local_variables())
        assert constants.e.dim == sympify('C', locals=_get_local_variables())

    def test_electron_mass_val(self):
        assert constants.electron_mass.val == 9.10938356 * 1e-31
        assert constants.m_e.val == 9.10938356 * 1e-31

    def test_electron_mass_dim(self):
        assert constants.electron_mass.dim == sympify('kg', locals=_get_local_variables())
        assert constants.m_e.dim == sympify('kg', locals=_get_local_variables())


class TestPhysicalConstantsDerived(object):
    def test_electronvolt_val(self):
        assert constants.electronvolt.val == constants.elementary_charge.val
        assert constants.eV.val == constants.elementary_charge.val

    def test_electronvolt_dim(self):
        assert constants.electronvolt.dim == sympify('J', locals=_get_local_variables())
        assert constants.eV.dim == sympify('J', locals=_get_local_variables())

    def test_electric_constant_val(self):
        assert constants.electric_constant.val == \
               1 / (constants.mu_0.val * constants.c.val * constants.c.val)
        assert constants.epsilon_0.val == \
               1 / (constants.mu_0.val * constants.c.val * constants.c.val)

    def test_electric_constant_dim(self):
        assert constants.electric_constant.dim == sympify('F*m^(-1)', locals=_get_local_variables())
        assert constants.epsilon_0.dim == sympify('F*m^(-1)', locals=_get_local_variables())

    def test_hbar_val(self):
        assert constants.hbar.val == constants.h.val / (2*pi)

    def test_hbar_dim(self):
        assert constants.hbar.dim == sympify('J*s', locals=_get_local_variables())

    def test_planck_mass_val(self):
        assert constants.planck_mass.val == \
               sqrt((constants.hbar.val * constants.c.val)/constants.G.val)
        assert constants.m_p.val == \
               sqrt((constants.hbar.val * constants.c.val)/constants.G.val)

    def test_planck_mass_dim(self):
        assert constants.planck_mass.dim == sympify('kg', locals=_get_local_variables())
        assert constants.m_p.dim == sympify('kg', locals=_get_local_variables())
        
    def test_planck_temperature_val(self):
        assert constants.planck_temperature.val == \
               sqrt((constants.hbar.val * constants.c.val ** 5)/constants.G.val) / constants.k_B.val
        assert constants.T_p.val == \
               sqrt((constants.hbar.val * constants.c.val ** 5) / constants.G.val) / constants.k_B.val

    def test_planck_temperature_dim(self):
        assert constants.planck_temperature.dim == sympify('K', locals=_get_local_variables())
        assert constants.T_p.dim == sympify('K', locals=_get_local_variables())
        
    def test_planck_length_val(self):
        assert constants.planck_length.val == \
               constants.hbar.val / (constants.m_p.val*constants.c.val)
        assert constants.l_p.val == \
               constants.hbar.val / (constants.m_p.val*constants.c.val)
        
    def test_planck_length_dim(self):
        assert constants.planck_length.dim == sympify('m', locals=_get_local_variables())
        assert constants.l_p.dim == sympify('m', locals=_get_local_variables())

    def test_planck_time_val(self):
        assert constants.planck_time.val == constants.l_p.val / constants.c.val
        assert constants.t_p.val == constants.l_p.val / constants.c.val

    def test_planck_time_dim(self):
        assert constants.planck_time.dim == sympify('s', locals=_get_local_variables())
        assert constants.t_p.dim == sympify('s', locals=_get_local_variables())

    def test_boltzmann_constant_val(self):
        assert constants.boltzmann_constant.val == constants.R.val / constants.N_A.val
        assert constants.k_B.val == constants.R.val / constants.N_A.val

    def test_boltzmann_constant_dim(self):
        assert constants.boltzmann_constant.dim == sympify('J*K^(-1)', locals=_get_local_variables())
        assert constants.k_B.dim == sympify('J*K^(-1)', locals=_get_local_variables())

    def test_magnetic_flux_quantum_val(self):
        assert constants.magnetic_flux_quantum.val == \
               constants.h.val / (2 * constants.elementary_charge.val)
        assert constants.phi_0.val == \
               constants.h.val / (2 * constants.elementary_charge.val)

    def test_magnetic_flux_quantum_dim(self):
        assert constants.magnetic_flux_quantum.dim == sympify('Wb', locals=_get_local_variables())
        assert constants.phi_0.dim == sympify('Wb', locals=_get_local_variables())
        
    def test_conductance_quantum_val(self):
        assert constants.conductance_quantum.val == (2 * constants.e.val**2) / constants.h.val
        assert constants.G_0.val == (2 * constants.e.val**2) / constants.h.val

    def test_conductance_quantum_dim(self):
        assert constants.conductance_quantum.dim == sympify('S', locals=_get_local_variables())
        assert constants.G_0.dim == sympify('S', locals=_get_local_variables())

    def test_josephson_constant_val(self):
        assert constants.josephson_constant.val == (2 * constants.e.val) / constants.h.val
        assert constants.K_J.val == (2 * constants.e.val) / constants.h.val

    def test_josephson_constant_dim(self):
        assert constants.josephson_constant.dim == sympify('Hz*V^(-1)', locals=_get_local_variables())
        assert constants.K_J.dim == sympify('Hz*V^(-1)', locals=_get_local_variables())
        
    def test_von_klitzing_constant_val(self):
        assert constants.von_klitzing_constant.val == constants.h.val / (constants.e.val**2)
        assert constants.R_K.val == constants.h.val / (constants.e.val**2)

    def test_von_klitzing_constant_dim(self):
        assert constants.von_klitzing_constant.dim == sympify('Omega', locals=_get_local_variables())
        assert constants.R_K.dim == sympify('Omega', locals=_get_local_variables())
        
    def test_bohr_magneton_val(self):
        assert constants.bohr_magneton.val == \
               (constants.e.val * constants.hbar.val) / (2 * constants.m_e.val)
        assert constants.mu_B.val == \
               (constants.e.val * constants.hbar.val) / (2 * constants.m_e.val)

    def test_bohr_magneton_dim(self):
        assert constants.bohr_magneton.dim == sympify('J*T^(-1)', locals=_get_local_variables())
        assert constants.mu_B.dim == sympify('J*T^(-1)', locals=_get_local_variables())
        
    def test_nuclear_magneton_val(self):
        assert constants.nuclear_magneton.val == \
               (constants.e.val * constants.hbar.val) / (2 * constants.m_p.val)
        assert constants.mu_N.val== \
               (constants.e.val * constants.hbar.val) / (2 * constants.m_p.val)

    def test_nuclear_magneton_dim(self):
        assert constants.nuclear_magneton.dim == sympify('J*T^(-1)', locals=_get_local_variables())
        assert constants.mu_N.dim == sympify('J*T^(-1)', locals=_get_local_variables())
