"""Collection of mathematical and physical constants and conversion factors."""

import math as _math
from scitorch.core import Tensor

# SI prefixes
"""Source: http://www.npl.co.uk/si-units/"""

yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deka = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21
yocto = 1e-24

# binary prefixes
kibi = 2**10
mebi = 2**20
gibi = 2**30
tebi = 2**40
pebi = 2**50
exbi = 2**60
zebi = 2**70
yobi = 2**80

# mathematical constants

pi = _math.pi

# physical constants
"""Source: https://physics.nist.gov/cuu/Constants/"""

# constants with only scalar values

c = speed_of_light = Tensor(299792458, 'm*s^(-1)')
mu_0 = magnetic_constant = Tensor(4 * pi * 1e-7, 'N*A^(-2)')
G = gravitational_constant = Tensor(6.67408 * 1e-11, 'm^(3)*kg^(-1)*s^(-2)')
h = planck_constant = Tensor(6.626070040 * 1e-34, 'J*s')
R = gas_constant = Tensor(8.3144598, 'J*mol^(-1)*K^(-1)')
N_A = avogadro_constant = Tensor(6.022140857 * 1e23, 'mol^(-1)')
e = elementary_charge = Tensor(1.6021766208 * 1e-19, 'C')
m_e = electron_mass = Tensor(9.10938356 * 1e-31, 'kg')

# "derived" constants

eV = electronvolt = Tensor(elementary_charge.val, 'J')
k_B = boltzmann_constant = Tensor(R.val / N_A.val, 'J*K^(-1)')
epsilon_0 = electric_constant = Tensor(1 / (mu_0.val * c.val * c.val), 'F*m^(-1)')
hbar = Tensor(h.val / (2 * pi), 'J*s')
m_p = planck_mass = Tensor(_math.sqrt((hbar.val * c.val) / G.val), 'kg')
T_p = planck_temperature = Tensor(_math.sqrt((hbar.val * c.val**5) / G.val) / k_B.val, 'K')
l_p = planck_length = Tensor(hbar.val / (m_p.val * c.val), 'm')
t_p = planck_time = Tensor(l_p.val / c.val, 's')
phi_0 = magnetic_flux_quantum = Tensor(h.val / (2 * e.val), 'Wb')
G_0 = conductance_quantum = Tensor((2 * e.val**2) / h.val, 'S')
K_J = josephson_constant = Tensor((2 * e.val) / h.val, 'Hz*V^(-1)')
R_K = von_klitzing_constant = Tensor(h.val/(e.val**2), 'Omega')
mu_B = bohr_magneton = Tensor((e.val * hbar.val) / (2 * m_e.val), 'J*T^(-1)')
mu_N = nuclear_magneton = Tensor((e.val * hbar.val) / (2 * m_p.val), 'J*T^(-1)')


