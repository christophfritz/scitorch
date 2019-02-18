"""Collection of mathematical and physical constants and conversion factors."""

import math as _math

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

c = speed_of_light = dict(
    val=299792458,
    dim='m*s^(-1)')

mu_0 = magnetic_constant = dict(
    val=4 * pi * 1e-7,
    dim='N*A^(-2)')

G = gravitational_constant = dict(
    val=6.67408 * 1e-11,
    dim='m^(3)*kg^(-1)*s^(-2)')

h = planck_constant = dict(
    val=6.626070040 * 1e-34,
    dim='J*s')

R = gas_constant = dict(
    val=8.3144598,
    dim='J*mol^(-1)*K^(-1)')

N_A = avogadro_constant = dict(
    val=6.022140857 * 1e23,
    dim='mol^(-1)')

e = elementary_charge = dict(
    val=1.6021766208 * 1e-19,
    dim='C')

m_e = electron_mass = dict(
    val=9.10938356 * 1e-31,
    dim='kg')

# "derived" constants

eV = electronvolt = dict(
    val=elementary_charge.get('val'),
    dim='J')

k_B = boltzmann_constant = dict(
    val=R.get('val') / N_A.get('val'),
    dim='J*K^(-1)')

epsilon_0 = electric_constant = dict(
    val=1 / (mu_0.get('val') * c.get('val') * c.get('val')),
    dim='F*m^(-1)')

hbar = dict(
    val=h.get('val') / (2 * pi),
    dim='J*s')

m_p = planck_mass = dict(
    val=_math.sqrt((hbar.get('val') * c.get('val')) / G.get('val')),
    dim='kg')

T_p = planck_temperature = dict(
    val=_math.sqrt((hbar.get('val') * c.get('val')**5) / G.get('val')) / k_B.get('val'),
    dim='K')

l_p = planck_length = dict(
    val=hbar.get('val') / (m_p.get('val') * c.get('val')),
    dim='m')

t_p = planck_time = dict(
    val=l_p.get('val') / c.get('val'),
    dim='s')

phi_0 = magnetic_flux_quantum = dict(
    val=h.get('val') / (2 * e.get('val')),
    dim='Wb')

G_0 = conductance_quantum = dict(
    val=(2 * e.get('val')**2) / h.get('val'),
    dim='S')

K_J = josephson_constant = dict(
    val=(2 * e.get('val')) / h.get('val'),
    dim='Hz*V^(-1)')

R_K = von_klitzing_constant = dict(
    val=h.get('val')/(e.get('val')**2),
    dim='Omega')

mu_B = bohr_magneton = dict(
    val=(e.get('val') * hbar.get('val')) / (2 * m_e.get('val')),
    dim='J*T^(-1)')

mu_N = nuclear_magneton = dict(
    val=(e.get('val') * hbar.get('val')) / (2 * m_p.get('val')),
    dim='J*T^(-1)')


