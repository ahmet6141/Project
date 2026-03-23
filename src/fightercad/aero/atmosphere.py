"""International Standard Atmosphere (ISA) model.

Provides temperature, pressure, density, and speed of sound as functions
of geometric altitude.  Valid for 0–86 km (troposphere + stratosphere).
"""

from __future__ import annotations

import math


# Constants
_T0 = 288.15  # Sea-level temperature (K)
_P0 = 101325.0  # Sea-level pressure (Pa)
_RHO0 = 1.225  # Sea-level density (kg/m³)
_G = 9.80665  # Gravitational acceleration (m/s²)
_R = 287.05287  # Specific gas constant for air (J/(kg·K))
_GAMMA = 1.4  # Ratio of specific heats
_LAPSE_TROPO = -0.0065  # Temperature lapse rate in troposphere (K/m)
_H_TROPO = 11000.0  # Tropopause altitude (m)
_T_TROPO = _T0 + _LAPSE_TROPO * _H_TROPO  # Temperature at tropopause
_P_TROPO = _P0 * (_T_TROPO / _T0) ** (-_G / (_LAPSE_TROPO * _R))


def temperature(altitude_m: float) -> float:
    """ISA temperature at altitude (K)."""
    if altitude_m <= _H_TROPO:
        return _T0 + _LAPSE_TROPO * altitude_m
    else:
        return _T_TROPO  # Isothermal above tropopause (up to ~20 km)


def pressure(altitude_m: float) -> float:
    """ISA pressure at altitude (Pa)."""
    if altitude_m <= _H_TROPO:
        T = temperature(altitude_m)
        return _P0 * (T / _T0) ** (-_G / (_LAPSE_TROPO * _R))
    else:
        dh = altitude_m - _H_TROPO
        return _P_TROPO * math.exp(-_G * dh / (_R * _T_TROPO))


def density(altitude_m: float) -> float:
    """ISA density at altitude (kg/m³)."""
    return pressure(altitude_m) / (_R * temperature(altitude_m))


def speed_of_sound(altitude_m: float) -> float:
    """Speed of sound at altitude (m/s)."""
    return math.sqrt(_GAMMA * _R * temperature(altitude_m))


def dynamic_pressure(mach: float, altitude_m: float) -> float:
    """Dynamic pressure q = 0.5·γ·p·M² (Pa)."""
    return 0.5 * _GAMMA * pressure(altitude_m) * mach ** 2


def true_airspeed(mach: float, altitude_m: float) -> float:
    """True airspeed V = M·a (m/s)."""
    return mach * speed_of_sound(altitude_m)


def reynolds_number(mach: float, altitude_m: float, length_m: float) -> float:
    """Reynolds number Re = ρ·V·L / μ."""
    rho = density(altitude_m)
    V = true_airspeed(mach, altitude_m)
    T = temperature(altitude_m)
    # Sutherland's law for dynamic viscosity
    mu_ref = 1.716e-5
    T_ref = 273.15
    S_suth = 110.4
    mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S_suth) / (T + S_suth)
    return rho * V * length_m / mu
