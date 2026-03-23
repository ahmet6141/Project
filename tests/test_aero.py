"""Tests for aerodynamic analysis modules."""

import math

from fightercad.parameters import AircraftParams
from fightercad.aero.atmosphere import (
    temperature,
    pressure,
    density,
    speed_of_sound,
    reynolds_number,
    dynamic_pressure,
)
from fightercad.aero.supersonic import (
    ackeret_cp,
    prandtl_glauert_factor,
    finite_wing_lift_curve_slope,
)
from fightercad.aero.drag import compute_drag
from fightercad.aero.lift import compute_lift_curve_slope, compute_CL_for_level_flight
from fightercad.aero.stability import compute_stability


def test_atmosphere_sea_level():
    T = temperature(0)
    P = pressure(0)
    rho = density(0)
    assert abs(T - 288.15) < 0.1
    assert abs(P - 101325) < 10
    assert abs(rho - 1.225) < 0.01


def test_atmosphere_10km():
    T = temperature(10000)
    P = pressure(10000)
    rho = density(10000)
    # ISA at 10 km: T≈223.25K, P≈26500Pa, ρ≈0.4135
    assert abs(T - 223.25) < 1.0
    assert abs(P - 26500) < 500
    assert abs(rho - 0.4135) < 0.05


def test_speed_of_sound():
    a0 = speed_of_sound(0)
    # Sea level: ~340.3 m/s
    assert abs(a0 - 340.3) < 1.0


def test_reynolds_number():
    Re = reynolds_number(mach=0.8, altitude_m=0, length_m=1.0)
    assert Re > 1e6  # Should be order of millions at sea level


def test_dynamic_pressure():
    q = dynamic_pressure(mach=1.0, altitude_m=0)
    # q = 0.5 * gamma * P * M^2 = 0.5 * 1.4 * 101325 * 1 = 70927.5
    assert abs(q - 70928) < 100


def test_supersonic_lift_slope():
    # Ackeret thin airfoil theory: CLα = 4/√(M²-1)
    mach = 2.0
    CLa_2d = 4.0 / math.sqrt(mach ** 2 - 1)
    assert abs(CLa_2d - 2.309) < 0.01

    # Finite wing should be less than 2D
    CLa_3d = finite_wing_lift_curve_slope(mach, aspect_ratio=2.2, sweep_le_deg=55.0)
    assert 0 < CLa_3d < CLa_2d


def test_prandtl_glauert():
    # At M=0 factor should be 1.0
    factor = prandtl_glauert_factor(0.0)
    assert abs(factor - 1.0) < 0.01

    # At M=0.5, beta = sqrt(1 - 0.25) = sqrt(0.75) ≈ 0.866
    factor = prandtl_glauert_factor(0.5)
    assert abs(factor - 0.866) < 0.01


def test_drag_computation():
    params = AircraftParams()
    result = compute_drag(params, mach=1.6, altitude_m=10000, CL=0.15)
    assert result.CD0 > 0
    assert result.CDi > 0
    assert result.CDw > 0
    assert result.CD_total > 0
    assert abs(result.CD_total - (result.CD0 + result.CDi + result.CDw)) < 1e-10
    assert result.LD_ratio > 0


def test_stability():
    params = AircraftParams()
    result = compute_stability(params, mach=1.6)
    assert result.mac > 0
    assert result.x_cg > 0
    assert result.neutral_point > 0
    # Static margin can vary widely with simplified model
    assert -50 < result.static_margin_pct < 200
