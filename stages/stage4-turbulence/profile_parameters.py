"""Calculate profile parameters with T3D self-collision formulas and Stage 4 proton mass units."""
import math

from netCDF4 import Dataset
from scipy.constants import elementary_charge, proton_mass

CONVENTION = "t3d-74a06df-self-gkx-fb55997-proton-v1"


def positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value}")
    return value


def scaling_factor(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("collisionality_scaling_factor must be finite and nonnegative")
    return value


def geometry_scales(path: str, *, need_field: bool) -> tuple[float, float | None]:
    with Dataset(path) as data:
        required = ["Aminor_p", "phi"] if need_field else ["Aminor_p"]
        for name in required:
            if name not in data.variables:
                raise ValueError(f"VMEC profile parameters require {name} in {path}")
        length = positive(data.variables["Aminor_p"][:], f"VMEC Aminor_p in {path}")
        field = None
        if need_field:
            flux = float(data.variables["phi"][-1])
            field = positive(abs(flux) / (math.pi * length**2), f"VMEC reference field in {path}")
    return length, field


def reference_speed(temperature: float, context: str) -> float:
    return math.sqrt(1000 * elementary_charge * positive(temperature, context + " reference temperature") / proton_mass)


def reference_beta(density: float, temperature: float, field: float, context: str) -> float:
    n = positive(density, context + " reference density")
    t = positive(temperature, context + " reference temperature")
    b = positive(field, context + " reference field")
    return positive(0.0403 * n * t / b**2, context + " beta")


def self_collision(
    density: float, temperature: float, mass: float, charge: float, *, context: str,
) -> tuple[float, float | None]:
    """Return the self-collision rate in s^-1 and the Coulomb logarithm.

    Density is in 1e20 m^-3, temperature in keV and mass in proton masses.
    The ion formula simplifies T3D Species.logLambda for collisions within one species.
    """
    n = float(density)
    if not math.isfinite(n) or n < 0:
        raise ValueError(f"{context} density must be finite and nonnegative")
    t = positive(temperature, context + " temperature")
    a = positive(mass, context + " mass")
    z = float(charge)
    positive(abs(z), context + " absolute charge")
    if n == 0:
        return 0.0, None
    ncgs, tev = n * 1e14, t * 1000
    if z < 0:
        log_lambda = 23.5 - math.log(math.sqrt(ncgs) / tev**1.25) - math.sqrt(1e-5 + (math.log(tev) - 2)**2 / 16)
    else:
        log_lambda = 23 - math.log(z**2 / tev * math.sqrt(2 * ncgs * z**2 / tev))
    positive(log_lambda, context + " Coulomb logarithm")
    rate = 285 * z**4 * n * log_lambda / (math.sqrt(a) * t**1.5)
    return positive(rate, context + " self-collision rate"), log_lambda
