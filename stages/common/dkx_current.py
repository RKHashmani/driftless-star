"""Measured Stage 3 current diagnostics and their SI contract."""

from __future__ import annotations

from typing import Any

import numpy as np


CURRENT_CONTRACT_VERSION = 1
CURRENT_FIELDS = ("s", "rho", "JdotB", "B2", "J")
CURRENT_UNITS = {"s": "1", "rho": "1", "JdotB": "A T m^-2", "B2": "T^2", "J": "A m^-2"}
# The reference values in the pinned DKX units module. These do not use B0OverBBar.
CURRENT_NORMALIZATION = {
    "reference_n_m3": 1.0e20,
    "reference_v_ms": float(np.sqrt(2 * 1.0e3 * 1.602176634e-19 / 1.67262192369e-27)),
    "reference_B_T": 1.0,
}
CURRENT_NORMALIZATION["JdotB_scale_to_SI"] = (
    1.602176634e-19 * CURRENT_NORMALIZATION["reference_n_m3"]
    * CURRENT_NORMALIZATION["reference_v_ms"] * CURRENT_NORMALIZATION["reference_B_T"]
)
CURRENT_NORMALIZATION["B2_scale_to_SI"] = CURRENT_NORMALIZATION["reference_B_T"] ** 2


def check_current(data: dict[str, Any], attrs: dict[str, Any], *, wout_sha256: str | None = None) -> list[str]:
    """Return errors in baseline current samples, independently of the padded flux grid.

    Parameters
    ----------
    data : dict
        Current arrays indexed by CURRENT_FIELDS, in the units specified by CURRENT_UNITS.
    attrs : dict
        Current-group contract version, units, normalization, and equilibrium provenance.
    wout_sha256 : str, optional
        Expected equilibrium digest when checking against a particular WOUT.

    Returns
    -------
    list of str
        Contract violations, or an empty list for valid samples.
    """
    problems = []
    if attrs.get("contract_version") != CURRENT_CONTRACT_VERSION:
        problems.append("current contract is missing or incompatible. Regenerate Stage 3 with prepare and run-one")
    for name, units in CURRENT_UNITS.items():
        if attrs.get(f"{name}_units") != units:
            problems.append(f"current {name}_units must be {units!r}")
    for name, value in CURRENT_NORMALIZATION.items():
        actual = attrs.get(name)
        if not isinstance(actual, (float, int, np.number)) or not np.isfinite(actual) or actual != value:
            problems.append(f"current normalization {name} differs from the pinned DKX reference")
    if "wout_sha256" not in attrs or (wout_sha256 is not None and attrs["wout_sha256"] != wout_sha256):
        problems.append("current equilibrium provenance must match wout_sha256")
    arrays = {}
    for name in CURRENT_FIELDS:
        if data.get(name) is None:
            problems.append(f"missing required field 'current/{name}'. Regenerate Stage 3")
            continue
        arr = np.asarray(data[name], dtype=float)
        if arr.ndim != 1 or not arr.size or not np.all(np.isfinite(arr)):
            problems.append(f"current/{name} must be a nonempty finite 1D array")
        else:
            arrays[name] = arr
    if len(arrays) != len(CURRENT_FIELDS):
        return problems
    if len({arr.size for arr in arrays.values()}) != 1:
        return problems + ["current arrays must have equal lengths"]
    s, rho = arrays["s"], arrays["rho"]
    for name in ("s", "rho"):
        arr = arrays[name]
        if np.any((arr < 0) | (arr > 1)) or np.any(np.diff(arr) <= 0):
            problems.append(f"current/{name} must increase strictly within [0, 1]")
    if not np.allclose(s, rho**2, rtol=1e-12, atol=1e-14):
        problems.append("current native coordinates must satisfy s=rho**2")
    if np.any(arrays["B2"] <= 0):
        problems.append("current/B2 must be positive")
    elif not np.allclose(arrays["J"], arrays["JdotB"] / np.sqrt(arrays["B2"]), rtol=1e-12, atol=1e-14):
        problems.append("current/J must equal JdotB/sqrt(B2)")
    return problems
