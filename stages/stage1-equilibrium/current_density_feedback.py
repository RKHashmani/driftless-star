"""Convert measured DKX current into the next VMEX enclosed-current input."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
from numpy.typing import ArrayLike

from common.atomic_output import atomic_output_path
from common.current_density_feedback_contract import CURRENT_EDGE_TOL, artifact, build_record, resolve_settings
from common.dkx_current import CURRENT_FIELDS, check_current
from common.vmec_namelist import rewrite_current_profile

MU0 = 4.0e-7 * np.pi
EXPORT_ABS_TOL = 1.0e-8
EXPORT_REL_TOL = 1.0e-6
logger = logging.getLogger(__name__)


def load_vmex_helpers() -> SimpleNamespace:
    """Keep the pinned private collocation dependency behind one import boundary.

    Returns
    -------
    SimpleNamespace
        VMEX geometry, profile, input, and collocation helpers with float64 enabled.
    """
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        from vmex.core.bootstrap import (
            _picard_dds_matrix, redl_geometry_from_wout, vmec_j_dot_B_from_wout,
        )
        from vmex.core.input import VmecInput
        from vmex.core.profiles import current
        from vmex.core.setup import flux_profiles, radial_grids
        from vmex.core.wout import read_wout
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Current density feedback requires the pinned Stage 1 VMEX image and bootstrap helpers"
        ) from exc
    return SimpleNamespace(
        dds_matrix=_picard_dds_matrix, geometry=redl_geometry_from_wout,
        jdotb=vmec_j_dot_B_from_wout, input_type=VmecInput, current=current,
        flux_profiles=flux_profiles, radial_grids=radial_grids, read_wout=read_wout,
    )


def _vector(value: ArrayLike, name: str, *, size: int | None = None) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or (size is not None and values.size != size):
        raise ValueError(f"{name} must be a finite 1-D array" + (f" of length {size}" if size else ""))
    return values


def read_current(path: Path, wout_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read native measured surfaces and verify their equilibrium and SI contract.

    Parameters
    ----------
    path, wout_path : Path
        Stage 3 aggregate and the equilibrium against which it was calculated.

    Returns
    -------
    tuple of ndarray
        Native normalized flux, JdotB in A T/m², and B2 in T².
    """
    with h5py.File(path, "r") as h5:
        group = h5.get("current")
        if group is None:
            raise ValueError("Stage 3 current contract is missing or incompatible. Regenerate Stage 3.")
        data = {key: group[key][:] if key in group else None for key in CURRENT_FIELDS}
        errors = check_current(data, dict(group.attrs), wout_sha256=artifact(wout_path)["sha256"])
        if errors:
            raise ValueError("Invalid Stage 3 current. " + "; ".join(errors))
    s = data["s"]
    if s.size < 2:
        raise ValueError("Current density feedback requires at least two measured current surfaces")
    if abs(s[-1] - 1.0) > CURRENT_EDGE_TOL:
        raise ValueError(
            f"Current density feedback requires measured edge coverage at s=1 within {CURRENT_EDGE_TOL:g}, "
            f"but the last measured surface is s={s[-1]:.16g}. Edge extrapolation is unsupported."
        )
    return s, data["JdotB"], data["B2"]


def validate_mapping(inp: Any, wout: Any) -> np.ndarray:
    """Restrict feedback to the standard normalized toroidal-flux mesh.

    Parameters
    ----------
    inp, wout : VMEX objects
        Applied input and its equilibrium.

    Returns
    -------
    ndarray
        Uniform full-mesh normalized toroidal flux coordinates.
    """
    aphi = _vector(inp.aphi, "APHI")
    if aphi.size == 0 or aphi[0] != 1.0 or np.any(aphi[1:] != 0.0) or float(inp.bloat) != 1.0:
        raise ValueError("Current density feedback supports only standard APHI=(1,0,...) and BLOAT=1")
    ns = int(wout.ns)
    if ns < 5:
        raise ValueError("Current inversion requires a WOUT radial mesh with ns >= 5")
    s = np.linspace(0.0, 1.0, ns)
    phi = _vector(wout.phi, "WOUT phi", size=ns)
    if phi[-1] == 0 or not np.allclose(phi / phi[-1], s, rtol=1e-11, atol=1e-13):
        raise ValueError("WOUT phi must describe the standard signed toroidal-flux mesh")
    if float(wout.signgs) not in (-1.0, 1.0):
        raise ValueError("WOUT signgs must be -1 or 1")
    _vector(wout.pres, "WOUT pres", size=ns)
    _vector(wout.buco, "WOUT buco", size=ns)
    return s


def linear_extrapolate(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> np.ndarray:
    """Interpolate linearly, extending the first and last measured slopes.

    Parameters
    ----------
    x : array-like
        Evaluation coordinates.
    xp, fp : array-like
        Strictly increasing sample coordinates and corresponding values.

    Returns
    -------
    ndarray
        Interpolated or extrapolated values at x.
    """
    x, xp, fp = np.asarray(x), np.asarray(xp), np.asarray(fp)
    index = np.clip(np.searchsorted(xp, x, side="left"), 1, len(xp) - 1)
    fraction = (x - xp[index - 1]) / (xp[index] - xp[index - 1])
    return fp[index - 1] + fraction * (fp[index] - fp[index - 1])


def pressure_gradient(wout: Any) -> np.ndarray:
    """Use the stencil in pinned VMEX core.bootstrap.self_consistent_bootstrap.

    Parameters
    ----------
    wout : VMEX WOUT object
        Equilibrium carrying half-mesh pressure in pascals.

    Returns
    -------
    ndarray
        Full-mesh pressure derivative with respect to normalized toroidal flux.
    """
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    ds = s[1] - s[0]
    pres = _vector(wout.pres, "WOUT pres", size=ns)[1:]
    dp_half = np.empty_like(pres)
    dp_half[1:-1] = (pres[2:] - pres[:-2]) / (2 * ds)
    dp_half[0] = (pres[1] - pres[0]) / ds
    dp_half[-1] = (pres[-1] - pres[-2]) / ds
    return np.interp(s, s[1:] - 0.5 * ds, dp_half)


def enclosed_from_wout(wout: Any, surfaces: ArrayLike) -> np.ndarray:
    """Recover physical enclosed current, including VMEX's half-mesh edge extrapolation.

    Parameters
    ----------
    wout : VMEX WOUT object
        Equilibrium carrying buco and the Jacobian sign.
    surfaces : array-like
        Normalized toroidal flux coordinates at which to evaluate current.

    Returns
    -------
    ndarray
        Enclosed toroidal current in amperes, with I(0)=0.
    """
    ns = int(wout.ns)
    half = (np.arange(1, ns) - 0.5) / (ns - 1)
    values = float(wout.signgs) * 2.0 * np.pi / MU0 * _vector(wout.buco, "WOUT buco", size=ns)[1:]
    current = linear_extrapolate(surfaces, half, values)
    return np.where(np.asarray(surfaces) == 0.0, 0.0, current)


def invert_current(wout: Any, measured_s: ArrayLike, measured_jdotb: ArrayLike,
                   helpers: SimpleNamespace) -> tuple[np.ndarray, np.ndarray]:
    """Solve <B²>I' + mu0 p'I = phi_edge<J.B>, imposing I(0)=0.

    Parameters
    ----------
    wout : VMEX WOUT object
        Equilibrium supplying geometry, pressure, and signed edge flux.
    measured_s, measured_jdotb : array-like
        Measured normalized toroidal flux and JdotB in A T/m².
    helpers : SimpleNamespace
        Pinned VMEX helpers from load_vmex_helpers.

    Returns
    -------
    tuple of ndarray
        Full-mesh coordinates and enclosed toroidal current in amperes.
    """
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    b2 = _vector(helpers.geometry(wout, s).fsa_B2, "VMEX geometry B2", size=ns)
    if np.any(b2 <= 0):
        raise ValueError("VMEX inversion geometry requires positive B2")
    jdotb = linear_extrapolate(s, measured_s, measured_jdotb)
    dds = np.asarray(helpers.dds_matrix(ns, s[1] - s[0]), dtype=float)
    matrix = b2[:, None] * dds + np.diag(MU0 * pressure_gradient(wout))
    rhs = float(np.asarray(wout.phi)[-1]) * jdotb
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    rhs[0] = 0.0
    current = np.linalg.solve(matrix, rhs)
    if not np.all(np.isfinite(current)):
        raise ValueError("Current inversion returned nonfinite values")
    current[0] = 0.0
    return s, current


def compare_current(measured_jdotb: ArrayLike, vmex_jdotb: ArrayLike, measured_b2: ArrayLike,
                    settings: dict) -> tuple[float, float]:
    """Return the current residual and threshold in A/m² using DKX B2 for both currents.

    Parameters
    ----------
    measured_jdotb, vmex_jdotb : array-like
        JdotB in A T/m² on the same measured surfaces.
    measured_b2 : array-like
        DKX B2 in T² on those surfaces.
    settings : dict
        Validated absolute and relative convergence tolerances.

    Returns
    -------
    tuple of float
        Infinity-norm residual and its allowed bound in A/m².
    """
    root_b2 = np.sqrt(measured_b2)
    measured_j = np.asarray(measured_jdotb) / root_b2
    vmex_j = np.asarray(vmex_jdotb) / root_b2
    if measured_j.shape != vmex_j.shape or not np.all(np.isfinite(vmex_j)):
        raise ValueError("VMEX current must be finite on every measured DKX surface")
    residual = float(np.max(np.abs(measured_j - vmex_j)))
    threshold = settings["abs_tol"] + settings["rel_tol"] * float(np.max(np.abs(measured_j)))
    return residual, threshold


def _current_bound(intended: ArrayLike) -> float:
    return EXPORT_ABS_TOL + EXPORT_REL_TOL * float(np.max(np.abs(intended)))


def serialized_current(inp: Any, wout: Any, helpers: SimpleNamespace) -> tuple[np.ndarray, np.ndarray]:
    """Return half-mesh coordinates and enclosed current from VMEX's input evaluator.

    Parameters
    ----------
    inp, wout : VMEX objects
        Parsed input and equilibrium supplying radial resolution and orientation.
    helpers : SimpleNamespace
        Pinned VMEX input-profile evaluators.

    Returns
    -------
    tuple of ndarray
        Half-mesh coordinates without the dummy axis and enclosed current in amperes.
    """
    grids = helpers.radial_grids(int(wout.ns))
    profiles = helpers.flux_profiles(inp, grids, r00=1.0, signgs=int(wout.signgs))
    half = np.asarray(grids.s_half, dtype=np.float64)[1:]
    realized = int(wout.signgs) * 2 * np.pi / MU0 * np.asarray(profiles["icurv"], dtype=np.float64)[1:]
    return half, realized


def validate_export(path: Path, wout: Any, s: np.ndarray, current: np.ndarray, helpers: SimpleNamespace) -> dict:
    """Validate serialized knots and VMEX's CURTOR normalization in float64.

    Parameters
    ----------
    path : Path
        Candidate input already validated and written by rewrite_current_profile.
    wout : VMEX WOUT object
        Equilibrium supplying the radial mesh and orientation.
    s, current : ndarray
        Intended spline knots and enclosed current in amperes.
    helpers : SimpleNamespace
        Pinned VMEX input and profile helpers.

    Returns
    -------
    dict
        Normalization error, bound, and intended spline samples for the record.
    """
    inp = helpers.input_type.from_file(path)
    validate_mapping(inp, wout)
    if int(inp.ncurr) != 1 or inp.pcurr_type != "akima_spline_i":
        raise ValueError("Feedback input must use NCURR=1 and akima_spline_i")
    half, realized = serialized_current(inp, wout, helpers)
    intended = np.asarray(helpers.current("akima_spline_i", np.zeros(21), s, current, half, bloat=1.0))
    bound = _current_bound(intended)
    error = float(np.max(np.abs(realized - intended)))
    if not np.all(np.isfinite(realized)) or not np.isfinite(error) or error > bound:
        raise ValueError(f"Serialized VMEX current normalization failed, error {error:.6g} A exceeds {bound:.6g} A")
    return {"normalization_error_A": error, "normalization_bound_A": bound,
            "knots_s": s.tolist(), "enclosed_current_A": current.tolist()}


def validate_applied_equilibrium(inp: Any, wout: Any, helpers: SimpleNamespace) -> dict:
    """Check a feedback-generated equilibrium against the applied current, independently of DKX.

    Parameters
    ----------
    inp, wout : VMEX objects
        Applied NCURR=1 Akima input and its solved equilibrium.
    helpers : SimpleNamespace
        Pinned VMEX profile helpers.

    Returns
    -------
    dict
        Current realization error and bound in amperes.
    """
    if int(inp.ncurr) != 1 or inp.pcurr_type != "akima_spline_i":
        raise ValueError("The preceding feedback input lost its NCURR=1 current spline")
    half, intended = serialized_current(inp, wout, helpers)
    realized = enclosed_from_wout(wout, half)
    if np.any(intended != 0.0) and np.dot(realized, intended) < 0.0:
        raise ValueError("Feedback equilibrium has the opposite current sign to its applied input")
    error = float(np.max(np.abs(realized - intended)))
    bound = _current_bound(intended)
    if not np.isfinite(error) or error > bound:
        raise ValueError(
            f"Feedback equilibrium does not realize the applied current, error {error:.6g} A exceeds {bound:.6g} A"
        )
    edge = float(enclosed_from_wout(wout, np.array([1.0]))[0])
    if not np.isfinite(wout.ctor) or abs(float(wout.ctor) - edge) > _current_bound(np.array([edge])):
        raise ValueError("WOUT ctor disagrees with its half-mesh current edge extrapolation")
    return {"equilibrium_error_A": error, "equilibrium_bound_A": bound}


def _write_candidate(applied: Path, candidate: Path, equilibrium: Any, measured: tuple,
                     settings: dict, helpers: SimpleNamespace) -> dict:
    """Validate a relaxed current input before replacing the candidate output."""
    measured_s, measured_jdotb, _ = measured
    full_s, solved = invert_current(equilibrium, measured_s, measured_jdotb, helpers)
    prior = enclosed_from_wout(equilibrium, full_s)
    relaxed = (1.0 - settings["relaxation"]) * prior + settings["relaxation"] * solved
    knots = measured_s.copy() if measured_s[0] == 0.0 else np.r_[0.0, measured_s]
    knots[-1] = 1.0  # The measured edge passed the roundoff bound. Preserve native s in the record.
    current = np.interp(knots, full_s, relaxed)
    current[0] = 0.0
    candidate.write_text(rewrite_current_profile(applied.read_text(), knots, current))
    return validate_export(candidate, equilibrium, knots, current, helpers)


def _write_record(path: Path, result: dict) -> None:
    """Publish a complete record with normal umask permissions or the existing file's mode."""
    with atomic_output_path(path) as temporary:
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def _vmex_revision() -> str | None:
    """Read the installed solver revision for the completion record."""
    try:
        provenance = json.loads(importlib.metadata.distribution("vmex").read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        return None
    return provenance.get("vcs_info", {}).get("commit_id")


def run_adapter(*, aggregate: Path, wout: Path, applied_input: Path, candidate: Path,
                record: Path, settings_path: Path, boozer: Path) -> dict:
    """Compare measured current and publish the next input plus convergence evidence.

    Parameters
    ----------
    aggregate, wout, applied_input, boozer : Path
        Accepted Stage 3 samples, matching equilibrium, applied input, and Boozer artifact.
    candidate, record : Path
        Candidate VMEX input and JSON completion destinations.
    settings_path : Path
        JSON containing normalized settings and configuration provenance paths.

    Returns
    -------
    dict
        Completion evidence with current samples, residual, settings, and artifact hashes.
    """
    context = json.loads(settings_path.read_text())
    settings = resolve_settings({"current_density_feedback": context["settings"]})
    if not settings["enabled"]:
        raise ValueError("The current density feedback adapter requires enabled feedback")
    context_inputs = context.get("inputs")
    required = {"common_config", "dkx_config", "effective_config"}
    if not isinstance(context_inputs, dict) or not required <= context_inputs.keys():
        raise ValueError("Current density feedback context lacks common, DKX, or effective configuration provenance")
    paths = dict(context_inputs)
    paths.update(aggregate=aggregate, wout=wout, boozer=boozer, applied_input=applied_input,
                 context=settings_path)
    destinations = {candidate.resolve(), record.resolve()}
    if len(destinations) != 2 or destinations.intersection(Path(path).resolve() for path in paths.values()):
        raise ValueError(
            "Candidate input and completion record must be distinct from each other and every source input"
        )
    for path in paths.values():
        if not Path(path).is_file():
            raise ValueError(f"Current density feedback source file is missing, {path}")
    measured_s, measured_jdotb, measured_b2 = read_current(aggregate, wout)
    helpers = load_vmex_helpers()
    equilibrium = helpers.read_wout(wout)
    inp = helpers.input_type.from_file(applied_input)
    validate_mapping(inp, equilibrium)
    diagnostics = {}
    if int(inp.ncurr) == 1 and inp.pcurr_type == "akima_spline_i":
        diagnostics.update(validate_applied_equilibrium(inp, equilibrium, helpers))
    geometry = helpers.geometry(equilibrium, measured_s)
    vmex_jdotb = np.asarray(helpers.jdotb(equilibrium, measured_s, geom=geometry), dtype=float)
    residual, threshold = compare_current(measured_jdotb, vmex_jdotb, measured_b2, settings)
    converged = residual <= threshold
    with atomic_output_path(candidate) as temporary:
        if converged:
            temporary.write_bytes(applied_input.read_bytes())
        else:
            diagnostics.update(_write_candidate(
                applied_input, temporary, equilibrium, (measured_s, measured_jdotb, measured_b2), settings, helpers,
            ))
    paths["candidate_input"] = candidate
    result = build_record(
        status="converged" if converged else "continue", residual=residual, threshold=threshold,
        settings=settings, paths=paths, measured_s=measured_s.tolist(),
        j_dkx=(measured_jdotb / np.sqrt(measured_b2)).tolist(),
        j_vmex=(vmex_jdotb / np.sqrt(measured_b2)).tolist(),
        vmex_revision=_vmex_revision(), **diagnostics,
    )
    _write_record(record, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the adapter CLI and propagate conversion failures.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments, defaulting to the process arguments.

    Returns
    -------
    int
        Zero on success and one on conversion failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("aggregate", "wout", "input", "candidate", "record", "settings", "boozer"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_adapter(aggregate=args.aggregate, wout=args.wout, applied_input=args.input,
                             candidate=args.candidate, record=args.record, settings_path=args.settings,
                             boozer=args.boozer)
    except Exception:
        logger.exception("Current density feedback failed")
        return 1
    print(f"Current density feedback {result['status']}: residual {result['residual']:.6g}, "
          f"threshold {result['threshold']:.6g} A/m^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
