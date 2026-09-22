#!/usr/bin/env python3
"""Run a radial dkx flux scan from NEOPAX-style profile inputs.

This script connects NEOPAX profile definitions and outputs to DKX. It:

1. reads a NEOPAX ``transport_solution.h5`` file, analytical profile parameters
   from the NEOPAX TOML, or prescribed profile arrays from that TOML,
2. extracts one saved time slice, defaulting to the final one, when using
   ``transport_h5`` profiles,
3. builds a baseline run and optional gradient-response siblings at each selected radius,
4. launches those runs in parallel on CPUs or pinned GPUs,
5. collects particle flux, heat flux, and parallel-flow diagnostics,
6. writes an HDF5 profile file with datasets ``r``, ``Gamma``, ``Q``, and
   ``Upar`` that can be read by NEOPAX's ``FluxesRFileTransportModel``.
7. optionally writes PNG summary plots for ``Gamma``, ``Q``, and ``Upar``.

Default dkx resolution overrides used by this bridge (quickrun smoke test):
- ``Ntheta = 5``
- ``Nzeta = 11``
- ``Nxi = 12``
- ``NL = 3``
- ``Nx = 4``
- ``solverTolerance = 1e-6``

The script requires DKX's ``particleFlux_vm_rHat``, ``heatFlux_vm_rHat``,
``FSABFlow``, ``rHat``, and ``B0OverBBar`` diagnostics.
It uses the final iteration for each species.
It applies the existing conversions to NEOPAX units.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import h5py
import numpy as np
from common.neopax_geometry import read_neopax_minor_radius
from common.neopax_profiles import (
    NEOPAX_DENSITY_REFERENCE_M3,
    NEOPAX_TEMPERATURE_REFERENCE_EV,
    FaceState,
    SpeciesMeta,
    build_analytical_face_state,
    build_prescribed_face_state,
    read_transport_face_state,
)
from scipy.constants import elementary_charge, proton_mass

SFINCS_REFERENCE_R_M = 1.0
SFINCS_REFERENCE_MASS_KG = proton_mass
SFINCS_REFERENCE_T_J = NEOPAX_TEMPERATURE_REFERENCE_EV * elementary_charge
SFINCS_REFERENCE_V_MS = float(np.sqrt(2.0 * SFINCS_REFERENCE_T_J / SFINCS_REFERENCE_MASS_KG))
SFINCS_GAMMA_TO_NEOPAX = NEOPAX_DENSITY_REFERENCE_M3 * SFINCS_REFERENCE_V_MS / SFINCS_REFERENCE_R_M
SFINCS_Q_TO_NEOPAX = (
    NEOPAX_DENSITY_REFERENCE_M3
    * SFINCS_REFERENCE_MASS_KG
    * SFINCS_REFERENCE_V_MS**3
    / SFINCS_REFERENCE_R_M
    / elementary_charge
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


SCRIPT_DIR = Path(__file__).resolve().parent
STAGES_DIR = SCRIPT_DIR.parent

DEFAULT_COMMON_CONFIG = STAGES_DIR.parent / "inputs" / "quick_run" / "common_input.toml"
DEFAULT_DKX_TEMPLATE = STAGES_DIR.parent / "inputs" / "quick_run" / "sfincs_input.HSX_vacuum_ns201_quickrun"
DEFAULT_OUTPUT_DIR = STAGES_DIR.parent / "outputs" / "quick_run" / "stage3_neoclassical"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _infer_neopax_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.parent.name == "examples":
        return parent.parent.parent
    return parent


def _resolve_relative(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _default_transport_solution_path(config_path: Path, cfg: dict[str, Any]) -> Path:
    output_cfg = cfg.get("transport_output", {})
    output_dir = _resolve_relative(_infer_neopax_root(config_path), output_cfg.get("transport_output_dir"))
    if output_dir is None:
        raise ValueError("Could not resolve [transport_output].transport_output_dir from the NEOPAX config.")
    return output_dir / "transport_solution.h5"


def _parse_species_from_config(cfg: dict[str, Any]) -> list[SpeciesMeta]:
    species_cfg = cfg.get("species", {})
    names = list(species_cfg.get("names", []))
    charges = list(species_cfg.get("charge_qp", []))
    masses = list(species_cfg.get("mass_mp", []))
    if not names or len(names) != len(charges) or len(names) != len(masses):
        raise ValueError("NEOPAX [species] must define matching names, charge_qp, and mass_mp arrays.")
    return [
        SpeciesMeta(name=str(name), charge=float(charge), mass_mp=float(mass))
        for name, charge, mass in zip(names, charges, masses)
    ]


def _parse_index_list(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def _choose_radius_indices(
    rho: np.ndarray,
    *,
    explicit: list[int] | None,
    rho_min: float | None,
    rho_max: float | None,
    num_radii: int | None,
) -> list[int]:
    if explicit is not None:
        idxs = sorted(set(int(v) for v in explicit))
        for idx in idxs:
            if idx < 0 or idx >= rho.size:
                raise IndexError(f"rho index {idx} out of range [0, {rho.size - 1}]")
        return idxs

    # By default, skip the magnetic axis point. A local dkx solve at
    # rho=0 is usually not the most useful first-pass transport postprocessing
    # target, and users can still include it explicitly via --rho-indices.
    mask = np.ones_like(rho, dtype=bool)
    mask &= ~np.isclose(rho, 0.0)
    if rho_min is not None:
        mask &= rho >= float(rho_min)
    if rho_max is not None:
        mask &= rho <= float(rho_max)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise ValueError("No radii satisfy the requested rho filter.")
    if num_radii is None or int(num_radii) <= 0 or int(num_radii) >= candidates.size:
        return [int(v) for v in candidates]
    picks = np.linspace(0, candidates.size - 1, int(num_radii))
    return sorted(set(int(candidates[int(round(p))]) for p in picks))


def _bool_literal(value: bool) -> str:
    return ".true." if bool(value) else ".false."


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_literal(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.16g}"


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(_format_scalar(v) for v in np.asarray(value).ravel())
    return _format_scalar(value)


def _patch_group_value(*, text: str, group: str, key: str, value: Any) -> str:
    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", text)
    if start is None:
        raise ValueError(f"Missing namelist group &{group}")
    end = re.search(r"(?m)^\s*/\s*$", text[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{group}")
    end_pos = start.end() + end.start()
    group_txt = text[start.end() : end_pos]

    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    line = f"  {key} = {_format_value(value)}"
    match = pat.search(group_txt)
    if match is not None:
        group_txt2 = group_txt.replace(match.group(0), line)
    else:
        if not group_txt.endswith("\n"):
            group_txt = group_txt + "\n"
        group_txt2 = group_txt + line + "\n"
    return text[: start.end()] + group_txt2 + text[end_pos:]


def _drop_group_key(*, text: str, group: str, key: str) -> str:
    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", text)
    if start is None:
        raise ValueError(f"Missing namelist group &{group}")
    end = re.search(r"(?m)^\s*/\s*$", text[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{group}")
    end_pos = start.end() + end.start()
    group_txt = text[start.end() : end_pos]

    # Locate assignments outside comments and strings. Values may continue onto
    # another line or share a line with the next assignment.
    masked = re.sub(
        r"'[^']*(?:''[^']*)*'|\"[^\"]*(?:\"\"[^\"]*)*\"|![^\n\r]*",
        lambda match: " " * len(match.group()),
        group_txt,
    )
    assignments = list(re.finditer(r"\b([A-Za-z_]\w*)\s*(?:\([^)]*\)\s*)?=", masked))
    group_txt2 = group_txt
    for index in range(len(assignments) - 1, -1, -1):
        assignment = assignments[index]
        if assignment.group(1).lower() == key.lower():
            stop = assignments[index + 1].start() if index + 1 < len(assignments) else len(group_txt)
            group_txt2 = group_txt2[:assignment.start()] + group_txt2[stop:]
    return text[: start.end()] + group_txt2 + text[end_pos:]


FD_CHANNELS = {"density_gradient": "fd_n", "temperature_gradient": "fd_t"}


def _response_settings(args: argparse.Namespace, species: list[SpeciesMeta]) -> dict[str, Any]:
    """Validate response controls and resolve species selections.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed response mode, species selections, and step controls.
    species : list[SpeciesMeta]
        Species names from the common input, in solver order.

    Returns
    -------
    dict[str, Any]
        Response settings with canonical species names, or empty selections
        when response mode is disabled.

    Raises
    ------
    ValueError
        An enabled response has invalid steps or species selections.
    """
    settings = {
        "response_mode": args.response_mode,
        "perturb_density_species": [],
        "perturb_temperature_species": [],
        "dkap_density": args.dkap_density,
        "dkap_temperature": args.dkap_temperature,
        "perturb_rel_step": args.perturb_rel_step,
    }
    if args.response_mode == "none":
        return settings
    names = {sp.name.lower(): sp.name for sp in species}
    if len(names) != len(species):
        raise ValueError("Species names must be unique ignoring case.")
    for key in ("dkap_density", "dkap_temperature", "perturb_rel_step"):
        value = settings[key]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    for key in ("perturb_density_species", "perturb_temperature_species"):
        for token in getattr(args, key).split(","):
            token = token.strip()
            if not token:
                continue
            name = names.get(token.lower())
            if name is None:
                raise ValueError(f"Unknown {key} species {token!r}.")
            if re.fullmatch(r"\w+", name) is None:
                raise ValueError(f"Species {name!r} cannot name a response run directory.")
            if name not in settings[key]:
                settings[key].append(name)
    if not (settings["perturb_density_species"] or settings["perturb_temperature_species"]):
        raise ValueError("fd_gradients requires at least one perturbation species.")
    return settings


def _response_states(
    *,
    snapshot: FaceState,
    species: list[SpeciesMeta],
    radius_indices: list[int],
    settings: dict[str, Any],
) -> list[tuple[int, FaceState, dict[str, Any]]]:
    """Prepare all states with kappa = -d(log profile)/d(rho).

    Density steps change kappa_n by delta and kappa_T by -delta to preserve
    the pressure gradient. Temperature steps change only kappa_T.

    Parameters
    ----------
    snapshot : FaceState
        Baseline profiles and gradients on the radial faces.
    species : list[SpeciesMeta]
        Species metadata in the same order as the profile arrays.
    radius_indices : list[int]
        Face indices selected for the scan.
    settings : dict[str, Any]
        Validated settings from ``_response_settings``.

    Returns
    -------
    list[tuple[int, FaceState, dict[str, Any]]]
        Radius index, profile state, and response metadata for each run.

    Raises
    ------
    ValueError
        A selected profile is invalid or its effective step is zero or nonfinite.
    """
    species_names = [sp.name for sp in species]
    pairs = [
        (channel, species_names.index(name))
        for channel, selection in (
            ("density_gradient", "perturb_density_species"),
            ("temperature_gradient", "perturb_temperature_species"),
        )
        for name in settings[selection]
    ]
    states = []
    for radius_index in radius_indices:
        states.append((radius_index, snapshot, {
            "response_label": "base", "perturb_species": "none", "perturb_delta": 0.0,
        }))
        for channel, si in pairs:
            name = species[si].name
            location = f"species {name!r} at rho={snapshot.rho[radius_index]} (index {radius_index})"
            n, t = snapshot.density[si, radius_index], snapshot.temperature[si, radius_index]
            dn, dt = snapshot.density_grad[si, radius_index], snapshot.temperature_grad[si, radius_index]
            if not np.all(np.isfinite([n, t, dn, dt])) or n <= 0 or t <= 0:
                raise ValueError(
                    f"Gradient response for {location} requires finite gradients "
                    "and positive local density and temperature."
                )
            density_grad = snapshot.density_grad.copy()
            temperature_grad = snapshot.temperature_grad.copy()
            if channel == "density_gradient":
                delta = -max(settings["dkap_density"], settings["perturb_rel_step"] * abs(dn / n))
                density_grad[si, radius_index] -= n * delta
                temperature_grad[si, radius_index] += t * delta
            else:
                delta = max(settings["dkap_temperature"], settings["perturb_rel_step"] * abs(dt / t))
                temperature_grad[si, radius_index] -= t * delta
            if not np.isfinite(delta) or delta == 0:
                raise ValueError(f"Gradient response for {location} must have a finite nonzero step.")
            state = replace(snapshot, density_grad=density_grad, temperature_grad=temperature_grad)
            states.append((radius_index, state, {
                "response_label": channel, "perturb_species": name, "perturb_delta": delta,
            }))
    return states


def _prepare_input_text(
    *,
    template_text: str,
    species: list[SpeciesMeta],
    snapshot: FaceState,
    radius_index: int,
    include_phi1: bool | None,
    resolution_overrides: dict[str, int | None],
    solver_tolerance: float | None,
) -> str:
    rho = snapshot.rho
    density = snapshot.density
    temperature = snapshot.temperature
    er = snapshot.er
    # sfincs reads dNHatdrNs and dTHatdrNs against rN, which is rho.
    # Therefore, the namelist uses the per-rho face gradients from the profile state.
    dndr = snapshot.density_grad
    dtdr = snapshot.temperature_grad

    text = template_text
    text = _patch_group_value(text=text, group="general", key="RHSMode", value=1)
    text = _patch_group_value(text=text, group="geometryParameters", key="inputRadialCoordinate", value=3)
    # Let dkx infer gradient coordinates separately:
    # species from dNHatdrNs/dTHatdrNs -> mode 3, Phi from Er -> mode 4.
    text = _drop_group_key(text=text, group="geometryParameters", key="inputRadialCoordinateForGradients")
    text = _patch_group_value(text=text, group="geometryParameters", key="rN_wish", value=float(rho[radius_index]))

    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="Zs",
        value=[sp.charge for sp in species],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="mHats",
        value=[sp.mass_mp for sp in species],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="nHats",
        value=density[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="THats",
        value=temperature[:, radius_index],
    )
    # DKX prefers rHat, psiHat, and psiN entries over rN. Remove template
    # gradients, including indexed entries, before writing the profile values.
    for field in ("N", "T"):
        for coordinate in ("psiHat", "psiN", "rHat", "rN"):
            text = _drop_group_key(
                text=text, group="speciesParameters", key=f"d{field}Hatd{coordinate}s"
            )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="dNHatdrNs",
        value=dndr[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="dTHatdrNs",
        value=dtdr[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="physicsParameters",
        key="Er",
        value=float(er[radius_index]),
    )

    if include_phi1 is not None:
        text = _patch_group_value(
            text=text,
            group="physicsParameters",
            key="includePhi1",
            value=bool(include_phi1),
        )

    for key, value in resolution_overrides.items():
        if value is not None:
            text = _patch_group_value(text=text, group="resolutionParameters", key=key, value=int(value))

    if solver_tolerance is not None:
        text = _patch_group_value(
            text=text,
            group="resolutionParameters",
            key="solverTolerance",
            value=float(solver_tolerance),
        )

    return text


def _infer_wout_path(config_path: Path, cfg: dict[str, Any], explicit: str | None) -> Path | None:
    if explicit:
        # Resolve a command-line override as an absolute path or relative to the current directory.
        # Do not resolve it relative to the NEOPAX root.
        expanded = os.path.expandvars(os.path.expanduser(explicit))
        return Path(expanded).resolve()
    geometry_cfg = cfg.get("geometry", {})
    return _resolve_relative(_infer_neopax_root(config_path), geometry_cfg.get("vmec_file"))


def _infer_booz_path(config_path: Path, cfg: dict[str, Any], explicit: str | None) -> Path | None:
    if explicit:
        # CLI override: standard path semantics (absolute or relative to CWD),
        # not relative to the NEOPAX root.
        expanded = os.path.expandvars(os.path.expanduser(explicit))
        return Path(expanded).resolve()
    geometry_cfg = cfg.get("geometry", {})
    return _resolve_relative(_infer_neopax_root(config_path), geometry_cfg.get("boozer_file"))


def _last_species_vector(arr: np.ndarray, n_species: int) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 2 and data.shape[1] > 0:
        data = data[:, -1]
    if data.ndim == 1:
        if data.shape[0] != n_species:
            raise ValueError(f"Expected {n_species} species values, got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("DKX final diagnostics contain non-finite values.")
        return data
    raise ValueError(f"Unsupported diagnostic shape {data.shape} for n_species={n_species}")


def _last_scalar(arr: np.ndarray) -> float:
    data = np.asarray(arr, dtype=np.float64)
    if data.size != 1 or not np.all(np.isfinite(data)):
        raise ValueError("Expected one finite DKX diagnostic value.")
    return float(data.reshape(-1)[0])


def _extract_flux_triplet(
    results: dict[str, Any],
    n_species: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, str]]:
    gamma_key = "particleFlux_vm_rHat"
    q_key = "heatFlux_vm_rHat"
    required = (gamma_key, q_key, "FSABFlow", "rHat", "B0OverBBar")
    missing = [key for key in required if key not in results]
    if missing:
        raise KeyError(f"DKX output is missing required diagnostics {missing}.")

    gamma_hat = _last_species_vector(np.asarray(results[gamma_key]), n_species)
    q_hat = _last_species_vector(np.asarray(results[q_key]), n_species)
    gamma = SFINCS_GAMMA_TO_NEOPAX * gamma_hat
    q = SFINCS_Q_TO_NEOPAX * q_hat

    fsab_flow = _last_species_vector(np.asarray(results["FSABFlow"]), n_species)
    b0_over_bbar = _last_scalar(np.asarray(results["B0OverBBar"], dtype=np.float64))
    # NTX fixed-field parallel-flow audit bridge:
    # NEOPAX's physical Upar closure matches the SFINCS hat-normalized FSABFlow
    # after restoring the historical factor 2 * B0OverBBar / sqrt(pi).
    flow_bridge = 2.0 * float(b0_over_bbar) / float(np.sqrt(np.pi))
    upar = flow_bridge * fsab_flow
    meta = {
        "Gamma_key": gamma_key,
        "Q_key": q_key,
        "Upar_key": "FSABFlow",
        "Gamma_scale_to_neopax": f"{SFINCS_GAMMA_TO_NEOPAX:.16g}",
        "Q_scale_to_neopax": f"{SFINCS_Q_TO_NEOPAX:.16g}",
        "Upar_bridge": "2*B0OverBBar/sqrt(pi)",
    }
    r_hat = _last_scalar(np.asarray(results["rHat"], dtype=np.float64))
    return gamma, q, upar, r_hat, meta


def _build_worker_env(args: argparse.Namespace, *, gpu_id: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if str(args.backend).lower() == "gpu":
        # On NVIDIA-backed JAX installs, using the generic "gpu" alias can
        # still let JAX probe other accelerator backends such as ROCm. Force
        # CUDA explicitly for these worker processes.
        env["JAX_PLATFORMS"] = "cuda"
        env["JAX_PLATFORM_NAME"] = "cuda"
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Drop accelerator-selection variables inherited from the parent shell
        # that can redirect JAX toward a different backend family.
        env.pop("PJRT_DEVICE", None)
        env.pop("JAX_BACKEND_TARGET", None)
        env.pop("ROCM_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
    else:
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
        # Some recent JAX/XLA builds reject the legacy CPU-thread flags that older
        # shells or dkx opt-in settings may add. Keep other XLA flags, but
        # scrub the unsupported CPU-thread knobs for per-worker CPU launches.
        xla_flags = env.get("XLA_FLAGS", "")
        filtered_xla_flags = " ".join(
            token
            for token in xla_flags.split()
            if not token.startswith("--xla_cpu_parallelism_threads=")
            and not token.startswith("--xla_cpu_multi_thread_eigen_num_threads=")
            and not token.startswith("--xla_cpu_multi_thread_eigen=")
        ).strip()
        if filtered_xla_flags:
            env["XLA_FLAGS"] = filtered_xla_flags
        else:
            env.pop("XLA_FLAGS", None)
        cores = int(args.cores_per_run)
        threads = max(1, cores)
        if cores > 0:
            env["DKX_CORES"] = str(cores)
        # Pin native thread pools so N parallel workers do not each try to use
        # the whole machine. This matters a lot for medium/heavy CPU scans.
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(threads)
        env["NUMEXPR_NUM_THREADS"] = str(threads)
    return env


def _solver_identity() -> dict[str, str]:
    """Read the installed solver identity without importing DKX or initializing JAX."""
    package = distribution("dkx")
    direct_url = json.loads(package.read_text("direct_url.json") or "{}")
    return {
        "name": "dkx",
        "version": package.version,
        "revision": direct_url.get("vcs_info", {}).get("commit_id", ""),
    }


def _wout_digest(path: Path | None) -> str | None:
    """Identify the equilibrium contents used by this scan."""
    if path is None:
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_summary(summary: dict[str, Any], expected: dict[str, Any]) -> None:
    """Check one completion record against its prepared surface."""
    for key in ("solver", "wout_sha256", "radius_index", "rho"):
        if key not in summary or summary[key] != expected[key]:
            raise ValueError(f"DKX result has a mismatched {key}. Run prepare and solve the surface again.")
    r_hat = _last_scalar(np.asarray(summary["rHat"], dtype=float))
    rho = float(summary["rho"])
    if not np.isfinite(rho) or rho < 0 or r_hat < 0 or (rho == 0) != (r_hat == 0):
        raise ValueError("DKX result radii must be nonnegative and agree on the magnetic axis.")
    for key in ("Gamma", "Q", "Upar"):
        values = np.asarray(summary[key], dtype=float)
        if values.shape != (expected["n_species"],) or not np.all(np.isfinite(values)):
            raise ValueError(f"DKX result {key} must contain one finite value per species.")


def _run_single_worker_from_payload(payload_path: Path) -> int:
    with payload_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    input_path = Path(payload["input_path"])
    output_path = Path(payload["output_path"])
    result_json = Path(payload["result_json"])
    result_json.unlink(missing_ok=True)
    wout_path = payload.get("wout_path")
    n_species = int(payload["n_species"])
    benchmark_repeats = max(0, int(payload.get("benchmark_repeats", 0)))
    benchmark_warmup = max(0, int(payload.get("benchmark_warmup", 0)))

    if payload.get("solver") != _solver_identity():
        raise ValueError("The installed DKX differs from the prepared scan. Run prepare again.")
    resolved_wout = None if wout_path in (None, "") else Path(wout_path)
    if payload.get("wout_sha256") != _wout_digest(resolved_wout):
        raise ValueError("The equilibrium changed after preparation. Run prepare again.")

    from dkx.api import read_output, write_output

    solve_count = 1 if benchmark_repeats <= 0 else benchmark_warmup + benchmark_repeats
    elapsed_s: list[float] = []
    for _ in range(solve_count):
        t0 = time.perf_counter()
        write_output(
            input_path,
            output_path,
            wout_path=resolved_wout,
            overwrite=True,
            emit=print if payload.get("verbose", False) else None,
        )
        elapsed_s.append(time.perf_counter() - t0)
    results = read_output(output_path)
    gamma, q, upar, r_hat, meta = _extract_flux_triplet(results, n_species)
    summary = {
        "solver": payload["solver"],
        "wout_sha256": payload["wout_sha256"],
        "radius_index": int(payload["radius_index"]),
        "rho": float(payload["rho"]),
        "rHat": float(r_hat),
        "Gamma": gamma.tolist(),
        "Q": q.tolist(),
        "Upar": upar.tolist(),
        "meta": meta,
    }
    if benchmark_repeats > 0:
        warm_runs = elapsed_s[benchmark_warmup:]
        summary["benchmark"] = {
            "warmup_runs": int(benchmark_warmup),
            "repeats": int(benchmark_repeats),
            "all_runs_s": [float(v) for v in elapsed_s],
            "cold_run_s": float(elapsed_s[0]),
            "warm_runs_s": [float(v) for v in warm_runs],
            "warm_mean_s": float(np.mean(warm_runs)) if warm_runs else float("nan"),
            "warm_min_s": float(np.min(warm_runs)) if warm_runs else float("nan"),
        }
    # Unit conversions can overflow even when native diagnostics are finite.
    _validate_summary(summary, payload)
    with result_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


def _existing_result_is_usable(
    result_json: Path,
    *,
    expected: dict[str, Any],
    require_benchmark: bool,
) -> bool:
    if not result_json.exists() or not expected["solver"]["revision"] or not expected["wout_sha256"]:
        return False
    try:
        summary = json.loads(result_json.read_text(encoding="utf-8"))
        _validate_summary(summary, expected)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if require_benchmark and "benchmark" not in summary:
        return False
    return True


def _write_summary_plots(
    *,
    output_dir: Path,
    rho: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    upar: np.ndarray,
    species: list[SpeciesMeta],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Plotting was requested but matplotlib is not installed."
        ) from exc

    traces = (
        ("Gamma", np.asarray(gamma, dtype=np.float64), output_dir / "Gamma_vs_rho.png"),
        ("Q", np.asarray(q, dtype=np.float64), output_dir / "Q_vs_rho.png"),
        ("Upar", np.asarray(upar, dtype=np.float64), output_dir / "Upar_vs_rho.png"),
    )

    for label, values, path in traces:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        for i, sp in enumerate(species):
            ax.plot(rho, values[i], marker="o", linewidth=1.5, markersize=4.0, label=sp.name)
        ax.set_xlabel("rho")
        ax.set_ylabel(label)
        ax.set_title(f"{label} from dkx radial scan")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _suppress_benign_worker_stderr(stderr: str, args: argparse.Namespace) -> str:
    if str(args.backend).lower() != "cpu":
        return stderr
    text = (stderr or "").strip()
    if not text:
        return ""
    benign_markers = (
        "Jax plugin configuration error",
        "jax_plugins.xla_cuda12.initialize()",
        "cuda_device_count()",
        "operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE",
    )
    if all(marker in text for marker in benign_markers):
        return ""
    return stderr


def _launch_one_subprocess(
    *,
    script_path: Path,
    payload_path: Path,
    env: dict[str, str],
    stream_output: bool,
) -> subprocess.Popen[str]:
    cmd = [sys.executable, str(script_path), "--worker-payload", str(payload_path)]
    kwargs: dict[str, Any] = {
        "env": env,
        "text": True,
    }
    if not stream_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        cmd,
        **kwargs,
    )


def _terminate_worker_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, 15)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return


def _kill_worker_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, 9)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            return


def _cleanup_worker_processes(active: list[dict[str, Any]]) -> None:
    for item in active:
        _terminate_worker_process(item["proc"])
    deadline = time.time() + 5.0
    while time.time() < deadline:
        remaining = [item for item in active if item["proc"].poll() is None]
        if not remaining:
            return
        time.sleep(0.1)
    for item in active:
        _kill_worker_process(item["proc"])
    for item in active:
        proc = item["proc"]
        try:
            proc.communicate(timeout=1.0)
        except Exception:
            pass


def _collect_worker_result(
    proc: subprocess.Popen[str],
    *,
    stream_output: bool,
) -> tuple[int, str, str]:
    if stream_output:
        code = proc.wait()
        return int(code), "", ""
    stdout, stderr = proc.communicate()
    return int(proc.returncode), stdout or "", stderr or ""


def _run_tasks_in_parallel(
    *,
    task_payloads: list[Path],
    args: argparse.Namespace,
    gpu_ids: list[str],
) -> None:
    max_parallel = max(1, int(args.max_parallel))
    script_path = Path(__file__).resolve()
    total = len(task_payloads)
    completed = 0

    def _spawn(payload_path: Path, slot: int) -> dict[str, Any]:
        gpu_id = None
        if str(args.backend).lower() == "gpu":
            gpu_id = gpu_ids[slot % len(gpu_ids)]
        env = _build_worker_env(args, gpu_id=gpu_id)
        stream_output = bool(args.verbose_workers) and max_parallel == 1
        proc = _launch_one_subprocess(
            script_path=script_path,
            payload_path=payload_path,
            env=env,
            stream_output=stream_output,
        )
        return {
            "proc": proc,
            "payload_path": payload_path,
            "gpu_id": gpu_id,
            "stream_output": stream_output,
        }

    active: list[dict[str, Any]] = []
    pending = list(task_payloads)
    slot = 0
    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                payload_path = pending.pop(0)
                worker = _spawn(payload_path, slot)
                active.append(worker)
                rho_value = None
                try:
                    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
                    rho_value = float(payload.get("rho"))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    rho_value = None
                rho_note = f" rho={rho_value:.4f}" if rho_value is not None else ""
                gpu_note = f" gpu={worker['gpu_id']}" if worker["gpu_id"] is not None else ""
                print(
                    f"[dkx-scan] launched {completed + len(active)}/{total}:{rho_note}{gpu_note}",
                    flush=True,
                )
                slot += 1
            time.sleep(0.1)
            finished: list[dict[str, Any]] = []
            for item in active:
                if item["proc"].poll() is not None:
                    finished.append(item)
            if not finished:
                continue
            active = [item for item in active if item not in finished]
            for item in finished:
                proc = item["proc"]
                payload_path = item["payload_path"]
                gpu_id = item["gpu_id"]
                stream_output = item["stream_output"]
                code, stdout, stderr = _collect_worker_result(proc, stream_output=stream_output)
                label = str(payload_path) if payload_path is not None else "<payload>"
                if code != 0:
                    _cleanup_worker_processes(active)
                    msg = [
                        f"dkx worker failed for {label}",
                    ]
                    if gpu_id is not None:
                        msg.append(f"gpu={gpu_id}")
                    if code < 0 or code in (137, 143):
                        msg.append(
                            f"worker exited via signal (returncode={code}) "
                            "with no Python traceback. This signature usually "
                            "means the container ran out of memory and the "
                            "kernel killed the worker. Try one of:"
                        )
                        msg.append(
                            f"  - lower --max-parallel "
                            f"(current run: {int(args.max_parallel)})"
                        )
                        msg.append(
                            "  - increase the RAM available to the workers "
                            "(either on the host or in the container runtime)"
                        )
                    if stdout.strip():
                        msg.append("stdout:\n" + stdout.strip())
                    if stderr.strip():
                        msg.append("stderr:\n" + stderr.strip())
                    raise RuntimeError("\n".join(msg))
                if bool(args.verbose_workers) and not stream_output:
                    if stdout.strip():
                        print(f"[dkx-worker stdout] {label}\n{stdout.strip()}", flush=True)
                    stderr_to_print = _suppress_benign_worker_stderr(stderr, args)
                    if stderr_to_print.strip():
                        print(f"[dkx-worker stderr] {label}\n{stderr_to_print.strip()}", flush=True)
                completed += 1
                rho_value = None
                if payload_path is not None:
                    try:
                        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
                        rho_value = float(payload.get("rho"))
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        rho_value = None
                rho_note = f" rho={rho_value:.4f}" if rho_value is not None else ""
                gpu_note = f" gpu={gpu_id}" if gpu_id is not None else ""
                print(f"[dkx-scan] completed {completed}/{total}:{rho_note}{gpu_note}", flush=True)
    except KeyboardInterrupt:
        _cleanup_worker_processes(active)
        raise
    except Exception:
        _cleanup_worker_processes(active)
        raise


def _prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list[Path]]:
    """Build per-surface dkx inputs and the scan manifest.

    Loads profiles, selects flux surfaces, and writes an ``input.namelist`` plus ``payload.json`` under
    each ``output_dir/runs/<run_subdir>`` directory, then records ``output_dir/manifest.json`` describing
    the scan. Reuse detection is preserved: a surface whose namelist is byte-identical to the existing
    one and whose ``result.json`` is usable is treated as already complete and is not returned as
    pending.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the prepare phase.

    Returns
    -------
    tuple[dict[str, Any], list[Path]]
        The manifest dict (also written to ``manifest.json``) and the ``payload.json`` paths of surfaces
        that still need to run. Pending-ness is in-memory only and is never stored in the manifest. Each
        ``runs`` entry records only paths relative to ``output_dir`` so a host scheduler can rebuild real
        paths under its own view of the output directory.
    """
    config_path = Path(args.common_config).resolve()
    cfg = _load_toml(config_path)
    species = _parse_species_from_config(cfg)
    response_settings = _response_settings(args, species)
    wout_path = _infer_wout_path(config_path, cfg, args.wout_path)
    booz_path = _infer_booz_path(config_path, cfg, args.boozer_path)
    profiles_source = str(args.profiles_source).lower()
    if profiles_source == "analytical":
        transport_solution = None
        analytical_n_radii = args.analytical_n_radii
        if analytical_n_radii is None or int(analytical_n_radii) <= 0:
            # [geometry].n_radial counts transport cells. The scan samples their bounding faces.
            # Therefore, the default is the face count. An explicit analytical radius count is also
            # a face count. It can give NEOPAX a finer or coarser interpolation grid.
            analytical_n_radii = int(cfg.get("geometry", {}).get("n_radial", 51)) + 1
        if wout_path is None or booz_path is None:
            raise ValueError(
                "--profiles-source=analytical reconstructs the transport faces on NEOPAX's metre grid, so "
                "[geometry].vmec_file and [geometry].boozer_file must both resolve"
            )
        snapshot = build_analytical_face_state(
            cfg,
            species=species,
            n_faces=int(analytical_n_radii),
            minor_radius=read_neopax_minor_radius(wout_path, booz_path),
        )
    elif profiles_source == "prescribed":
        transport_solution = None
        snapshot = build_prescribed_face_state(cfg, n_species=len(species))
    else:
        transport_solution = (
            Path(args.neopax_result).resolve()
            if args.neopax_result is not None
            else _default_transport_solution_path(config_path, cfg)
        )
        if not transport_solution.exists():
            raise FileNotFoundError(
                f"Could not find NEOPAX transport result at {transport_solution}. "
                "If needed, enable [transport_output].transport_write_hdf5 = true and rerun NEOPAX."
            )
        snapshot = read_transport_face_state(transport_solution, time_index=int(args.time_index))
    explicit_indices = _parse_index_list(args.rho_indices)
    radius_indices = _choose_radius_indices(
        snapshot.rho,
        explicit=explicit_indices,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        num_radii=args.num_radii,
    )

    template_path = Path(args.dkx_template).resolve()
    template_text = template_path.read_text(encoding="utf-8")
    solver = _solver_identity()
    wout_sha256 = _wout_digest(wout_path)
    if not solver["revision"] or not wout_sha256:
        missing = []
        if not solver["revision"]:
            missing.append("installed DKX Git revision")
        if not wout_sha256:
            missing.append("WOUT content digest")
        print(f"[dkx-scan] result reuse disabled. Missing {' and '.join(missing)}.", flush=True)

    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolution_overrides = {
        "Ntheta": args.ntheta,
        "Nzeta": args.nzeta,
        "Nxi": args.nxi,
        "NL": args.nl,
        "Nx": args.nx,
    }

    pending_task_payloads: list[Path] = []
    manifest_runs: list[dict[str, Any]] = []
    response_runs = _response_states(
        snapshot=snapshot, species=species, radius_indices=radius_indices, settings=response_settings,
    )
    for radius_index, state, response in response_runs:
        rho_value = float(snapshot.rho[radius_index])
        run_name = f"rho_{radius_index:03d}_r{rho_value:.4f}".replace(".", "p")
        if response["response_label"] != "base":
            run_name += f"_{FD_CHANNELS[response['response_label']]}_{response['perturb_species']}"
        surface_dir = run_dir / run_name
        surface_dir.mkdir(parents=True, exist_ok=True)

        input_text = _prepare_input_text(
            template_text=template_text,
            species=species,
            snapshot=state,
            radius_index=radius_index,
            include_phi1=args.include_phi1,
            resolution_overrides=resolution_overrides,
            solver_tolerance=args.solver_tolerance,
        )
        input_path = surface_dir / "input.namelist"
        result_json = surface_dir / "result.json"
        existing_input_text = None
        if input_path.exists():
            try:
                existing_input_text = input_path.read_text(encoding="utf-8")
            except Exception:
                existing_input_text = None
        payload = {
            "solver": solver,
            "wout_sha256": wout_sha256,
            "radius_index": int(radius_index),
            "rho": rho_value,
            "input_path": str(input_path),
            "output_path": str(surface_dir / "sfincsOutput.h5"),
            "result_json": str(result_json),
            "wout_path": None if wout_path is None else str(wout_path),
            "n_species": len(species),
            "verbose": bool(args.verbose_workers),
            "benchmark_repeats": int(args.benchmark_repeats) if response["response_label"] == "base" else 0,
            "benchmark_warmup": int(args.benchmark_warmup),
        }
        can_reuse = (
            existing_input_text == input_text
            and _existing_result_is_usable(
                result_json,
                expected=payload,
                require_benchmark=payload["benchmark_repeats"] > 0,
            )
        )
        input_path.write_text(input_text, encoding="utf-8")
        payload_path = surface_dir / "payload.json"
        with payload_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if not can_reuse:
            result_json.unlink(missing_ok=True)
            pending_task_payloads.append(payload_path)
        manifest_runs.append(
            {
                **response,
                "index": len(manifest_runs),
                "radius_index": int(radius_index),
                "rho": rho_value,
                "run_subdir": run_name,
                "payload": "payload.json",
                "result_json": "result.json",
            }
        )

    manifest = {
        "schema_version": 1,
        **response_settings,
        "solver": solver,
        "wout_sha256": wout_sha256,
        "profiles_source": str(args.profiles_source),
        "source_transport_solution": None if transport_solution is None else str(transport_solution),
        "source_dkx_template": str(template_path),
        "time_index": int(args.time_index),
        "time_value": None if snapshot.time_value is None else float(snapshot.time_value),
        "include_phi1": args.include_phi1,
        "backend": str(args.backend).lower(),
        "max_parallel": int(args.max_parallel),
        "benchmark_repeats": int(args.benchmark_repeats),
        "benchmark_warmup": int(args.benchmark_warmup),
        "species_meta": [
            {"name": sp.name, "charge": sp.charge, "mass_mp": sp.mass_mp} for sp in species
        ],
        "wout_path": None if wout_path is None else str(wout_path),
        "boozer_path": None if booz_path is None else str(booz_path),
        "runs": manifest_runs,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return manifest, pending_task_payloads


def _collect_responses(
    manifest: dict[str, Any],
    responses: dict[tuple[int, str, str], dict[str, Any]],
    baseline_runs: list[dict[str, Any]],
    n_species: int,
    axis_padded: bool,
) -> dict[str, Any]:
    """Collect sibling fluxes and prepared increments in configured pair order.

    Parameters
    ----------
    manifest : dict[str, Any]
        Prepared run list and response settings.
    responses : dict[tuple[int, str, str], dict[str, Any]]
        Perturbed results and signed steps keyed by radius index, channel, and input species.
    baseline_runs : list[dict[str, Any]]
        Baseline manifest entries sorted by radius.
    n_species : int
        Number of output species.
    axis_padded : bool
        Whether the aggregate starts with a synthetic zero-flux axis.

    Returns
    -------
    dict[str, Any]
        Response datasets, or an empty mapping for a baseline scan.

    Raises
    ------
    KeyError
        A configured response is missing from the prepared results.
    """
    pairs = []
    if manifest["response_mode"] == "fd_gradients":
        for channel, key in (
            ("density_gradient", "perturb_density_species"),
            ("temperature_gradient", "perturb_temperature_species"),
        ):
            pairs.extend((channel, name) for name in manifest[key])
    if not pairs:
        return {}
    nr = len(baseline_runs) + int(axis_padded)
    data = {
        "Gamma_perturbed": np.zeros((len(pairs), n_species, nr)),
        "Q_perturbed": np.zeros((len(pairs), n_species, nr)),
        "perturb_delta": np.zeros((len(pairs), nr)),
        "perturb_present": np.zeros((len(pairs), nr), dtype=bool),
        "response_label": [pair[0] for pair in pairs],
        "perturb_species": [pair[1] for pair in pairs],
    }
    for ri, run in enumerate(baseline_runs, start=int(axis_padded)):
        for pi, (channel, name) in enumerate(pairs):
            result = responses[(run["radius_index"], channel, name)]
            data["Gamma_perturbed"][pi, :, ri] = result["Gamma"]
            data["Q_perturbed"][pi, :, ri] = result["Q"]
            data["perturb_delta"][pi, ri] = result["perturb_delta"]
            data["perturb_present"][pi, ri] = True
    return data


def _collect(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    """Reduce per-surface worker results into the Stage 3 flux-profile HDF5.

    Reads each surface's ``result.json`` (located relative to ``output_dir`` from the manifest run
    entries), stacks the flux arrays, applies the magnetic-axis zero-padding, and writes
    ``dkx_flux_profiles.h5``. Provenance attributes are taken from the manifest rather than live
    arguments so this phase can run independently of the process that prepared the scan.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Collection uses ``output_dir``, ``output``, and ``plot``.
    manifest : dict[str, Any]
        The scan manifest produced by :func:`_prepare`.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / "runs"
    species = [
        SpeciesMeta(name=meta["name"], charge=meta["charge"], mass_mp=meta["mass_mp"])
        for meta in manifest["species_meta"]
    ]
    if not manifest["runs"]:
        raise ValueError("The DKX manifest has no surfaces to collect.")

    rho_out = []
    rhat_out = []
    gamma_out = []
    q_out = []
    upar_out = []
    raw_meta = {"Gamma_key": None, "Q_key": None, "Upar_key": None}
    benchmark_rows: list[dict[str, Any]] = []
    responses = {}
    baseline_runs = []
    for run in sorted(manifest["runs"], key=lambda r: r["rho"]):
        result_json = run_dir / run["run_subdir"] / run["result_json"]
        if not result_json.exists():
            raise FileNotFoundError(
                f"Missing dkx worker result for run {run['run_subdir']} at {result_json}. "
                "Run the surface before collecting."
            )
        summary = json.loads(result_json.read_text(encoding="utf-8"))
        _validate_summary(summary, {
            **run,
            "solver": manifest["solver"],
            "wout_sha256": manifest["wout_sha256"],
            "n_species": len(species),
        })
        if run["response_label"] != "base":
            identity = (run["radius_index"], run["response_label"], run["perturb_species"])
            responses[identity] = {**summary, "perturb_delta": run["perturb_delta"]}
            continue
        baseline_runs.append(run)
        rho_out.append(float(summary["rho"]))
        rhat_value = summary.get("rHat")
        if rhat_value is None:
            raise KeyError(f"Worker summary {result_json} is missing rHat.")
        rhat_out.append(float(rhat_value))
        gamma_out.append(np.asarray(summary["Gamma"], dtype=np.float64))
        q_out.append(np.asarray(summary["Q"], dtype=np.float64))
        upar_out.append(np.asarray(summary["Upar"], dtype=np.float64))
        raw_meta = dict(summary.get("meta", raw_meta))
        if "benchmark" in summary:
            benchmark_rows.append(
                {
                    "rho": float(summary["rho"]),
                    "radius_index": int(summary["radius_index"]),
                    **dict(summary["benchmark"]),
                }
            )

    rho_arr = np.asarray(rho_out, dtype=np.float64)
    rhat_arr = np.asarray(rhat_out, dtype=np.float64)
    gamma_arr = np.stack(gamma_out, axis=1)
    q_arr = np.stack(q_out, axis=1)
    upar_arr = np.stack(upar_out, axis=1)
    axis_padded = False
    # NEOPAX evaluates the axis at exactly zero. A positive radius does not cover it.
    if rho_arr[0] != 0:
        zero_flux = np.zeros((len(species), 1), dtype=np.float64)
        rho_arr = np.concatenate([np.asarray([0.0], dtype=np.float64), rho_arr])
        rhat_arr = np.concatenate([np.asarray([0.0], dtype=np.float64), rhat_arr])
        gamma_arr = np.concatenate([zero_flux, gamma_arr], axis=1)
        q_arr = np.concatenate([zero_flux, q_arr], axis=1)
        upar_arr = np.concatenate([zero_flux, upar_arr], axis=1)
        axis_padded = True

    if rho_arr.size < 2 or np.any(np.diff(rho_arr) <= 0) or np.any(np.diff(rhat_arr) <= 0):
        raise ValueError("DKX collection needs at least two strictly increasing radii.")

    response_data = _collect_responses(manifest, responses, baseline_runs, len(species), axis_padded)
    time_value = manifest["time_value"]
    include_phi1 = manifest["include_phi1"]
    out_h5 = Path(args.output).resolve() if args.output else output_dir / "dkx_flux_profiles.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("r", data=rhat_arr)
        f.create_dataset("rHat", data=rhat_arr)
        f.create_dataset("rho", data=rho_arr)
        f.create_dataset("Gamma", data=gamma_arr)
        f.create_dataset("Q", data=q_arr)
        f.create_dataset("Upar", data=upar_arr)
        for key, values in response_data.items():
            if key in ("response_label", "perturb_species"):
                f.create_dataset(key, data=values, dtype=h5py.string_dtype("utf-8"))
            else:
                f.create_dataset(key, data=values)
        if response_data:
            f.attrs["response_note"] = (
                "kappa=-d(log profile)/d(rho). Density steps preserve the pressure gradient. "
                "Use (F_perturbed-F_base)/perturb_delta where perturb_present is true."
            )
        f.create_dataset("species_names", data=np.asarray([sp.name.encode("utf-8") for sp in species]))
        f.attrs["profiles_source"] = str(manifest["profiles_source"])
        f.attrs["source_transport_solution"] = (
            "" if manifest["source_transport_solution"] is None else str(manifest["source_transport_solution"])
        )
        f.attrs["source_dkx_template"] = str(manifest["source_dkx_template"])
        f.attrs["time_index"] = int(manifest["time_index"])
        f.attrs["time_value"] = np.nan if time_value is None else float(time_value)
        f.attrs["backend"] = str(manifest["backend"])
        f.attrs["max_parallel"] = int(manifest["max_parallel"])
        for key, value in manifest["solver"].items():
            f.attrs[f"solver_{key}"] = value
        f.attrs["wout_sha256"] = manifest["wout_sha256"] or ""
        f.attrs["include_phi1"] = bool(include_phi1) if include_phi1 is not None else -1
        f.attrs["axis_zero_padded"] = bool(axis_padded)
        for key in (
            "Gamma_key", "Q_key", "Upar_key",
            "Gamma_scale_to_neopax", "Q_scale_to_neopax", "Upar_bridge",
        ):
            value = raw_meta.get(key)
            if value is not None:
                f.attrs[f"raw_{key}"] = str(value)
        f.attrs["Upar_note"] = (
            "Upar is derived from dkx FSABFlow using the NTX fixed-field "
            "parallel-flow bridge factor 2*B0OverBBar/sqrt(pi)."
        )
        f.attrs["normalization_note"] = (
            "Gamma is converted from dkx particleFlux_vm_* using nbar*vbar/Rbar "
            "with nbar=1e20 m^-3, Tbar=1 keV, mbar=mp, Rbar=1 m. "
            "Q is converted from heatFlux_vm_* using (nbar*mbar*vbar^3/Rbar)/e so the written "
            "values match NEOPAX's eV-based physical heat-flux convention. "
            "Upar uses the NTX archive-backed observable bridge rather than a raw FSABFlow alias."
        )
        f.attrs["radius_note"] = (
            "The saved coordinate r/rHat uses dkx's rHat output. "
            "rho is also saved separately from the source NEOPAX transport file."
        )
        if benchmark_rows:
            f.attrs["benchmark_note"] = (
                "Timing benchmark mode was enabled. cold_run_s includes first-run JAX startup/compile effects; "
                "warm_* fields are repeated same-process timings after the configured warmup runs."
            )
            f.create_dataset(
                "benchmark_rho",
                data=np.asarray([row["rho"] for row in benchmark_rows], dtype=np.float64),
            )
            f.create_dataset(
                "benchmark_cold_run_s",
                data=np.asarray([row["cold_run_s"] for row in benchmark_rows], dtype=np.float64),
            )
            f.create_dataset(
                "benchmark_warm_mean_s",
                data=np.asarray([row["warm_mean_s"] for row in benchmark_rows], dtype=np.float64),
            )

    if bool(args.plot):
        _write_summary_plots(
            output_dir=output_dir,
            rho=rho_arr,
            gamma=gamma_arr,
            q=q_arr,
            upar=upar_arr,
            species=species,
        )

    if benchmark_rows:
        print("[dkx-scan] benchmark summary (same-worker repeated solves):", flush=True)
        for row in benchmark_rows:
            print(
                "[dkx-scan] "
                f"rho={float(row['rho']):.4f} cold={float(row['cold_run_s']):.3f}s "
                f"warm_mean={float(row['warm_mean_s']):.3f}s "
                f"warm_min={float(row['warm_min_s']):.3f}s "
                f"repeats={int(row['repeats'])}",
                flush=True,
            )

    print(f"wrote {out_h5}", flush=True)
    return 0


def cmd_main(args: argparse.Namespace) -> int:
    manifest, pending_task_payloads = _prepare(args)

    backend = str(args.backend).lower()
    gpu_ids = [token.strip() for token in str(args.gpu_ids).split(",") if token.strip()]
    if backend == "gpu" and not gpu_ids:
        gpu_ids = ["0"]

    runs = manifest["runs"]
    total_runs = len(runs)
    reused_runs = total_runs - len(pending_task_payloads)
    rho_values = [run["rho"] for run in runs]
    rho_min_val = float(min(rho_values))
    rho_max_val = float(max(rho_values))
    backend_note = f"backend={backend}"
    parallel_note = f"max_parallel={int(args.max_parallel)}"
    if backend == "gpu":
        placement_note = f"gpu_ids={','.join(gpu_ids)}"
    else:
        placement_note = f"cores_per_run={int(args.cores_per_run)}"
    print(
        "[dkx-scan] "
        f"prepared {total_runs} runs over rho in [{rho_min_val:.4f}, {rho_max_val:.4f}] "
        f"({backend_note}, {parallel_note}, {placement_note})"
    , flush=True)
    if reused_runs:
        print(
            "[dkx-scan] "
            f"reusing {reused_runs}/{total_runs} existing completed runs; "
            f"launching {len(pending_task_payloads)} new workers.",
            flush=True,
        )

    if pending_task_payloads:
        _run_tasks_in_parallel(
            task_payloads=pending_task_payloads,
            args=args,
            gpu_ids=gpu_ids,
        )

    return _collect(args, manifest)


def cmd_prepare(args: argparse.Namespace) -> int:
    _prepare(args)
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    payload_path = Path(args.payload).resolve()
    gpu_id = None
    if str(args.backend).lower() == "gpu":
        gpu_tokens = [token.strip() for token in str(args.gpu_ids).split(",") if token.strip()]
        gpu_id = gpu_tokens[0] if gpu_tokens else "0"
    env = _build_worker_env(args, gpu_id=gpu_id)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-payload", str(payload_path)],
        env=env,
    )
    return proc.returncode


def cmd_collect(args: argparse.Namespace) -> int:
    manifest_path = Path(args.output_dir).resolve() / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found at {manifest_path}. Run the prepare phase before collecting."
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return _collect(args, manifest)


def _add_output_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for runs and collected fluxes.")


def _add_output(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--output", default=None,
        help="Aggregate output path. Defaults to dkx_flux_profiles.h5 under --output-dir.",
    )


def _add_io_and_shaping(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--common-config",
        default=str(DEFAULT_COMMON_CONFIG),
        help="Path to the shared common_input TOML.",
    )
    p.add_argument(
        "--profiles-source",
        choices=("transport_h5", "analytical", "prescribed"),
        default="analytical",
        help=(
            "Choose whether profiles come from transport_solution.h5, from the TOML analytical profile "
            "block, or from prescribed profile arrays in the TOML [profiles] block."
        ),
    )
    p.add_argument(
        "--neopax-result",
        default=None,
        help="Optional explicit path to transport_solution.h5. Used only for profiles-source=transport_h5.",
    )
    p.add_argument(
        "--analytical-n-radii",
        type=int,
        default=None,
        help=(
            "Number of analytical cell faces; defaults to [geometry].n_radial + 1 from the NEOPAX config, "
            "the transport face grid"
        ),
    )
    p.add_argument(
        "--dkx-template",
        default=str(DEFAULT_DKX_TEMPLATE),
        help="Template dkx input.namelist.",
    )
    _add_output_dir(p)
    p.add_argument("--time-index", type=int, default=-1, help="Time index in transport_solution.h5. Default: final.")
    p.add_argument("--rho-indices", default=None, help="Comma-separated explicit rho indices.")
    p.add_argument("--rho-min", type=float, default=None, help="Minimum rho to include.")
    p.add_argument("--rho-max", type=float, default=None, help="Maximum rho to include.")
    p.add_argument("--num-radii", type=int, default=None, help="Number of radii to sample inside the rho filter.")
    p.add_argument(
        "--response-mode", choices=("none", "fd_gradients"), default="none",
        help="Use baseline fluxes only (none, default) or add gradient-response siblings (fd_gradients).",
    )
    p.add_argument(
        "--perturb-density-species", default="",
        help="Comma-separated density-response species. Default empty.",
    )
    p.add_argument(
        "--perturb-temperature-species", default="",
        help="Comma-separated temperature-response species. Default empty.",
    )
    p.add_argument(
        "--dkap-density", type=float, default=0.5,
        help="Minimum density-gradient step magnitude. Default 0.5.",
    )
    p.add_argument(
        "--dkap-temperature", type=float, default=0.5,
        help="Minimum temperature-gradient step magnitude. Default 0.5.",
    )
    p.add_argument(
        "--perturb-rel-step", type=float, default=0.5,
        help="Relative gradient-step factor. Default 0.5.",
    )
    p.add_argument("--wout-path", default=None, help="Optional VMEC equilibrium override for dkx.")
    p.add_argument(
        "--boozer-path",
        default=None,
        help="Optional Boozer equilibrium override for the NEOPAX minor radius the analytical faces are built on.",
    )
    p.add_argument("--include-phi1", dest="include_phi1", action="store_true", help="Force includePhi1 = true.")
    p.add_argument("--no-include-phi1", dest="include_phi1", action="store_false", help="Force includePhi1 = false.")
    p.set_defaults(include_phi1=None)
    p.add_argument("--ntheta", type=int, default=5, help="Override Ntheta. Default: 5.")
    p.add_argument("--nzeta", type=int, default=11, help="Override Nzeta. Default: 11.")
    p.add_argument("--nxi", type=int, default=12, help="Override Nxi. Default: 12.")
    p.add_argument("--nl", type=int, default=3, help="Override NL. Default: 3.")
    p.add_argument("--nx", type=int, default=4, help="Override Nx. Default: 4.")
    p.add_argument("--solver-tolerance", type=float, default=1.0e-6, help="Override solverTolerance. Default: 1e-6.")


def _add_backend(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu", help="Parallel execution backend.")


def _add_gpu_ids(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids for backend=gpu.")


def _add_max_parallel(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-parallel", type=int, default=8, help="Maximum concurrent dkx runs.")


def _add_cores_per_run(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cores-per-run", type=int, default=1,
        help="CPU cores per run. Nonpositive values leave DKX's environment or automatic default in control.",
    )


def _add_benchmark(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--benchmark-repeats",
        type=int,
        default=0,
        help="Repeat each selected surface this many extra times inside one worker and report warm timings.",
    )
    p.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        help="Number of same-worker warmup solves to discard before benchmark repeats.",
    )


def _add_plot(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plot", dest="plot", action="store_true", help="Write PNG plots of Gamma, Q, and Upar versus rho.")
    p.add_argument("--no-plot", dest="plot", action="store_false", help="Skip PNG plots.")
    p.set_defaults(plot=True)


def _add_verbose_workers(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--verbose-workers",
        dest="verbose_workers",
        action="store_true",
        help="Allow verbose dkx worker logging.",
    )
    p.add_argument(
        "--no-verbose-workers",
        dest="verbose_workers",
        action="store_false",
        help="Silence dkx worker logging.",
    )
    p.set_defaults(verbose_workers=True)


def _add_worker_payload(p: argparse.ArgumentParser) -> None:
    p.add_argument("--worker-payload", default=None, help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            __doc__
            + "\n\n"
            + "Unless overridden on the command line, this script forces the following "
            + "dkx resolution settings (quickrun smoke test): "
            + "Ntheta=5, Nzeta=11, Nxi=12, NL=3, Nx=4, solverTolerance=1e-6."
        )
    )
    _add_io_and_shaping(p)
    _add_output(p)
    _add_backend(p)
    _add_gpu_ids(p)
    _add_max_parallel(p)
    _add_cores_per_run(p)
    _add_benchmark(p)
    _add_plot(p)
    _add_verbose_workers(p)
    _add_worker_payload(p)
    p.set_defaults(func=cmd_main)

    sub = p.add_subparsers(dest="command", required=False)

    prepare = sub.add_parser("prepare", help="Write per-surface dkx inputs and the scan manifest.")
    _add_io_and_shaping(prepare)
    _add_backend(prepare)
    _add_max_parallel(prepare)
    _add_benchmark(prepare)
    _add_verbose_workers(prepare)
    prepare.set_defaults(func=cmd_prepare)

    run_one = sub.add_parser("run-one", help=argparse.SUPPRESS)
    run_one.add_argument("--payload", required=True, help="Path to a per-surface payload.json to execute.")
    _add_backend(run_one)
    _add_gpu_ids(run_one)
    _add_cores_per_run(run_one)
    run_one.set_defaults(func=cmd_run_one)

    collect = sub.add_parser("collect", help="Reduce per-surface results into the flux-profile HDF5.")
    _add_output_dir(collect)
    _add_output(collect)
    _add_plot(collect)
    collect.set_defaults(func=cmd_collect)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_payload:
        return _run_single_worker_from_payload(Path(args.worker_payload).resolve())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
