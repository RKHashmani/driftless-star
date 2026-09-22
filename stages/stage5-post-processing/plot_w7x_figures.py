"""Draw the five W7-X poster figures from the five committed W7-X runs.

``ion_temperature_comparison.png``
    Final ion temperature of every run at its latest completed iteration, over the initial
    profiles the runs started from.
``iteration_ion_temperature.png``, ``iteration_collisional_exchange.png``
    The benchmark run's ion temperature and electron-to-ion collisional exchange power density at
    the end of every iteration's time window, opening on the profile the loop started from as
    iteration 0.
``ion_heat_flux_power.png``, ``ion_heat_flux_power_log.png``
    Every benchmark iteration's Stage 4 ion heat flux as power crossing each flux surface, on a
    linear and on a logarithmic axis. Stage 4 runs ahead of Stage 5 within an iteration, so
    iteration 1's flux is already the flux of the initial profile and iteration 0 reuses it.

Each run lives at ``<outputs-root>/<case>/loop/iter_N/output/``. ``--t3d-nc`` adds the T3D + GX
reference curves to every figure; without it the figures carry the runs alone.

NEOPAX evaluates the collisional exchange from the source models each iteration's saved
common_input_updated.toml selects, so its installed Git revision must match ``--neopax-rev`` or the
current stages/pixi.toml pin.

Run from the repository root, inside the stage-5 container (it has matplotlib and NEOPAX)::

    python stages/stage5-post-processing/plot_w7x_figures.py \\
        --t3d-nc outputs/issue89_all_runs_results/t3d_reference/run_gx/tests/regression/test-w7x-gx.nc
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import logging
import re
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np

from fit_vmec_pressure_from_transport_h5 import _saved_time_count

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_ROOT = Path("outputs/w7-x")
DEFAULT_OUTDIR = DEFAULT_OUTPUTS_ROOT / "plots"

SPECIES_ORDER = ("e", "ion")
"""Species order of the ``(n_time, n_species, n_rho)`` arrays in ``transport_solution.h5``."""

ION = SPECIES_ORDER.index("ion")

LOOP_STATUSES = ("continue", "converged", "horizon", "halted")
"""Statuses Stage 5 post-processing writes into ``converge_status.json``."""

ELEMENTARY_CHARGE = 1.602176634e-19
"""J per eV. Stage 4 stores the heat flux in eV m^-2 s^-1."""

STATE_POWER_TO_MW_PER_M3 = 1e20 * 1e3 * ELEMENTARY_CHARGE / 1e6
"""NEOPAX pressure-source unit to MW/m^3."""

EXCHANGE_COMPONENTS = frozenset({"power_exchange", "power_exchange_temperature_equilibration"})
"""Pressure-source components that carry the electron-to-ion collisional exchange."""

CASES: list[dict[str, Any]] = [
    {"directory": "t3d_benchmark", "label": "Benchmark", "color": "#555555", "marker": "D", "linestyle": "--"},
    {"directory": "mhd_off_neoclassical_off", "label": "MHD Off, Neoclassical Off", "color": "#E68613",
     "marker": "^"},
    {"directory": "mhd_on_neoclassical_off", "label": "MHD On, Neoclassical Off", "color": "#2C9B40",
     "marker": "v", "linestyle": "--"},
    {"directory": "mhd_off_neoclassical_on", "label": "MHD Off, Neoclassical On", "color": "#A00060",
     "marker": "o"},
    {"directory": "mhd_on_neoclassical_on", "label": "MHD On, Neoclassical On", "color": "#00B8D9",
     "marker": "P", "linestyle": (0, (3, 7)), "markerfacecolor": "none", "linewidth": 2.4, "markersize": 7.0},
]
"""The five runs, benchmark first. The other four share one initial profile and are the matched runs."""

BENCHMARK_LABEL = "Driftless Star Benchmark"

POSTER_RC = {
    "font.size": 17.0,
    "axes.titlesize": 18.0,
    "axes.labelsize": 18.0,
    "xtick.labelsize": 17.0,
    "ytick.labelsize": 17.0,
    "legend.fontsize": 16.0,
    "lines.linewidth": 2.2,
    "lines.markersize": 5.5,
}
"""Type and line weights for a printed poster, read from a couple of metres."""

COMPARISON_RC = {**POSTER_RC, "axes.titlesize": 21.0, "axes.labelsize": 20.0}
"""Comparison-figure type, one step up from the iteration figures."""

RHO_LABEL = r"$\rho$ (normalized $\sqrt{\mathrm{toroidal\ flux}}$)"
T3D_COLOR = "tab:blue"
SHARED_INITIAL_COLOR = "#969696"
COMPARISON_FIGURE_SIZE_IN = (13.0, 6.0)
ITERATION_FIGURE_SIZE_IN = (5.5, 5.0)
INITIAL_LINEWIDTH = 2.6
NONFINAL_ALPHA = 0.3
FINAL_LINEWIDTH = 2.6
TITLE_SIZE = 19.0


class TransportState(NamedTuple):
    """The written prefix of a NEOPAX ``transport_solution.h5``.

    Attributes
    ----------
    rho : numpy.ndarray
        Radial coordinate, shape ``(n_rho,)``.
    ts : numpy.ndarray
        Time in seconds, shape ``(n_time,)``, strictly increasing.
    density, temperature, pressure : numpy.ndarray
        Shape ``(n_time, n_species, n_rho)`` in 1e20 m^-3, keV and 1e20 m^-3 keV.
    er : numpy.ndarray
        Radial electric field, shape ``(n_time, n_rho)``.
    """

    rho: np.ndarray
    ts: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    er: np.ndarray


class T3DReference(NamedTuple):
    """The accepted time slices of a T3D run, restricted to the quantities the figures draw.

    Attributes
    ----------
    rho : numpy.ndarray
        T3D's radial grid, shape ``(n_rho,)``.
    ts : numpy.ndarray
        Accepted times in seconds, shape ``(n_accepted,)``.
    ti : numpy.ndarray
        Bulk ion temperature in keV, shape ``(n_accepted, n_rho)``.
    exchange_mw : numpy.ndarray
        Collisional exchange power density in MW/m^3, same shape.
    rho_mid : numpy.ndarray
        Staggered midpoint grid the flux is written on, shape ``(n_mid,)``.
    power_mw : numpy.ndarray
        Ion heat flux as power through each midpoint surface in MW at the last accepted record.
    """

    rho: np.ndarray
    ts: np.ndarray
    ti: np.ndarray
    exchange_mw: np.ndarray
    rho_mid: np.ndarray
    power_mw: np.ndarray


class Endpoints(NamedTuple):
    """A run's first and last ion temperature profile.

    Attributes
    ----------
    rho : numpy.ndarray
        Transport grid, shape ``(n_rho,)``.
    initial, final : numpy.ndarray
        Ion temperature in keV at ``t = 0`` and at the end of the latest completed iteration.
    iteration : int
        The latest completed iteration.
    time : float
        Transport time in seconds at the end of that iteration.
    """

    rho: np.ndarray
    initial: np.ndarray
    final: np.ndarray
    iteration: int
    time: float


class Benchmark(NamedTuple):
    """The benchmark run's full iteration history.

    Attributes
    ----------
    rho : numpy.ndarray
        Transport grid, shape ``(n_rho,)``.
    iterations : tuple of int
        Iteration numbers, ascending.
    ti_initial, exchange_initial : numpy.ndarray
        The profiles the loop started from, shape ``(n_rho,)``, in keV and MW/m^3.
    ti, exchange : numpy.ndarray
        The same quantities at the end of each iteration's time window, shape
        ``(n_iterations, n_rho)``.
    rho_flux : numpy.ndarray
        Stage 4 flux grid, shape ``(n_rho_flux,)``.
    power_mw : numpy.ndarray
        Each iteration's ion heat flux as power through each flux surface in MW, shape
        ``(n_iterations, n_rho_flux)``.
    """

    rho: np.ndarray
    iterations: tuple[int, ...]
    ti_initial: np.ndarray
    ti: np.ndarray
    exchange_initial: np.ndarray
    exchange: np.ndarray
    rho_flux: np.ndarray
    power_mw: np.ndarray


def read_transport_state(h5_path: Path) -> TransportState:
    """Read a ``transport_solution.h5``, keeping only the time slots the solve actually wrote.

    Parameters
    ----------
    h5_path : Path
        A driftless-star ``transport_solution.h5``.

    Returns
    -------
    TransportState
        Profiles truncated to the strictly increasing prefix of ``ts``.

    Raises
    ------
    KeyError
        If a required dataset is missing.
    ValueError
        If a profile is not 3-D over ``(time, species, rho)`` consistent with :data:`SPECIES_ORDER`
        and the grid it is written on, or if ``Er`` is not ``(time, rho)``.
    """
    required = ("rho", "ts", "density", "temperature", "pressure", "Er")
    with h5py.File(h5_path, "r") as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise KeyError(f"{h5_path}: missing required dataset(s) {missing}")
        arrays = {name: np.asarray(f[name][()], dtype=float) for name in required}

    rho, ts = arrays["rho"], arrays["ts"]
    n_slots = int(ts.size)
    expected = (n_slots, len(SPECIES_ORDER), rho.size)
    for name in ("density", "temperature", "pressure"):
        if arrays[name].shape != expected:
            raise ValueError(
                f"{h5_path}: '{name}' has shape {arrays[name].shape}, expected {expected} from ts, "
                f"the species order {SPECIES_ORDER} and the grid it is written on"
            )
    if arrays["Er"].shape != (n_slots, rho.size):
        raise ValueError(f"{h5_path}: 'Er' has shape {arrays['Er'].shape}, expected {(n_slots, rho.size)}")

    n_written = _saved_time_count(ts)
    if n_written < n_slots:
        logger.warning(
            "%s: only %d of %d time slots were written; the trailing slots hold preallocated values "
            "and are dropped. The solve did not run to completion.", h5_path, n_written, n_slots,
        )
    return TransportState(
        rho=rho,
        ts=ts[:n_written],
        density=arrays["density"][:n_written],
        temperature=arrays["temperature"][:n_written],
        pressure=arrays["pressure"][:n_written],
        er=arrays["Er"][:n_written],
    )


def read_ion_heat_flux(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the Stage 4 turbulent ion heat flux from a ``neopax_fluxes.h5``.

    Parameters
    ----------
    h5_path : Path
        A Stage 4 ``neopax_fluxes.h5``.

    Returns
    -------
    tuple of numpy.ndarray
        The radial grid and the ion row of ``Q``, in the file's own eV m^-2 s^-1, both shape
        ``(n_rho,)``.

    Raises
    ------
    KeyError
        If a required dataset is missing.
    ValueError
        If ``Q`` is not 2-D over ``(species, rho)``, or if the species names hold no ion or do not
        match its rows.
    """
    with h5py.File(h5_path, "r") as f:
        missing = [name for name in ("rho", "Q") if name not in f]
        if missing:
            raise KeyError(f"{h5_path}: missing required dataset(s) {missing}")
        rho = np.asarray(f["rho"][()], dtype=float)
        q = np.asarray(f["Q"][()], dtype=float)
        meta = f.get("meta")
        names = meta["species_names"][()] if meta is not None and "species_names" in meta else None

    if q.ndim != 2 or q.shape[1] != rho.size:
        raise ValueError(f"{h5_path}: 'Q' must be (species, {rho.size}), got shape {q.shape}")
    species = SPECIES_ORDER if names is None else tuple(
        name.decode() if isinstance(name, bytes) else str(name) for name in names)
    if "ion" not in species:
        raise ValueError(f"{h5_path}: species names {species} hold no 'ion' row")
    if len(species) != q.shape[0]:
        raise ValueError(f"{h5_path}: species names {species} do not match the {q.shape[0]} rows of 'Q'")
    return rho, q[species.index("ion")]


def flux_surface_areas(wout_path: Path, rho: np.ndarray, n_theta: int = 192, n_zeta: int = 192) -> np.ndarray:
    """Surface area of each VMEC flux surface, in m^2.

    Integrates ``|e_theta x e_zeta|`` over the surface, where the cross product of the two
    covariant tangent vectors has magnitude ``sqrt(W^2 + R^2 (R_theta^2 + Z_theta^2))`` with
    ``W = R_theta Z_zeta - Z_theta R_zeta``. VMEC's radial label is the normalized toroidal
    flux ``s``, and the benchmark's ``rho`` is ``sqrt(s)``, so surfaces are taken at ``s = rho^2``.

    Parameters
    ----------
    wout_path : Path
        A VMEC ``wout_*.nc`` equilibrium.
    rho : numpy.ndarray
        Radii to evaluate, as sqrt(normalized toroidal flux).
    n_theta, n_zeta : int
        Poloidal and toroidal sample counts for the surface integral.

    Returns
    -------
    numpy.ndarray
        Areas in m^2, same shape as ``rho``.
    """
    from netCDF4 import Dataset

    with Dataset(wout_path, "r") as ds:
        n_s = int(ds["ns"][...])
        xm, xn = np.asarray(ds["xm"][:]), np.asarray(ds["xn"][:])
        rmnc, zmns = np.asarray(ds["rmnc"][:]), np.asarray(ds["zmns"][:])

    s_full = np.linspace(0.0, 1.0, n_s)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
    grid_t, grid_z = np.meshgrid(theta, zeta, indexing="ij")

    out = np.zeros(np.shape(rho), dtype=float)
    for i, r in enumerate(np.atleast_1d(rho)):
        s = float(r) ** 2
        R = np.zeros_like(grid_t)
        R_t = np.zeros_like(grid_t)
        Z_t = np.zeros_like(grid_t)
        R_z = np.zeros_like(grid_t)
        Z_z = np.zeros_like(grid_t)
        for k in range(xm.size):
            rc = np.interp(s, s_full, rmnc[:, k])
            zs = np.interp(s, s_full, zmns[:, k])
            angle = xm[k] * grid_t - xn[k] * grid_z
            R += rc * np.cos(angle)
            R_t += -rc * xm[k] * np.sin(angle)
            Z_t += zs * xm[k] * np.cos(angle)
            R_z += rc * xn[k] * np.sin(angle)
            Z_z += -zs * xn[k] * np.cos(angle)
        W = R_t * Z_z - Z_t * R_z
        out[i] = np.sqrt(W**2 + R**2 * (R_t**2 + Z_t**2)).mean() * (2.0 * np.pi) ** 2
    return out


def accepted_record_indices(t: np.ndarray) -> np.ndarray:
    """Select the accepted time slices of a T3D run, dropping intermediate Newton iterates.

    Keeps the first record at each distinct time. T3D labels the Newton iterates of the step
    leaving ``t_k`` with time ``t_k`` itself, so they always follow the accepted record for ``t_k``.

    Parameters
    ----------
    t : numpy.ndarray
        The ``/time/t`` axis, shape ``(n_steps,)``. Non-decreasing.

    Returns
    -------
    numpy.ndarray
        Integer indices of the accepted records, in order.

    Raises
    ------
    ValueError
        If ``t`` is not a 1-D non-empty axis.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError(f"T3D time axis must be 1-D and non-empty, got shape {t.shape}")
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = t[1:] != t[:-1]
    return np.flatnonzero(keep)


def read_t3d(nc_path: Path) -> T3DReference:
    """Read the accepted time slices of a T3D run for the reference curves.

    ``Sp_coll_<tag>`` is converted to MW/m^3 with ``norms/P_ref_MWm3``. The flux sits on the
    staggered midpoints between grid points, hence its own radial grid.

    Parameters
    ----------
    nc_path : Path
        A T3D NetCDF output.

    Returns
    -------
    T3DReference
        Ion temperature and collisional exchange at the accepted times, and the final ion heat
        flux power.
    """
    from netCDF4 import Dataset

    with Dataset(nc_path, "r") as ds:
        tag = str(ds["species/bulk_ion_tag"][...])
        rho = np.asarray(ds["grid/rho"][:], dtype=float)
        t_ref = float(ds["norms/t_ref"][...])
        p_ref = float(ds["norms/P_ref_MWm3"][...])
        t_norm = np.asarray(ds["time/t"][:], dtype=float)
        accepted = accepted_record_indices(t_norm)
        ti = np.asarray(ds[f"species/T_{tag}"][:], dtype=float)[accepted]
        exchange = np.asarray(ds[f"species/Sp_coll_{tag}"][:], dtype=float)[accepted] * p_ref
        rho_mid = np.asarray(ds["grid/midpoints"][:], dtype=float)
        power_mw = np.asarray(ds[f"species/Q_MW_{tag}"][int(accepted[-1])], dtype=float)

    logger.info("%s: %d records -> %d accepted time slices", nc_path, t_norm.size, accepted.size)
    return T3DReference(rho=rho, ts=t_norm[accepted] * t_ref, ti=ti, exchange_mw=exchange,
                        rho_mid=rho_mid, power_mw=power_mw)


def completed_iterations(run_dir: Path) -> tuple[int, ...]:
    """Find every loop iteration of a run that finished with all three outputs the figures need.

    Parameters
    ----------
    run_dir : Path
        A run directory, i.e. the parent of ``loop/iter_N/``.

    Returns
    -------
    tuple of int
        Iteration numbers, ascending. Each has a ``converge_status.json`` naming one of
        :data:`LOOP_STATUSES`, a ``transport_solution.h5`` and a ``neopax_fluxes.h5``.

    Raises
    ------
    FileNotFoundError
        If no iteration under ``run_dir`` qualifies.
    """
    numbers = []
    for directory in run_dir.glob("loop/iter_*"):
        match = re.fullmatch(r"iter_(\d+)", directory.name)
        if match is not None and directory.is_dir():
            numbers.append(int(match[1]))

    found = []
    for number in sorted(numbers):
        output = run_dir / f"loop/iter_{number}/output"
        signal = output / "stage5_post_processing/converge_status.json"
        absent = [path.name for path in (output / "stage5_transport/transport_solution.h5",
                                         output / "stage4_turbulence/neopax_fluxes.h5") if not path.is_file()]
        if absent:
            logger.warning("skipping iteration %d of %s: no %s", number, run_dir, ", ".join(absent))
            continue
        try:
            status = json.loads(signal.read_text()).get("status")
        except FileNotFoundError:
            logger.warning("skipping iteration %d of %s: no %s", number, run_dir, signal.name)
            continue
        except json.JSONDecodeError:
            logger.warning("skipping iteration %d of %s: %s does not parse as JSON", number, run_dir, signal.name)
            continue
        if status not in LOOP_STATUSES:
            logger.warning("skipping iteration %d of %s: status %r is not one of %s", number, run_dir,
                           status, ", ".join(LOOP_STATUSES))
            continue
        found.append(number)
    if not found:
        raise FileNotFoundError(
            f"{run_dir}: no iteration carries a completion status, a transport solution and a Stage 4 flux")
    logger.info("found %d completed iteration(s) under %s", len(found), run_dir)
    return tuple(found)


def frozen_equilibrium(run_dir: Path) -> Path:
    """Locate the benchmark's frozen iteration-1 VMEC equilibrium.

    The benchmark does not rerun Stage 1, so all fluxes use this geometry.

    Parameters
    ----------
    run_dir : Path
        A run directory.

    Returns
    -------
    Path
        The ``wout_*.nc`` from iteration 1.

    Raises
    ------
    FileNotFoundError
        If iteration 1 holds no VMEC equilibrium.
    """
    wout = sorted(run_dir.glob("loop/iter_1/output/stage1_equilibrium/wout_*.nc"))
    if not wout:
        raise FileNotFoundError(f"{run_dir}: no loop/iter_1/output/stage1_equilibrium/wout_*.nc")
    logger.info("flux-surface geometry from %s", wout[0])
    return wout[0]


def _require_neopax_revision(expected: str | None) -> None:
    """Require a Git install matching the full SHA override or the current ``stages/pixi.toml`` pin."""
    if expected is None:
        manifest = Path(__file__).resolve().parents[1] / "pixi.toml"
        expected = tomllib.loads(manifest.read_text())["feature"]["neopax"]["pypi-dependencies"]["neopax"]["rev"]
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("The expected NEOPAX revision must be a full 40-character Git commit.")
    try:
        dist = metadata.distribution("neopax")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Evaluating the collisional exchange requires NEOPAX at {expected}. "
            "Install the pinned NEOPAX environment."
        ) from exc
    actual = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
    if actual != expected:
        raise RuntimeError(
            f"NEOPAX revision mismatch. Expected {expected}, installed {actual or 'unverified build'}. "
            "Install the run's commit and use --neopax-rev for a historical run."
        )
    logger.info("source evaluation NEOPAX commit %s, package %s", actual, dist.version)


def exchange_power(state: TransportState, transport_h5: Path) -> np.ndarray:
    """Evaluate the electron-to-ion collisional exchange at every saved state, through NEOPAX.

    A configuration with no temperature source gives zero rather than an invented source.

    Parameters
    ----------
    state : TransportState
        A transport solution already truncated to its written time slots.
    transport_h5 : Path
        The solution the state came from. Its ``common_input_updated.toml`` sits beside it.

    Returns
    -------
    numpy.ndarray
        Power gained by the ions in MW/m^3 on the saved cell centers, shape ``(n_time, n_rho)``.

    Raises
    ------
    FileNotFoundError
        If no source configuration sits beside ``transport_h5``.
    ValueError
        If the saved species, masses or charges disagree with :data:`SPECIES_ORDER` and the state.
    RuntimeError
        If the installed NEOPAX source API is unavailable or incompatible.
    """
    config_path = transport_h5.with_name("common_input_updated.toml")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing saved source configuration {config_path}. Source models cannot be inferred "
            "from the HDF5 fields."
        )
    cfg = tomllib.loads(config_path.read_text())
    species_cfg = cfg["species"]
    names = tuple(species_cfg["names"])
    if names != SPECIES_ORDER:
        raise ValueError(f"Saved config species {names} do not match the reader order {SPECIES_ORDER}.")
    mass = np.asarray(species_cfg["mass_mp"], dtype=float)
    charge = np.asarray(species_cfg["charge_qp"], dtype=float)
    if mass.shape != (len(names),) or charge.shape != (len(names),):
        raise ValueError("Saved species masses and charges must match the species names.")
    try:
        import jax.numpy as jnp
        from NEOPAX import Species, build_source_models_from_config
        from NEOPAX import TransportState as NeopaxState
        from NEOPAX._source_models import assemble_pressure_source_components
    except ImportError as exc:
        raise RuntimeError("The installed NEOPAX source API is unavailable or incompatible.") from exc

    species = Species(number_species=len(names), species_indices=jnp.arange(len(names)),
                      mass_mp=jnp.asarray(mass), charge_qp=jnp.asarray(charge), names=names)
    temperature_source = (build_source_models_from_config(cfg, species=species) or {}).get("temperature")
    logger.info("source config %s, temperature sources %s", config_path,
                cfg.get("sources", {}).get("temperature", []))
    if temperature_source is None:
        return np.zeros((state.ts.size, state.rho.size))

    series = []
    for density, pressure, er in zip(state.density, state.pressure, state.er, strict=True):
        saved = NeopaxState(density=jnp.asarray(density), pressure=jnp.asarray(pressure), Er=jnp.asarray(er))
        components = assemble_pressure_source_components(temperature_source(saved), saved, species)
        total = sum((np.asarray(value, dtype=float) for name, value in components.items()
                     if name in EXCHANGE_COMPONENTS), start=np.zeros_like(density))
        if total.shape != density.shape or not np.isfinite(total).all():
            raise ValueError("NEOPAX returned an exchange of the wrong shape or with non-finite values.")
        series.append(total[ION] * STATE_POWER_TO_MW_PER_M3)
    return np.asarray(series)


def read_endpoints(run_dir: Path) -> Endpoints:
    """Read a run's starting ion temperature and the one ending its latest completed iteration.

    Parameters
    ----------
    run_dir : Path
        A run directory.

    Returns
    -------
    Endpoints
        Both profiles on the run's transport grid.

    Raises
    ------
    ValueError
        If iteration 1 did not complete, if the first iteration does not start at ``t = 0``, if the
        two iterations use different radial grids, or if either profile is not finite.
    """
    iterations = completed_iterations(run_dir)
    if 1 not in iterations:
        raise ValueError(f"{run_dir}: iteration 1 did not complete, so the run has no starting profile")
    latest = iterations[-1]
    first = read_transport_state(run_dir / "loop/iter_1/output/stage5_transport/transport_solution.h5")
    last = read_transport_state(run_dir / f"loop/iter_{latest}/output/stage5_transport/transport_solution.h5")
    if not np.isclose(first.ts[0], 0.0, atol=1e-14, rtol=0.0):
        raise ValueError(f"{run_dir}: iteration 1 starts at t = {first.ts[0]:g} s, not at zero")
    if not np.array_equal(first.rho, last.rho):
        raise ValueError(f"{run_dir}: iteration {latest} uses a different transport grid than iteration 1")
    initial, final = first.temperature[0, ION], last.temperature[-1, ION]
    if not (np.isfinite(initial).all() and np.isfinite(final).all()):
        raise ValueError(f"{run_dir}: the saved ion temperature holds non-finite values")
    logger.info("%s: iteration %d ends at t = %.8g s, core T_i %.6f keV",
                run_dir.name, latest, last.ts[-1], final[0])
    return Endpoints(rho=first.rho, initial=initial, final=final, iteration=latest, time=float(last.ts[-1]))


def load_benchmark(run_dir: Path, expected_rev: str | None = None) -> Benchmark:
    """Read every iteration of the benchmark run and reduce it to the profiles the figures draw.

    Parameters
    ----------
    run_dir : Path
        The benchmark run directory.
    expected_rev : str or None
        Full NEOPAX commit the run used. Defaults to the ``stages/pixi.toml`` pin.

    Returns
    -------
    Benchmark
        Per-iteration ion temperature, collisional exchange and heat flux power.

    Raises
    ------
    ValueError
        If iteration 1 did not complete or does not start at ``t = 0``, or if the iterations do not
        share one transport grid or one flux grid, which would make a single radial axis wrong for
        some of them.
    RuntimeError
        If the installed NEOPAX does not match the requested SHA or the repository pin.
    """
    iterations = completed_iterations(run_dir)
    if iterations[0] != 1:
        raise ValueError(f"{run_dir}: iteration 1 did not complete, so the run has no starting profile")
    _require_neopax_revision(expected_rev)
    ti, exchange, q_ion = [], [], []
    rho, rho_flux, initial = None, None, None
    for number in iterations:
        output = run_dir / f"loop/iter_{number}/output"
        solution = output / "stage5_transport/transport_solution.h5"
        state = read_transport_state(solution)
        if rho is None:
            rho = state.rho
        elif not np.array_equal(rho, state.rho):
            raise ValueError(f"iteration {number} uses a different transport grid than iteration 1")
        power = exchange_power(state, solution)
        # Each pass starts from the previous one's answer, so only iteration 1's first slot holds a
        # profile no transport solve has touched.
        if initial is None:
            if not np.isclose(state.ts[0], 0.0, atol=1e-14, rtol=0.0):
                raise ValueError(f"{run_dir}: iteration 1 starts at t = {state.ts[0]:g} s, not at zero")
            initial = (state.temperature[0, ION], power[0])
        ti.append(state.temperature[-1, ION])
        exchange.append(power[-1])
        logger.info("iteration %d: %d saved slots, core T_i %.4f keV", number, state.ts.size, ti[-1][0])

        grid, q = read_ion_heat_flux(output / "stage4_turbulence/neopax_fluxes.h5")
        if rho_flux is None:
            rho_flux = grid
        elif not np.array_equal(rho_flux, grid):
            raise ValueError(f"iteration {number} uses a different flux grid than iteration 1")
        q_ion.append(q)

    areas = flux_surface_areas(frozen_equilibrium(run_dir), rho_flux)
    return Benchmark(
        rho=rho, iterations=iterations, ti_initial=initial[0], ti=np.asarray(ti),
        exchange_initial=initial[1], exchange=np.asarray(exchange), rho_flux=rho_flux,
        power_mw=np.asarray(q_ion) * ELEMENTARY_CHARGE * areas / 1.0e6,
    )


def _new_figure(size: tuple[float, float]) -> tuple[Any, Any]:
    """Return an Agg figure and axis with constrained layout and size in inches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=size, layout="constrained")


def _draw_iterations(ax: Any, rho: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Draw one curve per iteration, fading all but the last.

    ``values`` has shape ``(n_iterations, n_rho)``. Return RGBA colours of shape
    ``(n_iterations, 4)``, from amber to red, avoiding pale yellow that prints faint.
    """
    import matplotlib.pyplot as plt

    colours = plt.get_cmap("autumn")(np.linspace(0.62, 0.0, len(values)))
    for colour, profile in zip(colours[:-1], values[:-1]):
        ax.plot(rho, profile, "o-", color=colour, ms=4.0, lw=1.6, alpha=NONFINAL_ALPHA, zorder=2)
    ax.plot(rho, values[-1], "o-", color=colours[-1], ms=4.0, lw=FINAL_LINEWIDTH, zorder=3.5)
    return colours


def _end_label(ax: Any, text: str, colour: Any, on_top: bool) -> None:
    """Write a curve label in the reserved band at the top or bottom of the axes."""
    ax.text(0.025, 0.965 if on_top else 0.035, text, transform=ax.transAxes, color=colour,
            fontsize=12.5, fontweight="bold", va="top" if on_top else "bottom", zorder=4)


def _legend_below(fig: Any, colours: np.ndarray, iterations: tuple[int, ...], reference_handle: Any) -> None:
    """Put the legend below the axes, opening with one swatch per iteration.

    ``HandlerTuple`` splits the swatch tuple into one sub-box per patch, which is what turns it
    into a colour bar rather than a stack of overlapping patches.
    """
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Rectangle

    swatches = tuple(
        Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=colour, alpha=1.0 if i == len(colours) - 1 else NONFINAL_ALPHA)
        for i, colour in enumerate(colours)
    )
    handles: list[Any] = [swatches]
    labels = [f"{BENCHMARK_LABEL}\niterations {iterations[0]}-{iterations[-1]}"]
    if reference_handle is not None:
        handles.append(reference_handle)
        labels.append("T3D + GX")
    fig.legend(handles, labels, loc="outside lower center", ncol=1, frameon=False,
               handlelength=3.0, columnspacing=1.4,
               handler_map={tuple: HandlerTuple(ndivide=len(swatches), pad=0.12)})


def _save(fig: Any, out_path: Path, dpi: int) -> None:
    """Write and close a figure, creating its directory, and log where it landed."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_ion_temperature_comparison(endpoints: list[Endpoints], reference: T3DReference | None,
                                    out_path: Path, *, dpi: int) -> None:
    """Draw every run's final ion temperature over the profiles the runs started from.

    The four matched runs share one initial profile, drawn once. With a reference, the benchmark
    and T3D initial profiles are drawn as one curve through the sorted union of their grids,
    because both sample the same analytic profile on different grids.

    Parameters
    ----------
    endpoints : list of Endpoints
        One per entry of :data:`CASES`, in that order.
    reference : T3DReference or None
        The T3D + GX reference, or None to draw the runs alone.
    out_path : Path
        Figure destination.
    dpi : int
        Figure resolution.

    Raises
    ------
    ValueError
        If the four matched runs do not share one initial profile on one grid.
    """
    import matplotlib.pyplot as plt

    benchmark, matched = endpoints[0], endpoints[1:]
    for other in matched[1:]:
        if not (np.array_equal(other.rho, matched[0].rho) and np.array_equal(other.initial, matched[0].initial)):
            raise ValueError("the matched runs do not share one initial ion temperature profile and grid")

    with plt.rc_context(COMPARISON_RC):
        fig, ax = _new_figure(COMPARISON_FIGURE_SIZE_IN)
        initial_handles, latest_handles = [], []
        if reference is None:
            initial_handles.append(ax.plot(
                benchmark.rho, benchmark.initial, color=CASES[0]["color"], linestyle=":",
                linewidth=INITIAL_LINEWIDTH + 0.3, label=f"{CASES[0]['label']}, iteration 0")[0])
        else:
            merged_rho = np.concatenate((benchmark.rho, reference.rho))
            merged_ti = np.concatenate((benchmark.initial, reference.ti[0]))
            order = np.argsort(merged_rho)
            initial_handles.append(ax.plot(
                merged_rho[order], merged_ti[order], color=T3D_COLOR, linestyle="--",
                linewidth=INITIAL_LINEWIDTH, label="T3D + GX and\nBenchmark\ninitial (iteration 0)")[0])
        initial_handles.append(ax.plot(
            matched[0].rho, matched[0].initial, color=SHARED_INITIAL_COLOR, linestyle="--",
            linewidth=INITIAL_LINEWIDTH, label="All matched runs\niteration 0")[0])
        if reference is not None:
            latest_handles.append(ax.plot(
                reference.rho, reference.ti[-1], color=T3D_COLOR, marker="s", markersize=6.5,
                linewidth=2.8, label=f"T3D + GX, final\nt = {reference.ts[-1]:.5f} s")[0])
        for case, run in zip(CASES, endpoints):
            display = case["label"].replace(", Neoclassical", ",\nNeoclassical")
            latest_handles.append(ax.plot(
                run.rho, run.final, color=case["color"], linestyle=case.get("linestyle", "-"),
                marker=case["marker"], markerfacecolor=case.get("markerfacecolor", case["color"]),
                markersize=case.get("markersize", 5.5), linewidth=case.get("linewidth", 2.4),
                label=f"{display}\niter {run.iteration}, {run.time:.5f} s")[0])
        ax.set(title="Ion Temperature Comparison", xlabel=RHO_LABEL, ylabel=r"$T_i$ [keV]", xlim=(0, 1))
        ax.grid(alpha=0.3)
        ax.margins(y=0.08)
        fig.legend(handles=latest_handles + initial_handles, loc="outside right center", ncol=2,
                   fontsize=18, frameon=False, handlelength=1.6, columnspacing=0.7, labelspacing=0.7)
        _save(fig, out_path, dpi)


def plot_iteration_family(rho: np.ndarray, initial: np.ndarray, values: np.ndarray,
                          iterations: tuple[int, ...], reference: tuple[np.ndarray, np.ndarray] | None,
                          out_path: Path, *, ylabel: str, title: str, dpi: int,
                          title_size: float = TITLE_SIZE) -> None:
    """Draw one profile per loop iteration, opening on the profile the loop started from.

    Each iteration advances the transport clock across one time window, starting where the previous
    iteration ended, so the family is successive states of a single time march.

    Parameters
    ----------
    rho : numpy.ndarray
        Radial grid, shape ``(n_rho,)``.
    initial : numpy.ndarray
        The starting profile, drawn as iteration 0, shape ``(n_rho,)``.
    values : numpy.ndarray
        One profile per iteration, shape ``(n_iterations, n_rho)``.
    iterations : tuple of int
        Iteration numbers matching the first axis of ``values``, without the prepended zero.
    reference : tuple of numpy.ndarray, or None
        The reference's radial grid and profile, or None to draw the run alone.
    out_path : Path
        Figure destination.
    ylabel, title : str
        Axis label and title.
    title_size : float
        Title size in points.
    dpi : int
        Figure resolution.
    """
    import matplotlib.pyplot as plt

    family = np.vstack((initial, values))
    numbers = (0, *iterations)
    with plt.rc_context(POSTER_RC):
        fig, ax = _new_figure(ITERATION_FIGURE_SIZE_IN)
        colours = _draw_iterations(ax, rho, family)
        # Reserve a band above and below the family so neither end label can land on a curve.
        ax.margins(y=0.16)
        first_on_top = float(np.median(family[0] - family[-1])) > 0.0
        _end_label(ax, "iteration 0 (initial)", colours[0], first_on_top)
        _end_label(ax, f"iteration {numbers[-1]} (final)", colours[-1], not first_on_top)
        handle = None if reference is None else ax.plot(*reference, "s-", color=T3D_COLOR, zorder=3)[0]
        ax.set(xlabel=RHO_LABEL, ylabel=ylabel)
        ax.set_title(title, fontsize=title_size)
        ax.grid(alpha=0.3)
        _legend_below(fig, colours, numbers, handle)
        _save(fig, out_path, dpi)


def plot_iteration_flux(rho: np.ndarray, power_mw: np.ndarray, iterations: tuple[int, ...],
                        reference: tuple[np.ndarray, np.ndarray] | None, out_path: Path, *,
                        scale: str, dpi: int) -> None:
    """Draw every iteration's Stage 4 ion heat flux as power crossing each flux surface.

    Stage 4 runs ahead of Stage 5 within an iteration, so iteration 1's flux is already the flux of
    the profile the loop started from and iteration 0 repeats it.

    Parameters
    ----------
    rho : numpy.ndarray
        Flux grid, shape ``(n_rho,)``.
    power_mw : numpy.ndarray
        One flux profile per iteration in MW, shape ``(n_iterations, n_rho)``.
    iterations : tuple of int
        Iteration numbers matching the first axis of ``power_mw``, without the prepended zero.
    reference : tuple of numpy.ndarray, or None
        The reference's midpoint grid and flux power, or None to draw the run alone.
    out_path : Path
        Figure destination.
    scale : str
        ``"linear"`` or ``"log"``. The logarithmic axis masks non-positive power.
    dpi : int
        Figure resolution.

    Raises
    ------
    ValueError
        If ``scale`` names neither axis.
    """
    import matplotlib.pyplot as plt

    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log', got {scale!r}")
    values = np.vstack((power_mw[0], power_mw))
    numbers = (0, *iterations)
    ref = None if reference is None else reference[1]

    with plt.rc_context(POSTER_RC):
        fig, ax = _new_figure(ITERATION_FIGURE_SIZE_IN)
        if scale == "log":
            ax.set_yscale("log")
            values = np.where(values > 0.0, values, np.nan)
            ref = None if ref is None else np.where(ref > 0.0, ref, np.nan)
        colours = _draw_iterations(ax, rho, values)
        handle = None if ref is None else ax.plot(reference[0], ref, "s-", color=T3D_COLOR, zorder=3)[0]
        drawn = [values] if ref is None else [values, ref]
        peak = max(float(np.nanmax(part)) for part in drawn)
        if scale == "log":
            lowest = min(float(np.nanmin(part)) for part in drawn)
            ax.set_ylim(10 ** (np.floor(np.log10(lowest)) - 1), 10 ** (np.ceil(np.log10(peak)) + 1))
        else:
            # Reserve space below the zero-flux curve for its initial-state label.
            ax.set_ylim(-0.18 * peak, 1.2 * peak)
            ax.set_yticks(np.arange(0, peak + 0.5, 0.5))
        _end_label(ax, "iterations 0 and 1 (initial)", colours[0], on_top=False)
        _end_label(ax, f"iteration {numbers[-1]} (final)", colours[-1], on_top=True)
        ax.set(xlabel=RHO_LABEL, ylabel="ion heat flux [MW]")
        ax.set_title(f"Ion heat flux over {len(iterations)} iterations", fontsize=TITLE_SIZE)
        ax.grid(alpha=0.3)
        _legend_below(fig, colours, numbers, handle)
        _save(fig, out_path, dpi)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the run root, the output directory, the optional reference and figure options.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT,
                        help="parent of the five W7-X run directories")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="directory for the five PNGs")
    parser.add_argument("--t3d-nc", type=Path, default=None,
                        help="T3D+GX NetCDF output; without it the figures carry no reference curves")
    parser.add_argument("--neopax-rev", default=None,
                        help="Full NEOPAX commit the runs used. Defaults to the stages/pixi.toml pin.")
    parser.add_argument("--dpi", type=int, default=600, help="figure resolution")
    return parser


def main() -> None:
    """Read the five runs and write the five figures."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.t3d_nc is None:
        logger.info("no --t3d-nc given, so the figures are drawn from the runs alone")
    reference = None if args.t3d_nc is None else read_t3d(args.t3d_nc)

    endpoints = [read_endpoints(args.outputs_root / case["directory"]) for case in CASES]
    plot_ion_temperature_comparison(endpoints, reference, args.outdir / "ion_temperature_comparison.png",
                                    dpi=args.dpi)
    benchmark = load_benchmark(args.outputs_root / CASES[0]["directory"], args.neopax_rev)
    n_iterations = len(benchmark.iterations)
    plot_iteration_family(
        benchmark.rho, benchmark.ti_initial, benchmark.ti, benchmark.iterations,
        None if reference is None else (reference.rho, reference.ti[-1]),
        args.outdir / "iteration_ion_temperature.png", ylabel=r"$T_i$ [keV]",
        title=f"Ion temperature over {n_iterations} iterations", dpi=args.dpi)
    plot_iteration_family(
        benchmark.rho, benchmark.exchange_initial, benchmark.exchange, benchmark.iterations,
        None if reference is None else (reference.rho, reference.exchange_mw[-1]),
        args.outdir / "iteration_collisional_exchange.png", ylabel=r"ion heating [MW/m$^3$]",
        title=f"Collisional heating over {n_iterations} iterations", dpi=args.dpi, title_size=17.0)
    flux_reference = None if reference is None else (reference.rho_mid, reference.power_mw)
    plot_iteration_flux(benchmark.rho_flux, benchmark.power_mw, benchmark.iterations, flux_reference,
                        args.outdir / "ion_heat_flux_power.png", scale="linear", dpi=args.dpi)
    plot_iteration_flux(benchmark.rho_flux, benchmark.power_mw, benchmark.iterations, flux_reference,
                        args.outdir / "ion_heat_flux_power_log.png", scale="log", dpi=args.dpi)


if __name__ == "__main__":
    main()
