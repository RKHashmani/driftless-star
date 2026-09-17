"""Tests for the Stage 3 dkx radial-scan helpers.

These reuse the pure functions from ``dkx_radial_scan.py``, loaded by path.
``_choose_radius_indices`` selects which flux surfaces to solve, and
``_prepare_input_text`` writes one surface's face state into a sfincs namelist. The
profile sources those helpers consume are shared with Stage 4 and live in
``common.neopax_profiles``, covered by ``tests/common/test_neopax_profiles.py``.
Pure helpers load without DKX. Preparation reads installed DKX metadata.
These tests replace that lookup with a fixed solver identity.
Workers import the solver only when they run.
"""

from __future__ import annotations

import argparse
import re
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.io_contracts import validate_dkx_flux
from tests.helpers.stage_import import load_stage_module

scan = load_stage_module("stages/stage3-neoclassical/dkx_radial_scan.py")

SOLVER = {"name": "dkx", "version": "2.4.0", "revision": "test-revision"}


@pytest.fixture(autouse=True)
def installed_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply a fixed solver identity for pure helper tests without importing DKX."""
    monkeypatch.setattr(scan, "_solver_identity", lambda: SOLVER.copy())


# Radius selection

# `_choose_radius_indices` picks which flux surfaces the scan will solve. With no options given, it should skip the
# magnetic axis (rho = 0, at index 0) because a solve there is rarely useful. Given 5 radii, this asserts it returns
# indices [1, 2, 3, 4], i.e. everything except the axis.
def test_choose_radius_default_skips_axis() -> None:
    rho = np.linspace(0.0, 1.0, 5)  # index 0 is rho = 0
    idxs = scan._choose_radius_indices(rho, explicit=None, rho_min=None, rho_max=None, num_radii=None)
    assert idxs == [1, 2, 3, 4]


def test_choose_radius_explicit_sorted_and_deduped() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    idxs = scan._choose_radius_indices(rho, explicit=[3, 1, 1], rho_min=None, rho_max=None, num_radii=None)
    assert idxs == [1, 3]


def test_choose_radius_explicit_out_of_range_raises() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    with pytest.raises(IndexError):
        scan._choose_radius_indices(rho, explicit=[7], rho_min=None, rho_max=None, num_radii=None)


# `num_radii` thins the candidate surfaces down to a smaller evenly-spaced sample. After skipping the axis the
# candidates are [1, 2, 3, 4]; asking for 2 should keep the two endpoints, so this asserts the result is [1, 4]. This
# lets a user run a cheaper, coarser scan.
def test_choose_radius_num_radii_subsamples() -> None:
    rho = np.linspace(0.0, 1.0, 5)  # candidates after axis skip: [1, 2, 3, 4]
    idxs = scan._choose_radius_indices(rho, explicit=None, rho_min=None, rho_max=None, num_radii=2)
    assert idxs == [1, 4]  # endpoints of the candidate span


# The rho range is 0 to 1, so a `rho_min` of 2.0 filters out every surface. Rather than return an empty list (which
# would make the scan do nothing), this asserts it raises ValueError so the impossible filter is reported clearly.
def test_choose_radius_empty_filter_raises() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        scan._choose_radius_indices(rho, explicit=None, rho_min=2.0, rho_max=None, num_radii=None)


# Prepare dispatch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Complete prescribed input for prepare tests. The [species] block supplies charge and mass. The
# [profiles] arrays use SI values on five cells. Face arrays are linear in rho, with constant
# gradients per unit rho.
PRESCRIBED_TOML = """\
[species]
names = ["e", "D", "T"]
charge_qp = [-1.0, 1.0, 1.0]
mass_mp = [0.000544617, 2.0, 3.0]

[geometry]
n_radial = 5

[profiles]
model = "prescribed"
density = [[1.0e19, 1.1e19, 1.2e19, 1.3e19, 1.4e19], [5.0e18, 5.5e18, 6.0e18, 6.5e18, 7.0e18], [5.0e18, 5.5e18, 6.0e18, 6.5e18, 7.0e18]]
temperature = [[1000.0, 900.0, 800.0, 700.0, 600.0], [950.0, 850.0, 750.0, 650.0, 550.0], [940.0, 840.0, 740.0, 640.0, 540.0]]
Er = [0.0, 1.0, 2.0, 3.0, 4.0]
density_face = [[0.95e19, 1.05e19, 1.15e19, 1.25e19, 1.35e19, 1.45e19], [4.75e18, 5.25e18, 5.75e18, 6.25e18, 6.75e18, 7.25e18], [4.75e18, 5.25e18, 5.75e18, 6.25e18, 6.75e18, 7.25e18]]
temperature_face = [[1050.0, 950.0, 850.0, 750.0, 650.0, 550.0], [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0], [990.0, 890.0, 790.0, 690.0, 590.0, 490.0]]
Er_face = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
density_grad_face = [[5.0e18, 5.0e18, 5.0e18, 5.0e18, 5.0e18, 5.0e18], [2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18], [2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18]]
temperature_grad_face = [[-500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0]]
"""


# Shared tests call ``build_prescribed_face_state`` directly. This test covers dispatch from the
# prescribed profile-source option. It uses no transport solution. The real prepare step records the
# source and scans the face grid without the magnetic axis. Here, [geometry].n_radial gives five
# cells and six faces. Removing the axis leaves five scan surfaces.
def test_prepare_dispatches_prescribed_source(tmp_path: Path) -> None:
    config = tmp_path / "common_input.toml"
    config.write_text(PRESCRIBED_TOML)
    args = scan.build_parser().parse_args([
        "prepare",
        "--common-config", str(config),
        "--dkx-template", str(REPO_ROOT / "inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun"),
        "--output-dir", str(tmp_path / "out"),
        "--profiles-source", "prescribed",
    ])
    manifest, pending = scan._prepare(args)
    assert manifest["profiles_source"] == "prescribed"
    assert manifest["source_transport_solution"] is None
    assert [run["rho"] for run in manifest["runs"]] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert len(pending) == 5


# sfincs input text

def _namelist_values(text: str, key: str) -> np.ndarray:
    """Read one namelist entry back out of the emitted input text as a float array."""
    match = re.search(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n\r]+)$", text)
    assert match is not None, f"{key} is absent from the emitted namelist"
    return np.asarray([float(token) for token in match.group(1).split()], dtype=float)


# sfincs reads ``dNHatdrNs`` and ``dTHatdrNs`` against rho. The snapshot stores gradients per unit
# rho. The writer must not scale, negate or recalculate them. The test uses distinct constant
# gradients for both channels. It detects incorrect units, signs and channel selection.
def test_prepare_input_text_writes_the_snapshot_gradients_unchanged() -> None:
    cfg = tomllib.loads(PRESCRIBED_TOML)
    snapshot = scan.build_prescribed_face_state(cfg, n_species=3)
    species = scan._parse_species_from_config(cfg)
    template = (REPO_ROOT / "inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun").read_text()
    radius_index = 3
    text = scan._prepare_input_text(
        template_text=template,
        species=species,
        snapshot=snapshot,
        radius_index=radius_index,
        include_phi1=None,
        resolution_overrides={},
        solver_tolerance=None,
    )
    assert_allclose(_namelist_values(text, "dNHatdrNs"), snapshot.density_grad[:, radius_index], rtol=1e-12)
    assert_allclose(_namelist_values(text, "dTHatdrNs"), snapshot.temperature_grad[:, radius_index], rtol=1e-12)
    assert_allclose(_namelist_values(text, "dNHatdrNs"), [0.05, 0.025, 0.025], rtol=1e-12)
    assert_allclose(_namelist_values(text, "dTHatdrNs"), [-0.5, -0.5, -0.5], rtol=1e-12)
    assert "dnhatdrhats" not in text.lower()
    assert "dthatdrhats" not in text.lower()


@pytest.mark.parametrize("coordinate", ["psiHat", "psiN", "rHat", "rN"])
def test_prepare_replaces_indexed_template_gradients(coordinate: str) -> None:
    cfg = tomllib.loads(PRESCRIBED_TOML)
    template = (REPO_ROOT / "inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun").read_text()
    template = template.replace(
        "&speciesParameters",
        f"&speciesParameters\n  withAdiabatic = .false., DnHaTd{coordinate}s(1) = 999 ! stale density\n"
        f"  dTHatd{coordinate}s = 999\n    999 999 ! continued temperature",
    )
    text = scan._prepare_input_text(
        template_text=template, species=scan._parse_species_from_config(cfg),
        snapshot=scan.build_prescribed_face_state(cfg, n_species=3),
        radius_index=3, include_phi1=None, resolution_overrides={}, solver_tolerance=None,
    )
    assert not re.search(r"(?i)d[nt]Hatd\w+s\s*\(", text)
    assert "withAdiabatic = .false." in text
    assert "continued temperature" not in text
    assert_allclose(_namelist_values(text, "dNHatdrNs"), [0.05, 0.025, 0.025])
    assert_allclose(_namelist_values(text, "dTHatdrNs"), [-0.5, -0.5, -0.5])


def _prepared_scan(tmp_path: Path):
    config = tmp_path / "common.toml"
    config.write_text(PRESCRIBED_TOML)
    wout = tmp_path / "wout.nc"
    wout.write_bytes(b"equilibrium for cache tests")
    args = scan.build_parser().parse_args([
        "prepare", "--common-config", str(config), "--profiles-source", "prescribed",
        "--wout-path", str(wout), "--num-radii", "2", "--output-dir", str(tmp_path / "scan"),
    ])
    manifest, pending = scan._prepare(args)
    return args, manifest, pending


def _summary(payload: dict) -> dict:
    return {
        "solver": payload["solver"], "wout_sha256": payload["wout_sha256"],
        "radius_index": payload["radius_index"], "rho": payload["rho"],
        "rHat": payload["rho"] * 0.8,
        "Gamma": [1.0, 2.0, 3.0], "Q": [4.0, 5.0, 6.0], "Upar": [7.0, 8.0, 9.0],
    }


def _complete(pending: list[Path]) -> list[Path]:
    results = []
    for path in pending:
        payload = json.loads(path.read_text())
        result = Path(payload["result_json"])
        result.write_text(json.dumps(_summary(payload)))
        results.append(result)
    return results


@pytest.mark.parametrize("change", [None, "malformed", "solver", "equilibrium", "namelist", "radius"])
def test_prepare_reuses_only_matching_results(tmp_path: Path, monkeypatch, change) -> None:
    args, _, pending = _prepared_scan(tmp_path)
    results = _complete(pending)
    if change == "malformed":
        for path in results:
            summary = json.loads(path.read_text())
            del summary["solver"]
            path.write_text(json.dumps(summary))
    elif change == "solver":
        monkeypatch.setattr(scan, "_solver_identity", lambda: {**SOLVER, "revision": "new-revision"})
    elif change == "equilibrium":
        Path(args.wout_path).write_bytes(b"different geometry at the same path")
    elif change == "namelist":
        args.nxi = 16
    elif change == "radius":
        for path in results:
            summary = json.loads(path.read_text())
            summary["rHat"] = -1.0
            path.write_text(json.dumps(summary))
    _, pending = scan._prepare(args)
    assert len(pending) == (0 if change is None else 2)
    assert all(path.exists() == (change is None) for path in results)


@pytest.mark.parametrize("missing", ["revision", "digest"])
def test_prepare_explains_disabled_reuse(tmp_path: Path, monkeypatch, capsys, missing: str) -> None:
    if missing == "revision":
        monkeypatch.setattr(scan, "_solver_identity", lambda: {**SOLVER, "revision": ""})
    else:
        monkeypatch.setattr(scan, "_wout_digest", lambda path: None)
    args, _, pending = _prepared_scan(tmp_path)
    results = _complete(pending)
    capsys.readouterr()
    _, pending = scan._prepare(args)
    assert len(pending) == 2
    assert all(not result.exists() for result in results)
    message = capsys.readouterr().out
    assert message.count("result reuse disabled") == 1
    assert missing in message


@pytest.mark.parametrize("cores, inherited, expected", [(0, None, None), (0, "0", "0"), (4, "0", "4")])
def test_worker_cpu_threads_preserve_automatic_modes(monkeypatch, cores, inherited, expected) -> None:
    monkeypatch.delenv("DKX_CORES", raising=False)
    if inherited is not None:
        monkeypatch.setenv("DKX_CORES", inherited)
    args = scan.build_parser().parse_args(["--backend", "cpu", "--cores-per-run", str(cores)])
    env = scan._build_worker_env(args, gpu_id=None)
    assert env.get("DKX_CORES") == expected
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == str(max(1, cores))


def test_last_species_vector_checks_only_the_final_iteration() -> None:
    history = np.array([[np.nan, 10.0], [np.inf, 20.0], [-np.inf, 30.0]])
    assert_allclose(scan._last_species_vector(history, 3), [10.0, 20.0, 30.0])


def _native_diagnostics() -> dict:
    return {
        "particleFlux_vm_rHat": np.array([[1, 10], [2, 20], [3, 30]]),
        "heatFlux_vm_rHat": np.array([[4, 40], [5, 50], [6, 60]]),
        "FSABFlow": np.array([[7, 70], [8, 80], [9, 90]]),
        "rHat": 0.16, "B0OverBBar": 2.0,
    }


@pytest.mark.parametrize("problem", ["coordinate", "field", "shape", "nonfinite"])
def test_extract_rejects_unusable_native_diagnostics(problem: str) -> None:
    native = _native_diagnostics()
    if problem == "coordinate":
        native["particleFlux_vm_rN"] = native.pop("particleFlux_vm_rHat")
    elif problem == "field":
        del native["B0OverBBar"]
    elif problem == "shape":
        native["heatFlux_vm_rHat"] = np.ones((2, 3))
    else:
        native["FSABFlow"] = np.full((3, 1), np.nan)
    with pytest.raises((ValueError, KeyError)):
        scan._extract_flux_triplet(native, 3)


@pytest.mark.parametrize("failure", [None, "solve", "diagnostics", "conversion_overflow"])
def test_worker_uses_public_api_and_clears_completion_on_failure(tmp_path: Path, monkeypatch, failure) -> None:
    _, _, pending = _prepared_scan(tmp_path)
    payload = json.loads(pending[0].read_text())
    result = Path(payload["result_json"])
    result.write_text("old success")
    calls = []
    api = ModuleType("dkx.api")
    native = _native_diagnostics()
    if failure == "conversion_overflow":
        native["particleFlux_vm_rHat"] = np.full((3, 1), 1.0e300)

    def write_output(*args, **kwargs):
        calls.append((args, kwargs))
        if failure == "solve":
            raise RuntimeError("did not converge")

    api.write_output = write_output
    api.read_output = lambda path: {} if failure == "diagnostics" else native
    monkeypatch.setitem(sys.modules, "dkx", ModuleType("dkx"))
    monkeypatch.setitem(sys.modules, "dkx.api", api)
    if failure:
        with np.errstate(over="ignore"), pytest.raises((RuntimeError, KeyError, ValueError)):
            scan._run_single_worker_from_payload(pending[0])
        assert not result.exists()
    else:
        assert scan._run_single_worker_from_payload(pending[0]) == 0
        summary = json.loads(result.read_text())
        assert_allclose(summary["Gamma"], scan.SFINCS_GAMMA_TO_NEOPAX * np.array([10, 20, 30]))
        assert_allclose(summary["Q"], scan.SFINCS_Q_TO_NEOPAX * np.array([40, 50, 60]))
        assert_allclose(summary["Upar"], 4 / np.sqrt(np.pi) * np.array([70, 80, 90]))
        assert not {"rHat", "B0OverBBar"} & summary["meta"].keys()
    assert calls == [((Path(payload["input_path"]), Path(payload["output_path"])), {
        "wout_path": Path(payload["wout_path"]), "overwrite": True, "emit": print,
    })]


@pytest.mark.parametrize("problem", [None, "missing", "solver", "species", "radius", "axis_only"])
def test_collect_checks_results_and_honors_output(tmp_path: Path, problem) -> None:
    prepare_args, manifest, pending = _prepared_scan(tmp_path)
    results = _complete(pending)
    last = json.loads(results[-1].read_text())
    last["meta"] = {"Gamma_key": "particleFlux_vm_rHat", "B0OverBBar": "2.0", "rHat": str(last["rHat"])}
    results[-1].write_text(json.dumps(last))
    destination = tmp_path / "custom" / "handoff flux.h5"
    destination.parent.mkdir()
    destination.write_bytes(b"existing aggregate")
    args = scan.build_parser().parse_args([
        "collect", "--output-dir", prepare_args.output_dir,
        "--output", str(destination), "--no-plot",
    ])
    if problem == "missing":
        results[0].unlink()
    elif problem:
        summary = json.loads(results[0].read_text())
        if problem == "solver":
            summary["solver"]["revision"] = "wrong"
        elif problem == "species":
            summary["Gamma"] = [1.0]
        elif problem == "radius":
            summary["rHat"] = 0.0
        else:
            manifest["runs"] = [manifest["runs"][0]]
            manifest["runs"][0]["rho"] = 0.0
            summary["rho"] = summary["rHat"] = 0.0
        results[0].write_text(json.dumps(summary))
    if problem:
        with pytest.raises((ValueError, FileNotFoundError)):
            scan._collect(args, manifest)
        assert destination.read_bytes() == b"existing aggregate"
    else:
        assert scan._collect(args, manifest) == 0
        with h5py.File(destination) as output:
            assert output["Gamma"].shape == (3, 3)
            assert_allclose(output["Gamma"][:, 0], 0)
            assert list(output["species_names"][:]) == [b"e", b"D", b"T"]
            assert output.attrs["axis_zero_padded"]
            assert output.attrs["raw_Gamma_key"] == "particleFlux_vm_rHat"
            assert "raw_B0OverBBar" not in output.attrs
            assert "raw_rHat" not in output.attrs
            assert not {"Gamma_perturbed", "Q_perturbed", "perturb_delta", "perturb_present",
                        "response_label", "perturb_species"}.intersection(output)
            assert not {"response_mode", "response_note"}.intersection(output.attrs)
        validate_dkx_flux(destination)


def _prepared_responses(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Any], list[Path]]:
    args, _, _ = _prepared_scan(tmp_path)
    args.response_mode = "fd_gradients"
    args.perturb_density_species = "D,e"
    args.perturb_temperature_species = "T,D"
    manifest, pending = scan._prepare(args)
    return args, manifest, pending


@pytest.mark.parametrize("floor, relative", [(0.5, 0.5), (0.01, 2.0)])
def test_response_gradients_apply_expected_steps(
    tmp_path: Path, floor: float, relative: float,
) -> None:
    args, manifest, _ = _prepared_responses(tmp_path)
    args.dkap_density = args.dkap_temperature = floor
    args.perturb_rel_step = relative
    manifest, _ = scan._prepare(args)
    state = scan.build_prescribed_face_state(tomllib.loads(PRESCRIBED_TOML), n_species=3)
    runs_root = Path(args.output_dir) / "runs"
    assert len(manifest["runs"]) == 10
    for run in manifest["runs"]:
        text = (runs_root / run["run_subdir"] / "input.namelist").read_text()
        idx = run["radius_index"]
        n, t = state.density[:, idx], state.temperature[:, idx]
        dn, dt = state.density_grad[:, idx], state.temperature_grad[:, idx]
        assert_allclose(_namelist_values(text, "nHats"), n)
        assert_allclose(_namelist_values(text, "THats"), t)
        assert_allclose(_namelist_values(text, "Er"), state.er[idx])
        assert_allclose(_namelist_values(text, "rN_wish"), state.rho[idx])
        actual_n = -_namelist_values(text, "dNHatdrNs") / n
        actual_t = -_namelist_values(text, "dTHatdrNs") / t
        expected_n, expected_t = -dn / n, -dt / t
        if run["response_label"] != "base":
            si = ["e", "D", "T"].index(run["perturb_species"])
            delta = run["perturb_delta"]
            kappa = expected_n[si] if run["response_label"] == "density_gradient" else expected_t[si]
            assert abs(delta) == max(floor, relative * abs(kappa))
            if relative == 2.0:
                assert abs(delta) > floor
            if run["response_label"] == "density_gradient":
                assert delta < 0
                assert_allclose(actual_n + actual_t, expected_n + expected_t)
                expected_n[si] += delta
                expected_t[si] -= delta
                assert_allclose(
                    _namelist_values(text, "dNHatdrNs") * t + _namelist_values(text, "dTHatdrNs") * n,
                    dn * t + dt * n,
                )
            else:
                assert delta > 0
                expected_t[si] += delta
        assert_allclose(actual_n, expected_n)
        assert_allclose(actual_t, expected_t)


@pytest.mark.parametrize("axis", [False, True])
def test_response_collection_order_and_axis(tmp_path: Path, axis: bool) -> None:
    args, manifest, pending = _prepared_responses(tmp_path)
    if axis:
        args.rho_indices = "0,1,5"
        manifest, pending = scan._prepare(args)
    results = _complete(pending)
    for i, path in enumerate(results):
        result = json.loads(path.read_text())
        result["Gamma"] = [i + 1., i + 2., i + 3.]
        result["Q"] = [i + 4., i + 5., i + 6.]
        path.write_text(json.dumps(result))
    # The collector uses the requested pair order, regardless of manifest order.
    manifest["runs"].reverse()
    collect_args = scan.build_parser().parse_args(["collect", "--output-dir", args.output_dir, "--no-plot"])
    scan._collect(collect_args, manifest)
    validate_dkx_flux(Path(args.output_dir) / "dkx_flux_profiles.h5")
    with h5py.File(Path(args.output_dir) / "dkx_flux_profiles.h5") as f:
        assert f["Gamma"].shape == (3, 3)
        assert f["Gamma_perturbed"].shape == (4, 3, 3)
        assert list(f["response_label"].asstr()[:]) == ["density_gradient"] * 2 + ["temperature_gradient"] * 2
        assert list(f["perturb_species"].asstr()[:]) == ["D", "e", "T", "D"]
        assert_allclose(f["r"][:], f["rho"][:] * 0.8)
        assert f["perturb_present"].dtype == np.dtype(bool)
        assert np.all(f["perturb_present"][:, 1:])
        assert np.all(f["perturb_present"][:, 0] == axis)
        if not axis:
            assert_allclose(f["Gamma_perturbed"][:, :, 0], 0)
            assert_allclose(f["Q_perturbed"][:, :, 0], 0)
            assert_allclose(f["perturb_delta"][:, 0], 0)
        for run in manifest["runs"]:
            result = json.loads((Path(args.output_dir) / "runs" / run["run_subdir"] / "result.json").read_text())
            ri = np.flatnonzero(f["rho"][:] == run["rho"])[0]
            if run["response_label"] == "base":
                assert_allclose(f["Gamma"][:, ri], result["Gamma"])
            else:
                pi = [("density_gradient", "D"), ("density_gradient", "e"),
                      ("temperature_gradient", "T"), ("temperature_gradient", "D")].index(
                          (run["response_label"], run["perturb_species"]))
                assert_allclose(f["Gamma_perturbed"][pi, :, ri], result["Gamma"])
                assert_allclose(f["Q_perturbed"][pi, :, ri], result["Q"])
                assert f["perturb_delta"][pi, ri] == run["perturb_delta"]


@pytest.mark.parametrize("problem", ["missing_result", "rho", "nonfinite"])
def test_response_collection_rejects_invalid_siblings(tmp_path: Path, problem: str) -> None:
    args, manifest, pending = _prepared_responses(tmp_path)
    results = _complete(pending)
    path = results[1]
    if problem == "missing_result":
        path.unlink()
    else:
        result = json.loads(path.read_text())
        if problem == "rho":
            result["rho"] *= 2
        else:
            result["Q"][0] = float("nan")
        path.write_text(json.dumps(result))
    collect_args = scan.build_parser().parse_args(["collect", "--output-dir", args.output_dir, "--no-plot"])
    with pytest.raises((ValueError, FileNotFoundError)):
        scan._collect(collect_args, manifest)
    assert not (Path(args.output_dir) / "dkx_flux_profiles.h5").exists()


def test_response_reuse_is_specific_to_step_and_channel(tmp_path: Path) -> None:
    args, manifest, pending = _prepared_responses(tmp_path)
    results = _complete(pending)
    assert scan._prepare(args)[1] == []
    args.dkap_density = 2.0
    updated, pending = scan._prepare(args)
    assert len(pending) == 4
    assert all("_fd_n_" in str(path) for path in pending)
    for before, after in zip(manifest["runs"], updated["runs"]):
        changed = before["perturb_delta"] != after["perturb_delta"]
        assert changed == (before["response_label"] == "density_gradient")
    assert all(result.exists() == ("_fd_n_" not in str(result)) for result in results)


def test_response_benchmarks_only_baseline_and_reuses_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _prepared_responses(tmp_path)
    args.num_radii = 1
    args.benchmark_repeats, args.benchmark_warmup = 2, 1
    manifest, pending = scan._prepare(args)
    calls = []
    api = ModuleType("dkx.api")
    api.write_output = lambda *a, **kw: calls.append(a[0].parent.name)
    api.read_output = lambda path: _native_diagnostics()
    monkeypatch.setitem(sys.modules, "dkx", ModuleType("dkx"))
    monkeypatch.setitem(sys.modules, "dkx.api", api)
    for payload_path in pending:
        scan._run_single_worker_from_payload(payload_path)
    for run in manifest["runs"]:
        assert calls.count(run["run_subdir"]) == (3 if run["response_label"] == "base" else 1)
    assert scan._prepare(args)[1] == []
    collect_args = scan.build_parser().parse_args(["collect", "--output-dir", args.output_dir, "--no-plot"])
    scan._collect(collect_args, manifest)
    with h5py.File(Path(args.output_dir) / "dkx_flux_profiles.h5") as output:
        assert output["benchmark_rho"].shape == (1,)


@pytest.mark.parametrize("key,value", [("dkap_density", -1), ("dkap_temperature", float("nan")),
                                      ("perturb_rel_step", float("inf")), ("perturb_density_species", "missing")])
def test_response_rejects_invalid_controls(tmp_path: Path, key: str, value: float | str) -> None:
    args, _, _ = _prepared_responses(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError):
        scan._prepare(args)


def test_response_rejects_zero_step_and_empty_selection(tmp_path: Path) -> None:
    args, _, _ = _prepared_responses(tmp_path)
    args.dkap_density = args.perturb_rel_step = 0
    with pytest.raises(ValueError, match="nonzero"):
        scan._prepare(args)
    args.perturb_density_species = args.perturb_temperature_species = ""
    with pytest.raises(ValueError, match="at least one"):
        scan._prepare(args)


@pytest.mark.parametrize("name", ["He-3", "a/b", "D ion"])
def test_response_rejects_unschedulable_species(name: str) -> None:
    args = scan.build_parser().parse_args([
        "--response-mode", "fd_gradients", "--perturb-density-species", name,
    ])
    with pytest.raises(ValueError, match="run directory"):
        scan._response_settings(args, [scan.SpeciesMeta(name=name, charge=1, mass_mp=3)])


@pytest.mark.parametrize("field,value", [("density", 0), ("temperature", -1),
                                        ("density_grad", np.nan), ("temperature_grad", np.inf)])
def test_response_rejects_invalid_local_profiles(field: str, value: float) -> None:
    state = scan.build_prescribed_face_state(tomllib.loads(PRESCRIBED_TOML), n_species=3)
    getattr(state, field)[1, 1] = value
    species = scan._parse_species_from_config(tomllib.loads(PRESCRIBED_TOML))
    args = scan.build_parser().parse_args([
        "--response-mode", "fd_gradients", "--perturb-density-species", "D",
    ])
    with pytest.raises(ValueError, match="species 'D' at rho=0.2.*positive local"):
        scan._response_states(
            snapshot=state, species=species, radius_indices=[1], settings=scan._response_settings(args, species),
        )


@pytest.mark.parametrize("names", ['"e", "D", "d"', '"e", "He-3", "none"'])
def test_baseline_ignores_response_validation(tmp_path: Path, names: str) -> None:
    args, _, _ = _prepared_scan(tmp_path)
    Path(args.common_config).write_text(PRESCRIBED_TOML.replace('"e", "D", "T"', names))
    args.perturb_density_species = "unknown"
    args.perturb_temperature_species = "also_unknown"
    args.dkap_density = -1
    args.dkap_temperature = float("nan")
    args.perturb_rel_step = float("inf")
    manifest, pending = scan._prepare(args)
    assert len(pending) == len(manifest["runs"]) == 2
    assert all(run["response_label"] == "base" for run in manifest["runs"])


def test_response_rejects_case_collisions() -> None:
    cfg = tomllib.loads(PRESCRIBED_TOML.replace('"e", "D", "T"', '"e", "D", "d"'))
    args = scan.build_parser().parse_args([
        "--response-mode", "fd_gradients", "--perturb-density-species", "D",
    ])
    with pytest.raises(ValueError, match="unique ignoring case"):
        scan._response_settings(args, scan._parse_species_from_config(cfg))


def test_response_normalizes_and_deduplicates_names(tmp_path: Path) -> None:
    args, _, _ = _prepared_scan(tmp_path)
    args.response_mode = "fd_gradients"
    args.perturb_density_species = " d,D, E,e,, "
    args.perturb_temperature_species = "t, T,d,D"
    manifest, pending = scan._prepare(args)
    assert manifest["perturb_density_species"] == ["D", "e"]
    assert manifest["perturb_temperature_species"] == ["T", "D"]
    assert len(pending) == 10
    assert [run["perturb_species"] for run in manifest["runs"][:5]] == ["none", "D", "e", "T", "D"]


def test_response_rejects_overflowing_step_before_writing_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _prepared_scan(tmp_path)
    state = scan.build_prescribed_face_state(tomllib.loads(PRESCRIBED_TOML), n_species=3)
    state.density_grad[1, 5] = -2 * state.density[1, 5]
    monkeypatch.setattr(scan, "build_prescribed_face_state", lambda *a, **kw: state)
    args.output_dir = str(tmp_path / "overflow")
    args.response_mode = "fd_gradients"
    args.perturb_density_species = "D"
    args.perturb_rel_step = 1e308
    with np.errstate(over="ignore"), pytest.raises(
        ValueError, match="species 'D' at rho=1.0.*finite nonzero",
    ):
        scan._prepare(args)
    assert not list(Path(args.output_dir).rglob("input.namelist"))
    assert not (Path(args.output_dir) / "manifest.json").exists()


def test_linear_flux_model_recovers_pressure_preserving_slopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, manifest, pending = _prepared_responses(tmp_path)
    # Rows identify output species. Columns identify the species whose gradient changes.
    density_coeff = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    temperature_coeff = np.array([[9., 7., 5.], [8., 6., 4.], [3., 2., 1.]])
    native_outputs: dict[Path, dict[str, Any]] = {}

    def write_output(input_path: Path, output_path: Path, **kwargs: Any) -> None:
        text = input_path.read_text()
        kappa_n = -_namelist_values(text, "dNHatdrNs") / _namelist_values(text, "nHats")
        kappa_t = -_namelist_values(text, "dTHatdrNs") / _namelist_values(text, "THats")
        native_outputs[output_path] = {
            "particleFlux_vm_rHat": (density_coeff @ kappa_n + temperature_coeff @ kappa_t + 10)
            / scan.SFINCS_GAMMA_TO_NEOPAX,
            "heatFlux_vm_rHat": (2 * density_coeff @ kappa_n + 3 * temperature_coeff @ kappa_t - 4)
            / scan.SFINCS_Q_TO_NEOPAX,
            "FSABFlow": np.ones(3),
            "rHat": float(_namelist_values(text, "rN_wish")[0]) * 0.8,
            "B0OverBBar": 1.0,
        }

    api = ModuleType("dkx.api")
    api.write_output = write_output
    api.read_output = native_outputs.__getitem__
    monkeypatch.setitem(sys.modules, "dkx", ModuleType("dkx"))
    monkeypatch.setitem(sys.modules, "dkx.api", api)
    for path in pending:
        assert scan._run_single_worker_from_payload(path) == 0
        payload = json.loads(path.read_text())
        Path(payload["input_path"]).unlink()
    collect_args = scan.build_parser().parse_args(["collect", "--output-dir", args.output_dir, "--no-plot"])
    assert scan._collect(collect_args, manifest) == 0
    with h5py.File(Path(args.output_dir) / "dkx_flux_profiles.h5") as output:
        pairs = zip(output["response_label"].asstr(), output["perturb_species"].asstr())
        for pi, (channel, name) in enumerate(pairs):
            si = ["e", "D", "T"].index(name)
            present = output["perturb_present"][pi]
            delta = output["perturb_delta"][pi, present]
            for flux, a, b in (("Gamma", density_coeff, temperature_coeff),
                               ("Q", 2 * density_coeff, 3 * temperature_coeff)):
                slope = (output[f"{flux}_perturbed"][pi][:, present] - output[flux][:, present]) / delta
                expected = a[:, si] - b[:, si] if channel == "density_gradient" else b[:, si]
                assert_allclose(slope, np.repeat(expected[:, None], present.sum(), axis=1), atol=1e-12)
