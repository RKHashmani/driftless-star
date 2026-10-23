"""Numerical and completion invariants for the current adapter."""

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import h5py
import numpy as np
from numpy.testing import assert_allclose
import pytest

from common.current_density_feedback_contract import artifact, resolve_settings
from common.dkx_current import CURRENT_NORMALIZATION, CURRENT_UNITS
from tests.helpers.stage_import import load_stage_module

m = load_stage_module("stages/stage1-equilibrium/current_density_feedback.py")


def wout_fixture(ns=9, *, phi=-2.0, signgs=-1, slope=0.0):
    full = np.linspace(0, 1, ns)
    half = (np.arange(1, ns) - 0.5) / (ns - 1)
    return SimpleNamespace(ns=ns, phi=phi * full, signgs=signgs,
                           pres=np.r_[0.0, 100.0 + slope * half],
                           buco=np.r_[0.0, signgs * m.MU0 / (2 * np.pi) * 3 * half], ctor=3.0)


def polynomial_matrix(ns, ds):
    x = np.arange(ns) * ds
    v = np.vander(x, N=ns, increasing=True)
    derivative = np.column_stack([np.zeros(ns)] + [power * x**(power - 1) for power in range(1, ns)])
    return derivative @ np.linalg.inv(v)


@pytest.mark.parametrize("phi", [-2.0, 2.0])
@pytest.mark.parametrize("slope", [0.0, -1e6])
def test_analytical_inversion_with_pressure_and_signed_flux(phi, slope):
    wout = wout_fixture(phi=phi, slope=slope)
    measured = np.linspace(0.1, 1, 10)
    source = (4.0 * 7 + m.MU0 * slope * 7 * measured) / phi
    helpers = SimpleNamespace(dds_matrix=polynomial_matrix,
                              geometry=lambda w, s: SimpleNamespace(fsa_B2=np.full(s.shape, 4.0)))
    s, current = m.invert_current(wout, measured, source, helpers)
    assert_allclose(current, 7 * s, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("zero", [True, False])
def test_zero_and_sign_changing_inversions(zero):
    wout = wout_fixture()
    measured = np.linspace(0, 1, 9)
    source = np.zeros(9) if zero else 4 * (2 * measured - .5) / wout.phi[-1]
    helpers = SimpleNamespace(dds_matrix=polynomial_matrix,
                              geometry=lambda w, s: SimpleNamespace(fsa_B2=np.full(s.shape, 4.0)))
    s, current = m.invert_current(wout, measured, source, helpers)
    assert_allclose(current, np.zeros(9) if zero else s * (s - .5), atol=1e-10)


def test_buco_current_uses_signed_geometry_and_half_mesh_edge():
    for signgs in (-1, 1):
        wout = wout_fixture(signgs=signgs)
        assert_allclose(m.enclosed_from_wout(wout, np.array([0, 0.2, 1])), [0, 0.6, 3])
        naive_last = signgs * 2 * np.pi / m.MU0 * wout.buco[-1]
        assert naive_last != pytest.approx(wout.ctor)


def test_compare_uses_same_dkx_field_for_both_profiles():
    settings = {"abs_tol": 1.0, "rel_tol": 0.1}
    residual, threshold = m.compare_current(
        np.array([2.0, -12.0]), np.array([0.0, -6.0]), np.array([4.0, 9.0]), settings,
    )
    assert residual == 2.0
    assert threshold == 1.4


@pytest.mark.parametrize("aphi,bloat", [([2, 0], 1), ([1, .1], 1), ([1, 0], 2)])
def test_unsupported_radial_mapping(aphi, bloat):
    with pytest.raises(ValueError, match="APHI"):
        m.validate_mapping(SimpleNamespace(aphi=aphi, bloat=bloat), wout_fixture())


def test_invalid_wout_grid():
    inp = SimpleNamespace(aphi=[1, 0], bloat=1)
    for wout in (wout_fixture(ns=4), wout_fixture(phi=0)):
        with pytest.raises(ValueError):
            m.validate_mapping(inp, wout)
    wout = wout_fixture()
    wout.phi[3] += .1
    with pytest.raises(ValueError, match="mesh"):
        m.validate_mapping(inp, wout)


def write_current(path, wout_path, values):
    with h5py.File(path, "w") as f:
        g = f.create_group("current")
        s = np.linspace(.1, 1, len(values))
        for key, data in dict(s=s, rho=np.sqrt(s), JdotB=values, B2=np.ones(len(s)), J=values).items():
            g[key] = data
        g.attrs.update(contract_version=1, wout_sha256=artifact(wout_path)["sha256"], **CURRENT_NORMALIZATION)
        for key, unit in CURRENT_UNITS.items():
            g.attrs[f"{key}_units"] = unit


def test_read_current_contract_and_provenance(tmp_path):
    wout = tmp_path / "wout.nc"
    wout.write_bytes(b"equilibrium")
    path = tmp_path / "current.h5"
    write_current(path, wout, [1, -2, 3])
    _, jdotb, b2 = m.read_current(path, wout)
    assert_allclose(jdotb, [1, -2, 3])
    with h5py.File(path, "a") as f:
        f["current"].attrs["reference_B_T"] = 2.0
    with pytest.raises(ValueError, match="normalization"):
        m.read_current(path, wout)
    write_current(path, wout, [1, -2, 3])
    wout.write_bytes(b"different equilibrium")
    with pytest.raises(ValueError, match="provenance"):
        m.read_current(path, wout)


@pytest.mark.parametrize("surfaces,message", [([1.0], "at least two"), ([.2, .98], "Edge extrapolation")])
def test_read_current_requires_two_samples_and_measured_edge(tmp_path, surfaces, message):
    wout = tmp_path / "wout.nc"
    wout.write_bytes(b"equilibrium")
    path = tmp_path / "current.h5"
    write_current(path, wout, np.ones(len(surfaces)))
    with h5py.File(path, "a") as h5:
        h5["current/s"][:] = surfaces
        h5["current/rho"][:] = np.sqrt(surfaces)
    with pytest.raises(ValueError, match=message):
        m.read_current(path, wout)


def adapter_case(tmp_path, monkeypatch, *, prediction=3.0):
    paths = {name: tmp_path / name for name in ("aggregate", "wout", "applied_input", "candidate", "record", "boozer")}
    paths["wout"].write_bytes(b"wout")
    paths["boozer"].write_bytes(b"boozer")
    paths["applied_input"].write_text("&INDATA NCURR=0, AI=0.4, AM=1, PHIEDGE=2 /\n")
    write_current(paths["aggregate"], paths["wout"], np.full(4, prediction))
    inputs = {}
    for name in ("common_config", "dkx_config", "effective_config"):
        file = tmp_path / name
        file.write_text(name)
        inputs[name] = str(file)
    settings_path = tmp_path / "settings.json"
    settings = resolve_settings({"current_density_feedback": {"enabled": True, "relaxation": .25}})
    settings_path.write_text(json.dumps({"settings": settings, "inputs": inputs}))
    paths["settings_path"] = settings_path
    wout = wout_fixture()
    inp = SimpleNamespace(aphi=[1.0], bloat=1.0, ncurr=0)
    helpers = SimpleNamespace(read_wout=lambda path: wout,
                              input_type=SimpleNamespace(from_file=lambda path: inp),
                              geometry=lambda w, s: SimpleNamespace(fsa_B2=np.ones_like(s)),
                              jdotb=lambda w, s, geom: np.full_like(s, 3.0))
    monkeypatch.setattr(m, "load_vmex_helpers", lambda: helpers)
    return paths


def test_already_converged_copies_applied_input_without_rewrite(tmp_path, monkeypatch):
    paths = adapter_case(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("unnecessary rewrite")
    monkeypatch.setattr(m, "invert_current", forbidden)
    result = m.run_adapter(**paths)
    assert result["status"] == "converged"
    assert result["residual"] == 0
    assert paths["candidate"].read_bytes() == paths["applied_input"].read_bytes()
    assert "NCURR=0" in paths["candidate"].read_text()
    assert set(result["artifacts"]) >= {"aggregate", "wout", "boozer", "applied_input", "candidate_input", "context"}


def test_underrelax_against_realized_buco_even_ncurr_zero(tmp_path, monkeypatch):
    paths = adapter_case(tmp_path, monkeypatch, prediction=12.0)
    monkeypatch.setattr(m, "invert_current", lambda w, s, j, h: (np.array([0, 1]), np.array([0, 11.])))
    exported = {}
    def export(path, w, s, current, h):
        exported.update(s=s, current=current)
        return {}
    monkeypatch.setattr(m, "validate_export", export)
    result = m.run_adapter(**paths)
    assert result["status"] == "continue"
    assert_allclose(exported["current"], 5 * exported["s"])
    text = paths["candidate"].read_text()
    assert "NCURR = 1" in text
    assert "AI=0.4" in text and "AM=1" in text and "PHIEDGE=2" in text


def test_invalid_export_preserves_existing_candidate_and_record(tmp_path, monkeypatch):
    paths = adapter_case(tmp_path, monkeypatch, prediction=12.0)
    paths["candidate"].write_text("accepted candidate")
    paths["record"].write_text("accepted record")
    monkeypatch.setattr(m, "invert_current", lambda w, s, j, h: (np.array([0, 1]), np.array([0, 11.])))
    def invalid(*args):
        raise ValueError("invalid export")
    monkeypatch.setattr(m, "validate_export", invalid)
    with pytest.raises(ValueError, match="invalid export"):
        m.run_adapter(**paths)
    assert paths["candidate"].read_text() == "accepted candidate"
    assert paths["record"].read_text() == "accepted record"


@pytest.mark.parametrize("output", ["candidate", "record"])
@pytest.mark.parametrize("source", ["aggregate", "wout", "boozer", "settings_path", "applied_input", "common_config"])
def test_output_cannot_overwrite_any_source(tmp_path, monkeypatch, output, source):
    paths = adapter_case(tmp_path, monkeypatch)
    path = tmp_path / "common_config" if source == "common_config" else paths[source]
    original = path.read_bytes()
    paths[output] = path
    with pytest.raises(ValueError, match="distinct"):
        m.run_adapter(**paths)
    assert path.read_bytes() == original


@pytest.mark.parametrize("inputs", [{}, [], None])
def test_incomplete_context_preserves_outputs(tmp_path, monkeypatch, inputs):
    paths = adapter_case(tmp_path, monkeypatch)
    context = json.loads(paths["settings_path"].read_text())
    context["inputs"] = inputs
    paths["settings_path"].write_text(json.dumps(context))
    paths["candidate"].write_text("accepted candidate")
    paths["record"].write_text("accepted record")
    with pytest.raises(ValueError, match="context"):
        m.run_adapter(**paths)
    assert paths["candidate"].read_text() == "accepted candidate"
    assert paths["record"].read_text() == "accepted record"


def test_equilibrium_sign_error_rejected_even_for_small_current(monkeypatch):
    wout = wout_fixture()
    inp = SimpleNamespace(ncurr=1, pcurr_type="akima_spline_i")
    half = np.array([.2, .5])
    monkeypatch.setattr(m, "serialized_current", lambda *args: (half, np.array([-1e-12, -1e-12])))
    with pytest.raises(ValueError, match="opposite current sign"):
        m.validate_applied_equilibrium(inp, wout, None)


def test_feedback_profile_is_checked_at_new_outer_boundary(tmp_path, monkeypatch):
    paths = adapter_case(tmp_path, monkeypatch)
    helpers = m.load_vmex_helpers()
    helpers.input_type.from_file = lambda path: SimpleNamespace(aphi=[1], bloat=1, ncurr=1, pcurr_type="akima_spline_i")
    def fail(*args):
        raise ValueError("outer boundary sign mismatch")
    monkeypatch.setattr(m, "validate_applied_equilibrium", fail)
    with pytest.raises(ValueError, match="outer boundary sign"):
        m.run_adapter(**paths)


@pytest.mark.parametrize("umask,existing_mode,expected", [(0o022, None, 0o644), (0o007, None, 0o660),
                                                         (0o077, 0o640, 0o640)])
def test_adapter_outputs_use_umask_and_preserve_existing_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, umask: int, existing_mode: int | None, expected: int,
) -> None:
    paths = adapter_case(tmp_path, monkeypatch)
    if existing_mode is not None:
        for key in ("candidate", "record"):
            paths[key].write_text("previous output")
            paths[key].chmod(existing_mode)
    original_umask = os.umask(umask)
    try:
        m.run_adapter(**paths)
    finally:
        os.umask(original_umask)
    for key in ("candidate", "record"):
        assert stat.S_IMODE(paths[key].stat().st_mode) == expected


def test_measured_edge_roundoff_is_preserved_in_record_and_normalized_only_for_export(tmp_path, monkeypatch):
    paths = adapter_case(tmp_path, monkeypatch, prediction=12.0)
    edge = np.nextafter(1.0, 0.0)
    with h5py.File(paths["aggregate"], "a") as h5:
        h5["current/s"][-1] = edge
        h5["current/rho"][-1] = np.sqrt(edge)
    monkeypatch.setattr(m, "invert_current", lambda *args: (np.array([0., 1.]), np.array([0., 11.])))
    exported = {}
    def export(path, wout, s, current, helpers):
        exported["s"] = s
        return {}
    monkeypatch.setattr(m, "validate_export", export)
    result = m.run_adapter(**paths)
    assert result["measured_s"][-1] == edge
    assert exported["s"][-1] == 1.0
