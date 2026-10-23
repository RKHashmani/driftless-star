"""Current inversion and serialized profiles evaluated with the pinned VMEX."""

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
from numpy.testing import assert_allclose
import pytest

from common.vmec_namelist import rewrite_current_profile
from tests.helpers.stage_import import load_stage_module

pytestmark = pytest.mark.stage_env
m = load_stage_module("stages/stage1-equilibrium/current_density_feedback.py")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def helpers():
    h = m.load_vmex_helpers()
    provenance = json.loads(importlib.metadata.distribution("vmex").read_text("direct_url.json"))
    pin = tomllib.loads((ROOT / "stages/pixi.toml").read_text())["feature"]["vmex"]["pypi-dependencies"]["vmex"]["rev"]
    assert provenance["vcs_info"]["commit_id"] == pin
    return h


def synthetic_wout(ns=33, *, signgs=-1, phi=-2.0, quadratic=0.0, pressure_slope=0.0):
    full = np.linspace(0.0, 1.0, ns)
    half = (np.arange(1, ns) - .5) / (ns - 1)
    current = 3 * half + quadratic * half**2
    w = SimpleNamespace(
        ns=ns, phi=phi * full, signgs=signgs, nfp=1, lasym=False,
        pres=np.r_[0., 1e5 + pressure_slope * half],
        buco=np.r_[0., signgs * m.MU0 / (2 * np.pi) * current],
        bvco=np.full(ns, 5.), iotas=np.full(ns, .4), xm_nyq=np.array([0.]),
        xn_nyq=np.array([0.]), bmnc=np.full((ns, 1), 2.), gmnc=np.ones((ns, 1)),
    )
    w.ctor = signgs * 2 * np.pi / m.MU0 * (1.5 * w.buco[-1] - .5 * w.buco[-2])
    return w


def test_private_collocation_helper_and_precision(helpers):
    import jax
    assert jax.config.jax_enable_x64
    assert helpers.dds_matrix.__module__ == "vmex.core.bootstrap"
    s = np.linspace(0, 1, 9)
    dds = helpers.dds_matrix(len(s), s[1] - s[0])
    for power in range(1, 5):
        assert_allclose(dds @ s**power, power * s**(power - 1), atol=2e-14)


def test_missing_private_helper_fails_at_import_boundary(helpers, monkeypatch):
    import vmex.core.bootstrap as bootstrap
    monkeypatch.delattr(bootstrap, "_picard_dds_matrix")
    with pytest.raises(RuntimeError, match="pinned Stage 1 VMEX"):
        m.load_vmex_helpers()


@pytest.mark.parametrize("phi", [-2.0, 2.0])
@pytest.mark.parametrize("signgs", [-1, 1])
def test_wout_identity_recovers_enclosed_current_and_ctor(helpers, phi, signgs):
    w = synthetic_wout(phi=phi, signgs=signgs, pressure_slope=-1e5)
    samples = np.linspace(.04, 1, 30)
    geom = helpers.geometry(w, samples)
    jdotb = np.asarray(helpers.jdotb(w, samples, geom=geom))
    s, current = m.invert_current(w, samples, jdotb, helpers)
    assert_allclose(current, 3 * s, rtol=2e-11, atol=1e-11)
    assert_allclose(current[-1], w.ctor, rtol=2e-11)
    assert_allclose(geom.fsa_B2, 4.0)


def test_wout_quadratic_closure_improves_with_radial_refinement(helpers):
    errors, edge_errors = [], []
    for ns in (17, 33, 65):
        w = synthetic_wout(ns, quadratic=2.0, pressure_slope=-1e5)
        samples = np.linspace(0, 1, ns)
        jdotb = np.asarray(helpers.jdotb(w, samples))
        s, current = m.invert_current(w, samples, jdotb, helpers)
        errors.append(float(np.max(np.abs(current - (3 * s + 2 * s**2)))))
        edge_errors.append(abs(w.ctor - 5.0))
    assert errors[1] < .6 * errors[0]
    assert errors[2] < .6 * errors[1]
    assert errors[-1] < .002
    assert_allclose(np.array(edge_errors[:-1]) / edge_errors[1:], 4.0, rtol=1e-9)


@pytest.mark.parametrize("values", [
    [0, 1, 2, 3, 4], [0, -1, -2, -3, -4], [0, -1, 1, 2, 1],
    [0, 1e-14, 2e-14, 3e-14, 4e-14], [0, 1, 2, 1, 1e-9], [0, 0, 0, 0, 0],
])
def test_serialized_current_profiles(helpers, tmp_path, values):
    s = np.array([0, .1, .4, .7, 1.0])
    current = np.array(values, dtype=float)
    source = (ROOT / "inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun").read_text()
    path = tmp_path / "input.vmec"
    path.write_text(rewrite_current_profile(source, s, current))
    result = m.validate_export(path, synthetic_wout(), s, current, helpers)
    assert result["normalization_error_A"] <= result["normalization_bound_A"]
    original = helpers.input_type.from_file(ROOT / "inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun")
    changed = helpers.input_type.from_file(path)
    for name in ("am", "am_aux_s", "am_aux_f", "rbc", "zbs", "ai"):
        assert_allclose(getattr(changed, name), getattr(original, name), rtol=0, atol=0)


def test_tiny_cancelling_edge_is_rejected_if_vmex_cannot_represent_it(helpers, tmp_path):
    s = np.array([0, .1, .4, .7, 1.0])
    current = np.array([0, 1, 2, 1, 1e-18])
    path = tmp_path / "input.vmec"
    path.write_text(rewrite_current_profile("&INDATA PHIEDGE=2 /", s, current))
    with pytest.raises(ValueError, match="normalization"):
        m.validate_export(path, synthetic_wout(), s, current, helpers)


def test_feedback_equilibrium_validation_uses_applied_profile(helpers, tmp_path):
    path = tmp_path / "input.vmec"
    s = np.linspace(0, 1, 5)
    path.write_text(rewrite_current_profile("&INDATA PHIEDGE=-2 /", s, 3 * s))
    inp = helpers.input_type.from_file(path)
    wout = synthetic_wout()
    result = m.validate_applied_equilibrium(inp, wout, helpers)
    assert result["equilibrium_error_A"] < 1e-12
    wout.buco *= -1
    with pytest.raises(ValueError, match="opposite current sign"):
        m.validate_applied_equilibrium(inp, wout, helpers)
