"""Check VMEC-route DKX signs and SI normalization against pinned VMEX.

Both solver packages must be installed at their repository revisions. Redl is
compared with the kinetic current on a shared axisymmetric equilibrium, using
identical hydrogen/electron profiles and zero electric field. The equilibrium's
own current is not a reference for the kinetic prediction.
"""

import importlib.metadata
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np
from numpy.testing import assert_allclose
import pytest

from common.current_density_feedback_contract import artifact
from common.vmec_namelist import rewrite_current_profile
from tests.helpers.stage_import import load_stage_module

pytestmark = pytest.mark.stage_env
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def solvers(monkeypatch):
    features = tomllib.loads((ROOT / "stages/pixi.toml").read_text())["feature"]
    for name in ("dkx", "vmex"):
        try:
            package = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("This cross-solver check requires both pinned DKX and VMEX")
        provenance = json.loads(package.read_text("direct_url.json") or "{}")
        expected = features[name]["pypi-dependencies"][name]["rev"]
        assert provenance.get("vcs_info", {}).get("commit_id") == expected
    monkeypatch.setenv("DKX_CORES", "2")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import dkx
    dkx.configure()
    adapter = load_stage_module("stages/stage1-equilibrium/current_density_feedback.py")
    helpers = adapter.load_vmex_helpers()
    return dkx, adapter, helpers


@pytest.mark.parametrize("flux_sign", [-1, 1])
def test_vmec_route_current_matches_redl_and_drives_vmex(tmp_path, solvers, flux_sign):
    dkx, adapter, helpers = solvers
    from vmex.core.bootstrap import KineticProfiles, j_dot_B_redl

    runner = load_stage_module("stages/stage1-equilibrium/run_vmex.py")
    scan = load_stage_module("stages/stage3-neoclassical/dkx_radial_scan.py")
    applied, wout_path = tmp_path / "input.initial", tmp_path / "wout.initial.nc"
    # A circular axisymmetric surface is within Redl's intended geometry. B is
    # about 2.5 T, so a missing magnetic normalization cannot pass this check.
    applied.write_text(
        "&INDATA\n NFP=1, MPOL=3, NTOR=0, NS_ARRAY=9, NITER_ARRAY=1000, FTOL_ARRAY=1e-10,\n"
        f" NCURR=0, AI=0.4, AM=0, PHIEDGE={flux_sign * 2.5 * np.pi * .1**2:.17g},\n"
        " RBC(0,0)=1, RBC(0,1)=0.1, ZBS(0,1)=0.1,\n/\n"
    )
    assert runner.run_vmex(applied, wout_path, "cpu") == 0
    wout = helpers.read_wout(wout_path)
    surfaces = np.array([.3, .7, 1.0])
    profiles = KineticProfiles(
        ne_coeffs=np.array([1e19, -.05e19]),
        Te_coeffs=np.array([2e3, -.1e3]), Ti_coeffs=np.array([2e3, -.1e3]),
    )
    geometry = helpers.geometry(wout, surfaces)
    redl, _ = j_dot_B_redl(profiles, geometry, helicity_n=0)
    measured = []
    for index, s in enumerate(surfaces):
        native_path = tmp_path / f"dkx_{index}.h5"
        run = dkx.run(
            out=native_path, geometryScheme=5, equilibriumFile=str(wout_path),
            VMECRadialOption=0, inputRadialCoordinate=1, psiN_wish=float(s),
            inputRadialCoordinateForGradients=1, Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4],
            nHats=[.1 * (1 - .05 * s)] * 2, THats=[2.0 * (1 - .05 * s)] * 2,
            dNHatdpsiNs=[-.005] * 2, dTHatdpsiNs=[-.1] * 2,
            Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=0.0, collisionOperator=0,
            includePhi1=False, Ntheta=17, Nzeta=1, Nxi=16, NL=4, Nx=5,
            solverTolerance=1e-8, emit=None,
        )
        assert run.solve_result.converged
        native = dkx.read_output(native_path)
        measured.append(scan._extract_current(native, 2))
        # Geometry scheme 5 keeps VMEC's signed flux and Jacobian. The separate
        # Boozer-route handedness adjustment must not be applied a second time.
        assert_allclose(np.asarray(native["psiAHat"]).item(), wout.phi[-1] / (2 * np.pi), rtol=1e-14)
    jdotb = np.array([row["JdotB"] for row in measured])
    b2 = np.array([row["B2"] for row in measured])
    assert_allclose(b2, geometry.fsa_B2, rtol=1e-3)
    assert np.all(b2 > 5.0)
    assert np.all(np.sign(jdotb[:2]) == np.sign(np.asarray(redl)[:2]))
    # This modest angular/velocity grid checks the convention and scale. A
    # 50% bound allows kinetic discretization and fit error, while rejecting a
    # sign reversal or omitted 2*pi or magnetic normalization factor.
    ratio = jdotb[:2] / np.asarray(redl)[:2]
    assert_allclose(ratio, 1.0, rtol=.5)

    aggregate = tmp_path / "current.h5"
    with h5py.File(aggregate, "w") as stream:
        group = stream.create_group("current")
        group.attrs.update(measured[0]["meta"], wout_sha256=artifact(wout_path)["sha256"])
        for name in ("s", "rho", "JdotB", "B2", "J"):
            group[name] = [row[name] for row in measured]
    s, target_jdotb, _ = adapter.read_current(aggregate, wout_path)
    full, target = adapter.invert_current(wout, s, target_jdotb, helpers)
    assert np.sign(target[-1]) == np.sign(wout.phi[-1] * target_jdotb[-1])
    knots = np.r_[0.0, s]
    relaxed = .5 * adapter.enclosed_from_wout(wout, full) + .5 * target
    enclosed = np.interp(knots, full, relaxed)
    candidate, updated_path = tmp_path / "input.candidate", tmp_path / "wout.candidate.nc"
    candidate.write_text(rewrite_current_profile(applied.read_text(), knots, enclosed))
    adapter.validate_export(candidate, wout, knots, enclosed, helpers)
    assert runner.run_vmex(candidate, updated_path, "cpu") == 0
    adapter.validate_applied_equilibrium(
        helpers.input_type.from_file(candidate), helpers.read_wout(updated_path), helpers,
    )
    print(f"flux_sign={flux_sign}, DKX/Redl={ratio.tolist()}, target_edge_A={target[-1]:.8g}")
