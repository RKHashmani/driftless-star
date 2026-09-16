"""Explore spline derivatives and overshoot with pinned VMEX. Run inside the Stage 1 image."""

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from vmex.core.input import VmecInput
from vmex.core.profiles import pressure


def main():
    jax.config.update("jax_enable_x64", True)
    pin = "35e7170a24ab33fce4d4d25dc7f448279c279e7f"
    provenance = json.loads(importlib.metadata.distribution("vmex").read_text("direct_url.json"))
    assert provenance["vcs_info"]["commit_id"] == pin, provenance
    root = Path(__file__).resolve().parents[1]
    writer_path = root / "stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py"
    spec = importlib.util.spec_from_file_location("pressure_writer", writer_path)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    template = root / "inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun"
    reports = []
    with tempfile.TemporaryDirectory() as directory:
        for kind in ("akima_spline", "cubic_spline"):
            for name, rho, values in (
                ("smooth", np.linspace(0, 1, 6), np.array([10., 9.8, 8., 5., 2., .1])),
                ("sharp", np.linspace(0, 1, 6), np.array([10., 10., 10., .1, .1, .1])),
                ("two_knots", np.array([0., 1.]), np.array([10., .1])),
            ):
                values = values * 16021.76634
                target = Path(directory) / "input.spline"
                writer._write_vmec_input_with_pressure_spline(template, rho, values, profile_type=kind, output_path=target)
                inp = VmecInput.from_file(target)
                assert inp.pmass_type == kind
                assert inp.pres_scale == 1.
                # VMEX pads its input arrays to 101 slots. The evaluator uses
                # the valid knot prefix, as the solver does.
                knots = np.asarray(inp.am_aux_s)
                valid = knots >= 0.
                knots, pressures = knots[valid], np.asarray(inp.am_aux_f)[valid]
                assert_allclose(knots, rho**2, rtol=0, atol=0)
                assert_allclose(pressures, values, rtol=0, atol=0)
                evaluate = lambda s: pressure(inp.pmass_type, inp.am, knots, pressures, s, pres_scale=inp.pres_scale)
                assert_allclose(evaluate(jnp.asarray(knots)), values, rtol=2e-13, atol=1e-8)
                interior = np.concatenate([np.linspace(a, b, 51)[1:-1] for a, b in zip(knots[:-1], knots[1:])])
                sampled = np.asarray(evaluate(jnp.asarray(interior)))
                gradients = np.asarray(jax.vmap(jax.grad(evaluate))(jnp.asarray(interior)))
                assert np.all(np.isfinite(sampled)) and np.all(np.isfinite(gradients))
                delta = 1e-6
                numerical = (np.asarray(evaluate(interior + delta)) - np.asarray(evaluate(interior - delta))) / (2 * delta)
                assert_allclose(gradients, numerical, rtol=1e-5, atol=2e-3)
                lower = np.repeat(np.minimum(values[:-1], values[1:]), 49)
                upper = np.repeat(np.maximum(values[:-1], values[1:]), 49)
                overshoot = np.maximum(np.maximum(lower - sampled, sampled - upper), 0.)
                reports.append(dict(profile_type=kind, sample=name, knot_count=len(knots),
                                    minimum_pa=float(sampled.min()), maximum_pa=float(sampled.max()),
                                    interval_overshoot_pa=float(overshoot.max()),
                                    minimum_dp_ds=float(gradients.min()), maximum_dp_ds=float(gradients.max())))
    print(json.dumps(dict(vmex_commit=pin, checks=reports), indent=2))


if __name__ == "__main__":
    main()
