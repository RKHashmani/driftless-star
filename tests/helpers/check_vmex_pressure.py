"""Check exported spline inputs with pinned VMEX. Run inside the Stage 1 image."""

import importlib.metadata
import json
from pathlib import Path
import tempfile

import jax
import numpy as np
from numpy.testing import assert_allclose

from tests.helpers.stage_import import load_stage_module
from vmex.core.input import VmecInput
from vmex.core.profiles import pressure


def main():
    jax.config.update("jax_enable_x64", True)
    pin = "35e7170a24ab33fce4d4d25dc7f448279c279e7f"
    provenance = json.loads(importlib.metadata.distribution("vmex").read_text("direct_url.json"))
    assert provenance["vcs_info"]["commit_id"] == pin, provenance
    writer = load_stage_module("stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py")
    root = Path(__file__).resolve().parents[2]
    template = root / "inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun"
    representative_s = np.array([.02, .5, .9])
    # These fixed results come from the pinned VMEX revision.
    # They use NEOPAX pressure units.
    representative_values = {
        "akima_spline": np.array([9.975, 3.2661224489795937, .7085754628129912]),
        "cubic_spline": np.array([9.928105355799968, 3.3392846300900842, .40619267892573296]),
    }
    checked = []
    with tempfile.TemporaryDirectory() as directory:
        for kind in ("akima_spline", "cubic_spline"):
            for name, rho, native_values, expected in (
                ("smooth", np.linspace(0, 1, 6), np.array([10., 9.8, 8., 5., 2., .1]),
                 representative_values[kind]),
                ("two_knots", np.array([0., 1.]), np.array([10., .1]),
                 10. - 9.9 * representative_s),
            ):
                values = native_values * 16021.76634
                target = Path(directory) / "input.spline"
                writer._write_vmec_input_with_pressure_spline(template, rho, values, profile_type=kind, output_path=target)
                inp = VmecInput.from_file(target)
                assert inp.pmass_type == kind
                assert inp.pres_scale == 1.
                # VMEX pads the arrays to 101 slots. Use the valid knot prefix.
                valid = np.asarray(inp.am_aux_s) >= 0.
                knots, pressures = np.asarray(inp.am_aux_s)[valid], np.asarray(inp.am_aux_f)[valid]
                assert_allclose(knots, rho**2, rtol=0, atol=0)
                assert_allclose(pressures, values, rtol=0, atol=0)
                evaluate = lambda s: pressure(inp.pmass_type, inp.am, knots, pressures, s, pres_scale=inp.pres_scale)
                assert_allclose(evaluate(knots), values, rtol=2e-13, atol=1e-8)
                assert_allclose(evaluate(representative_s), expected * 16021.76634, rtol=2e-13, atol=1e-8)
                checked.append(dict(profile_type=kind, sample=name, knot_count=len(knots)))
    print(json.dumps(dict(vmex_commit=pin, checks=checked), indent=2))


if __name__ == "__main__":
    main()
