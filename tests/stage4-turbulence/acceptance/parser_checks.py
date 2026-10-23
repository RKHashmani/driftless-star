"""Load shipped templates and generated runtime TOMLs with pinned GKX and real geometry.

Run explicitly in Stage 4 with PYTHONPATH=stages:stages/stage4-turbulence.
"""
import argparse
import json
from pathlib import Path

import gkx_radial_scan as scan
from solver_provenance import current_provenance
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.workflows.runtime.startup import build_runtime_geometry, build_runtime_linear_params


def _prepare_inputs(out: Path, vmec: Path, template: Path) -> dict:
    common = out / "common.toml"
    common.write_text('''[species]
names = ["e", "D", "T"]
charge_qp = [-1.0, 1.0, 1.0]
mass_mp = [0.000544617, 2.0, 3.0]
[geometry]
n_radial = 2
[profiles]
model = "prescribed"
density = [[1e19, 1e19], [5e18, 5e18], [5e18, 5e18]]
temperature = [[1000.0, 1000.0], [1000.0, 1000.0], [1000.0, 1000.0]]
Er = [0.0, 0.0]
density_face = [[1e19, 1e19, 1e19], [5e18, 5e18, 5e18], [5e18, 5e18, 5e18]]
temperature_face = [[1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0]]
Er_face = [0.0, 0.0, 0.0]
density_grad_face = [[-1e19, -1e19, -1e19], [-5e18, -5e18, -5e18], [-5e18, -5e18, -5e18]]
temperature_grad_face = [[-1000.0, -1000.0, -1000.0], [-1000.0, -1000.0, -1000.0], [-1000.0, -1000.0, -1000.0]]
''')
    prepare = scan.build_parser().parse_args(["prepare", "--common-config", str(common),
        "--gkx-template", str(template), "--profiles-source", "prescribed",
        "--vmec-file-override", str(vmec.resolve()),
        "--rho-star-physical", "0.01", "--rho-indices", "1", "--output-dir", str(out / "prepared"),
        "--response-mode", "fd_gradients", "--perturb-density-species", "D", "--perturb-temperature-species", "D",
        "--nx", "4", "--ny", "4", "--nz", "8", "--ntheta", "8", "--nl", "2", "--nm", "4",
        "--t-max", "0.00001", "--dt", "0.00001", "--fixed-dt", "--no-resolved-diagnostics"])
    if scan.cmd_prepare(prepare):
        raise RuntimeError("Preparation failed")
    return json.loads((out / "prepared/manifest.json").read_text())


def main() -> None:
    """Check template parsing and prepared baseline and perturbation inputs.

    Raises
    ------
    RuntimeError
        Preparation fails or a generated runtime TOML has an unresolved VMEC path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = current_provenance()
    templates = [
        root / "inputs/quick_run/HSX_vacuum_ns201_quickrun.toml",
        *sorted((root / "inputs/w7-x").glob("*/w7x_t3d_reconstruction.toml")),
    ]
    for template in templates:
        load_runtime_from_toml(template)
    manifest = _prepare_inputs(out, args.vmec, templates[0])
    for run in manifest["runs"]:
        config, _ = load_runtime_from_toml(run["config_path"])
        if not Path(config.geometry.vmec_file).is_file():
            raise RuntimeError("Unresolved VMEC path")
        geom = build_runtime_geometry(config)
        build_runtime_linear_params(config, Nm=4, geom=geom)
    report = dict(
        provenance=provenance, templates=len(templates),
        generated_runtime_tomls=len(manifest["runs"]), real_geometry=True,
    )
    (out / "parser_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
