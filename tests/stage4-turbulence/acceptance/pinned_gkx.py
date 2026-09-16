"""Check profile parameters with three one-step CPU runs of pinned GKX.

From the repository root, set GKX_SOURCE to a checkout of PINNED_REVISION,
STAGE4_PYTHON to a Python interpreter with Stage 4 dependencies,
STAGE4_MANIFEST to a prepared manifest, and STAGE4_VMEC to a real equilibrium.

    export MPLCONFIGDIR=/tmp/stage4-mpl
    export JAX_PLATFORMS=cpu
    export PYTHONPATH="${GKX_SOURCE:?}/src:$PWD/stages:$PWD/stages/stage4-turbulence"
    "${STAGE4_PYTHON:?}" tests/stage4-turbulence/acceptance/pinned_gkx.py --manifest "${STAGE4_MANIFEST:?}" --vmec "${STAGE4_VMEC:?}" --out /tmp/stage4-pinned-repro

Checks remain active with Python -O. Each solver run has a 180-second limit.
Cases use one kinetic deuterium species and adiabatic electrons. They check
fixed inputs, profile collisions, and profile beta with electromagnetic fields.
They do not establish converged transport or kinetic-electron behavior.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np

PINNED_REVISION = "fb5599745573773d422b6cdae914d8f8435ef0d2"


class StartupGeometry:
    """Supply only the geometry scalar needed for parameter assertions.

    The solver subprocess separately builds the complete real VMEC geometry.
    """

    def gradpar(self) -> float:
        return 1.0


def require(condition: bool, message: str) -> None:
    """Raise an acceptance failure even when Python optimization is enabled."""
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vmec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import gkx
    import gkx_radial_scan as scan
    import profile_parameters as profiles
    from gkx.workflows.runtime.toml import load_runtime_from_toml
    from gkx.workflows.runtime.startup import (
        build_runtime_linear_params,
        build_runtime_linear_terms,
    )

    source_root = Path(gkx.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    require(revision == PINNED_REVISION, f"Expected GKX {PINNED_REVISION}, imported {revision} from {source_root}")
    print("Verified pinned GKX source", revision, flush=True)

    manifest = json.loads(args.manifest.read_text())
    vmec_file = args.vmec.resolve()
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["vmec_file"] = str(vmec_file)
    manifest["grid"].update(Nx=4, Ny=4, Nz=8, ntheta=8, nperiod=1)
    manifest["time"].update(
        t_max=0.00001, dt=0.00001, chunk_steps=1, sample_stride=1,
        diagnostics_stride=1, fixed_dt=True, state_sharding=None,
    )
    manifest["run"].update(Nl=2, Nm=4)
    manifest["physics"].update(adiabatic_electrons=True, collisions=True)
    manifest["terms"].update(collisions=1.0)
    manifest["output"]["resolved_diagnostics"] = False
    base_run = copy.deepcopy(manifest["runs"][0])
    base_run["torflux"] = 0.25
    base_run["runtime_species"] = [dict(
        name="ion", charge=1.0, mass=2.0, density=1.0, temperature=1.0,
        tprim=1.0, fprim=1.0, nu=0.01,
    )]
    length, field = profiles.geometry_scales(vmec_file, need_field=True)
    beta = profiles.reference_beta(1.0, 1.0, field, "acceptance")
    rate, _ = profiles.self_collision(1.0, 1.0, 2.0, 1.0, context="acceptance")
    collision_frequency = rate * length / profiles.reference_speed(1.0, "acceptance")
    results = []

    for name in ["fixed", "profile_collisions", "profile_em"]:
        case_dir = output_dir / name
        case_dir.mkdir(exist_ok=True)
        case_manifest = copy.deepcopy(manifest)
        case_run = copy.deepcopy(base_run)
        electromagnetic = name == "profile_em"
        case_run["geometry_file_toml"] = str(case_dir / "geometry.eik.nc")
        case_run["beta"] = beta if electromagnetic else 0.0
        case_run["runtime_species"][0]["nu"] = (
            0.01 if name == "fixed" else collision_frequency
        )
        case_manifest["physics"].update(
            electromagnetic=electromagnetic,
            use_apar=electromagnetic,
            use_bpar=electromagnetic,
        )
        case_manifest["terms"].update(apar=1.0, bpar=1.0)
        config_path = case_dir / "runtime.toml"
        config_path.write_text(scan._runtime_toml_text(case_manifest, case_run))
        config, _ = load_runtime_from_toml(config_path)
        params = build_runtime_linear_params(config, Nm=4, geom=StartupGeometry())
        terms = build_runtime_linear_terms(config)
        require(config.species[0].nu == case_run["runtime_species"][0]["nu"], f"{name} TOML collision frequency")
        require(math.isclose(float(params.nu[0]), config.species[0].nu, rel_tol=1e-6), f"{name} runtime frequency")
        require(float(params.beta) == case_run["beta"], f"{name} runtime beta")
        require(terms.apar == terms.bpar == float(electromagnetic), f"{name} electromagnetic terms")
        require(terms.collisions == 1.0, f"{name} collision term")

        disabled = replace(
            config,
            physics=replace(config.physics, electromagnetic=False, collisions=False),
        )
        disabled_params = build_runtime_linear_params(
            disabled, Nm=4, geom=StartupGeometry()
        )
        disabled_terms = build_runtime_linear_terms(disabled)
        require(disabled_params.beta == 0, f"{name} disabled beta")
        require(disabled_terms.collisions == 0, f"{name} disabled collision term")
        require(disabled_terms.apar == disabled_terms.bpar == 0, f"{name} disabled electromagnetic terms")

        disabled_manifest = copy.deepcopy(case_manifest)
        disabled_manifest["physics"].update(electromagnetic=False, collisions=False)
        zero_run = copy.deepcopy(case_run)
        for species in zero_run["runtime_species"]:
            species["nu"] = 0.0
        zero_config = replace(config, species=tuple(replace(species, nu=0.0) for species in config.species))
        for audit_manifest, audit_run, solver_config in (
            (case_manifest, case_run, config),
            (disabled_manifest, case_run, disabled),
            (case_manifest, zero_run, zero_config),
        ):
            solver_terms = build_runtime_linear_terms(solver_config)
            for term in ("apar", "bpar", "collisions"):
                require(
                    scan._effective_term(audit_manifest, audit_run, term) == getattr(solver_terms, term),
                    f"{name} audit {term} must match pinned build_runtime_linear_terms",
                )

        command = [
            sys.executable, "-m", "gkx.cli", "run", "--config", str(config_path),
            "--out", str(case_dir / "smoke"), "--steps", "1", "--no-progress",
        ]
        try:
            process = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180,
            )
            (case_dir / "solver.log").write_text(process.stdout)
            result = dict(
                case=name, returncode=process.returncode, nu=config.species[0].nu,
                beta=config.physics.beta, startup_checks=True,
            )
        except subprocess.TimeoutExpired as error:
            (case_dir / "solver.log").write_bytes(error.stdout or b"")
            result = dict(case=name, timeout=180, startup_checks=True)
        results.append(result)
        print(json.dumps(result), flush=True)

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    for result in results:
        require(result.get("returncode") == 0, f"Solver failed with {result}")
        diagnostics = np.genfromtxt(
            output_dir / result["case"] / "smoke.diagnostics.csv",
            delimiter=",", names=True,
        )
        require(
            all(np.isfinite(diagnostics[name]).all() for name in diagnostics.dtype.names),
            f"Non-finite diagnostics for {result['case']}",
        )
    print("All three bounded CPU cases passed with finite diagnostics.")


if __name__ == "__main__":
    main()
