"""Check profile parameters with bounded solver runs of pinned GKX.

From the repository root, set STAGE4_PYTHON to a Python interpreter with Stage 4 dependencies,
STAGE4_MANIFEST to a prepared manifest, and STAGE4_VMEC to a real equilibrium.

    export MPLCONFIGDIR=/tmp/stage4-mpl
    export JAX_PLATFORMS=cpu
    export PYTHONPATH="$PWD/stages:$PWD/stages/stage4-turbulence"
    "${STAGE4_PYTHON:?}" tests/stage4-turbulence/acceptance/pinned_gkx.py --manifest "${STAGE4_MANIFEST:?}" --vmec "${STAGE4_VMEC:?}" --out /tmp/stage4-pinned-repro

Installed VCS metadata or a clean source checkout must match stages/pixi.toml.
To use a checkout, prepend its src directory to PYTHONPATH. Use a fresh output directory.
Checks remain active with Python -O. Each solver run has a 180-second limit.
Cases use deuterium or D/T with adiabatic electrons. They check fixed inputs,
profile collisions, electromagnetic fields, and explicit Landau selection.
They do not establish converged transport or kinetic-electron behavior.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np

from solver_provenance import current_provenance
from run_validity import input_fingerprint, completion_status


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
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--cases", nargs="+", choices=("fixed", "profile_collisions", "profile_em", "dt_species", "landau"),
        default=["fixed", "profile_collisions", "profile_em", "dt_species", "landau"],
    )
    args = parser.parse_args()
    import gkx_radial_scan as scan
    import profile_parameters as profiles
    from gkx.workflows.runtime.toml import load_runtime_from_toml
    from gkx.workflows.runtime.startup import (
        build_runtime_linear_params,
        build_runtime_linear_terms,
    )

    provenance = current_provenance()
    print("Verified executing GKX", provenance, flush=True)

    manifest = json.loads(args.manifest.read_text())
    vmec_file = args.vmec.resolve()
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest["vmec_file"] = str(vmec_file)
    manifest["booz_file"] = None
    manifest["schema_version"] = 4
    manifest["solver_provenance"] = provenance
    manifest["grid"].update(Nx=4, Ny=4, Nz=8, ntheta=8, nperiod=1)
    manifest["time"].update(
        t_max=0.00001, dt=0.00001, sample_stride=1, run_to="t_max",
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
        density_reference_physical=1.0, temperature_reference_physical=1.0,
    )]
    length, field = profiles.geometry_scales(vmec_file, need_field=True)
    beta = profiles.reference_beta(1.0, 1.0, field, "acceptance")
    rate, _ = profiles.self_collision(1.0, 1.0, 2.0, 1.0, context="acceptance")
    collision_frequency = rate * length / profiles.reference_speed(1.0, "acceptance")
    results = []

    for name in args.cases:
        case_dir = output_dir / name
        case_dir.mkdir(exist_ok=True)
        case_manifest = copy.deepcopy(manifest)
        case_run = copy.deepcopy(base_run)
        electromagnetic = name == "profile_em"
        case_run.update(index=0, run_dir=str(case_dir), output_prefix=str(case_dir / "smoke"),
                        config_path=str(case_dir / "runtime.toml"), geometry_file=str(case_dir / "geometry.eik.nc"),
                        geometry_file_toml=str(case_dir / "geometry.eik.nc"), response_label="base")
        case_manifest["time"]["collision_operator"] = "landau" if name == "landau" else "lenard_bernstein"
        if name == "dt_species":
            case_run["runtime_species"][0]["name"] = "D"
            case_run["runtime_species"].append(dict(case_run["runtime_species"][0], name="T", mass=3.0))
        case_manifest["runtime_species_names"] = [sp["name"] for sp in case_run["runtime_species"]]
        case_manifest["species_meta"] = [
            dict(name=sp["name"], mass_mp=sp["mass"], charge=sp["charge"]) for sp in case_run["runtime_species"]
        ]
        case_manifest["normalization"]["reference_species_name"] = case_run["runtime_species"][0]["name"]
        case_manifest["source_rho"] = []
        case_manifest["runs"] = [case_run]
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
        require(math.isclose(float(params.beta), case_run["beta"], rel_tol=1e-6), f"{name} runtime beta")
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

        case_run["input_fingerprint"] = input_fingerprint(case_manifest, case_run)
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(json.dumps(case_manifest, indent=2))
        command = [sys.executable, str(Path(scan.__file__)), "run-one", "--manifest", str(manifest_path),
                   "--index", "0", "--backend", args.backend]
        if args.backend == "gpu" and os.environ.get("CUDA_VISIBLE_DEVICES"):
            command.extend(["--gpu-ids", os.environ["CUDA_VISIBLE_DEVICES"]])
        try:
            process = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180,
            )
            (case_dir / "solver.log").write_text(process.stdout)
            result = dict(
                case=name, returncode=process.returncode, nu=config.species[0].nu,
                beta=config.physics.beta, startup_checks=True, provenance=provenance, backend=args.backend,
            )
        except subprocess.TimeoutExpired as error:
            (case_dir / "solver.log").write_bytes(error.stdout or b"")
            result = dict(case=name, timeout=180, startup_checks=True)
        if result.get("returncode") == 0:
            require(completion_status(case_manifest, case_run)[0], f"{name} completion certification")
            collect_args = scan.build_parser().parse_args(["collect", "--manifest", str(manifest_path),
                "--out", str(case_dir / "flux_summary.h5"), "--neopax-flux-out", str(case_dir / "neopax_fluxes.h5"),
                "--no-plot", "--no-plot-run-heat-traces"])
            require(scan.cmd_collect(collect_args) == 0, f"{name} NEOPAX collection")
            import h5py
            with h5py.File(case_dir / "neopax_fluxes.h5") as output:
                require(output["Q"].shape == (len(case_run["runtime_species"]), 1), f"{name} species shape")
                require(np.all(np.isfinite(output["Q"][...])), f"{name} finite collected heat flux")
            columns = np.genfromtxt(case_dir / "smoke.diagnostics.csv", delimiter=",", names=True).dtype.names
            for index in range(len(case_run["runtime_species"])):
                require(
                    f"heat_flux_s{index}" in columns and f"particle_flux_s{index}" in columns,
                    f"{name} species columns",
                )
            result["completion_and_collection"] = True
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
    print(f"All {len(results)} bounded cases passed with finite diagnostics, completion certification and collection.")


if __name__ == "__main__":
    main()
