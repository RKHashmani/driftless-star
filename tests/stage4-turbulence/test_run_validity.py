import json
import os
from pathlib import Path

import pytest

from tests.helpers.stage_import import load_stage_module

validity = load_stage_module("stages/stage4-turbulence/run_validity.py")


@pytest.fixture
def prepared(tmp_path):
    config = tmp_path / "input.toml"
    config.write_text("beta = 0.01\nnu = 0.02\n")
    equilibrium = tmp_path / "wout.nc"
    equilibrium.write_bytes(b"equilibrium")
    manifest = {
        "schema_version": 3,
        "vmec_file": str(equilibrium),
    }
    run = {"config_path": str(config), "output_prefix": str(tmp_path / "result")}
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    return manifest, run


def write_diagnostics(run, text="t,heat_flux,particle_flux\n0,1,2\n1,2,3\n"):
    validity.diagnostics_path(run).write_text(text)


def complete(manifest, run):
    token = validity.begin_attempt(manifest, run)
    write_diagnostics(run)
    validity.certify_completion(manifest, run, token)


def test_unchanged_inputs_reuse_trusted_completion(prepared):
    manifest, run = prepared
    complete(manifest, run)
    assert validity.completion_status(manifest, run) == (True, "trusted completion")
    sidecar = json.loads(validity.completion_path(run).read_text())
    assert sidecar["input_fingerprint"] == run["input_fingerprint"]


@pytest.mark.parametrize("target", ["config_path", "vmec_file"])
def test_changed_input_contents_require_reprepare(prepared, target):
    manifest, run = prepared
    complete(manifest, run)
    Path((run if target == "config_path" else manifest)[target]).write_bytes(b"changed")
    assert not validity.completion_status(manifest, run)[0]
    with pytest.raises(ValueError, match="Prepare the scan again"):
        validity.begin_attempt(manifest, run)
    assert validity.completion_path(run).exists()
    assert validity.diagnostics_path(run).exists()


@pytest.mark.parametrize("target,key,value", [
    ("run", "rho_star_physical", 0.003),
    ("run", "a_minor", 0.75),
    ("run", "rho", 0.5),
    ("manifest", "geometry", {"a_minor": 0.75}),
    ("manifest", "species_meta", [{"name": "H", "mass_mp": 1.0}]),
    ("manifest", "source_rho", [0.0, 0.5, 1.0]),
])
def test_changed_flux_interpretation_rejects_completion(prepared, target, key, value):
    manifest, run = prepared
    complete(manifest, run)
    (run if target == "run" else manifest)[key] = value
    assert validity.completion_status(manifest, run) == (False, "runtime inputs changed after preparation")


@pytest.mark.parametrize("key", ["density_reference_physical", "temperature_reference_physical", "mass"])
def test_changed_reference_species_conversion_rejects_completion(prepared, key):
    manifest, run = prepared
    run["runtime_species"] = [{"name": "D", "mass": 2.0, "density_reference_physical": 1.0,
                               "temperature_reference_physical": 1.0}]
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    complete(manifest, run)
    run["runtime_species"][0][key] = 3.0
    assert validity.completion_status(manifest, run) == (False, "runtime inputs changed after preparation")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_flux_conversion_cannot_be_prepared(prepared, value):
    manifest, run = prepared
    manifest["snapshot_time"] = None
    run["rho_star_physical"] = value
    with pytest.raises(ValueError, match="JSON compliant"):
        validity.input_fingerprint(manifest, run)


def test_unknown_snapshot_time_does_not_block_preparation(prepared):
    manifest, run = prepared
    manifest["snapshot_time"] = None
    assert validity.input_fingerprint(manifest, run) == run["input_fingerprint"]


def test_boozer_contents_are_fingerprinted(prepared, tmp_path):
    manifest, run = prepared
    booz = tmp_path / "booz.nc"
    booz.write_bytes(b"old field")
    manifest["booz_file"] = str(booz)
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    complete(manifest, run)
    booz.write_bytes(b"new field")
    assert validity.completion_status(manifest, run) == (False, "runtime inputs changed after preparation")


def test_equilibrium_digest_reused_but_runtime_toml_is_always_read(prepared, monkeypatch):
    manifest, run = prepared
    validity._cached_equilibrium_digest.cache_clear()
    original = validity._digest
    reads = []

    def digest(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(validity, "_digest", digest)
    validity.input_fingerprint(manifest, run)
    validity.input_fingerprint(manifest, run)
    assert reads.count(Path(manifest["vmec_file"])) == 1
    assert reads.count(Path(run["config_path"])) == 2


def test_equilibrium_mutation_with_same_size_and_mtime_invalidates_cache(prepared):
    manifest, run = prepared
    complete(manifest, run)
    equilibrium = Path(manifest["vmec_file"])
    original = equilibrium.stat()
    equilibrium.write_bytes(b"EQUILIBRIUM")
    os.utime(equilibrium, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert equilibrium.stat().st_size == original.st_size
    assert validity.completion_status(manifest, run) == (False, "runtime inputs changed after preparation")


def test_unreadable_inputs_leave_previous_outputs_intact(prepared, monkeypatch):
    manifest, run = prepared
    complete(manifest, run)
    summary = Path(f"{run['output_prefix']}.summary.json")
    summary.write_text('{"converged": true}\n')
    geometry = summary.with_suffix(".eik.nc")
    geometry.write_bytes(b"geometry")
    run["geometry_file"] = str(geometry)
    preserved = [validity.completion_path(run), validity.diagnostics_path(run), summary, geometry]
    contents = {path: path.read_bytes() for path in preserved}

    def unreadable(*args):
        raise OSError("equilibrium temporarily unavailable")

    monkeypatch.setattr(validity, "input_fingerprint", unreadable)
    with pytest.raises(OSError, match="temporarily unavailable"):
        validity.begin_attempt(manifest, run)
    assert {path: path.read_bytes() for path in preserved} == contents


def test_reprepare_rejects_old_success(prepared):
    manifest, run = prepared
    complete(manifest, run)
    Path(run["config_path"]).write_text("beta = 0.02\n")
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    assert validity.completion_status(manifest, run) == (False, "completion belongs to different inputs")
    complete(manifest, run)
    assert validity.completion_status(manifest, run)[0]


def test_failed_retry_cannot_reuse_old_diagnostics(prepared):
    manifest, run = prepared
    complete(manifest, run)
    token = validity.begin_attempt(manifest, run)
    assert not validity.completion_status(manifest, run)[0]
    with pytest.raises(ValueError, match="missing diagnostics"):
        validity.certify_completion(manifest, run, token)
    write_diagnostics(run)
    assert validity.completion_status(manifest, run) == (False, "missing trusted completion marker")


@pytest.mark.parametrize("text", ["t,heat_flux,particle_flux\n", "t,heat_flux\n0,1\n", "t,heat_flux,particle_flux\n0,nan,1\n"])
def test_invalid_fresh_diagnostics_not_certified(prepared, text):
    manifest, run = prepared
    token = validity.begin_attempt(manifest, run)
    write_diagnostics(run, text)
    with pytest.raises(ValueError):
        validity.certify_completion(manifest, run, token)
    assert not validity.completion_path(run).exists()


def test_mutation_during_worker_not_certified(prepared):
    manifest, run = prepared
    token = validity.begin_attempt(manifest, run)
    write_diagnostics(run)
    Path(run["config_path"]).write_text("beta = 0.03\n")
    with pytest.raises(ValueError, match="changed during"):
        validity.certify_completion(manifest, run, token)


def test_diagnostics_mutation_rejects_completion(prepared):
    manifest, run = prepared
    complete(manifest, run)
    write_diagnostics(run, "t,heat_flux,particle_flux\n0,5,5\n")
    assert validity.completion_status(manifest, run) == (False, "diagnostics changed after successful completion")


def test_legacy_results_remain_readable_but_reprepare_needs_success(prepared):
    manifest, run = prepared
    manifest["schema_version"] = 2
    write_diagnostics(run, "t\n1\n")
    assert validity.completion_status(manifest, run)[0]
    manifest["schema_version"] = 3
    write_diagnostics(run)
    assert validity.completion_status(manifest, run) == (False, "missing trusted completion marker")
    complete(manifest, run)
    assert validity.completion_status(manifest, run)[0]


@pytest.fixture
def worker(prepared, tmp_path):
    import argparse
    manifest, run = prepared
    run.update(index=0, rho=0.5, rho_index=1, torflux=0.25, Er=0.0,
               run_dir=str(tmp_path), geometry_file=str(tmp_path / "generated_geometry.nc"))
    manifest.update(runs=[run], runtime_species_names=["D"], electron_model="adiabatic",
                    neopax_result="", common_config="")
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = argparse.Namespace(manifest=str(manifest_path), index=0, verbose_worker=False)
    return manifest, run, args


def test_attempt_discards_generated_geometry_but_legacy_preserves_it(worker):
    manifest, run, _ = worker
    path = Path(run["geometry_file"])
    path.write_bytes(b"cached old equilibrium")
    manifest["schema_version"] = 2
    validity.begin_attempt(manifest, run)
    assert path.exists()
    manifest["schema_version"] = 3
    validity.begin_attempt(manifest, run)
    assert not path.exists()


def test_worker_certifies_success_and_reuses_it(worker, monkeypatch, capsys):
    from types import SimpleNamespace
    scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
    manifest, run, args = worker
    calls = []

    def solver(*positional, **kwargs):
        calls.append(positional)
        write_diagnostics(run)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(scan.subprocess, "run", solver)
    assert scan.cmd_run_one(args) == 0
    assert validity.completion_status(manifest, run)[0]
    assert scan.cmd_run_one(args) == 0
    assert len(calls) == 1
    assert "already_completed" in capsys.readouterr().out


@pytest.mark.parametrize("exit_code,write_fresh", [(1, False), (0, False), (1, True)])
def test_worker_retry_never_certifies_failed_or_missing_fresh_output(worker, monkeypatch, exit_code, write_fresh):
    from types import SimpleNamespace
    scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
    manifest, run, args = worker
    complete(manifest, run)
    Path(run["config_path"]).write_text("beta = 0.02\nnu = 0.03\n")
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    Path(args.manifest).write_text(json.dumps(manifest))

    def solver(*positional, **kwargs):
        assert not validity.completion_path(run).exists()
        assert not validity.diagnostics_path(run).exists()
        if write_fresh:
            write_diagnostics(run)
        return SimpleNamespace(returncode=exit_code, stdout="", stderr="")

    monkeypatch.setattr(scan.subprocess, "run", solver)
    assert scan.cmd_run_one(args) != 0
    assert not validity.completion_path(run).exists()
    assert not validity.completion_status(manifest, run)[0]


def test_worker_rejects_inputs_changed_after_preparation(worker, monkeypatch, capsys):
    scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
    manifest, run, args = worker
    complete(manifest, run)
    Path(run["config_path"]).write_text("beta = 0.4\n")
    monkeypatch.setattr(scan.subprocess, "run", lambda *a, **kw: pytest.fail("solver must not start"))
    assert scan.cmd_run_one(args) == 2
    assert "Prepare the scan again" in capsys.readouterr().err
    assert validity.completion_path(run).exists()
    assert validity.diagnostics_path(run).exists()


def test_launcher_reuses_only_trusted_results(worker, monkeypatch):
    from types import SimpleNamespace
    scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
    manifest, run, args = worker
    launch_args = SimpleNamespace(manifest=args.manifest, max_parallel=1, backend="cpu", threads_per_run=1)
    complete(manifest, run)
    monkeypatch.setattr(scan, "_launch_subprocess", lambda **kw: pytest.fail("trusted result should be reused"))
    assert scan.cmd_run(launch_args) == 0
    validity.completion_path(run).unlink()
    launched = []

    class FailedWorker:
        def poll(self):
            return 1

        def communicate(self):
            return "", "fake worker failed"

    def launch(**kwargs):
        launched.append(kwargs)
        return FailedWorker()

    monkeypatch.setattr(scan, "_launch_subprocess", launch)
    assert scan.cmd_run(launch_args) == 1
    assert len(launched) == 1


@pytest.mark.parametrize("invalidity", ["no_marker", "changed_inputs", "changed_diagnostics"])
def test_collector_rejects_untrusted_results_without_replacing_flux_files(worker, tmp_path, capsys, invalidity):
    from types import SimpleNamespace
    scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
    manifest, run, args = worker
    complete(manifest, run)
    if invalidity == "no_marker":
        validity.completion_path(run).unlink()
    elif invalidity == "changed_inputs":
        Path(run["config_path"]).write_text("beta = 0.5\n")
    else:
        write_diagnostics(run, "t,heat_flux,particle_flux\n0,9,9\n")
    collect_args = SimpleNamespace(
        manifest=args.manifest, out=str(tmp_path / "flux_summary.h5"),
        neopax_flux_out=str(tmp_path / "neopax_fluxes.h5"), average_window=5.0,
        t_final=None, plot=False, plot_run_heat_traces=False,
    )
    for path in (collect_args.out, collect_args.neopax_flux_out):
        Path(path).write_bytes(b"previous flux artifact")
    assert scan.cmd_collect(collect_args) == 2
    for path in (collect_args.out, collect_args.neopax_flux_out):
        assert Path(path).read_bytes() == b"previous flux artifact"
    assert "Cannot collect" in capsys.readouterr().err


def test_explicit_incomplete_collection_marks_invalid_perturbation_absent(worker, tmp_path):
    from types import SimpleNamespace
    import h5py
    scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
    manifest, run, args = worker
    run["runtime_species"] = [{"name": "D", "mass": 2.0, "density_reference_physical": 1.0,
                               "temperature_reference_physical": 1.0}]
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    token = validity.begin_attempt(manifest, run)
    write_diagnostics(run, "t,heat_flux,particle_flux,heat_flux_s0,particle_flux_s0\n0,1,2,1,2\n1,2,3,2,3\n")
    validity.certify_completion(manifest, run, token)
    perturbed = dict(run, index=1, response_label="temperature_gradient", perturb_species="D",
                     perturb_delta=0.2, output_prefix=str(tmp_path / "perturbed"))
    perturbed["input_fingerprint"] = validity.input_fingerprint(manifest, perturbed)
    write_diagnostics(perturbed)
    manifest["runs"].append(perturbed)
    Path(args.manifest).write_text(json.dumps(manifest))
    collect_args = SimpleNamespace(
        manifest=args.manifest, out=str(tmp_path / "flux_summary.h5"), allow_incomplete=True,
        neopax_flux_out=str(tmp_path / "neopax_fluxes.h5"), average_window=5.0,
        t_final=None, plot=False, plot_run_heat_traces=False,
    )
    assert scan.cmd_collect(collect_args) == 0
    with h5py.File(collect_args.neopax_flux_out) as output:
        assert output["perturb_present"][...].tolist() == [[False]]
        assert output["Q"][0, 0] > 0.0
        assert bool(output["meta"].attrs["incomplete_results"])
        assert "missing trusted completion marker" in output["meta"].attrs["invalid_runs_json"]


@pytest.mark.parametrize("key,value", [
    ("parameter_audit", {
        "beta_source": "profiles", "collisionality_source": "fixed",
        "collisionality_scaling_factor": 0.0, "calculation_convention": "next-pin", "fixed_beta": 999,
    }),
    ("beta", 999), ("tau_e", 999),
])
def test_unused_manifest_copies_do_not_invalidate_runtime_inputs(prepared, key, value):
    manifest, run = prepared
    complete(manifest, run)
    run[key] = value
    manifest["snapshot_time"] = 42
    manifest["profiles_source"] = "transport_h5"
    assert validity.completion_status(manifest, run)[0]


def test_changing_selected_reference_conversion_invalidates_result(prepared):
    manifest, run = prepared
    run["runtime_species"] = [
        dict(name="D", mass=2, density_reference_physical=1, temperature_reference_physical=1),
        dict(name="H", mass=1, density_reference_physical=1, temperature_reference_physical=1),
    ]
    manifest["normalization"] = {"reference_species_name": "D"}
    run["input_fingerprint"] = validity.input_fingerprint(manifest, run)
    complete(manifest, run)
    manifest["normalization"]["reference_species_name"] = "H"
    assert not validity.completion_status(manifest, run)[0]


@pytest.mark.parametrize("helper", ["read_diagnostics_csv", "input_fingerprint"])
def test_programming_errors_propagate_from_validity_checks(prepared, monkeypatch, helper):
    manifest, run = prepared
    complete(manifest, run)

    def broken(*args):
        raise KeyError("programming defect")

    monkeypatch.setattr(validity, helper, broken)
    with pytest.raises(KeyError, match="programming defect"):
        validity.completion_status(manifest, run)


@pytest.mark.parametrize("helper,error", [
    ("read_diagnostics_csv", OSError("unavailable diagnostics")),
    ("read_diagnostics_csv", ValueError("malformed diagnostics")),
    ("input_fingerprint", OSError("unavailable input")),
    ("input_fingerprint", ValueError("invalid input")),
])
def test_expected_io_errors_remain_untrusted_results(prepared, monkeypatch, helper, error):
    manifest, run = prepared
    complete(manifest, run)

    def unavailable(*args):
        raise error

    monkeypatch.setattr(validity, helper, unavailable)
    valid, reason = validity.completion_status(manifest, run)
    assert not valid
    assert str(error) in reason


def test_malformed_completion_json_is_an_untrusted_result(prepared):
    manifest, run = prepared
    complete(manifest, run)
    validity.completion_path(run).write_text("{broken JSON")
    valid, reason = validity.completion_status(manifest, run)
    assert not valid
    assert "cannot verify completion" in reason
