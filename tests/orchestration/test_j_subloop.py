"""Exercise inner current passes and the accepted handoff to outer transport."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from common.vmec_namelist import rewrite_current_profile
import src.j_subloop as j_subloop
import src.ouroboros as ouroboros
from src.utils import LOOP_STAGES, resolve_pipeline_paths
from stages.common.current_density_feedback_contract import build_record, resolve_settings
from tests.helpers.stage_import import load_stage_module
from tests.orchestration.test_ouroboros import _write_config

FLAGS = dict.fromkeys(LOOP_STAGES, True)
STAGE3 = {"profiles_source": "analytical", "dPhiHatdrHat": -2., "solver_tolerance": 3e-7,
          "resolution": {"Ntheta": 9, "Nzeta": 13}}
STAGE4 = {"profiles_source": "analytical", "max_steps": 40}


def _write_run(tmp_path, *, ncurr=0, max_iters=3):
    config_path = _write_config(tmp_path, loop={"pressure_profile_type": "akima_spline"})
    config = yaml.safe_load(config_path.read_text())
    config.update(current_density_feedback={"enabled": True, "max_iters": max_iters},
                  stage3={"dkx": deepcopy(STAGE3)}, stage4={"gkx": deepcopy(STAGE4)},
                  gpu_ids="4,5", jobs_per_gpu=2, container_runtime="apptainer")
    paths = resolve_pipeline_paths(config)
    Path(paths["s1_input"]).write_text(
        f"&INDATA\n NCURR={ncurr}\n AM=100, -100\n AI=.4\n RBC(0,0)=5.5\n/\n"
    )
    Path(paths["s3_config"]).write_text("&physicsParameters\n dPhiHatdrHat=-2\n/\n")
    Path(paths["s4_config"]).write_text("[solver]\nmax_steps=40\n")
    config_path.write_text(yaml.safe_dump(config))
    return config_path, config, paths


def _absolute(call, path):
    return Path(call["repo_root"]) / path


def _adapter_result(call, *, converged):
    """Stand in for solver execution while retaining the actual provenance contract."""
    config = yaml.safe_load(Path(call["config_path"]).read_text())
    paths = resolve_pipeline_paths(config)
    output = j_subloop.current_paths(paths["output_dir"])
    context_path = config["current_density_feedback_pass"]["context"]
    context = json.loads(_absolute(call, context_path).read_text())
    applied = _absolute(call, paths["s1_input"]).read_text()
    for key in ("s1_output", "s2_output", "s3_output"):
        destination = _absolute(call, paths[key])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{key} from {paths['s1_input']}\n")
    candidate = applied if converged else rewrite_current_profile(
        applied, np.array([0., .25, 1.]), np.array([0., -4., 10.]),
    )
    _absolute(call, output["candidate"]).parent.mkdir(parents=True, exist_ok=True)
    _absolute(call, output["candidate"]).write_text(candidate)
    artifacts = {
        "applied_input": paths["s1_input"], "candidate_input": output["candidate"],
        "wout": paths["s1_output"], "boozer": paths["s2_output"], "aggregate": paths["s3_output"],
        "context": context_path, **context["inputs"],
    }
    record = build_record(status="converged" if converged else "continue",
                          residual=.25 if converged else 20., threshold=1.5,
                          settings=resolve_settings(config),
                          measured_s=[.25, 1.], j_dkx=[50., 50.],
                          j_vmex=[49.75, 50.] if converged else [30., 50.],
                          paths={name: _absolute(call, path) for name, path in artifacts.items()})
    for name, path in artifacts.items():
        record["artifacts"][name]["path"] = path
    _absolute(call, call["target"]).write_text(json.dumps(record))
    return config, paths, record


def _run(config, paths, runner, root, **kwargs):
    return j_subloop.run_j_subloop(config=config, outer_paths=paths, enabled=FLAGS, rerun=FLAGS,
                                   repo_root=root, run_forward_pass=runner, cores=8, **kwargs)


@pytest.mark.parametrize("ncurr", [0, 1])
def test_first_input_preserved_then_candidate_applied_with_fixed_outer_settings(tmp_path, ncurr):
    _, config, paths = _write_run(tmp_path, ncurr=ncurr)
    source = Path(paths["s1_input"]).read_text()
    original_config = deepcopy(config)
    calls, consumed, generated = [], [], []

    def run(**call):
        calls.append(call)
        cfg, inner, record = _adapter_result(call, converged=len(calls) == 2)
        consumed.append(Path(inner["s1_input"]).read_text())
        generated.append((cfg, inner, record))

    extra = ["gpu_ids=4,5", "jobs_per_gpu=2", "container_runtime=apptainer"]
    accepted = _run(config, paths, run, tmp_path, extra_config=extra,
                    profile="executors/cluster", htcondor_jobdir="logs/current")
    assert len(calls) == 2
    assert consumed[0] == source
    assert "NCURR = 1" in consumed[1]
    assert "AM=100, -100" in consumed[1] and "AI=.4" in consumed[1]
    assert "RBC(0,0)=5.5" in consumed[1]
    assert consumed[1] == Path(generated[0][2]["artifacts"]["candidate_input"]["path"]).read_text()
    assert Path(paths["s1_input"]).read_text() == source
    assert config == original_config
    for call, (cfg, inner, _) in zip(calls, generated):
        assert cfg["stage3"]["dkx"] == STAGE3
        assert cfg["stage4"]["gkx"] == STAGE4
        assert cfg["loop"] == config["loop"]
        for key in ("s3_config", "s4_config", "s5_config"):
            assert Path(inner[key]).read_bytes() == Path(paths[key]).read_bytes()
        assert call["cores"] == 8
        assert call["extra_config"] == extra
        assert call["profile"] == "executors/cluster"
        assert call["htcondor_jobdir"] == "logs/current"
    assert all(set(cfg["current_density_feedback_pass"]) == {"context"} for cfg, _, _ in generated)
    assert accepted == {
        **{key: generated[-1][1][key] for key in ("s1_input", "s1_output", "s2_output", "s3_output")},
        "current_density_feedback_record": calls[-1]["target"],
    }


def test_max_iters_counts_initial_equilibrium_and_prevents_transport(tmp_path, monkeypatch):
    config_path, _, _ = _write_run(tmp_path, max_iters=1)
    calls = []

    def run(**call):
        calls.append(call)
        _adapter_result(call, converged=False)

    monkeypatch.setattr(ouroboros, "run_forward_pass", run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "1"])
    with pytest.raises(RuntimeError, match="after 1 equilibrium passes"):
        ouroboros.main()
    assert len(calls) == 1
    assert "j_subloop/iter_1/" in calls[0]["target"]
    assert not list((tmp_path / "out").rglob("transport_solution.h5"))


def test_solver_failure_stops_before_transport(tmp_path, monkeypatch):
    config_path, _, _ = _write_run(tmp_path)
    calls = []

    def run(**call):
        calls.append(call)
        raise subprocess.CalledProcessError(1, ["snakemake"])

    monkeypatch.setattr(ouroboros, "run_forward_pass", run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "1"])
    with pytest.raises(subprocess.CalledProcessError):
        ouroboros.main()
    assert len(calls) == 1
    assert "j_subloop/iter_1/" in calls[0]["target"]


def test_resume_rechecks_every_pass_with_snakemake_and_preserves_input_timestamps(tmp_path):
    _, config, paths = _write_run(tmp_path)
    initial = []

    def run(**call):
        initial.append(call)
        _adapter_result(call, converged=len(initial) == 2)

    accepted = _run(config, paths, run, tmp_path)
    files = [path for path in Path(paths["output_dir"]).rglob("*") if path.is_file()]
    timestamps = {path: path.stat().st_mtime_ns for path in files}
    resumed = []
    assert _run(config, paths, lambda **call: resumed.append(call), tmp_path) == accepted
    assert [call["target"] for call in resumed] == [call["target"] for call in initial]
    assert {path: path.stat().st_mtime_ns for path in files} == timestamps


@pytest.mark.parametrize("changed", ["settings", "s5_config", "s3_config", "s1_input"])
def test_resume_rejects_stale_records_even_when_all_outputs_exist(tmp_path, changed):
    _, config, paths = _write_run(tmp_path)
    _run(config, paths, lambda **call: _adapter_result(call, converged=True), tmp_path)
    if changed == "settings":
        config["current_density_feedback"]["relaxation"] = .25
    else:
        path = Path(paths[changed])
        path.write_text(path.read_text() + "\n! changed source\n")
    calls = []
    with pytest.raises(ValueError, match="Stale current density feedback"):
        _run(config, paths, lambda **call: calls.append(call), tmp_path)
    assert len(calls) == 1


def test_interruption_resumes_dependency_checks_before_accepting_later_record(tmp_path):
    _, config, paths = _write_run(tmp_path)
    first_calls = []

    def interrupted(**call):
        first_calls.append(call)
        if len(first_calls) == 2:
            raise subprocess.CalledProcessError(1, "snakemake")
        _adapter_result(call, converged=False)

    with pytest.raises(subprocess.CalledProcessError):
        _run(config, paths, interrupted, tmp_path)
    resumed = []

    def resume(**call):
        resumed.append(call)
        if len(resumed) == 2:
            _adapter_result(call, converged=True)

    accepted = _run(config, paths, resume, tmp_path)
    assert [call["target"] for call in resumed] == [call["target"] for call in first_calls]
    assert accepted["current_density_feedback_record"] == resumed[1]["target"]


def test_relative_artifact_paths_resolve_against_repo_root_from_another_cwd(tmp_path, monkeypatch):
    _, config, paths = _write_run(tmp_path)
    config["input_dir"] = str(Path(config["input_dir"]).relative_to(tmp_path))
    config["output_dir"] = str(Path(config["output_dir"]).relative_to(tmp_path))
    paths = resolve_pipeline_paths(config)
    other = tmp_path / "caller"
    other.mkdir()
    monkeypatch.chdir(other)
    accepted = _run(config, paths, lambda **call: _adapter_result(call, converged=True), tmp_path)
    assert not Path(accepted["s1_input"]).is_absolute()
    assert (tmp_path / accepted["current_density_feedback_record"]).exists()


def test_transport_requires_valid_current_evidence_and_declares_every_artifact(tmp_path):
    _, config, paths = _write_run(tmp_path)
    with pytest.raises(ValueError, match="Run Ouroboros"):
        j_subloop.convergence_inputs(config, {})
    accepted = _run(config, paths, lambda **call: _adapter_result(call, converged=True), tmp_path)
    final_config = {**config, "artifact_reuse": accepted}
    inputs = j_subloop.convergence_inputs(final_config, accepted)
    record = json.loads(Path(accepted["current_density_feedback_record"]).read_text())
    assert set(inputs) == {accepted["current_density_feedback_record"],
                           paths["s3_config"], paths["s5_config"],
                           *(item["path"] for item in record["artifacts"].values())}
    assert j_subloop.resolve_artifact_reuse(final_config) == accepted
    Path(accepted["s1_output"]).write_text("changed equilibrium\n")
    with pytest.raises(ValueError, match="Stale current density feedback artifact wout"):
        j_subloop.convergence_inputs(final_config, accepted)


@pytest.mark.parametrize("changed", ["stage3", "s5_config", "s3_config"])
def test_transport_rejects_settings_or_profiles_changed_after_acceptance(tmp_path, changed):
    _, config, paths = _write_run(tmp_path)
    accepted = _run(config, paths, lambda **call: _adapter_result(call, converged=True), tmp_path)
    if changed == "stage3":
        config["stage3"]["dkx"]["dPhiHatdrHat"] = 3.
    else:
        Path(paths[changed]).write_text("changed physics input\n")
    with pytest.raises(ValueError, match="differs|differ"):
        j_subloop.convergence_inputs({**config, "artifact_reuse": accepted}, accepted)


def test_disabled_feedback_needs_no_convergence_inputs():
    assert j_subloop.convergence_inputs({}, {}) == []
    assert j_subloop.resolve_artifact_reuse({}) == {}


@pytest.mark.parametrize("reuse", [{"s1_input": "input.vmec"}, []])
def test_partial_or_non_mapping_reuse_is_rejected(reuse):
    with pytest.raises(ValueError, match="artifact_reuse"):
        j_subloop.resolve_artifact_reuse({"current_density_feedback": {"enabled": True}, "artifact_reuse": reuse})


def test_disabled_driver_runs_transport_without_inner_passes(tmp_path, monkeypatch):
    config_path, config, _ = _write_run(tmp_path)
    config["current_density_feedback"]["enabled"] = False
    config_path.write_text(yaml.safe_dump(config))
    calls = []

    def forward(**call):
        calls.append(call)
        Path(call["target"]).parent.mkdir(parents=True, exist_ok=True)
        Path(call["target"]).write_text(json.dumps({"status": "converged"}))

    monkeypatch.setattr(ouroboros, "run_forward_pass", forward)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "1"])
    ouroboros.main()
    assert len(calls) == 1
    assert "j_subloop" not in calls[0]["target"]


def test_two_outer_iterations_preserve_accepted_current_and_switch_profiles_only_at_boundary(tmp_path, monkeypatch):
    config_path, base_config, _ = _write_run(tmp_path)
    pressure_writer = load_stage_module("stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py")
    calls, inner_configs, inner_inputs, transport_configs, exported_inputs = [], [], [], [], []

    def forward(**call):
        calls.append(call)
        config = yaml.safe_load(Path(call["config_path"]).read_text())
        if "current_density_feedback_pass" in config:
            cfg, paths, _ = _adapter_result(call, converged=len(inner_configs) % 2 == 1)
            inner_configs.append(cfg)
            inner_inputs.append(Path(paths["s1_input"]).read_text())
            return
        transport_configs.append(config)
        reuse = j_subloop.resolve_artifact_reuse(config)
        j_subloop.convergence_inputs(config, reuse)
        paths = resolve_pipeline_paths(config)
        Path(paths["s1_feedback"]).parent.mkdir(parents=True, exist_ok=True)
        pressure_writer._write_vmec_input_with_pressure_spline(
            Path(reuse["s1_input"]), np.array([0., .5, 1.]),
            np.array([200., 100., 0.]) * len(transport_configs),
            output_path=Path(paths["s1_feedback"]),
        )
        exported_inputs.append(Path(paths["s1_feedback"]).read_text())
        shutil.copyfile(paths["s5_config"], paths["s5_config_feedback"])
        with Path(paths["s5_config_feedback"]).open("a") as stream:
            stream.write(f"\n# transport profiles from outer iteration {len(transport_configs)}\n")
        Path(call["target"]).write_text(json.dumps({"status": "continue"}))

    monkeypatch.setattr(ouroboros, "run_forward_pass", forward)
    monkeypatch.setattr("sys.argv", [
        "ouroboros", "--config", str(config_path), "--max-iters", "2", "--cores", "12",
        "--gpu-ids", "2,3", "--jobs-per-gpu", "3", "--container-runtime", "apptainer",
        "--profile", "executors/test", "--htcondor-jobdir", "logs/test",
    ])
    ouroboros.main()
    assert len(calls) == 6
    assert len(transport_configs) == 2
    assert "NCURR=0" in inner_inputs[0]
    assert "NCURR = 1" in inner_inputs[1]
    assert inner_inputs[2] == exported_inputs[0]
    for input_text in exported_inputs:
        assert "NCURR = 1" in input_text
        assert "PCURR_TYPE = 'akima_spline_i'" in input_text
        assert "CURTOR = 1.0000000000000000E+01" in input_text
        assert "PMASS_TYPE = 'akima_spline'" in input_text
        assert "RBC(0,0)=5.5" in input_text
    assert [cfg["stage3"]["dkx"]["profiles_source"] for cfg in inner_configs] == [
        "analytical", "analytical", "prescribed", "prescribed",
    ]
    assert [cfg["stage4"]["gkx"]["profiles_source"] for cfg in inner_configs] == [
        "analytical", "analytical", "prescribed", "prescribed",
    ]
    for cfg in [*inner_configs, *transport_configs]:
        assert cfg["stage3"]["dkx"]["dPhiHatdrHat"] == STAGE3["dPhiHatdrHat"]
        assert cfg["stage3"]["dkx"]["solver_tolerance"] == STAGE3["solver_tolerance"]
        assert cfg["stage3"]["dkx"]["resolution"] == STAGE3["resolution"]
        assert cfg["gpu_ids"] == "2,3"
        assert cfg["jobs_per_gpu"] == 3
        assert cfg["loop"] == base_config["loop"]
    for call in calls:
        assert call["cores"] == 12
        assert call["profile"] == "executors/test"
        assert call["htcondor_jobdir"] == "logs/test"
        assert call["extra_config"] == ["gpu_ids=2,3", "jobs_per_gpu=3", "container_runtime=apptainer"]
    for index, cfg in enumerate(inner_configs):
        common = Path(resolve_pipeline_paths(cfg)["s5_config"]).read_text()
        assert ("transport profiles from outer iteration 1" in common) == (index >= 2)
