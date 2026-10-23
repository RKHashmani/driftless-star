"""Validate J subloop settings and content-bound completion evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from stages.common.current_density_feedback_contract import (
    DEFAULT_SETTINGS,
    REQUIRED_ARTIFACTS,
    build_record,
    resolve_settings,
    validate_record,
)


@pytest.fixture
def record_files(tmp_path):
    paths = {}
    for name in REQUIRED_ARTIFACTS:
        path = tmp_path / name
        path.write_text(f"original {name}\n")
        paths[name] = str(path)
    settings = resolve_settings({"current_density_feedback": {"enabled": True}})
    record = build_record(status="converged", residual=.25, threshold=1.5,
                          settings=settings, paths=paths, measured_s=[.25, 1.],
                          j_dkx=[50., 50.], j_vmex=[49.75, 50.])
    return record, paths, settings


@pytest.mark.parametrize("config", [{}, {"current_density_feedback": {}}])
def test_default_settings_disable_feedback(config):
    result = resolve_settings(config)
    assert result == DEFAULT_SETTINGS
    assert result is not DEFAULT_SETTINGS


def test_settings_keep_user_choices_and_do_not_mutate_them():
    config = {"current_density_feedback": {"enabled": True, "max_iters": 2, "rel_tol": 0,
                                   "abs_tol": 3, "relaxation": 1}}
    before = deepcopy(config)
    settings = resolve_settings(config)
    assert settings == config["current_density_feedback"]
    assert all(isinstance(settings[key], float) for key in ("rel_tol", "abs_tol", "relaxation"))
    assert config == before


@pytest.mark.parametrize("supplied", [False, ["enabled"]])
def test_non_mapping_feedback_settings_rejected(supplied):
    with pytest.raises(ValueError, match="mapping"):
        resolve_settings({"current_density_feedback": supplied})


@pytest.mark.parametrize("key,value", [
    ("enabled", 1), ("enabled", "false"), ("max_iters", 0),
    ("max_iters", True), ("max_iters", 2.5),
    ("rel_tol", -.01), ("abs_tol", -1), ("relaxation", 0), ("relaxation", 1.01),
    ("rel_tol", True), ("relaxation", "0.5"), ("rel_tol", float("nan")),
    ("iterations", 2),
])
def test_invalid_feedback_values_rejected(key, value):
    with pytest.raises(ValueError):
        resolve_settings({"current_density_feedback": {key: value}})


@pytest.mark.parametrize("frozen", ["stage1", "stage2", "stage3"])
def test_enabled_feedback_requires_evolving_upstream_stages(frozen):
    rerun = dict.fromkeys(("stage1", "stage2", "stage3"), True)
    rerun[frozen] = False
    with pytest.raises(ValueError, match="Stages 1 through 3"):
        resolve_settings({"current_density_feedback": {"enabled": True}}, rerun=rerun)


def test_enabled_feedback_requires_stage3_but_disabled_feedback_does_not():
    with pytest.raises(ValueError, match="Stage 3"):
        resolve_settings({"current_density_feedback": {"enabled": True}}, enabled={"stage3": False})
    assert not resolve_settings({}, enabled={"stage3": False}, rerun={"stage1": False})["enabled"]


def test_record_round_trip_requires_matching_settings_and_all_contents(record_files, tmp_path):
    record, paths, settings = record_files
    record_path = tmp_path / "completion.json"
    record_path.write_text(json.dumps(record))
    assert validate_record(record_path, settings=settings, expected_paths=paths,
                           require_converged=True) == record


@pytest.mark.parametrize("name", ["wout", "candidate_input", "effective_config"])
def test_changed_artifact_invalidates_existing_record(record_files, name):
    record, paths, settings = record_files
    Path(paths[name]).write_text("changed after completion\n")
    with pytest.raises(ValueError, match=f"Stale current density feedback artifact {name}"):
        validate_record(record, settings=settings)


def test_same_contents_at_a_different_expected_path_are_not_same_provenance(record_files, tmp_path):
    record, paths, _ = record_files
    alternate = tmp_path / "different-equilibrium"
    alternate.write_bytes(Path(paths["wout"]).read_bytes())
    with pytest.raises(ValueError, match="path mismatch for wout"):
        validate_record(record, expected_paths={"wout": str(alternate)})


def test_missing_artifact_is_not_accepted(record_files):
    record, paths, _ = record_files
    Path(paths["aggregate"]).unlink()
    with pytest.raises(FileNotFoundError):
        validate_record(record)


@pytest.mark.parametrize("key,value", [
    ("contract_version", 0), ("status", "continue"),
    ("residual", float("nan")), ("residual", -1), ("residual", True),
    ("threshold", "1.5"),
    ("artifacts", {}), ("settings", {"enabled": True}),
])
def test_invalid_completion_record_rejected(record_files, key, value):
    record, _, _ = record_files
    record[key] = value
    with pytest.raises(ValueError):
        validate_record(record)


def test_changed_settings_require_regeneration(record_files):
    record, _, settings = record_files
    changed = {**settings, "relaxation": .25}
    with pytest.raises(ValueError, match="Stale current density feedback settings"):
        validate_record(record, settings=changed)


def test_nonconverged_completion_can_resume_but_cannot_authorize_transport(record_files):
    record, _, _ = record_files
    record.update(status="continue", residual=2.0, j_vmex=[48., 50.])
    assert validate_record(record) == record
    with pytest.raises(ValueError, match="convergence evidence"):
        validate_record(record, require_converged=True)


def test_equality_at_threshold_is_converged(record_files):
    record, _, _ = record_files
    record["residual"] = record["threshold"]
    record["j_vmex"] = [48.5, 50.]
    assert validate_record(record, require_converged=True) == record


@pytest.mark.parametrize("key,values", [
    ("measured_s", None), ("j_dkx", [.25]), ("j_vmex", [[.25, 1.]]),
    ("measured_s", [.25, 1., 2.]), ("j_dkx", [.25, float("nan")]),
    ("j_vmex", [.25, float("inf")]), ("measured_s", [.25, True]), ("j_dkx", [.25, "1.0"]),
])
def test_malformed_sample_arrays_cannot_authorize_transport(record_files, key, values):
    record, _, _ = record_files
    record[key] = values
    with pytest.raises(ValueError):
        validate_record(record, require_converged=True)


@pytest.mark.parametrize("samples", [[-.1, 1.], [.25, 1.1], [.25, .9], [1., 1.], [.5, .25]])
def test_measured_coordinates_require_ordered_full_edge_coverage(record_files, samples):
    record, _, _ = record_files
    record["measured_s"] = samples
    with pytest.raises(ValueError):
        validate_record(record)


def test_edge_roundoff_is_accepted_without_extrapolating_a_truncated_grid(record_files):
    record, _, _ = record_files
    record["measured_s"][-1] = 1. - 5e-13
    assert validate_record(record, require_converged=True) == record


@pytest.mark.parametrize("mutation", [
    {"residual": .1},
    {"threshold": 10.},
])
def test_forged_residual_or_threshold_cannot_authorize_transport(record_files, mutation):
    record, _, _ = record_files
    record.update(mutation)
    with pytest.raises(ValueError):
        validate_record(record, require_converged=True)


def test_recomputed_bound_uses_maximum_absolute_signed_current(record_files):
    record, _, _ = record_files
    record.update(j_dkx=[-50., 10.], j_vmex=[-49.75, 10.])
    assert validate_record(record, require_converged=True) == record
