"""Check installed metadata, checkout verification, and fresh-interpreter probes."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from tests.helpers.stage_import import load_stage_module

provenance = load_stage_module("stages/stage4-turbulence/solver_provenance.py")
probe_current = provenance.current_provenance


@pytest.fixture(autouse=True)
def clear_probe_cache():
    probe_current.cache_clear()
    yield
    probe_current.cache_clear()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_probe_records_effective_solver_precision(monkeypatch, precision):
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        result = dict(installed_revision=provenance.expected_revision(), precision=precision)
        return SimpleNamespace(returncode=0, stdout=json.dumps(result))
    monkeypatch.setattr(provenance.subprocess, "run", probe)
    assert probe_current()["precision"] == precision
    assert probe_current()["precision"] == precision
    assert len(calls) == 1
    assert calls[0][-1] == str(Path(provenance.__file__).resolve())


def test_probe_rejects_wrong_installed_revision(monkeypatch):
    monkeypatch.setattr(provenance.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps(dict(installed_revision="old", precision="float32"))))
    with pytest.raises(ValueError, match="Expected GKX"):
        probe_current()


def test_probe_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(provenance.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stderr="no GKX metadata"))
    with pytest.raises(ValueError, match="Cannot verify executing GKX.*no GKX metadata"):
        probe_current()


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("probe", 60), OSError("cannot start probe")])
def test_probe_errors_become_actionable_value_errors(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(provenance.subprocess, "run", fail)
    with pytest.raises(ValueError, match="Cannot verify executing GKX") as caught:
        probe_current()
    assert caught.value.__cause__ is error


def test_precision_mismatch_names_values_and_environment(solver_identity):
    prepared = dict(solver_identity, precision="float64")
    with pytest.raises(ValueError, match="precision.*float64.*float32.*JAX_ENABLE_X64.*Prepare"):
        provenance.verify_provenance({"solver_provenance": prepared})


def test_distribution_metadata_must_match_imported_module(tmp_path, monkeypatch):
    origin = tmp_path / "gkx/__init__.py"
    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(origin)))
    distribution = SimpleNamespace(read_text=lambda name: json.dumps({"vcs_info": {"commit_id": "verified"}}),
                                   locate_file=lambda name: tmp_path / name)
    monkeypatch.setattr(provenance.importlib.metadata, "distribution", lambda name: distribution)
    assert provenance.installed_revision() == "verified"
    distribution.locate_file = lambda name: tmp_path / "another_install" / name
    with pytest.raises(ValueError, match="No verifiable GKX revision"):
        provenance.installed_revision()


@pytest.mark.parametrize("dirty", [False, True])
def test_source_checkout_must_be_clean(tmp_path, monkeypatch, dirty):
    root = tmp_path / "source"
    (root / ".git").mkdir(parents=True)
    origin = root / "src/gkx/__init__.py"
    monkeypatch.setattr(provenance.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(origin)))
    def missing(name):
        raise provenance.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(provenance.importlib.metadata, "distribution", missing)
    monkeypatch.setattr(provenance.subprocess, "check_output", lambda command, **kwargs:
                        (" M src/gkx/config.py" if dirty else "") if "status" in command else "verified")
    if dirty:
        with pytest.raises(ValueError, match="source changes"):
            provenance.installed_revision()
    else:
        assert provenance.installed_revision() == "verified"
