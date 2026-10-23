"""Keep unit tests independent of the separately installed Stage 4 solver."""
import pytest
from tests.helpers.stage_import import load_stage_module


@pytest.fixture(autouse=True)
def solver_identity(monkeypatch):
    provenance = load_stage_module("stages/stage4-turbulence/solver_provenance.py")
    scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
    identity = dict(expected_revision=provenance.expected_revision(),
                    installed_revision=provenance.expected_revision(), precision="float32")
    monkeypatch.setattr(provenance, "current_provenance", lambda: dict(identity))
    monkeypatch.setattr(scan, "current_provenance", lambda: dict(identity))
    return identity
