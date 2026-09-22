"""Check the committed W7-X run configs."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from tests.helpers.runs import W7X_RUNS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_run(run: str) -> tuple[dict, dict]:
    """Load one W7-X pipeline config and its common input."""
    run_dir = REPO_ROOT / "inputs" / run
    pipeline = yaml.safe_load((run_dir / "config.yaml").read_text())
    common = tomllib.loads((run_dir / "common_input.toml").read_text())
    return pipeline, common


@pytest.mark.parametrize("run", W7X_RUNS)
def test_w7x_scans_use_the_transport_face_grid(run: str) -> None:
    """Configured scans must derive their analytical grid from the transport face count."""
    pipeline, _ = _load_run(run)
    if "stage3" in pipeline:
        assert pipeline["stage3"]["dkx"]["analytical_n_radii"] is None
    assert pipeline["stage4"]["gkx"]["analytical_n_radii"] is None


@pytest.mark.parametrize("run", W7X_RUNS)
def test_w7x_transport_clock_saves_both_step_states(run: str) -> None:
    """A one-step pass must save the states at both ends of its clock interval."""
    _, common = _load_run(run)
    solver = common["transport_solver"]
    assert solver["stop_after_accepted_steps"] == 1
    assert solver["save_n"] == 2
    assert 0.0 <= solver["t0"] < solver["t_final"]
    assert solver["dt"] > 0.0
