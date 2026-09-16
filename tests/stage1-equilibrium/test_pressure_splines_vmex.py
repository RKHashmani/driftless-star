"""Validate exported pressures with the pinned VMEX container."""

from pathlib import Path
import subprocess

import pytest


@pytest.mark.docker
def test_pressure_splines_in_pinned_vmex():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{root}:/work:ro", "-w", "/work",
         "ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu",
         "python", "-m", "tests.helpers.check_vmex_pressure"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"vmex_commit": "35e7170' in result.stdout
