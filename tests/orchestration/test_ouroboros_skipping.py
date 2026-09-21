"""Exercise skipped producers through real loop input and feedback files."""

from __future__ import annotations

import json
import logging
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.ouroboros as ouroboros
from src.utils import LOOP_STAGES, resolve_pipeline_paths
from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution
from tests.orchestration.test_ouroboros import _write_config

REPO_ROOT = Path(__file__).resolve().parents[2]
writer = load_stage_module("stages/stage5-post-processing/write_prescribed_profiles_from_transport_h5.py")


def _write_skipping_config(tmp_path, enabled, loop=None):
    path = _write_config(tmp_path, loop=loop)
    config = yaml.safe_load(path.read_text())
    common = (REPO_ROOT / "inputs/quick_run/common_input.toml").read_text()
    for stage, section in (("stage3", "neoclassical"), ("stage4", "turbulence")):
        if not enabled[stage]:
            common = common.replace(f'[{section}]\nflux_model = "fluxes_r_file"',
                                    f'[{section}]\nflux_model = " NoNe "')
            del config["filenames"][f"s{stage[-1]}_config"]
            del config["filenames"][f"s{stage[-1]}_output"]
    path.write_text(yaml.safe_dump(config))
    paths = resolve_pipeline_paths(config, enabled=enabled)
    Path(paths["s5_config"]).write_text(common)
    Path(paths["s1_input"]).write_text("&INDATA\n AM=1\n/\n")
    for key in ("s3_config", "s4_config"):
        if key in paths:
            Path(paths[key]).write_text(f"producer input {key}\n")
    return path, config, paths


@pytest.mark.parametrize("stage3,stage4,freeze_skipped", [
    (True, True, False),
    (True, False, False),
    (False, True, False),
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (False, False, True),
])
def test_two_iterations_seed_only_enabled_producers(
    tmp_path, monkeypatch, caplog, stage3, stage4, freeze_skipped
):
    enabled = dict.fromkeys(LOOP_STAGES, True) | {"stage3": stage3, "stage4": stage4}
    flags = {stage: enabled[stage] or not freeze_skipped for stage in LOOP_STAGES}
    config_path, config, base = _write_skipping_config(tmp_path, enabled, loop={"rerun": flags})
    original = Path(base["s5_config"]).read_bytes()
    calls = []
    consumed = []

    def forward(*, input_dir, output_dir, target, extra_configfiles, **kwargs):
        paths = resolve_pipeline_paths(config, input_dir=input_dir, output_dir=output_dir, enabled=enabled)
        calls.append(extra_configfiles)
        consumed.append(tomllib.loads(Path(paths["s5_config"]).read_text()))
        for stage in ("stage3", "stage4"):
            key = f"s{stage[-1]}_config"
            if enabled[stage]:
                assert Path(paths[key]).read_bytes() == Path(base[key]).read_bytes()
            else:
                assert key not in paths
        transport = Path(paths["s5_output"])
        transport.parent.mkdir(parents=True, exist_ok=True)
        faces = np.linspace(0.0, 1.0, 6)
        write_transport_solution(
            transport, rho=(faces[1:] + faces[:-1]) / 2,
            density=np.ones((3, 5)), temperature=np.ones((3, 5)), er=np.zeros(5),
            rho_face=faces, r_grid_half=0.4 * faces,
            density_face=np.ones((3, 6)), temperature_face=np.ones((3, 6)), er_face=np.zeros(6),
            density_grad_face=np.zeros((3, 6)), temperature_grad_face=np.zeros((3, 6)),
            final_time=len(calls) * 4.0e-7, next_dt=3.0e-8,
        )
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", ["writer", str(transport), paths["s5_config"],
                                          "--output-toml", paths["s5_config_feedback"]])
            writer.main()
        Path(paths["s1_feedback"]).write_text("evolved boundary\n")
        Path(target).write_text(json.dumps({"status": "continue"}))

    monkeypatch.setattr(ouroboros, "run_forward_pass", forward)
    monkeypatch.setattr(sys, "argv", ["ouroboros", "--config", str(config_path), "--max-iters", "2"])
    with caplog.at_level(logging.INFO):
        ouroboros.main()

    assert len(calls) == 2
    assert calls[0] is None
    assert consumed[1]["transport_solver"]["t0"] == 4.0e-7
    assert consumed[1]["profiles"]["model"] == "prescribed"
    assert consumed[1]["profiles"]["density"] == [[1.0e20] * 5] * 3
    for section in ("neoclassical", "turbulence"):
        assert consumed[1][section] == consumed[0][section]
    expected = {stage: {section: {"profiles_source": "prescribed"}}
                for stage, section in (("stage3", "dkx"), ("stage4", "gkx")) if enabled[stage]}
    if expected:
        assert yaml.safe_load(calls[1][0].read_text()) == expected
    else:
        assert calls[1] is None
        assert not (tmp_path / "out/loop/iter_2/input/loop_overrides.yaml").exists()
    for stage in ("stage3", "stage4"):
        if not enabled[stage]:
            assert f"Stage {stage[-1]} is skipped" in caplog.text
    assert Path(base["s5_config"]).read_bytes() == original
    assert (tmp_path / "out/loop/iter_2/input/vmec_input.testrun").read_text() == "evolved boundary\n"


def test_overrides_reuse_only_enabled_frozen_stages_and_keep_all_flags(tmp_path):
    enabled = dict.fromkeys(LOOP_STAGES, True) | {"stage4": False}
    flags = dict.fromkeys(LOOP_STAGES, False) | {"stage5": True}
    path = ouroboros._write_loop_overrides(tmp_path, rerun=flags, enabled=enabled,
                                          reuse_output_dir="first/output")
    assert yaml.safe_load(path.read_text()) == {
        "loop": {"rerun": flags, "reuse_output_dir": "first/output"},
    }


def test_main_stops_when_every_enabled_stage_is_frozen(tmp_path, monkeypatch, caplog):
    enabled = dict.fromkeys(LOOP_STAGES, True) | {"stage3": False, "stage4": False}
    flags = {stage: not enabled[stage] for stage in LOOP_STAGES}
    config_path, _, _ = _write_skipping_config(tmp_path, enabled, loop={"rerun": flags})
    calls = []

    def forward(*, target, **kwargs):
        calls.append(kwargs)
        Path(target).parent.mkdir(parents=True)
        Path(target).write_text(json.dumps({"status": "continue"}))

    monkeypatch.setattr(ouroboros, "run_forward_pass", forward)
    monkeypatch.setattr(sys, "argv", ["ouroboros", "--config", str(config_path), "--max-iters", "3"])
    with caplog.at_level(logging.INFO):
        ouroboros.main()
    assert len(calls) == 1
    assert "Every enabled stage is frozen" in caplog.text
    assert not (tmp_path / "out/loop/iter_2").exists()


def test_empty_overrides_remove_stale_driver_file(tmp_path):
    enabled = dict.fromkeys(LOOP_STAGES, True) | {"stage3": False, "stage4": False}
    old = tmp_path / "loop_overrides.yaml"
    old.write_text("stage3:\n  dkx:\n    profiles_source: prescribed\n")
    assert ouroboros._write_loop_overrides(tmp_path, enabled=enabled) is None
    assert not old.exists()
