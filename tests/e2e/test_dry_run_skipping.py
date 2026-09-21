"""Plan optional producers without requiring their templates or YAML settings."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from src.utils import resolve_enabled_stages, resolve_pipeline_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
COMBINATIONS = [(True, True), (False, True), (True, False), (False, False)]


def _config(tmp_path: Path, stage3: bool, stage4: bool) -> tuple[dict, Path, str]:
    """Copy only the inputs required by the selected producers."""
    source_dir = REPO_ROOT / "inputs/quick_run"
    config = yaml.safe_load((source_dir / "config.yaml").read_text())
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config["input_dir"] = str(input_dir)
    config["output_dir"] = str(tmp_path / "out")
    config["loop"]["rerun"] = dict.fromkeys((f"stage{i}" for i in range(1, 6)), True)
    common = (source_dir / "common_input.toml").read_text()
    for stage, enabled, section in ((3, stage3, "neoclassical"), (4, stage4, "turbulence")):
        if not enabled:
            common = common.replace(f'[{section}]\nflux_model = "fluxes_r_file"',
                                    f'[{section}]\nflux_model = " NoNe "')
            del config[f"stage{stage}"]
            del config["filenames"][f"s{stage}_config"]
            del config["filenames"][f"s{stage}_output"]
    for key in ("s1_input", "s3_config", "s4_config"):
        if key in config["filenames"]:
            filename = config["filenames"][key].format(run_name=config["run_name"])
            shutil.copy2(source_dir / filename, input_dir / filename)
    common_path = input_dir / config["filenames"]["s5_config"]
    common_path.write_text(common)
    return config, common_path, common


def _dry_run(tmp_path: Path, config: dict, targets: list[str] | None = None,
             extra_args: list[str] | None = None) -> tuple[subprocess.CompletedProcess, str]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        ["snakemake", "-n", "-p", *(targets or []), "--configfile", str(config_path),
         "--workflow-profile", "none", "--runtime-source-cache-path", str(tmp_path / "srccache"),
         *(extra_args or [])],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return result, result.stdout + result.stderr


@pytest.mark.parametrize("stage3,stage4", COMBINATIONS)
def test_optional_producer_dags(tmp_path: Path, stage3: bool, stage4: bool) -> None:
    config, common_path, original = _config(tmp_path, stage3, stage4)
    enabled = resolve_enabled_stages(common_path)
    paths = resolve_pipeline_paths(config, enabled=enabled)
    result, output = _dry_run(tmp_path, config)
    assert result.returncode == 0, output
    for rule in ("stage1_vmex", "stage2_boozer", "stage5_neopax"):
        assert f"rule {rule}:" in output, output
    assert "rule stage5_post_processing:" not in output, output
    assert ("relabel_neopax_flux_radius.py" in output) == stage4, output
    resolved = tomllib.loads(Path(paths["s5_resolved_config"]).read_text())
    for stage, active, image, section, field, directory in (
        (3, stage3, "stage-3-dkx-cpu", "neoclassical", "neoclassical_file", "stage3_neoclassical"),
        (4, stage4, "stage-4-gkx-cpu", "turbulence", "turbulence_file", "stage4_turbulence"),
    ):
        assert (f"checkpoint stage{stage}_prepare:" in output) == active, output
        assert (f"rule stage{stage}_collect:" in output) == active, output
        assert (image in output) == active, output
        assert f"rule stage{stage}_run_one:" not in output, output
        if active:
            assert paths[f"s{stage}_output"] in output, output
            assert resolved[section][field]
        else:
            assert f"Stage {stage} is skipped" in output, output
            assert directory not in output, output
            assert not (Path(config["output_dir"]) / directory).exists()
            assert resolved[section][field] == ""
            assert resolved[section]["flux_model"] == " NoNe "
    assert common_path.read_text() == original
    stage5_job = output.split("rule stage5_neopax:", 1)[1].split("output:", 1)[0]
    assert str(common_path) in stage5_job, output
    assert paths["s5_resolved_config"] not in stage5_job, output


@pytest.mark.parametrize("stage3,stage4", COMBINATIONS[1:])
def test_first_pass_freezes_do_not_enable_skipped_producers(tmp_path: Path, stage3: bool, stage4: bool) -> None:
    config, common_path, _ = _config(tmp_path, stage3, stage4)
    # Rerun flags for skipped stages can stay true when all enabled stages are frozen.
    config["loop"]["rerun"] = {f"stage{i}": False for i in range(1, 6)}
    for stage, active in ((3, stage3), (4, stage4)):
        if not active:
            config["loop"]["rerun"][f"stage{stage}"] = True
    paths = resolve_pipeline_paths(config, enabled=resolve_enabled_stages(common_path))
    result, output = _dry_run(tmp_path, config, [paths["s5_signal"]])
    assert result.returncode == 0, output
    assert "rule stage1_vmex:" in output, output
    assert "rule stage2_boozer:" in output, output
    assert "rule stage5_neopax:" in output, output
    assert "rule stage5_post_processing:" in output, output
    assert "Pressure export skipped" in output, output
    assert "fit_vmec_pressure_from_transport_h5.py" not in output, output
    for stage, active in ((3, stage3), (4, stage4)):
        assert (f"checkpoint stage{stage}_prepare:" in output) == active, output


@pytest.mark.parametrize("stage3,stage4", COMBINATIONS[1:])
def test_reuse_ignores_skipped_producers(tmp_path: Path, stage3: bool, stage4: bool) -> None:
    config, common_path, _ = _config(tmp_path, stage3, stage4)
    enabled = resolve_enabled_stages(common_path)
    reuse = resolve_pipeline_paths(config, output_dir=str(tmp_path / "reuse"), enabled=enabled)
    for stage in range(1, 5):
        if enabled[f"stage{stage}"]:
            path = Path(reuse[f"s{stage}_output"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen artifact\n")
    config["loop"]["rerun"] = {f"stage{i}": i == 5 for i in range(1, 6)}
    config["loop"]["reuse_output_dir"] = reuse["output_dir"]
    result, output = _dry_run(tmp_path, config)
    assert result.returncode == 0, output
    assert "rule stage5_neopax:" in output, output
    for stage in range(1, 5):
        assert f"rule stage{stage}_" not in output, output
        assert f"checkpoint stage{stage}_" not in output, output
        if enabled[f"stage{stage}"]:
            assert reuse[f"s{stage}_output"] in output, output
    paths = resolve_pipeline_paths(config, enabled=enabled)
    resolved = tomllib.loads(Path(paths["s5_resolved_config"]).read_text())
    for stage, section, field in ((3, "neoclassical", "neoclassical_file"),
                                  (4, "turbulence", "turbulence_file")):
        expected = os.path.relpath(reuse[f"s{stage}_output"], paths["stage5_dir"]) if enabled[f"stage{stage}"] else ""
        assert resolved[section][field] == expected


@pytest.mark.parametrize("rule", ["stage3_prepare", "stage4_prepare"])
def test_explicit_skipped_rule_target_is_missing(tmp_path: Path, rule: str) -> None:
    config, _, _ = _config(tmp_path, False, False)
    result, output = _dry_run(tmp_path, config, [rule])
    assert result.returncode != 0, output
    assert "No rule to produce" in output, output
    assert rule in output, output


@pytest.mark.parametrize("stage", [3, 4])
def test_enabled_producer_still_requires_its_input(tmp_path: Path, stage: int) -> None:
    config, common_path, _ = _config(tmp_path, stage == 3, stage == 4)
    paths = resolve_pipeline_paths(config, enabled=resolve_enabled_stages(common_path))
    Path(paths[f"s{stage}_config"]).unlink()
    result, output = _dry_run(tmp_path, config)
    assert result.returncode != 0, output
    assert "MissingInputException" in output, output
    assert paths[f"s{stage}_config"] in output, output


def test_skipped_producer_settings_and_freeze_dependencies_are_ignored(tmp_path: Path) -> None:
    config, _, _ = _config(tmp_path, False, False)
    config["stage3"] = 12
    config["stage4"] = {"neopax_radius_relabel": "unsupported"}
    config["loop"]["rerun"].update(stage3=False, stage4=False)
    result, output = _dry_run(tmp_path, config)
    assert result.returncode == 0, output
    assert "rule stage1_vmex:" in output, output
    assert "rule stage5_neopax:" in output, output


def test_apptainer_resources_can_name_omitted_rules(tmp_path):
    config, _, _ = _config(tmp_path, False, False)
    config["container_runtime"] = "apptainer"
    result, output = _dry_run(tmp_path, config, extra_args=[
        "--set-resources", "stage3_prepare:mem_mb=64", "stage3_run_one:mem_mb=64",
        "stage3_collect:mem_mb=64", "stage4_prepare:mem_mb=64",
        "stage4_run_one:mem_mb=64", "stage4_collect:mem_mb=64",
    ])
    assert result.returncode == 0, output
    assert "apptainer run --unsquash" in output, output
    assert "docker run" not in output, output
    assert "stage-3-dkx" not in output and "stage-4-gkx" not in output, output


@pytest.mark.parametrize("stage,solver", [(3, "dkx"), (4, "gkx")])
def test_reenabled_producer_requires_its_settings(tmp_path, stage, solver):
    config, _, _ = _config(tmp_path, stage == 3, stage == 4)
    del config[f"stage{stage}"]
    result, output = _dry_run(tmp_path, config)
    assert result.returncode != 0, output
    assert f"config['stage{stage}']['{solver}'] must be a mapping" in output, output
