"""Check CPU helper jobs and GPU solver jobs in the planned workflow."""

from pathlib import Path
import re
import subprocess

import pytest
import yaml

from src.utils import resolve_pipeline_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
CPU_RULES = ("stage3_prepare", "stage3_collect", "stage4_prepare", "stage4_collect", "stage5_post_processing")
GPU_RULES = ("stage1_vmex", "stage2_boozer", "stage3_run_one", "stage4_run_one", "stage5_neopax")


@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
def test_gpu_run_keeps_helpers_on_cpu(tmp_path: Path, runtime: str) -> None:
    """Resolve profile resources and shell commands without a cluster or containers."""
    profile = yaml.safe_load(
        (REPO_ROOT / "executors/htcondor/profiles/htcondor-gpu/config.yaml").read_text()
    )
    # Use the real resource settings with the local executor so this also runs on macOS.
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "config.yaml").write_text(yaml.safe_dump({
        key: profile[key] for key in ("default-resources", "set-resources", "set-threads")
    }))
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    paths = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")
    overrides = tmp_path / "relabel.yaml"
    overrides.write_text(yaml.safe_dump({"stage4": {"neopax_radius_relabel": "boozer_volume"}}))
    surface = "rho_001_r0p2500"
    result = subprocess.run(
        [
            "snakemake", "-n", "-p", "--cores", "8",
            paths["s5_signal"],
            f"{paths['stage3_dir']}/runs/{surface}/result.json",
            f"{paths['stage4_dir']}/runs/{surface}/run.diagnostics.csv",
            "--profile", str(profile_dir), "--workflow-profile", "none",
            "--runtime-source-cache-path", f"{tmp_path}/srccache",
            "--configfile", "inputs/quick_run/config.yaml", str(overrides),
            "--config", f"output_dir={tmp_path}/out", "gpu_ids=all", f"container_runtime={runtime}",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for rule in (*CPU_RULES, *GPU_RULES):
        match = re.search(rf"(?:rule|checkpoint) {rule}:\n(.*?)(?=\n\n|\Z)", output, re.DOTALL)
        assert match, output
        block = match.group(1)
        gpu = rule in GPU_RULES
        assert f"request_gpus={int(gpu)}" in block, block
        assert "threads: 4" in block, block
        minimum_memory = "24GB" if rule == "stage4_run_one" else "12GB"
        assert f"gpus_minimum_memory={minimum_memory}" in block, block
        command = block.split("Shell command:", 1)[1]
        images = re.findall(r"ghcr.io/driftless-star/driftless-star:\S+", command)
        assert images and all(image.endswith("-gpu") for image in images), command
        if runtime == "apptainer":
            assert ("--nv " in command) == gpu, command
        else:
            assert ("--gpus " in command) == gpu, command
            assert ("src.gpu_slots" in command) == gpu, command
        if rule == "stage3_prepare":
            # This selects the later workers' backend and must survive CPU preparation.
            assert "--backend gpu" in command, command
        if rule == "stage4_collect":
            assert "relabel_neopax_flux_radius.py" in command, command
        if rule == "stage4_run_one":
            assert "request_memory=64GB" in block, block
