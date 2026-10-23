"""Run current density feedback to convergence inside one outer transport iteration."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
import logging
from pathlib import Path
import shutil

import yaml
from snakemake.utils import update_config

from stages.common.current_density_feedback_contract import artifact, resolve_settings, validate_record
from .utils import resolve_gpu_settings, resolve_pipeline_paths

logger = logging.getLogger(__name__)
REUSE_KEYS = {"s1_input", "s1_output", "s2_output", "s3_output", "current_density_feedback_record"}


def current_paths(output_dir: str) -> dict[str, str]:
    """Return the candidate input and completion-record paths for one pass.

    Parameters
    ----------
    output_dir : str
        Output directory for the equilibrium pass.

    Returns
    -------
    dict[str, str]
        Paths keyed by ``candidate`` and ``record``.
    """
    directory = f"{output_dir}/current_density_feedback"
    return {"candidate": f"{directory}/input.current", "record": f"{directory}/convergence.json"}


def write_if_changed(path: Path, text: str) -> None:
    """Keep dependency timestamps stable when resuming an unchanged pass.

    Parameters
    ----------
    path : Path
        Destination file. Missing parent directories are created.
    text : str
        Desired contents. An identical existing file is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _absolute(root: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _seed(root: Path, source: str, destination: str) -> None:
    source_path, target = _absolute(root, source), _absolute(root, destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or source_path.read_bytes() != target.read_bytes():
        shutil.copy2(source_path, target)


def merge_config(base: dict, overrides: dict) -> dict:
    """Apply the same recursive mapping overrides as Snakemake.

    Parameters
    ----------
    base, overrides : dict
        Base configuration and overrides to apply without modifying either.

    Returns
    -------
    dict
        Merged configuration with independent nested values.
    """
    result = deepcopy(base)
    update_config(result, deepcopy(overrides))
    return result


def resolve_artifact_reuse(config: dict) -> dict[str, str]:
    """Read the accepted pass without changing the outer evolution policy.

    Parameters
    ----------
    config : dict
        Run configuration with an optional driver-owned ``artifact_reuse`` map.

    Returns
    -------
    dict[str, str]
        Accepted artifact paths, or an empty mapping when reuse is absent.

    Raises
    ------
    ValueError
        If reuse has invalid keys or paths, or current density feedback is disabled.
    """
    reuse = config.get("artifact_reuse")
    if reuse is None:
        return {}
    if not isinstance(reuse, dict) or set(reuse) != REUSE_KEYS:
        raise ValueError(f"artifact_reuse must contain {sorted(REUSE_KEYS)}")
    if not resolve_settings(config)["enabled"]:
        raise ValueError("artifact_reuse requires enabled current density feedback")
    if any(not isinstance(path, str) or not path.strip() for path in reuse.values()):
        raise ValueError("artifact_reuse paths must be nonempty strings")
    return dict(reuse)


def _comparable_config(config: dict) -> dict:
    """Normalize GPU syntax while retaining physics and solver settings for comparison."""
    ignored = {"input_dir", "output_dir", "artifact_reuse", "current_density_feedback_pass", "gpu_ids", "jobs_per_gpu"}
    comparable = {key: value for key, value in config.items() if key not in ignored}
    comparable["gpu_settings"] = resolve_gpu_settings(config)
    return comparable


def convergence_inputs(config: dict, reuse: dict[str, str]) -> list[str]:
    """Require valid evidence only when the transport DAG is requested.

    Parameters
    ----------
    config : dict
        Fully merged configuration for the transport pass.
    reuse : dict[str, str]
        Accepted paths from ``resolve_artifact_reuse``.

    Returns
    -------
    list[str]
        Evidence and configuration files to declare as Snakemake inputs.
        Empty when current density feedback is disabled.

    Raises
    ------
    ValueError
        If evidence is absent, stale, unconverged, or inconsistent with transport.
    """
    settings = resolve_settings(config)
    if not settings["enabled"]:
        return []
    if not reuse:
        raise ValueError("Transport requires current convergence evidence. Run Ouroboros with --max-iters 1.")
    record_path = reuse["current_density_feedback_record"]
    record = validate_record(record_path, settings=settings, require_converged=True, expected_paths={
        "applied_input": reuse["s1_input"], "wout": reuse["s1_output"],
        "boozer": reuse["s2_output"], "aggregate": reuse["s3_output"],
    })
    paths = resolve_pipeline_paths(config, enabled={"stage4": False})
    for name, key in (("common_config", "s5_config"), ("dkx_config", "s3_config")):
        if artifact(paths[key])["sha256"] != record["artifacts"][name]["sha256"]:
            raise ValueError(f"Transport {name} differs from the accepted current pass")
    accepted_config = yaml.safe_load(Path(record["artifacts"]["effective_config"]["path"]).read_text())
    if _comparable_config(config) != _comparable_config(accepted_config):
        raise ValueError("Transport settings differ from the accepted current pass")
    return list(dict.fromkeys([
        record_path, paths["s3_config"], paths["s5_config"],
        *(item["path"] for item in record["artifacts"].values()),
    ]))


def run_j_subloop(*, config: dict, outer_paths: dict[str, str], enabled: dict[str, bool], rerun: dict[str, bool],
                  repo_root: Path, run_forward_pass: Callable[..., None], cores: int,
                  extra_config: list[str] | None = None, profile: str | None = None,
                  htcondor_jobdir: str | None = None) -> dict[str, str]:
    """Run equilibrium passes through Snakemake and return the accepted artifacts.

    Parameters
    ----------
    config : dict
        Fully merged outer configuration, including profile selection and solver settings.
    outer_paths : dict[str, str]
        Seeded outer inputs and output directory from ``resolve_pipeline_paths``.
    enabled, rerun : dict[str, bool]
        Enabled stages and the outer evolution policy.
    repo_root : Path
        Base directory for relative pipeline paths.
    run_forward_pass : callable
        Snakemake command dispatcher, called once per equilibrium pass.
    cores : int
        Core limit forwarded to every invocation.
    extra_config : list[str], optional
        Command-line configuration overrides forwarded unchanged.
    profile, htcondor_jobdir : str, optional
        Executor profile and job-log directory forwarded unchanged.

    Returns
    -------
    dict[str, str]
        Accepted Stage 1 through 3 outputs, applied input, and convergence record.

    Raises
    ------
    ValueError
        If settings or completion evidence are invalid.
    RuntimeError
        If the equilibrium pass limit is exhausted before convergence.
    """
    settings = resolve_settings(config, enabled=enabled, rerun=rerun)
    if not settings["enabled"]:
        raise ValueError("run_j_subloop requires enabled current density feedback")
    applied_input = outer_paths["s1_input"]
    for iteration in range(1, settings["max_iters"] + 1):
        directory = f"{outer_paths['output_dir']}/j_subloop/iter_{iteration}"
        paths = resolve_pipeline_paths(config, input_dir=f"{directory}/input",
                                       output_dir=f"{directory}/output", enabled=enabled)
        for key in ("s3_config", "s4_config", "s5_config"):
            if key in outer_paths:
                _seed(repo_root, outer_paths[key], paths[key])
        _seed(repo_root, applied_input, paths["s1_input"])
        pass_config = deepcopy(config)
        pass_config.update(input_dir=paths["input_dir"], output_dir=paths["output_dir"])
        context_path = f"{paths['input_dir']}/current_density_feedback_context.json"
        pass_config["current_density_feedback_pass"] = {"context": context_path}
        effective_path = f"{paths['input_dir']}/effective_config.yaml"
        write_if_changed(_absolute(repo_root, effective_path), yaml.safe_dump(pass_config, sort_keys=True))
        context = {"settings": settings, "inputs": {
            "common_config": paths["s5_config"], "dkx_config": paths["s3_config"],
            "effective_config": effective_path,
        }}
        write_if_changed(_absolute(repo_root, context_path), json.dumps(context, indent=2, sort_keys=True) + "\n")
        outputs = current_paths(paths["output_dir"])
        logger.info("Current pass %d of %d under %s", iteration, settings["max_iters"], directory)
        run_forward_pass(target=outputs["record"], input_dir=paths["input_dir"], output_dir=paths["output_dir"],
                         cores=cores, config_path=_absolute(repo_root, effective_path), repo_root=repo_root,
                         extra_config=extra_config, profile=profile, htcondor_jobdir=htcondor_jobdir)
        record = validate_record(
            _absolute(repo_root, outputs["record"]), settings=settings, base_dir=repo_root, expected_paths={
                "applied_input": paths["s1_input"], "candidate_input": outputs["candidate"],
                "wout": paths["s1_output"], "boozer": paths["s2_output"], "aggregate": paths["s3_output"],
                "context": context_path, **context["inputs"],
            },
        )
        logger.info("Current residual %.6g A/m², threshold %.6g A/m²", record["residual"], record["threshold"])
        if record["status"] == "converged":
            return {**{key: paths[key] for key in ("s1_input", "s1_output", "s2_output", "s3_output")},
                    "current_density_feedback_record": outputs["record"]}
        applied_input = outputs["candidate"]
    raise RuntimeError(
        f"Current density feedback did not converge after {settings['max_iters']} equilibrium passes. "
        "Transport stopped."
    )
