"""Pipeline path derivation

``resolve_pipeline_paths`` is the single source of truth mapping a run's config
to every file and directory the pipeline uses. The Snakefile and the closed-loop
driver (``src/ouroboros.py``) both import it so they never derive paths differently.
"""

from __future__ import annotations

# Output-file key -> its per-stage output subdirectory name.
_STAGE_SUBDIRS: dict[str, str] = {
    "s1_output": "stage1_equilibrium",
    "s2_output": "stage2_boozer",
    "s3_output": "stage3_neoclassical",
    "s4_output": "stage4_turbulence",
    "s5_output": "stage5_transport",
    "s5_signal": "stage5_post_processing",
}

# Basename of the resolved NEOPAX config written under the Stage 5 output dir.
RESOLVED_COMMON_CONFIG = "common_input_updated.toml"

# Basename of the per-surface scan manifest each prepare checkpoint writes under its stage dir.
MANIFEST_BASENAME = "manifest.json"


def resolve_pipeline_paths(
    config: dict,
    input_dir: str | None = None,
    output_dir: str | None = None,
    enabled: dict[str, bool] | None = None,
) -> dict[str, str]:
    """Turn one run's config into every concrete path the pipeline reads or writes.

    Parameters
    ----------
    config : dict
        Parsed run config; must contain ``run_name``, ``input_dir``,
        ``output_dir``, and ``filenames``. Filename entries are relative to their
        input or stage output directory. The directories can be absolute.
    input_dir, output_dir : str, optional
        Override the config's directories (``None`` keeps the config value). The
        loop driver passes these to run an iteration inside its own
        ``outputs/<run>/loop/iter_N/{input,output}`` sandbox.
    enabled : dict[str, bool], optional
        Enabled flags from ``resolve_enabled_stages``. If omitted, all stages are
        enabled. Skipped producers have no path entries.

    Returns
    -------
    dict[str, str]
        Paths relative to the repository or under absolute directories, in five groups.
        Stage 3 and Stage 4 entries are present only when that producer is enabled.

        - the resolved ``input_dir`` and ``output_dir``;
        - input files ``s1_input``, ``s3_config``, ``s4_config``, ``s5_config``
          (``s5_config`` is the shared ``common_input`` template);
        - output artifacts ``s1_output``..``s5_output``, ``s5_signal``;
        - per-stage output dirs ``stage1_dir``..``stage5_post_dir``;
        - paths generated under ``outputs/``: ``stage3_manifest`` and
          ``stage4_manifest`` (the per-surface scan manifests the prepare
          checkpoints write), ``s5_resolved_config`` (the path-resolved NEOPAX
          copy NEOPAX actually runs), ``s1_feedback`` (the evolved Stage 1
          boundary the loop feeds to the next iteration), and
          ``s5_config_feedback`` (the ``common_input`` copy carrying prescribed
          profiles from the transport solution, seeding the next iteration).

    Raises
    ------
    ValueError
        If the filenames mapping or a required filename is missing or invalid.

    Examples
    --------
    >>> config = {
    ...     "run_name": "demo",
    ...     "input_dir": "inputs/demo",
    ...     "output_dir": "outputs/demo",
    ...     "filenames": {
    ...         "s1_input": "vmec_input.{run_name}",
    ...         "s3_config": "sfincs_input.{run_name}",
    ...         "s4_config": "{run_name}.toml",
    ...         "s5_config": "common_input.toml",
    ...         "s1_output": "wout_{run_name}.nc",
    ...         "s2_output": "boozmn_{run_name}.nc",
    ...         "s3_output": "dkx_flux_profiles.h5",
    ...         "s4_output": "neopax_fluxes.h5",
    ...         "s5_output": "transport_solution.h5",
    ...         "s5_signal": "converge_status.json",
    ...     },
    ... }
    >>> resolve_pipeline_paths(config)["s1_output"]
    'outputs/demo/stage1_equilibrium/wout_demo.nc'
    >>> resolve_pipeline_paths(config, output_dir="outputs/demo/loop/iter_1/output")["s5_signal"]
    'outputs/demo/loop/iter_1/output/stage5_post_processing/converge_status.json'
    """
    enabled = {} if enabled is None else enabled
    input_dir = input_dir if input_dir is not None else config["input_dir"]
    output_dir = output_dir if output_dir is not None else config["output_dir"]

    def fn(key: str) -> str:
        return _filename(config, key)

    def stage_dir(key: str) -> str:
        return f"{output_dir}/{_STAGE_SUBDIRS[key]}"

    def out(key: str) -> str:
        return f"{stage_dir(key)}/{fn(key)}"

    paths = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        # Input files.
        "s1_input": f"{input_dir}/{fn('s1_input')}",
        "s5_config": f"{input_dir}/{fn('s5_config')}",
        # Output artifacts.
        "s1_output": out("s1_output"),
        "s2_output": out("s2_output"),
        "s5_output": out("s5_output"),
        "s5_signal": out("s5_signal"),
        # Per-stage output dirs.
        "stage1_dir": stage_dir("s1_output"),
        "stage2_dir": stage_dir("s2_output"),
        "stage5_dir": stage_dir("s5_output"),
        "stage5_post_dir": stage_dir("s5_signal"),
        # Generated under outputs/.
        "s5_resolved_config": f"{stage_dir('s5_output')}/{RESOLVED_COMMON_CONFIG}",
        "s1_feedback": f"{stage_dir('s5_signal')}/{fn('s1_input')}",
        "s5_config_feedback": f"{stage_dir('s5_signal')}/{fn('s5_config')}",
    }
    for stage in (3, 4):
        if enabled.get(f"stage{stage}", True):
            key = f"s{stage}_output"
            paths[f"s{stage}_config"] = f"{input_dir}/{fn(f's{stage}_config')}"
            paths[key] = out(key)
            paths[f"stage{stage}_dir"] = stage_dir(key)
            paths[f"stage{stage}_manifest"] = f"{stage_dir(key)}/{MANIFEST_BASENAME}"
    return paths


def _filename(config: dict, key: str) -> str:
    """Validate and expand one required filename."""
    files = config.get("filenames")
    if not isinstance(files, dict):
        raise ValueError("config['filenames'] must be a mapping of filename keys to relative paths.")
    value = files.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config['filenames']['{key}'] must be a nonempty path string, got {value!r}.")
    return value.format(run_name=config["run_name"])


def resolve_common_config_path(config: dict) -> str:
    """Resolve the shared template without requiring optional producer filenames.

    Parameters
    ----------
    config : dict
        Run config containing ``input_dir``, ``run_name``, and ``filenames.s5_config``.

    Returns
    -------
    str
        Shared template path under the configured input directory.

    Raises
    ------
    ValueError
        If the filenames mapping or shared template filename is invalid.
    """
    return f"{config['input_dir']}/{_filename(config, 's5_config')}"
