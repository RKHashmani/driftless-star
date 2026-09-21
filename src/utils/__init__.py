"""Shared utilities for the driftless-star Snakemake workflow."""

from .config_edit import apply_assignments
from .docker import resolve_docker_user
from .gpu import GpuSettings, parse_gpu_pool, resolve_gpu_settings
from .loop import LOOP_STAGES, resolve_enabled_stages, resolve_rerun_flags
from .paths import RESOLVED_COMMON_CONFIG, resolve_common_config_path, resolve_pipeline_paths

__all__ = [
    "GpuSettings",
    "LOOP_STAGES",
    "RESOLVED_COMMON_CONFIG",
    "apply_assignments",
    "parse_gpu_pool",
    "resolve_common_config_path",
    "resolve_docker_user",
    "resolve_enabled_stages",
    "resolve_gpu_settings",
    "resolve_pipeline_paths",
    "resolve_rerun_flags",
]
