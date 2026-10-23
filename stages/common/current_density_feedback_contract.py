"""Configuration and completion evidence for the optional current subloop."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CONTRACT_VERSION = 1
CURRENT_EDGE_TOL = 1.0e-12
DEFAULT_SETTINGS = {
    "enabled": False,
    "max_iters": 10,
    "rel_tol": 1.0e-2,
    "abs_tol": 1.0,
    "relaxation": 0.5,
}
REQUIRED_ARTIFACTS = {
    "aggregate", "wout", "boozer", "applied_input", "candidate_input", "context",
    "common_config", "dkx_config", "effective_config",
}


def resolve_settings(
    config: dict, *, enabled: dict[str, bool] | None = None, rerun: dict[str, bool] | None = None,
) -> dict:
    """Validate user settings before creating an iteration tree.

    Parameters
    ----------
    config : dict
        Run configuration with an optional ``current_density_feedback`` block.
    enabled, rerun : dict[str, bool], optional
        Stage selection and outer evolution flags used to check prerequisites.

    Returns
    -------
    dict
        Complete settings with defaults. Tolerances and relaxation are floats.

    Raises
    ------
    ValueError
        If settings are invalid or the required stages cannot run.
    """
    supplied = config.get("current_density_feedback")
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("current_density_feedback must be a mapping")
    unknown = supplied.keys() - DEFAULT_SETTINGS.keys()
    if unknown:
        raise ValueError(f"Unknown current_density_feedback settings {sorted(unknown)}")
    settings = {**DEFAULT_SETTINGS, **supplied}
    if not isinstance(settings["enabled"], bool):
        raise ValueError("current_density_feedback.enabled must be true or false")
    limit = settings["max_iters"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("current_density_feedback.max_iters must be a positive integer")
    for key in ("rel_tol", "abs_tol", "relaxation"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"current_density_feedback.{key} must be finite")
        settings[key] = float(value)
    if settings["rel_tol"] < 0 or settings["abs_tol"] < 0:
        raise ValueError("Current convergence tolerances must be nonnegative")
    if not 0 < settings["relaxation"] <= 1:
        raise ValueError("current_density_feedback.relaxation must be in (0, 1]")
    if settings["enabled"]:
        if enabled is not None and not enabled["stage3"]:
            raise ValueError("Current density feedback requires enabled Stage 3")
        if rerun is not None and not all(rerun[key] for key in ("stage1", "stage2", "stage3")):
            raise ValueError("Current density feedback requires loop.rerun for Stages 1 through 3")
    return settings


def artifact(path: str | Path) -> dict[str, str]:
    """Record file content, retaining portable paths relative to the workflow.

    Parameters
    ----------
    path : str or Path
        Existing artifact to hash.

    Returns
    -------
    dict[str, str]
        Original path and SHA-256 digest keyed by ``path`` and ``sha256``.
    """
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def build_record(*, status: str, residual: float, threshold: float, settings: dict,
                 paths: dict[str, str | Path], **diagnostics: object) -> dict:
    """Build a record after both adapter outputs have been validated.

    Parameters
    ----------
    status : str
        Current-loop status, ``continue`` or ``converged``.
    residual, threshold : float
        Maximum current mismatch and its acceptance bound in A/m².
    settings : dict
        Validated current density feedback settings.
    paths : dict[str, str | Path]
        Artifact names and existing files whose contents bind this record.
    **diagnostics : object
        Additional JSON-compatible evidence, including measured current samples.

    Returns
    -------
    dict
        Completion record with content hashes and the current contract version.
    """
    return {
        **diagnostics,
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "residual": float(residual),
        "threshold": float(threshold),
        "settings": settings,
        "artifacts": {name: artifact(path) for name, path in paths.items()},
    }


def validate_record(record: dict | str | Path, *, settings: dict | None = None,
                    expected_paths: dict[str, str | Path] | None = None, require_converged: bool = False,
                    base_dir: str | Path = ".") -> dict:
    """Reject stale settings, changed files, and inconsistent completion evidence.

    Parameters
    ----------
    record : dict, str, or Path
        Parsed completion record or path to its JSON file.
    settings : dict, optional
        Expected current density feedback settings.
    expected_paths : dict[str, str | Path], optional
        Expected artifact locations, keyed by artifact name.
    require_converged : bool, optional
        Require convergence before admitting transport into the DAG.
    base_dir : str or Path, optional
        Base directory for relative artifact paths. Defaults to the current directory.

    Returns
    -------
    dict
        Validated record, including measured samples and artifact provenance.

    Raises
    ------
    ValueError
        If settings, diagnostics, or artifact provenance are inconsistent.
    FileNotFoundError
        If the record or any referenced artifact is missing.
    """
    if not isinstance(record, dict):
        record = json.loads(Path(record).read_text())
    if not isinstance(record, dict):
        raise ValueError("Current density feedback record must be a mapping")
    if record.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Incompatible current density feedback record. Regenerate the current pass.")
    recorded_settings = resolve_settings({"current_density_feedback": record.get("settings")})
    if record.get("settings") != recorded_settings or not recorded_settings["enabled"]:
        raise ValueError("Invalid current density feedback record settings")
    if settings is not None and recorded_settings != settings:
        raise ValueError("Stale current density feedback settings. Regenerate the current pass.")
    residual, threshold = record.get("residual"), record.get("threshold")
    for value in (residual, threshold):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid current density feedback residual or threshold")
    arrays = []
    for name in ("measured_s", "j_dkx", "j_vmex"):
        values = record.get(name)
        if not isinstance(values, list) or len(values) < 2 or any(
            isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"Invalid current density feedback {name} samples")
        arrays.append(values)
    surfaces, dkx, vmex = arrays
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("Current density feedback sample arrays must have matching lengths")
    if (surfaces[0] < 0 or abs(surfaces[-1] - 1.0) > CURRENT_EDGE_TOL
            or any(b <= a for a, b in zip(surfaces, surfaces[1:]))):
        raise ValueError("Current density feedback surfaces must increase within [0, 1] with measured edge coverage")
    calculated_residual = max(abs(a - b) for a, b in zip(dkx, vmex))
    calculated_threshold = (
        recorded_settings["abs_tol"] + recorded_settings["rel_tol"] * max(abs(value) for value in dkx)
    )
    if not math.isclose(residual, calculated_residual, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Current density feedback residual disagrees with recorded current samples")
    if not math.isclose(threshold, calculated_threshold, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Current density feedback threshold disagrees with settings and recorded current samples")
    expected_status = "converged" if residual <= threshold else "continue"
    if record.get("status") != expected_status:
        raise ValueError("Current density feedback status disagrees with its residual")
    if require_converged and expected_status != "converged":
        raise ValueError("Transport requires current convergence evidence")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS <= artifacts.keys():
        raise ValueError("Current density feedback record lacks artifact provenance")
    for name, item in artifacts.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"Invalid current density feedback artifact {name}")
        expected = (expected_paths or {}).get(name, item["path"])
        expected = Path(base_dir) / expected
        if expected.resolve() != (Path(base_dir) / item["path"]).resolve():
            raise ValueError(f"Current density feedback artifact path mismatch for {name}")
        if artifact(expected)["sha256"] != item.get("sha256"):
            raise ValueError(f"Stale current density feedback artifact {name}. Regenerate the current pass.")
    return record
