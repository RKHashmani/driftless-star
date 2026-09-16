"""Verify that Stage 4 diagnostics come from a successful run with the prepared inputs."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@lru_cache(maxsize=32)
def _cached_equilibrium_digest(path: Path, identity: tuple[int, ...]) -> str:
    return _digest(path)


def _equilibrium_digest(path: Path) -> str:
    """Cache equilibrium reads within this process while checking file metadata."""
    path = path.resolve()
    status = path.stat()
    identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
    return _cached_equilibrium_digest(path, identity)


def completion_path(run: dict[str, Any]) -> Path:
    return Path(f"{run['output_prefix']}.completion.json")


def diagnostics_path(run: dict[str, Any]) -> Path:
    return Path(f"{run['output_prefix']}.diagnostics.csv")


def input_fingerprint(manifest: dict[str, Any], run: dict[str, Any]) -> str:
    """Hash solver inputs and values used by collection, excluding audit-only metadata."""
    runtime_species = run.get("runtime_species", [])
    reference_name = str(manifest.get("normalization", {}).get("reference_species_name", "")).strip().lower()
    reference = next(
        (species for species in runtime_species if str(species.get("name", "")).strip().lower() == reference_name),
        runtime_species[0] if runtime_species else {},
    )
    runtime_names = manifest.get("runtime_species_names", [])
    species_meta = manifest.get("species_meta", [])
    content = {
        "fingerprint_version": 3,
        "runtime_toml": _digest(Path(run["config_path"])),
        "collection": {
            "reference": {key: reference.get(key) for key in (
                "mass", "density_reference_physical", "temperature_reference_physical",
            )},
            "rho_star_physical": run.get("rho_star_physical", 1.0),
            "a_minor": run.get("a_minor", manifest.get("geometry", {}).get("a_minor", 1.0)),
            "axis_a_minor": manifest.get("geometry", {}).get("a_minor", 1.0),
            "species_names": [str(species["name"]) for species in species_meta] if species_meta else runtime_names,
            "runtime_species_names": runtime_names,
            "source_rho": manifest.get("source_rho", []),
            "source_er": manifest.get("source_er", []),
            "radius": {key: run.get(key) for key in ("rho", "r_physical", "rho_index", "torflux", "Er")},
            "response_label": run.get("response_label", "base"),
            "perturb_species": run.get("perturb_species", "none"),
            "perturb_delta": run.get("perturb_delta", 0.0),
        },
        "equilibria": {key: _equilibrium_digest(Path(manifest[key]))
                       for key in ("vmec_file", "booz_file") if manifest.get(key)},
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_diagnostics_csv(path: Path) -> dict[str, np.ndarray]:
    """Read numeric diagnostic columns as arrays, including files with one row."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return {str(name): np.atleast_1d(np.asarray(data[name], dtype=float)) for name in data.dtype.names or ()}


def _diagnostics_status(run: dict[str, Any], *, strict: bool) -> tuple[bool, str]:
    path = diagnostics_path(run)
    if not path.is_file():
        return False, f"missing diagnostics CSV {path}"
    try:
        data = read_diagnostics_csv(path)
        if "t" not in data or data["t"].size == 0:
            return False, "diagnostics have no time samples"
        if strict:
            for name in ("t", "heat_flux", "particle_flux"):
                if name not in data or not np.all(np.isfinite(data[name])):
                    return False, f"diagnostics have missing or nonfinite {name}"
    except (OSError, ValueError) as exc:
        return False, f"unreadable diagnostics CSV ({exc})"
    return True, "valid diagnostics"


def completion_status(manifest: dict[str, Any], run: dict[str, Any]) -> tuple[bool, str]:
    """Return whether the result is valid and the reason, for all workflow callers."""
    trusted = int(manifest.get("schema_version", 1)) >= 3
    valid, reason = _diagnostics_status(run, strict=trusted)
    if not valid or not trusted:
        return valid, reason
    try:
        expected = run.get("input_fingerprint")
        if not expected:
            return False, "manifest has no prepared input fingerprint"
        if input_fingerprint(manifest, run) != expected:
            return False, "runtime inputs changed after preparation"
        marker = completion_path(run)
        if not marker.is_file():
            return False, "missing trusted completion marker"
        completion = json.loads(marker.read_text(encoding="utf-8"))
        if completion.get("input_fingerprint") != expected:
            return False, "completion belongs to different inputs"
        if completion.get("diagnostics_sha256") != _digest(diagnostics_path(run)):
            return False, "diagnostics changed after successful completion"
    except (OSError, ValueError) as exc:
        return False, f"cannot verify completion ({exc})"
    return True, "trusted completion"


def _invalidate_outputs(run: dict[str, Any]) -> None:
    """Remove the marker first so an interrupted retry cannot trust old output."""
    for path in (
        completion_path(run),
        diagnostics_path(run),
        Path(f"{run['output_prefix']}.summary.json"),
    ):
        path.unlink(missing_ok=True)


def begin_attempt(manifest: dict[str, Any], run: dict[str, Any]) -> str | None:
    """Check prepared inputs before removing outputs for a worker attempt."""
    if int(manifest.get("schema_version", 1)) < 3:
        _invalidate_outputs(run)
        return None
    fingerprint = input_fingerprint(manifest, run)
    if not run.get("input_fingerprint") or fingerprint != run["input_fingerprint"]:
        raise ValueError("Runtime inputs changed after preparation. Prepare the scan again.")
    _invalidate_outputs(run)
    if run.get("geometry_file"):
        Path(run["geometry_file"]).unlink(missing_ok=True)
    return fingerprint


def certify_completion(manifest: dict[str, Any], run: dict[str, Any], token: str | None) -> None:
    """Certify fresh output only after the caller has checked solver success."""
    trusted = int(manifest.get("schema_version", 1)) >= 3
    valid, reason = _diagnostics_status(run, strict=trusted)
    if not valid:
        raise ValueError(reason)
    if not trusted:
        return
    if not token or token != run.get("input_fingerprint") or input_fingerprint(manifest, run) != token:
        raise ValueError("Runtime inputs changed during the worker invocation")
    marker = completion_path(run)
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "schema_version": 1,
        "input_fingerprint": token,
        "diagnostics_sha256": _digest(diagnostics_path(run)),
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
