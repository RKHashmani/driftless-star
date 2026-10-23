"""Identify the GKX code and precision used by the worker interpreter."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from functools import cache
from pathlib import Path
import subprocess
import sys
import tomllib


def expected_revision() -> str:
    """Read the solver pin from the Stage 4 dependency declaration.

    Returns
    -------
    str
        Expected GKX Git revision.
    """
    with (Path(__file__).resolve().parents[1] / "pixi.toml").open("rb") as stream:
        return tomllib.load(stream)["feature"]["gkx"]["pypi-dependencies"]["gkx"]["rev"]


def installed_revision() -> str:
    """Match the imported module to VCS metadata or a clean source checkout.

    Returns
    -------
    str
        Verified Git revision of the imported GKX package.

    Raises
    ------
    ValueError
        GKX is missing or its imported source has no verifiable revision.
    """
    spec = importlib.util.find_spec("gkx")
    if spec is None or spec.origin is None:
        raise ValueError("GKX is not installed in the worker interpreter")
    origin = Path(spec.origin).resolve()
    try:
        distribution = importlib.metadata.distribution("gkx")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        installed = Path(distribution.locate_file("gkx/__init__.py")).resolve()
        revision = direct.get("vcs_info", {}).get("commit_id")
        if installed == origin and revision:
            return str(revision)
    # PYTHONPATH checkouts must be the imported package's own repository.
    root = origin.parents[2]
    if origin != root / "src/gkx/__init__.py" or not (root / ".git").exists():
        raise ValueError(f"No verifiable GKX revision for {origin}")
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", "src", "pyproject.toml"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot verify GKX checkout {root}") from exc
    if dirty:
        raise ValueError(f"GKX checkout has source changes at {root}")
    return revision


@cache
def current_provenance() -> dict[str, str]:
    """Identify the executing solver once per command in a fresh interpreter.

    Returns
    -------
    dict of str to str
        Expected and installed revisions and effective solver precision.

    Raises
    ------
    ValueError
        The probe fails, times out, or reports a revision other than the pin.

    Notes
    -----
    Geometry imports can enable JAX x64 in the bridge process. The fresh
    interpreter measures the solver's precision under its inherited environment.
    The installed package and environment must remain fixed during a command.
    """
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            text=True, capture_output=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot verify executing GKX. {exc}") from exc
    if process.returncode:
        raise ValueError(f"Cannot verify executing GKX. {process.stderr.strip()}")
    result = json.loads(process.stdout)
    expected = expected_revision()
    if result["installed_revision"] != expected:
        raise ValueError(
            f"Expected GKX {expected}, found {result['installed_revision']}. "
            "Install the pinned environment and prepare again."
        )
    return {"expected_revision": expected, **result}


def verify_recorded_provenance(manifest: dict) -> None:
    """Check prepared solver identity without importing the solver.

    Parameters
    ----------
    manifest : dict
        Prepared scan manifest containing solver provenance.

    Raises
    ------
    ValueError
        Provenance is missing, incomplete, or inconsistent with the project pin.
    """
    prepared = manifest.get("solver_provenance")
    if not isinstance(prepared, dict):
        raise ValueError("Missing solver provenance. Prepare the scan again.")
    expected = expected_revision()
    for key in ("expected_revision", "installed_revision"):
        if prepared.get(key) != expected:
            raise ValueError(
                f"Solver {key} differs. Prepared {prepared.get(key)!r}, expected {expected!r}. "
                "Prepare the scan again."
            )
    if prepared.get("precision") not in ("float32", "float64"):
        raise ValueError(f"Invalid solver precision {prepared.get('precision')!r}. Prepare the scan again.")


def verify_provenance(manifest: dict) -> None:
    """Check that the executing solver matches the prepared identity.

    Parameters
    ----------
    manifest : dict
        Prepared scan manifest containing solver provenance.

    Raises
    ------
    ValueError
        Recorded provenance is invalid, the probe fails, or the solver differs.
    """
    verify_recorded_provenance(manifest)
    prepared = manifest.get("solver_provenance")
    for key, current in current_provenance().items():
        if prepared.get(key) != current:
            hint = " Check JAX_ENABLE_X64 in the solver environment." if key == "precision" else ""
            raise ValueError(
                f"Solver {key} differs. Prepared {prepared.get(key)!r}, executing {current!r}."
                f"{hint} Prepare the scan again."
            )


if __name__ == "__main__":
    revision = installed_revision()
    import jax
    print(json.dumps({"installed_revision": revision,
                      "precision": "float64" if jax.config.x64_enabled else "float32"}))
