"""Names of the committed run directories under ``inputs/``."""

from __future__ import annotations

W7X_RUNS = (
    "w7-x/t3d_benchmark",
    "w7-x/mhd_on_neoclassical_on",
    "w7-x/mhd_on_neoclassical_off",
    "w7-x/mhd_off_neoclassical_on",
    "w7-x/mhd_off_neoclassical_off",
)
"""The W7-X runs, each holding a ``config.yaml`` and a ``common_input.toml``."""

TRACKED_RUNS = ("quick_run", *W7X_RUNS)
"""Every committed run directory."""
