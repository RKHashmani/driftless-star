"""Replace selected VMEC INDATA assignments while preserving the rest of the input."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

import numpy as np
from numpy.typing import ArrayLike


PRESSURE_KEYS = frozenset({"PMASS_TYPE", "PRES_SCALE", "AM", "AM_AUX_S", "AM_AUX_F"})
CURRENT_KEYS = frozenset({"NCURR", "PCURR_TYPE", "AC", "AC_AUX_S", "AC_AUX_F", "CURTOR"})

# Mask quoted strings and comments before reading namelist syntax. A slash or
# assignment in a title or comment must not mark the end of the INDATA block.
_NAMELIST_LITERAL = re.compile(r"'(?:(?:'')|[^'])*'|\"(?:(?:\"\")|[^\"])*\"|![^\n]*")
_ASSIGNMENT = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*=")


def rewrite_assignments(text: str, keys: Iterable[str], assignments: Sequence[str]) -> str:
    """Replace complete selected assignments, including indexed and continued arrays.

    Parameters
    ----------
    text : str
        VMEC input containing an ``INDATA`` namelist.
    keys : iterable of str
        Assignment names to replace, matched without regard to case.
    assignments : sequence of str
        Replacement lines, including any array continuation lines.

    Returns
    -------
    str
        Updated input preserving comments and unrelated assignments.

    Raises
    ------
    ValueError
        If the ``INDATA`` block or its terminator is missing.
    """
    keys = {key.upper() for key in keys}
    masked = _NAMELIST_LITERAL.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    start = re.search(r"&\s*INDATA\b", masked, re.IGNORECASE)
    if start is None:
        raise ValueError("Input does not contain an &INDATA block")
    end = re.search(r"/|&end\b", masked[start.end():], re.IGNORECASE)
    if end is None:
        raise ValueError("Input does not contain a terminating '/' for &INDATA")
    body_end = start.end() + end.start()
    matches = list(_ASSIGNMENT.finditer(masked, start.end(), body_end))
    body = text[start.end():body_end]
    remove = np.zeros(len(body), dtype=bool)
    for i, match in enumerate(matches):
        if match.group(1).upper() not in keys:
            continue
        stop = matches[i + 1].start() if i + 1 < len(matches) else body_end
        stop = match.start() + len(text[match.start():stop].rstrip())
        remove[match.start() - start.end():stop - start.end()] = True
    # Keep comments, including those between continuation lines. Leave all
    # unrelated assignments, blank lines, and namelist blocks as supplied.
    for literal in _NAMELIST_LITERAL.finditer(body):
        if literal.group().startswith("!"):
            remove[literal.start():literal.end()] = False
    lines = []
    offset = 0
    for line in body.splitlines(keepends=True):
        deleted = remove[offset:offset + len(line)]
        kept = "".join(char for char, drop in zip(line, deleted) if not drop or char == "\n")
        if kept.strip() or not np.any(deleted):
            lines.append(kept)
        offset += len(line)
    body = "".join(lines)
    if not body.endswith("\n"):
        body += "\n"
    return text[:start.end()] + body + "".join("  " + line + "\n" for line in assignments) + text[body_end:]


def rewrite_current_profile(text: str, s: ArrayLike, enclosed_current: ArrayLike) -> str:
    """Write an enclosed toroidal current spline in amperes on normalized flux.

    Parameters
    ----------
    text : str
        Applied VMEC input whose current assignments will be replaced.
    s : array_like
        Two to 101 strictly increasing knots spanning normalized toroidal flux [0, 1].
    enclosed_current : array_like
        Finite enclosed currents in amperes, with zero current at the axis.

    Returns
    -------
    str
        Input with ``NCURR=1``, an ``akima_spline_i`` profile, and the edge current in ``CURTOR``.

    Raises
    ------
    ValueError
        If knots are invalid, or an exactly zero edge current has a nonzero interior.
    """
    s = np.asarray(s, dtype=float)
    current = np.asarray(enclosed_current, dtype=float)
    if s.ndim != 1 or s.size < 2 or current.shape != s.shape:
        raise ValueError("Current export requires matching 1-D arrays with at least two knots")
    if s.size > 101:
        raise ValueError(f"VMEX accepts at most 101 spline knots, got {s.size}")
    if not np.all(np.isfinite(s)) or not np.all(np.diff(s) > 0) or s[0] != 0.0 or s[-1] != 1.0:
        raise ValueError("Current knots must be finite, strictly increasing, and span [0, 1]")
    if not np.all(np.isfinite(current)) or current[0] != 0.0:
        raise ValueError("Enclosed current must be finite and zero at the axis")
    if current[-1] == 0.0 and np.any(current != 0.0):
        raise ValueError("An exactly zero edge current with nonzero interior cannot be normalized by VMEX")
    assignments = ["NCURR = 1", "PCURR_TYPE = 'akima_spline_i'"]
    for key, values in (("AC_AUX_S", s), ("AC_AUX_F", current)):
        for offset in range(0, len(values), 4):
            prefix = f"{key} = " if offset == 0 else "  "
            assignments.append(prefix + ", ".join(f"{float(v):.16E}" for v in values[offset:offset + 4]) + ",")
    assignments.append(f"CURTOR = {float(current[-1]):.16E}")
    return rewrite_assignments(text, CURRENT_KEYS, assignments)
