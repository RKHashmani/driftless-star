"""Tests for the ``dkx_flux_profiles.h5`` contract in ``src/io_contracts.py``.

Built to the Stage 3 writer: top-level ``r``/``rHat``/``rho`` and ``(n_species,
n_radii)`` ``Gamma``/``Q``/``Upar``, bytes ``species_names``, and a root
``axis_zero_padded`` attribute. Malformed cases run on the inner checker; the valid and
wrong-radius files exercise the h5 read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_dkx_flux, validate_dkx_flux
from tests.helpers.synthetic import write_dkx_flux
from stages.common.dkx_current import CURRENT_CONTRACT_VERSION, CURRENT_NORMALIZATION, CURRENT_UNITS


def _valid_data(n_species: int = 3, n_radii: int = 5) -> dict[str, np.ndarray | None]:
    rho = np.linspace(0.0, 1.0, n_radii)
    flux = np.ones((n_species, n_radii))
    flux[:, 0] = 0.0
    return {
        "current/s": rho[1:]**2, "current/rho": rho[1:], "current/JdotB": rho[1:],
        "current/B2": np.ones_like(rho[1:]), "current/J": rho[1:],
        "r": rho.copy(),
        "rHat": rho.copy(),
        "rho": rho,
        "Gamma": flux.copy(),
        "Q": flux.copy(),
        "Upar": np.zeros((n_species, n_radii)),
        "species_names": np.array([b"e", b"D", b"T"]),
    }


def _attrs(**values):
    return {"current": {"contract_version": CURRENT_CONTRACT_VERSION, "wout_sha256": "",
            **CURRENT_NORMALIZATION, **{f"{k}_units": v for k, v in CURRENT_UNITS.items()}}, **values}


def test_valid_inner_passes() -> None:
    assert _check_dkx_flux(_valid_data(), _attrs(axis_zero_padded=True)) == []


def test_missing_gamma_flagged() -> None:
    data = _valid_data()
    data["Gamma"] = None
    assert "missing required field 'Gamma'" in _check_dkx_flux(data, _attrs(axis_zero_padded=True))


# Flux arrays are shaped (n_species, n_radii). This gives `Q` only 2 species-rows while `species_names` lists 3 species,
# and asserts the checker flags the "species axis" mismatch.
def test_species_axis_mismatch_flagged() -> None:
    data = _valid_data(n_species=3, n_radii=5)
    data["Q"] = np.ones((2, 5))  # 2 species but species_names has 3
    problems = _check_dkx_flux(data, _attrs(axis_zero_padded=True))
    assert any("Q" in problem and "species axis" in problem for problem in problems)


# Passes valid data but an empty attributes dict (`{}`), so the required root attribute `axis_zero_padded` is absent,
# and asserts the checker reports it missing. Shows that file-level metadata, not just datasets, is part of the
# contract.
def test_missing_attr_flagged() -> None:
    assert "missing required root attribute 'axis_zero_padded'" in _check_dkx_flux(_valid_data(), {})


def test_nonfinite_gamma_flagged() -> None:
    data = _valid_data()
    data["Gamma"][0, 0] = np.nan
    assert "'Gamma' contains non-finite values" in _check_dkx_flux(data, _attrs(axis_zero_padded=True))


@pytest.mark.parametrize("padded", [False, True])
def test_valid_file_passes(tmp_path: Path, padded: bool) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    flux = np.ones((3, 5))
    if padded:
        flux[:, 0] = 0.0
    written = write_dkx_flux(tmp_path / "s.h5", rho=rho, gamma=flux, q=flux, axis_zero_padded=padded)
    assert validate_dkx_flux(written) is None


def test_one_radius_cannot_be_interpolated() -> None:
    problems = _check_dkx_flux(_valid_data(n_radii=1), _attrs(axis_zero_padded=True))
    assert any("at least two radii" in problem for problem in problems)


@pytest.mark.parametrize(
    "field, values, expected",
    [
        ("r", [-0.1, 0.25, 0.5, 0.75, 1.0], "'r' must be nonnegative"),
        ("rHat", [0.0, 0.25, 0.25, 0.75, 1.0], "'rHat' must be nonnegative and increase strictly"),
        ("rho", [0.0, 0.5, 0.25, 0.75, 1.0], "'rho' must be nonnegative and increase strictly"),
        ("rHat", [0.0, 0.25, 0.5, 0.75, 1.01], "'r' must match 'rHat'"),
    ],
)
def test_invalid_radius_contract(field: str, values: list[float], expected: str) -> None:
    data = _valid_data()
    data[field] = np.asarray(values)
    problems = _check_dkx_flux(data, _attrs(axis_zero_padded=False))
    assert any(expected in problem for problem in problems)


@pytest.mark.parametrize(
    "padded, field, expected",
    [
        (1, None, "'axis_zero_padded' must be Boolean"),
        (True, "rho", "axis padding requires 'rho' to start at zero"),
        (True, "Q", "axis padding requires 'Q' to start with zero flux"),
    ],
)
def test_inconsistent_axis_padding(padded: bool | int, field: str | None, expected: str) -> None:
    data = _valid_data()
    if field == "rho":
        data[field][0] = 0.1
    elif field == "Q":
        data[field][:, 0] = 1.0
    assert expected in _check_dkx_flux(data, _attrs(axis_zero_padded=padded))


# End-to-end failure path: writes a real file whose flux radius axis (4) disagrees with the length of `rho` (5), and
# asserts `validate_dkx_flux` raises a ContractError mentioning "radius axis". Confirms shape drift is caught when
# reading a real file, not just an in-memory dict.
def test_file_wrong_radius_axis_raises(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    flux = np.ones((3, 4))  # radius axis 4 != len(rho) 5
    written = write_dkx_flux(tmp_path / "s.h5", rho=rho, gamma=flux, q=flux, upar=np.zeros((3, 4)))
    with pytest.raises(ContractError, match="radius axis"):
        validate_dkx_flux(written)


@pytest.mark.parametrize("field, values, expected", [
    ("s", [0., .1, .25, .5625, 1.], "s=rho**2"),
    ("rho", [0., .25, .5, .5, 1.], "increase strictly"),
    ("B2", [1., 1., 0., 1., 1.], "must be positive"),
    ("J", [0., .25, .6, .75, 1.], "JdotB/sqrt(B2)"),
    ("JdotB", [0., .25, np.nan, .75, 1.], "finite 1D"),
])
def test_invalid_current_contract(field, values, expected) -> None:
    data = _valid_data()
    data[f"current/{field}"] = np.array(values)[1:]
    assert any(expected in p for p in _check_dkx_flux(data, _attrs(axis_zero_padded=True)))


def test_legacy_file_requests_regeneration(tmp_path: Path) -> None:
    import h5py
    written = write_dkx_flux(tmp_path / "old.h5", rho=np.linspace(0, 1, 5), gamma=np.ones((3, 5)), q=np.ones((3, 5)))
    with h5py.File(written, "a") as f:
        del f["current"]
    with pytest.raises(ContractError, match="Regenerate Stage 3"):
        validate_dkx_flux(written)
