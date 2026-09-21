"""Tests for the ``src.stage5_helper`` readers of the NEOPAX template.

NEOPAX is configured by a TOML file, not CLI flags. ``prepare_neopax_config``
writes a path-resolved copy of the shared template under the run's Stage 5 output
dir, rewriting its five path fields *relative to that copy's own directory* (NEOPAX
runs there) and never touching the committed template. These tests pin the rewrite
targets, the trailing slash on ``transport_output_dir``, and template immutability.
``read_rho_edge`` reads the radial grid's outer edge back out of the same template,
which is what keeps the Stage 4 relabelling step on the grid NEOPAX will build.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.stage5_helper import (
    prepare_neopax_config,
    read_rho_edge,
    resolve_pressure_convergence_method,
)

_TEMPLATE = (
    "[geometry]\n"
    'vmec_file = "PLACEHOLDER"\n'
    'boozer_file = "PLACEHOLDER"\n'
    "[neoclassical]\n"
    'neoclassical_file = "PLACEHOLDER"\n'
    "[turbulence]\n"
    'turbulence_file = "PLACEHOLDER"\n'
    "[transport_output]\n"
    'transport_output_dir = "PLACEHOLDER"\n'
)


# --- convergence method ---


@pytest.mark.parametrize(
    "config",
    [{}, {"convergence": {}}, {"convergence": {"pressure_rel_tol": 1.0e-2}}],
)
def test_pressure_convergence_method_defaults_to_pointwise(config: dict) -> None:
    assert resolve_pressure_convergence_method(config) == "pointwise"


@pytest.mark.parametrize("method", ["rms", "pointwise"])
def test_pressure_convergence_method_accepts_supported_values(method: str) -> None:
    assert resolve_pressure_convergence_method({"convergence": {"method": method}}) == method


@pytest.mark.parametrize("method", ["RMS", "maximum", True, None, 1])
def test_pressure_convergence_method_rejects_unsupported_values(method: object) -> None:
    with pytest.raises(ValueError, match=r"config\['convergence'\]\['method'\]"):
        resolve_pressure_convergence_method({"convergence": {"method": method}})


@pytest.mark.parametrize("convergence", [None, "rms", ["rms"]])
def test_pressure_convergence_method_requires_a_mapping(convergence: object) -> None:
    with pytest.raises(ValueError, match=r"config\['convergence'\] must be a mapping"):
        resolve_pressure_convergence_method({"convergence": convergence})


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a tmp run and resolve the NEOPAX config; return (template, resolved)."""
    template = tmp_path / "inputs" / "common_input.toml"
    template.parent.mkdir(parents=True)
    template.write_text(_TEMPLATE)

    out = tmp_path / "out"
    resolved = out / "stage5_transport" / "common_input_updated.toml"  # parent not pre-created
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(resolved),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "dkx_flux_profiles.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )
    return template, resolved


# NEOPAX (Stage 5) is configured by a TOML file, not CLI flags. `prepare_neopax_config` writes a copy of the template
# into the Stage 5 output dir and rewrites its five input-path fields so each points at the right upstream artifact,
# relative to that copy's own location. This reads the copy and asserts all five fields now hold the expected
# `../stageN/...` relative path, and that the output dir field resolves to `./` (the copy's own directory).
def test_rewrites_five_paths_relative_and_quoted(tmp_path: Path) -> None:
    _, resolved = _prepare(tmp_path)
    text = resolved.read_text()  # also asserts the parent dir was created
    assert 'vmec_file = "../stage1_equilibrium/wout.nc"' in text
    assert 'boozer_file = "../stage2_boozer/boozmn.nc"' in text
    assert 'neoclassical_file = "../stage3_neoclassical/dkx_flux_profiles.h5"' in text
    assert 'turbulence_file = "../stage4_turbulence/neopax_fluxes.h5"' in text
    # The output dir is the copy's own dir, so it resolves to "./" with a trailing slash.
    assert 'transport_output_dir = "./"' in text


# The rewrite must happen on the copy, never the shared template. After running the same preparation, this reads the
# original template file back and asserts its contents are byte-for-byte unchanged, proving the committed template is
# never mutated by a run.
def test_template_left_unmodified(tmp_path: Path) -> None:
    template, _ = _prepare(tmp_path)
    assert template.read_text() == _TEMPLATE


# --- read_rho_edge ---

def _write_template(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "common_input.toml"
    path.write_text(body)
    return path


def test_read_rho_edge_returns_the_configured_value(tmp_path: Path) -> None:
    assert read_rho_edge(str(_write_template(tmp_path, "[geometry]\nrho_edge = 0.7\n"))) == 0.7


# NEOPAX's own default is 1.0, so a template that never mentions rho_edge grids out to the boundary. Both an absent key
# and an absent [geometry] table have to resolve to it, or the relabelling step would reject a perfectly good grid.
@pytest.mark.parametrize("body", ["[geometry]\nn_radial = 5\n", "[species]\nn_species = 1\n"])
def test_read_rho_edge_defaults_to_one(tmp_path: Path, body: str) -> None:
    assert read_rho_edge(str(_write_template(tmp_path, body))) == 1.0


# rho_edge scales every stage's radial grid, so a value outside (0, 1] is never recoverable downstream. It is caught at
# Snakefile parse time, before any stage runs. A bool passes isinstance(x, int) in Python, hence its own case here.
@pytest.mark.parametrize("value", ['"0.7"', "true", "1.5", "0.0", "-0.5", "nan"])
def test_read_rho_edge_rejects_a_value_outside_the_unit_interval(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match=r"\[geometry\].rho_edge"):
        read_rho_edge(str(_write_template(tmp_path, f"[geometry]\nrho_edge = {value}\n")))

_SECTION_TEMPLATE = '''\
[geometry]
vmec_file = "stale.nc"
boozer_file = "stale.nc"
[neoclassical]
flux_model = " NoNe "
entropy_model = "fluxes_r_file"
neoclassical_file = "stale-neoclassical.h5"
[turbulence]
flux_model = "NONE"
turbulence_file = "stale-turbulence.h5"
[transport_output]
transport_output_dir = "stale-output/"
'''


def _prepare_optional_producers(
    tmp_path: Path,
    body: str,
    *,
    stage3: bool = False,
    stage4: bool = False,
    producer_root: Path | None = None,
) -> tuple[Path, Path]:
    template = tmp_path / "source.toml"
    template.write_text(body)
    out = tmp_path / "out"
    producer_root = producer_root or out
    resolved = out / "stage5_transport" / "resolved.toml"
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(resolved),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(producer_root / "stage3_neoclassical" / "flux.h5") if stage3 else None,
        s4_output=str(producer_root / "stage4_turbulence" / "flux.h5") if stage4 else None,
        s5_output_dir=str(resolved.parent),
    )
    return template, resolved


@pytest.mark.parametrize("stage3", [False, True])
@pytest.mark.parametrize("stage4", [False, True])
def test_optional_producer_paths_and_selectors(tmp_path: Path, stage3: bool, stage4: bool) -> None:
    template, resolved = _prepare_optional_producers(
        tmp_path, _SECTION_TEMPLATE, stage3=stage3, stage4=stage4
    )
    document = tomllib.loads(resolved.read_text())
    assert document["neoclassical"]["neoclassical_file"] == (
        "../stage3_neoclassical/flux.h5" if stage3 else ""
    )
    assert document["turbulence"]["turbulence_file"] == (
        "../stage4_turbulence/flux.h5" if stage4 else ""
    )
    assert document["neoclassical"]["flux_model"] == " NoNe "
    assert document["neoclassical"]["entropy_model"] == "fluxes_r_file"
    assert document["turbulence"]["flux_model"] == "NONE"
    assert document["geometry"]["vmec_file"] == "../stage1_equilibrium/wout.nc"
    assert document["geometry"]["boozer_file"] == "../stage2_boozer/boozmn.nc"
    assert document["transport_output"]["transport_output_dir"] == "./"
    assert template.read_text() == _SECTION_TEMPLATE


@pytest.mark.parametrize(
    "body",
    [
        '[neoclassical]\nflux_model = "none"\n[turbulence]\nflux_model = "none"\n',
        '[neoclassical]\n  "neoclassical_file" = ""\n[turbulence]\n  turbulence_file = ""\n',
        '[unrelated]\nneoclassical_file = "retain.h5"\nturbulence_file = "retain.h5"\n',
    ],
)
def test_absent_or_empty_skipped_fields_need_no_edits(tmp_path: Path, body: str) -> None:
    template, resolved = _prepare_optional_producers(tmp_path, body)
    assert resolved.read_text() == body
    assert template.read_text() == body


def test_frozen_producer_paths_resolve_to_previous_iteration(tmp_path: Path) -> None:
    # JSON surrogate escapes for non-BMP characters are invalid in TOML.
    producer_root = tmp_path / "previous_\U0001f680"
    _, resolved = _prepare_optional_producers(
        tmp_path, _SECTION_TEMPLATE, stage3=True, stage4=True, producer_root=producer_root
    )
    document = tomllib.loads(resolved.read_text())
    for section, field, directory in [
        ("neoclassical", "neoclassical_file", "stage3_neoclassical"),
        ("turbulence", "turbulence_file", "stage4_turbulence"),
    ]:
        assert (resolved.parent / document[section][field]).resolve() == (
            producer_root / directory / "flux.h5"
        )


@pytest.mark.parametrize("section,field,body,existing", [
    ("geometry", "vmec_file", '[{section}]\n  {field} = "old.nc"\n', False),
    ("geometry", "vmec_file", '[{section}]\n{field} = """\nold.nc\n"""\n', True),
    ("neoclassical", "neoclassical_file", '[{section}]\n"{field}" = "stale.h5"\n', True),
    ("neoclassical", "neoclassical_file", '{section}.{field} = "stale.h5"\n', False),
    ("turbulence", "turbulence_file", '{section} = {{{field} = "stale.h5"}}\n', True),
    ("geometry", "boozer_file", '[{section}]\n{field} = "old.nc"\n[other]\n{field} = "keep.nc"\n', True),
    ("transport_output", "transport_output_dir",
     '[{section}]\n{field} = "old/"\n[other]\ntext = \'\'\'\n{field} = "keep/"\n\'\'\'\n', False),
])
def test_unsafe_path_edits_fail_before_writing(
    tmp_path: Path, section: str, field: str, body: str, existing: bool
) -> None:
    body = body.format(section=section, field=field)
    resolved = tmp_path / "out" / "stage5_transport" / "resolved.toml"
    if existing:
        resolved.parent.mkdir(parents=True)
        resolved.write_text("previous resolved config\n")
    with pytest.raises(ValueError) as error:
        _prepare_optional_producers(tmp_path, body)
    message = str(error.value)
    assert str(tmp_path / "source.toml") in message
    assert f"[{section}].{field}" in message
    assert "unquoted key at the start of a line with a single-line value" in message
    assert (tmp_path / "source.toml").read_text() == body
    if existing:
        assert resolved.read_text() == "previous resolved config\n"
    else:
        assert not resolved.parent.exists()


def test_skipped_fields_allow_unrelated_nan(tmp_path: Path) -> None:
    body = _SECTION_TEMPLATE + '\n[other]\nvalue = nan\nvalues = [nan, +nan, -nan]\n'
    template, resolved = _prepare_optional_producers(tmp_path, body)
    document = tomllib.loads(resolved.read_text(), parse_float=str)
    assert document["neoclassical"]["neoclassical_file"] == ""
    assert document["turbulence"]["turbulence_file"] == ""
    assert document["other"] == {"value": "nan", "values": ["nan", "+nan", "-nan"]}
    assert template.read_text() == body


def test_invalid_template_reports_path_fields_without_overwriting(tmp_path: Path) -> None:
    resolved = tmp_path / "out" / "stage5_transport" / "resolved.toml"
    resolved.parent.mkdir(parents=True)
    resolved.write_text("previous resolved config\n")
    with pytest.raises(ValueError, match=r"source\.toml.*\[neoclassical\].neoclassical_file.*invalid TOML"):
        _prepare_optional_producers(tmp_path, "[neoclassical\n")
    assert resolved.read_text() == "previous resolved config\n"


def test_non_table_flux_section_fails_before_creating_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"source\.toml.*\[neoclassical\].neoclassical_file.*TOML table"):
        _prepare_optional_producers(tmp_path, 'neoclassical = "none"\n')
    assert not (tmp_path / "out").exists()


def test_relative_enabled_and_frozen_input_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    template = Path("source.toml")
    template.write_text(_SECTION_TEMPLATE)
    resolved = Path("current/stage5_transport/resolved.toml")
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(resolved),
        s1_output="current/stage1_equilibrium/wout.nc",
        s2_output="current/stage2_boozer/boozmn.nc",
        s3_output="previous/stage3_neoclassical/flux.h5",
        s4_output="current/stage4_turbulence/flux.h5",
        s5_output_dir=str(resolved.parent),
    )
    document = tomllib.loads(resolved.read_text())
    assert document["neoclassical"]["neoclassical_file"] == "../../previous/stage3_neoclassical/flux.h5"
    assert document["turbulence"]["turbulence_file"] == "../stage4_turbulence/flux.h5"
    assert template.read_text() == _SECTION_TEMPLATE
