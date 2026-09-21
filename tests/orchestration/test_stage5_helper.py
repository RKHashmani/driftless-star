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


# --- [profiles] validation ---

# Stage 5 is the last rule in the DAG, so a prescribed [profiles] block NEOPAX cannot load used to
# surface only after Stages 1-4 had run. The closed-loop driver writes such a block into the
# template from iteration 2 onward. These pin that a malformed one is rejected at parse time.

_SPECIES_AND_GEOMETRY = '[species]\nnames = ["e", "ion"]\n\n[geometry]\nn_radial = 3\n\n'


def _prepare_with_profiles(
    tmp_path: Path,
    profiles: str,
    structure: str = _SPECIES_AND_GEOMETRY,
) -> None:
    """Resolve a config with the supplied profile and dimension blocks."""
    template = tmp_path / "inputs" / "common_input.toml"
    template.parent.mkdir(parents=True, exist_ok=True)
    # TOML allows one [geometry] table, so the structure's geometry keys join the path template's.
    geometry_header = "[geometry]\n"
    species, _, geometry_body = structure.partition(geometry_header)
    template.write_text(
        geometry_header + geometry_body + _TEMPLATE.removeprefix(geometry_header) + "\n" + species + profiles
    )
    out = tmp_path / "out"
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(out / "stage5_transport" / "common_input_updated.toml"),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "sfincs_flux.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )


def test_scalar_analytical_parameters_are_accepted(tmp_path: Path) -> None:
    _prepare_with_profiles(tmp_path, '[profiles]\nmodel = "standard_analytical"\nn0 = 4.21\nT0 = 17.8\n')


# NEOPAX coerces per-species lists for every analytical parameter, shape exponents included, so
# parse-time validation must let them through.
def test_per_species_lists_under_the_analytical_model_are_accepted(tmp_path: Path) -> None:
    _prepare_with_profiles(
        tmp_path,
        '[profiles]\nmodel = "standard_analytical"\n'
        "n0 = [4.21, 4.21]\nn_edge = [0.4, 0.4]\n"
        "T0 = [6.7, 1.0]\nT_edge = [0.2, 0.2]\n"
        "density_shape_power = [1.0, 1.0]\ntemperature_shape_power = [2.0, 2.0]\n"
        "density_shape_alpha = [1.5, 1.5]\ntemperature_shape_alpha = [2.0, 1.0]\n",
    )


_VALID_PRESCRIBED = """\
[profiles]
model = "prescribed"
density = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
temperature = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
Er = [0.0, 0.1, 0.2]
density_face = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
temperature_face = [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
Er_face = [0.0, 0.1, 0.2, 0.3]
density_grad_face = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
temperature_grad_face = [[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]
"""

_REQUIRED_PRESCRIBED_ARRAYS = (
    "density",
    "temperature",
    "Er",
    "density_face",
    "temperature_face",
    "Er_face",
    "density_grad_face",
    "temperature_grad_face",
)

_SPECIES_RESOLVED_ARRAYS = (
    "density",
    "temperature",
    "density_face",
    "temperature_face",
    "density_grad_face",
    "temperature_grad_face",
)


def _replace_profile_value(profiles: str, key: str, value: str | None) -> str:
    """Replace or remove one assignment in a profile fixture."""
    prefix = f"{key} = "
    lines = profiles.splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    assert len(matches) == 1
    index = matches[0]
    if value is None:
        del lines[index]
    else:
        lines[index] = prefix + value
    return "\n".join(lines) + "\n"


def test_valid_prescribed_block_is_accepted(tmp_path: Path) -> None:
    _prepare_with_profiles(tmp_path, _VALID_PRESCRIBED)


@pytest.mark.parametrize("key", _REQUIRED_PRESCRIBED_ARRAYS)
def test_prescribed_block_requires_each_state_array(tmp_path: Path, key: str) -> None:
    """The validator must reject each missing center, face, or gradient array."""
    profiles = _replace_profile_value(_VALID_PRESCRIBED, key, None)
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key} is missing"):
        _prepare_with_profiles(tmp_path, profiles)


@pytest.mark.parametrize(
    ("key", "one_row"),
    [
        ("density", "[[1.0, 2.0, 3.0]]"),
        ("temperature", "[[1.0, 2.0, 3.0]]"),
        ("density_face", "[[1.0, 2.0, 3.0, 4.0]]"),
        ("temperature_face", "[[1.0, 2.0, 3.0, 4.0]]"),
        ("density_grad_face", "[[1.0, 2.0, 3.0, 4.0]]"),
        ("temperature_grad_face", "[[1.0, 2.0, 3.0, 4.0]]"),
    ],
)
def test_species_resolved_arrays_match_species_names(tmp_path: Path, key: str, one_row: str) -> None:
    """Each species-resolved array must contain one row per species name."""
    profiles = _replace_profile_value(_VALID_PRESCRIBED, key, one_row)
    with pytest.raises(ValueError, match=r"holds 1 species rows, expected 2"):
        _prepare_with_profiles(tmp_path, profiles)


@pytest.mark.parametrize(
    ("key", "short_value", "expected_size"),
    [
        ("density", "[[1.0, 2.0], [3.0, 4.0]]", 3),
        ("temperature", "[[1.0, 2.0], [3.0, 4.0]]", 3),
        ("Er", "[0.0, 0.1]", 3),
        ("density_face", "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]", 4),
        ("temperature_face", "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]", 4),
        ("Er_face", "[0.0, 0.1, 0.2]", 4),
        ("density_grad_face", "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]", 4),
        ("temperature_grad_face", "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]", 4),
    ],
)
def test_profile_arrays_match_their_radial_grid(
    tmp_path: Path,
    key: str,
    short_value: str,
    expected_size: int,
) -> None:
    """Center arrays use n_radial points and face arrays use one more point."""
    profiles = _replace_profile_value(_VALID_PRESCRIBED, key, short_value)
    with pytest.raises(ValueError, match=rf"rows must hold {expected_size} points"):
        _prepare_with_profiles(tmp_path, profiles)


@pytest.mark.parametrize(
    ("key", "wrong_rank", "ndim"),
    [
        *((key, "[1.0, 2.0, 3.0]", 2) for key in _SPECIES_RESOLVED_ARRAYS),
        ("Er", "[[0.0, 0.1, 0.2]]", 1),
        ("Er_face", "[[0.0, 0.1, 0.2, 0.3]]", 1),
    ],
)
def test_profile_arrays_have_the_required_rank(
    tmp_path: Path,
    key: str,
    wrong_rank: str,
    ndim: int,
) -> None:
    """The validator must reject vectors and tables in the wrong positions."""
    profiles = _replace_profile_value(_VALID_PRESCRIBED, key, wrong_rank)
    with pytest.raises(ValueError, match=rf"must be a {ndim}-D array"):
        _prepare_with_profiles(tmp_path, profiles)


def test_prescribed_block_requires_species_names(tmp_path: Path) -> None:
    """Species dimensions require a nonempty names array."""
    structure = "[geometry]\nn_radial = 3\n\n"
    with pytest.raises(ValueError, match=r"\[species\]\.names"):
        _prepare_with_profiles(tmp_path, _VALID_PRESCRIBED, structure)


def test_prescribed_block_requires_n_radial(tmp_path: Path) -> None:
    """Radial dimensions require an integer n_radial value."""
    structure = '[species]\nnames = ["e", "ion"]\n\n[geometry]\n\n'
    with pytest.raises(ValueError, match=r"\[geometry\]\.n_radial"):
        _prepare_with_profiles(tmp_path, _VALID_PRESCRIBED, structure)


# NEOPAX closes the outer boundary from the two outermost cells, and Stages 3 and 4 rebuild the same
# grid. A one-cell run therefore has no consumer. The n_radial check runs before any array shape
# check, so the fixture arrays stay at their own length here.
def test_prescribed_block_requires_at_least_two_radial_cells(tmp_path: Path) -> None:
    """The validator rejects a single-cell radial grid."""
    structure = '[species]\nnames = ["e", "ion"]\n\n[geometry]\nn_radial = 1\n\n'
    with pytest.raises(ValueError, match=r"\[geometry\]\.n_radial to be an integer of at least 2"):
        _prepare_with_profiles(tmp_path, _VALID_PRESCRIBED, structure)


# NEOPAX reads its centered arrays under either spelling, but the Stage 3 and Stage 4 reader accepts
# only "prescribed". A "given" block would therefore run NEOPAX on state the earlier stages never saw.
def test_given_is_rejected_in_favour_of_prescribed(tmp_path: Path) -> None:
    """The validator rejects the ``given`` synonym and names the accepted spelling."""
    profiles = _replace_profile_value(_VALID_PRESCRIBED, "model", '"given"')
    with pytest.raises(ValueError, match=r"\[profiles\]\.model is 'given'\. Use 'prescribed'"):
        _prepare_with_profiles(tmp_path, profiles)


# TOML parses a bare true or false into a Python bool, which is an int subclass. A numeric-element
# check that only excludes nested lists would let a bool reach the solver.
@pytest.mark.parametrize(
    ("key", "value", "ndim"),
    [
        ("density", '[["a", 2.0, 3.0], [4.0, 5.0, 6.0]]', 2),
        ("density_face", "[[true, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]", 2),
        ("Er", '[0.0, "0.1", 0.2]', 1),
        ("Er_face", "[0.0, 0.1, 0.2, false]", 1),
    ],
)
def test_profile_arrays_hold_only_numbers(tmp_path: Path, key: str, value: str, ndim: int) -> None:
    """The validator rejects string and boolean entries wherever they appear."""
    profiles = _replace_profile_value(_VALID_PRESCRIBED, key, value)
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key} must be a {ndim}-D array of numbers"):
        _prepare_with_profiles(tmp_path, profiles)


# Every tracked run config must pass the parse-time validation.
@pytest.mark.parametrize("run", ["w7-x_quick_run", "w7-x_t3d_validation", "quick_run"])
def test_tracked_configs_pass_validation(tmp_path: Path, run: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    template = repo_root / "inputs" / run / "common_input.toml"
    out = tmp_path / "out"
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(out / "stage5_transport" / "common_input_updated.toml"),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "sfincs_flux.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )


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
