"""Shared selector, optional path, and freeze dependency contracts."""

from copy import deepcopy
from itertools import product

import pytest

from src.utils import (
    LOOP_STAGES,
    resolve_common_config_path,
    resolve_enabled_stages,
    resolve_pipeline_paths,
    resolve_rerun_flags,
)


def _enabled(neo, turb):
    return dict(zip(LOOP_STAGES, (True, True, neo, turb, True)))


@pytest.mark.parametrize("neo,turb", product((True, False), repeat=2))
def test_selectors_and_optional_paths(tmp_path, config, neo, turb):
    template = tmp_path / "shared.toml"
    template.write_text(
        f'[neoclassical]\nflux_model = "{"fluxes_r_file" if neo else " NoNe "}"\n'
        f'[turbulence]\nflux_model = "{"model" if turb else "NONE"}"\n'
    )
    enabled = resolve_enabled_stages(template)
    assert enabled == _enabled(neo, turb)
    cfg = deepcopy(config)
    cfg["filenames"]["s5_config"] = "shared.{run_name}.toml"
    for n in (3, 4):
        if not enabled[f"stage{n}"]:
            del cfg["filenames"][f"s{n}_config"]
            cfg["filenames"][f"s{n}_output"] = ["ignored malformed path"]
    paths = resolve_pipeline_paths(cfg, input_dir="/custom/in", output_dir="/custom/out", enabled=enabled)
    assert paths["s5_config"] == f'/custom/in/shared.{cfg["run_name"]}.toml'
    for n in (3, 4):
        for key in (f"s{n}_config", f"s{n}_output", f"stage{n}_dir", f"stage{n}_manifest"):
            assert (key in paths) == enabled[f"stage{n}"]
    assert paths["stage1_dir"] == "/custom/out/stage1_equilibrium"
    assert paths["s1_output"].endswith(f'wout_{cfg["run_name"]}.nc')


@pytest.mark.parametrize("selector", [None, '', 'flux_model = ""'])
def test_missing_and_empty_selectors_keep_enabled(tmp_path, selector):
    template = tmp_path / "shared.toml"
    template.write_text("" if selector is None else f'[neoclassical]\n{selector}\n[turbulence]\n{selector}\n')
    assert resolve_enabled_stages(template) == dict.fromkeys(LOOP_STAGES, True)


def test_legacy_response_and_unrelated_none_do_not_skip(tmp_path):
    template = tmp_path / "shared.toml"
    template.write_text('''
flux_model = "none"
[neoclassical]
model = "none"
response_mode = "none"
entropy_model = "none"
[turbulence]
model = "none"
response_mode = "none"
[classical]
flux_model = "none"
''')
    assert all(resolve_enabled_stages(template).values())


@pytest.mark.parametrize("section", ["neoclassical", "turbulence"])
@pytest.mark.parametrize("body", [
    '{section} = 7',
    '[{section}]\nflux_model = true',
])
def test_selector_errors_name_template_and_setting(tmp_path, section, body):
    template = tmp_path / "bad.toml"
    template.write_text(body.format(section=section))
    with pytest.raises(ValueError) as error:
        resolve_enabled_stages(template)
    assert str(template) in str(error.value)
    assert f"[{section}].flux_model" in str(error.value)


def test_malformed_or_missing_template_fails(tmp_path):
    template = tmp_path / "bad.toml"
    template.write_text('[neoclassical\nflux_model = "none"')
    with pytest.raises(ValueError, match=r"bad.toml.*invalid TOML.*flux_model"):
        resolve_enabled_stages(template)
    with pytest.raises(FileNotFoundError):
        resolve_enabled_stages(tmp_path / "absent.toml")


def test_common_path_requires_only_shared_filename():
    config = {"run_name": "demo", "input_dir": "IN", "filenames": {"s5_config": "shared.{run_name}.toml"}}
    assert resolve_common_config_path(config) == "IN/shared.demo.toml"


@pytest.mark.parametrize("neo,turb,frozen,valid", [
    (False, False, ("stage3", "stage4"), True),
    (False, False, ("stage1", "stage2", "stage5"), True),
    (False, True, ("stage1", "stage2", "stage5"), False),
    (True, False, ("stage1", "stage2", "stage5"), False),
])
def test_freezing_uses_only_remaining_dependencies(neo, turb, frozen, valid):
    config = {"loop": {"rerun": dict.fromkeys(frozen, False)}}
    if valid:
        flags = resolve_rerun_flags(config, _enabled(neo, turb))
        assert {stage for stage, rerun in flags.items() if not rerun} == set(frozen)
    else:
        with pytest.raises(ValueError, match="stage5.*still rerun"):
            resolve_rerun_flags(config, _enabled(neo, turb))


def test_skipping_still_validates_all_flags():
    with pytest.raises(ValueError, match="stage3"):
        resolve_rerun_flags({"loop": {"rerun": {"stage3": "false"}}}, _enabled(False, False))
