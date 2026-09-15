"""Test the pressure export contract without an equilibrium or transport solver."""

import re
import sys

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tests.helpers.stage_import import load_stage_module

writer = load_stage_module("stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py")


def array(text, key):
    match = re.search(rf"\b{key}\s*=\s*([\dEe+.,\s-]+)", text)
    return np.array([float(v) for v in match[1].replace("\n", " ").split(",") if v.strip()])


RHO = np.array([0., .1, .4, .7, 1.])
NATIVE_PRESSURE = np.array([[9., 7., 5., 3., 1.2], [7., 5., 3., 1.5, .6]])


def transport_input(tmp_path, form="pressure"):
    """Write two distinct time slices with varying face pressure for each species."""
    density = np.broadcast_to(np.array([.5, 2.])[None, :, None], (2, 2, 5))
    temperature = np.array([
        [[10., 8., 6., 4., 2.], [2., 1.5, 1., .5, .1]],
        [[8., 6., 4., 2., 1.], [1.5, 1., .5, .25, .05]],
    ])
    source = tmp_path / "transport.h5"
    with h5py.File(source, "w") as f:
        f["rho_face"] = RHO
        if form == "pressure":
            f["pressure_faces"] = density * temperature
        else:
            f["density_faces"], f["temperature_faces"] = density, temperature
    return source


def run_export(tmp_path, monkeypatch, source, *options):
    template = tmp_path / "input"
    template.write_text("&INDATA\n AM=9,8,7\n PMASS_TYPE='power_series'\n PRES_SCALE=123\n/\n")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["writer", "write-input", str(source), str(template),
                                      "--output-input", str(output), *options])
    writer.main()
    return output.read_text()


@pytest.mark.parametrize("profile_type", [None, "akima_spline", "cubic_spline", "power_series"])
def test_cli_exports_selected_mode_in_pascals(tmp_path, monkeypatch, profile_type):
    options = [] if profile_type is None else ["--profile-type", profile_type]
    text = run_export(tmp_path, monkeypatch, transport_input(tmp_path), *options)
    profile_type = profile_type or "power_series"
    expected = NATIVE_PRESSURE[-1] * 16021.76634
    assert f"PMASS_TYPE = '{profile_type}'" in text
    assert "PRES_SCALE = 1.0000000000000000E+00" in text
    if profile_type == "power_series":
        assert_allclose(np.polynomial.polynomial.polyval(RHO**2, array(text, "AM")), expected)
        assert "AM_AUX" not in text
    else:
        assert_array_equal(array(text, "AM_AUX_S"), RHO**2)
        assert_array_equal(array(text, "AM_AUX_F"), expected)
        assert not re.search(r"\bAM\s*=", text)


@pytest.mark.parametrize("form", ["pressure", "density_temperature"])
def test_cli_converts_each_input_form_once(tmp_path, monkeypatch, form):
    source = transport_input(tmp_path, form)
    text = run_export(tmp_path, monkeypatch, source, "--profile-type", "akima_spline")
    assert "PMASS_TYPE = 'akima_spline'" in text
    assert_array_equal(array(text, "AM_AUX_F"), NATIVE_PRESSURE[-1] * 16021.76634)
    _, native, _ = writer._load_total_pressure(source, time_index=-1, final_time=False)
    assert_allclose(native, NATIVE_PRESSURE[-1])


@pytest.mark.parametrize("selection,expected_index", [
    ([], 1), (["--time-index", "0"], 0), (["--time-index", "0", "--final-time"], 1),
])
def test_cli_preserves_time_selection(tmp_path, monkeypatch, selection, expected_index):
    text = run_export(tmp_path, monkeypatch, transport_input(tmp_path), "--profile-type", "akima_spline", *selection)
    assert_array_equal(array(text, "AM_AUX_F"), NATIVE_PRESSURE[expected_index] * 16021.76634)


@pytest.mark.parametrize("rho", [
    [0., .3, .7], [.1, .5, 1.], [0., .7, .5, 1.], [0., .5, .5, 1.],
    [0., np.nan, 1.], [0., np.inf, 1.], [0.], [], [[0., 1.]],
    [-2e-12, .5, 1.], [0., .5, 1. + 2e-12], [0., .5, 1. - 2e-12],
    [-1e-13, -5e-14, .5, 1.], np.linspace(0, 1, 102),
])
def test_reject_invalid_grid_without_touching_input(tmp_path, rho):
    source = tmp_path / "input"
    source.write_text("&INDATA\n AM=1\n/\n")
    before = source.read_bytes()
    rho = np.array(rho)
    with pytest.raises(ValueError):
        writer._write_vmec_input_with_pressure_spline(source, rho, np.ones_like(rho))
    assert source.read_bytes() == before


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.])
def test_reject_invalid_pressure(tmp_path, value):
    with pytest.raises(ValueError, match="pressure"):
        writer._write_vmec_input_with_pressure_spline(tmp_path / "missing", np.array([0., .5, 1.]), np.array([1., value, 0.]))


def test_endpoint_roundoff_and_maximum_knots(tmp_path):
    source = tmp_path / "input"
    source.write_text("&INDATA\n/\n")
    rho = np.linspace(0, 1, 101)
    rho[0], rho[-1] = -5e-13, 1 + 5e-13
    pressure = np.linspace(100., 0., 101)
    writer._write_vmec_input_with_pressure_spline(source, rho, pressure)
    assert_array_equal(array(source.read_text(), "AM_AUX_S"), np.linspace(0, 1, 101)**2)
    assert_array_equal(array(source.read_text(), "AM_AUX_F"), pressure)
    assert rho[0] == -5e-13  # The writer leaves the source array unchanged.


@pytest.mark.parametrize("terminator", ["/", "&END"])
def test_rewrites_complete_assignments_and_preserves_other_input(tmp_path, terminator):
    source = tmp_path / "input"
    source.write_text("! preamble\n&INDATA\n"
                      "  TITLE='literal AM=8 / and ''quote'''\n\n"
                      "  am(0:2)=1,\n  2, 3, ! pressure note\n"
                      "  AM(3)=4, AI=0.4,0.1\n"
                      "  AM_AUX_S(1:3)=0, .5, 1\n"
                      "  AM_AUX_F=9,\n  8, 7\n"
                      "  pmass_type='cubic_spline', pres_scale=12\n"
                      "  RBC(0,0)=5.5\n" + terminator + "\n&OTHER\n AM=77\n/\n")
    rho, pressure = np.array([0., .2, .6, 1.]), np.array([4., 3.9, 2., .1])
    writer._write_vmec_input_with_pressure_spline(source, rho, pressure)
    first = source.read_text()
    writer._write_vmec_input_with_pressure_spline(source, rho, pressure)
    assert source.read_text() == first
    assert "TITLE='literal AM=8 / and ''quote'''\n\n" in first
    assert "! pressure note" in first
    assert "AI=0.4,0.1" in first
    assert "RBC(0,0)=5.5" in first
    assert first.endswith(terminator + "\n&OTHER\n AM=77\n/\n")
    assert "am(0:2)" not in first and "AM(3)" not in first
    assert_array_equal(array(first, "AM_AUX_S"), rho**2)
    assert_array_equal(array(first, "AM_AUX_F"), pressure)
    writer._write_vmec_input_with_pressure_fit(source, np.array([1., -1.]), output_path=None)
    assert "AM_AUX" not in source.read_text()
    writer._write_vmec_input_with_pressure_spline(source, rho, pressure)
    assert_array_equal(array(source.read_text(), "AM_AUX_F"), pressure)


@pytest.mark.parametrize("options", [["--degree", "2"], ["--drop-axis"]])
def test_splines_reject_polynomial_options(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["writer", "write-input", "missing.h5", "input", "--profile-type", "akima_spline", *options])
    with pytest.raises(SystemExit, match="2"):
        writer.main()


def test_pressure_conversion_rejects_overflow():
    with pytest.raises(ValueError, match="pascals must be finite"):
        writer._pressure_in_pascals(np.array([1e305]))
