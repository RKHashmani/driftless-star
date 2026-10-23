"""Check pressure and current edits on shared VMEC inputs."""

import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from common.vmec_namelist import rewrite_assignments, rewrite_current_profile
from tests.helpers.stage_import import load_stage_module

pressure_writer = load_stage_module("stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py")


VMEC_INPUT = """! preamble with NCURR=99 /
&INDATA
  TITLE='current AC=9 / and ''quoted'' text'
  MGRID_FILE="literal ! CURTOR=8 / and ""quote"" tail"

  NCURR = 0, PCURR_TYPE='power_series'
  AC(0:2) = 5, &
    ! current continuation
    & 4, 3,
  AC(3)=2, AI(0:1)=0.4,0.1
  AC_AUX_S(1:3)=0, .5, 1
  AC_AUX_F=8, 7,
    6
  CURTOR=123, ! old edge current
  PMASS_TYPE='cubic_spline'
  AM_AUX_S=0,.5,1
  AM_AUX_F=10,5,0
  PRES_SCALE=7
  PIOTA_TYPE='power_series'
  AI_AUX_S=0,1
  AI_AUX_F=.4,.6
  RBC(0,0)=5.5, ZBS(1,1)=0.02
/
&OTHER
 AC=99, AM=88
/
"""

S = np.array([0., .04, .25, .64, 1.])
CURRENT = np.array([0., -2., -5., -3., 1.])
PRESSURE = np.array([100., 99., 80., 40., 0.])


def assigned_array(text, key):
    match = re.search(rf"^\s*{key}\s*=\s*([\dEe+.,\s-]+)", text, re.MULTILINE)
    return np.array([float(v) for v in match[1].replace("\n", " ").split(",") if v.strip()])


def assert_preserved(text):
    for fragment in (
        "! preamble with NCURR=99 /\n",
        "  TITLE='current AC=9 / and ''quoted'' text'\n",
        '  MGRID_FILE="literal ! CURTOR=8 / and ""quote"" tail"\n\n',
        "! current continuation\n",
        "! old edge current\n",
        "AI(0:1)=0.4,0.1\n",
        "  PIOTA_TYPE='power_series'\n",
        "  AI_AUX_S=0,1\n",
        "  AI_AUX_F=.4,.6\n",
        "  RBC(0,0)=5.5, ZBS(1,1)=0.02\n",
        "&OTHER\n AC=99, AM=88\n/\n",
    ):
        assert fragment in text


def test_current_preserves_supplied_pressure():
    result = rewrite_current_profile(VMEC_INPUT, S, CURRENT)
    assert "  AM_AUX_F=10,5,0\n  PRES_SCALE=7\n" in result


@pytest.mark.parametrize("current_first", [True, False])
def test_pressure_and_current_updates_preserve_each_other(tmp_path, current_first):
    input_path = tmp_path / "input"
    input_path.write_text(VMEC_INPUT)

    def update_current():
        input_path.write_text(rewrite_current_profile(input_path.read_text(), S, CURRENT))

    def update_pressure():
        pressure_writer._write_vmec_input_with_pressure_spline(input_path, np.sqrt(S), PRESSURE)

    updates = (update_current, update_pressure) if current_first else (update_pressure, update_current)
    for update in updates:
        update()
    result = input_path.read_text()
    assert_preserved(result)
    assert "AC(0:2)" not in result and "AC(3)" not in result and "AC_AUX_S(1:3)" not in result
    assert_array_equal(assigned_array(result, "AC_AUX_S"), S)
    assert_array_equal(assigned_array(result, "AC_AUX_F"), CURRENT)
    assert_array_equal(assigned_array(result, "CURTOR"), [CURRENT[-1]])
    assert_array_equal(assigned_array(result, "AM_AUX_S"), np.sqrt(S)**2)
    assert_array_equal(assigned_array(result, "AM_AUX_F"), PRESSURE)
    assert "NCURR = 1" in result
    assert "PCURR_TYPE = 'akima_spline_i'" in result
    assert "PMASS_TYPE = 'akima_spline'" in result
    assert "PRES_SCALE = 1.0000000000000000E+00" in result
    for update in updates:
        update()
    assert input_path.read_text() == result


@pytest.mark.parametrize("text", ["TITLE='&INDATA /'", "&INDATA\n TITLE='/' ! /\n"])
def test_missing_real_namelist_delimiters_raise(text):
    with pytest.raises(ValueError, match="INDATA"):
        rewrite_assignments(text, {"NCURR"}, ["NCURR=1"])


@pytest.mark.parametrize("s,current", [
    ([0.], [0.]), ([[0., 1.]], [[0., 1.]]),
    ([0., 1.], [0.]), ([.1, 1.], [0., 1.]), ([0., .9], [0., 1.]),
    ([0., .5, .5, 1.], [0., 1., 2., 3.]),
    ([0., .7, .5, 1.], [0., 1., 2., 3.]),
    ([0., np.nan, 1.], [0., 1., 2.]), ([0., 1.], [1., 2.]),
    ([0., 1.], [0., np.inf]),
    (np.linspace(0., 1., 102), np.linspace(0., 2., 102)),
])
def test_current_rejects_invalid_knots(s, current):
    with pytest.raises(ValueError):
        rewrite_current_profile(VMEC_INPUT, s, current)


def test_current_rejects_exact_zero_edge_with_nonzero_interior_even_when_tiny():
    with pytest.raises(ValueError, match="exactly zero edge"):
        rewrite_current_profile(VMEC_INPUT, [0., .5, 1.], [0., 1e-30, 0.])


@pytest.mark.parametrize("current", [[0., 0., 0.], [0., 1., 1e-30], [0., -2., -1.]])
def test_current_accepts_zero_small_edge_and_signed_profiles(current):
    result = rewrite_current_profile(VMEC_INPUT, [0., .5, 1.], current)
    assert_array_equal(assigned_array(result, "AC_AUX_F"), current)
    assert_array_equal(assigned_array(result, "CURTOR"), [current[-1]])


def test_current_accepts_maximum_knot_count():
    s = np.linspace(0., 1., 101)
    current = 2. * s - s**2
    result = rewrite_current_profile(VMEC_INPUT, s, current)
    assert_array_equal(assigned_array(result, "AC_AUX_S"), s)
    assert_array_equal(assigned_array(result, "AC_AUX_F"), current)
