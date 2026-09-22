"""Check the Stage 1 adapter without requiring the VMEX solver environment."""

from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from tests.helpers.stage_import import load_stage_module


run_vmex = load_stage_module("stages/stage1-equilibrium/run_vmex.py")


@pytest.fixture
def vmex_cli(monkeypatch):
    cli = ModuleType("vmex.core.cli")
    cli.main = Mock()
    cli.resolve_wout_path = lambda *, input_path, outdir: Path(outdir) / "wout_solver.nc"
    monkeypatch.setitem(sys.modules, cli.__name__, cli)
    return cli


def test_missing_input_does_not_start_solver(tmp_path, vmex_cli):
    output_path = tmp_path / "outputs" / "custom.nc"
    with pytest.raises(FileNotFoundError):
        run_vmex.run_vmex(tmp_path / "missing", output_path)
    vmex_cli.main.assert_not_called()
    assert not output_path.parent.exists()


def test_success_moves_wout_to_custom_output_name(tmp_path, monkeypatch, vmex_cli):
    monkeypatch.chdir(tmp_path)
    input_path = Path("input files/vmec_input.HSX.quick.run")
    input_path.parent.mkdir()
    input_path.write_text("input")
    output_path = Path("output files/nested/custom equilibrium.nc")

    def solve(argv):
        assert Path(argv[0]) == input_path.resolve()
        assert argv[argv.index("--device") + 1] == "auto"
        outdir = Path(argv[argv.index("--outdir") + 1])
        vmex_cli.resolve_wout_path(input_path=input_path, outdir=outdir).write_bytes(b"equilibrium")
        return 0

    vmex_cli.main.side_effect = solve
    assert run_vmex.main(["--input", str(input_path), "--output", str(output_path)]) == 0
    assert output_path.read_bytes() == b"equilibrium"
    assert list(output_path.parent.iterdir()) == [output_path]


def test_nonzero_status_preserves_existing_output_after_partial_write(tmp_path, vmex_cli):
    input_path = tmp_path / "input.case"
    input_path.write_text("input")
    output_path = tmp_path / "wout_solver.nc"
    output_path.write_bytes(b"previous equilibrium")

    def solve(argv):
        assert argv[argv.index("--device") + 1] == "cuda"
        outdir = Path(argv[argv.index("--outdir") + 1])
        vmex_cli.resolve_wout_path(input_path=input_path, outdir=outdir).write_bytes(b"failed solve")
        return 2

    vmex_cli.main.side_effect = solve
    assert run_vmex.main([
        "--input", str(input_path), "--output", str(output_path), "--device", "cuda",
    ]) == 2
    assert output_path.read_bytes() == b"previous equilibrium"
    assert set(tmp_path.iterdir()) == {input_path, output_path}


def test_main_reports_missing_input(tmp_path, vmex_cli, caplog):
    assert run_vmex.main(["--input", str(tmp_path / "missing"), "--output", str(tmp_path / "wout.nc")]) == 1
    assert "Stage 1 input file does not exist" in caplog.text
    vmex_cli.main.assert_not_called()
