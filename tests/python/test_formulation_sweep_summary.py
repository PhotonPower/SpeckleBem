"""Unit tests of benchmarks/formulation_sweep_summary.py (WP22b2 summary script).

Pure Python: the script is imported by path (it is not part of the specklebem package), and the
records are small synthetic HISTORY / ROW lines in the format of
``mlfmm_scaling::history_record`` and ``mlfmm_scaling::row``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "formulation_sweep_summary.py"


@pytest.fixture(scope="module")
def fss():
    spec = importlib.util.spec_from_file_location("formulation_sweep_summary", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # standard importlib recipe: register before exec
    spec.loader.exec_module(module)
    return module


def _history_line(form="ictf", jacobi="0", it=25, stride=10, res=None, waist=None, L="3.2"):
    if res is None:
        res = ",".join(f"{2.0 ** -k:.4e}" for k in range(0, it + 1, stride))
        if it % stride:
            res += f",{2.0 ** -it:.4e}"
    w = "" if waist is None else f" waist={waist}"
    return (f"HISTORY material=ag L_um={L} box=50.000000 fine=0{w} beam=angular_spectrum d0=3 "
            f"tol=0.001 seed=1 N2=1234 form={form} jacobi={jacobi} it={it} conv=1 "
            f"stride={stride} res={res}")


def test_history_indices_match_the_cpp_rule(fss):
    assert fss.history_indices(25, 10) == [0, 10, 20, 25]
    assert fss.history_indices(20, 10) == [0, 10, 20]
    assert fss.history_indices(3, 10) == [0, 3]
    assert fss.history_indices(0, 10) == [0]
    assert fss.history_indices(4, 1) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        fss.history_indices(5, 0)


def test_history_reconstructs_the_sampled_iterations(fss):
    r = fss.parse_kv(_history_line(it=25, stride=10), "HISTORY ")
    ks, res = fss.history(r)
    assert ks == [0, 10, 20, 25]
    assert res == pytest.approx([1.0, 2.0**-10, 2.0**-20, 2.0**-25], rel=1e-4)
    # Exact samples are returned as stored; between samples log-linear interpolation.
    assert fss.value_at(ks, res, 10) == res[1]
    assert fss.value_at(ks, res, 5) == pytest.approx(2.0**-5, rel=1e-4)
    assert fss.value_at(ks, res, 26) is None


def test_history_length_mismatch_is_an_error(fss):
    r = fss.parse_kv(_history_line(it=25, stride=10, res="1,0.5,0.25"), "HISTORY ")
    with pytest.raises(ValueError, match="length mismatch"):
        fss.history(r)


def test_waist_defaults_to_4_for_old_records(fss):
    old = fss.parse_kv(_history_line(), "HISTORY ")
    new = fss.parse_kv(_history_line(waist="4"), "HISTORY ")
    other = fss.parse_kv(_history_line(waist="6"), "HISTORY ")
    assert fss.key(old) == fss.key(new)
    assert fss.key(old)[3] == "4"
    assert fss.key(other)[3] == "6"
    assert "w0 = L/6" in fss.label(fss.key(other))


def test_duplicate_records_are_refused(fss):
    a = fss.parse_kv(_history_line(), "HISTORY ")
    b = fss.parse_kv(_history_line(waist="4"), "HISTORY ")  # same run as a (waist 4 default)
    with pytest.raises(ValueError, match="duplicate"):
        fss.index([a, b], "HISTORY")
    # Different formulation, Jacobi, L or waist: distinct runs.
    distinct = [
        a,
        fss.parse_kv(_history_line(form="mctf"), "HISTORY "),
        fss.parse_kv(_history_line(jacobi="1"), "HISTORY "),
        fss.parse_kv(_history_line(L="5"), "HISTORY "),
        fss.parse_kv(_history_line(waist="6"), "HISTORY "),
    ]
    assert len(fss.index(distinct, "HISTORY")) == 5


def test_main_refuses_duplicates_and_prints_ascii(fss, tmp_path, capsys):
    h = tmp_path / "h.txt"
    h.write_text(_history_line() + "\n" + _history_line(form="pmchwt", it=3, stride=10) + "\n",
                 encoding="utf-8")
    rows = tmp_path / "rows.txt"
    rows.write_text("ROW material=ag L_um=3.2 box=50.000000 waist=4 it=25 conv=1 res=3e-8 "
                    "true_res=4.5e-8 solve_s=10 setup_s=2 peak_gb=1.5 form=ictf jacobi=0\n",
                    encoding="utf-8")
    assert fss.main([str(h), "--rows", str(rows)]) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    assert "| ag | 50 nm | 3.2 | 1234 | ICTF | 25 | yes |" in out
    assert "| 4.5e-08 |" in out  # true residual from the ROW line
    with h.open("a", encoding="utf-8") as f:
        f.write(_history_line() + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        fss.main([str(h)])
