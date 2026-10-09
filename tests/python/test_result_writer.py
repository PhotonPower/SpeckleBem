"""NumPy round trip of the C++ .npy directory writer (io::open_npy_directory).

The helper executable ``specklebem_npy_fixture`` (tests/support/npy_fixture_main.cpp) writes a
fixed dataset; this test reads it back with ``numpy.load`` and ``json.load``. The executable is
taken from the environment variable ``SPECKLEBEM_NPY_FIXTURE`` or, by default, from the build
tree of the imported ``specklebem`` package (``<build>/python/specklebem`` ->
``<build>/tests``). The test is skipped if neither exists (e.g. an installed package or a
build with SPECKLEBEM_BUILD_TESTS=OFF).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import specklebem as sb


def _fixture_executable() -> Path | None:
    env = os.environ.get("SPECKLEBEM_NPY_FIXTURE")
    if env:
        return Path(env)
    name = "specklebem_npy_fixture" + (".exe" if os.name == "nt" else "")
    candidate = Path(sb.__file__).resolve().parents[2] / "tests" / name
    return candidate if candidate.is_file() else None


@pytest.fixture(scope="module")
def npy_dir(tmp_path_factory):
    exe = _fixture_executable()
    if exe is None:
        pytest.skip(
            "specklebem_npy_fixture not found: set SPECKLEBEM_NPY_FIXTURE or use the build-tree "
            "package (PYTHONPATH=<build>/python) of a build with tests"
        )
    out = tmp_path_factory.mktemp("npy") / "result"
    env = dict(os.environ)
    if os.name == "nt":  # MinGW runtime DLLs live next to python.exe (MSYS2)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(exe), str(out)], check=True, env=env, timeout=60)
    return out


def _load(path: Path, dtype: str, shape: tuple) -> np.ndarray:
    with open(path, "rb") as f:
        assert np.lib.format.read_magic(f) == (1, 0)
        hdr_shape, fortran_order, hdr_dtype = np.lib.format.read_array_header_1_0(f)
    assert (hdr_shape, fortran_order, hdr_dtype) == (shape, False, np.dtype(dtype))
    arr = np.load(path, allow_pickle=False)
    assert arr.dtype == np.dtype(dtype) and arr.shape == shape
    assert arr.flags.c_contiguous
    return arr


def test_arrays(npy_dir):
    v = _load(npy_dir / "vector.npy", "<c16", (5,))
    k = np.arange(5)
    np.testing.assert_array_equal(v, (k + 0.5) - 1j * (k / 3.0))
    assert _load(npy_dir / "empty.npy", "<c16", (0,)).size == 0
    mr = _load(npy_dir / "matrix_real.npy", "<f8", (2, 3))
    i, j = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
    np.testing.assert_array_equal(mr, 10 * i + j + 0.25)
    mc = _load(npy_dir / "group" / "matrix_complex.npy", "<c16", (3, 2))
    i, j = np.meshgrid(np.arange(3), np.arange(2), indexing="ij")
    np.testing.assert_array_equal(mc, (i + 1) + 1j * (0.5 * j - i))


def test_mesh(npy_dir):
    v = _load(npy_dir / "mesh" / "vertices.npy", "<f8", (42, 3))
    t = _load(npy_dir / "mesh" / "triangles.npy", "<i8", (80, 3))
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 0.5e-6, rtol=1e-12)
    assert t.min() == 0 and t.max() == 41
    # Orientation preserved: counter-clockwise seen from outside -> positive volume.
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0 > 0.0


def test_attributes(npy_dir):
    with open(npy_dir / "attributes.json", encoding="utf-8") as f:
        attrs = json.load(f)
    assert list(attrs) == ["name", "wavelength", "third"]
    assert attrs["name"] == 'SpeckleBem "npy" \\ é\n'
    assert attrs["wavelength"] == 500e-9
    assert attrs["third"] == 1.0 / 3.0
