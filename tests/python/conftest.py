"""Shared fixtures of the Python tests.

``npy_dir``: a directory written by the C++ helper ``specklebem_npy_fixture``
(tests/support/npy_fixture_main.cpp) through ``io::open_npy_directory``, used by the NumPy round
trip (test_result_writer.py) and the HDF5 conversion (test_hdf5.py). The executable is taken
from the environment variable ``SPECKLEBEM_NPY_FIXTURE`` or, by default, from the build tree of
the imported ``specklebem`` package (``<build>/python/specklebem`` -> ``<build>/tests``). Tests
using it are skipped if the variable is unset and the build-tree helper does not exist (e.g. an
installed package or a build with SPECKLEBEM_BUILD_TESTS=OFF); they fail if the variable is set
but names no file (CI sets it, so the round trip cannot be skipped silently).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import specklebem as sb


def _fixture_executable() -> Path | None:
    env = os.environ.get("SPECKLEBEM_NPY_FIXTURE")
    if env:
        exe = Path(env).resolve()
        if not exe.is_file():
            pytest.fail(f"SPECKLEBEM_NPY_FIXTURE={env!r} does not name an existing file")
        return exe
    name = "specklebem_npy_fixture" + (".exe" if os.name == "nt" else "")
    candidate = Path(sb.__file__).resolve().parents[2] / "tests" / name
    return candidate if candidate.is_file() else None


@pytest.fixture(scope="session")
def npy_dir(tmp_path_factory):
    exe = _fixture_executable()
    if exe is None:
        pytest.skip(
            "specklebem_npy_fixture not found: set SPECKLEBEM_NPY_FIXTURE or use the build-tree "
            "package (PYTHONPATH=<build>/python) of a build with tests"
        )
    # Non-ASCII directory name: UTF-8 paths end to end (ADR 0007, WP-P1).
    out = tmp_path_factory.mktemp("npy") / "result_Jürgen_µm_路径"
    env = dict(os.environ)
    if os.name == "nt":  # MinGW runtime DLLs live next to python.exe (MSYS2)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(exe), str(out)], check=True, env=env, timeout=60)
    return out
