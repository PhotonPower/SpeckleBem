"""Convenience wrapper around the Gaussian rough-surface generator (no physics in Python)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _specklebem as _core

if TYPE_CHECKING:
    import numpy as np


class RoughSurface:
    """Gaussian random rough surface on an L x L patch, closed by a box below it.

    All lengths in metres. ``L``, ``sigma``, ``Lc``, ``mesh_size``, ``seed`` and ``use_fft``
    select the height map (``generate_gaussian_height_map``); ``box_depth``, ``box_mesh_size``
    and ``box_fine_depth`` the closing box (``make_mesh_from_height_map``, None = automatic).
    ``seed=0`` draws a random seed that is only logged by the C++ core (reproducible only via
    the logged seed); pass a non-zero seed for reproducible surfaces.
    """

    def __init__(
        self,
        L: float,
        sigma: float,
        Lc: float,
        mesh_size: float,
        seed: int = 0,
        *,
        box_depth: float | None = None,
        box_mesh_size: float | None = None,
        box_fine_depth: float | None = None,
        use_fft: bool = True,
    ) -> None:
        self._heights = _core.generate_gaussian_height_map(
            L=L, sigma=sigma, Lc=Lc, mesh_size=mesh_size, seed=seed, use_fft=use_fft
        )
        self._box = dict(
            box_depth=box_depth, box_mesh_size=box_mesh_size, box_fine_depth=box_fine_depth
        )
        self._mesh: _core.TriangleMesh | None = None
        self._z: np.ndarray | None = None

    @property
    def heights(self) -> _core.HeightMap:
        """The C++ height map object (z, dx, dy, rms(), estimated_correlation_length())."""
        return self._heights

    @property
    def height_map(self) -> np.ndarray:
        """Heights z(i, j) [m] at (x_i, y_j), shape (n_x, n_y), float64, read-only."""
        if self._z is None:
            self._z = self._heights.z
            self._z.flags.writeable = False
        return self._z

    @property
    def dx(self) -> float:
        """Grid spacing along x [m]."""
        return self._heights.dx

    @property
    def dy(self) -> float:
        """Grid spacing along y [m]."""
        return self._heights.dy

    @property
    def mesh(self) -> _core.TriangleMesh:
        """Closed box mesh (built on first access, then cached)."""
        if self._mesh is None:
            self._mesh = _core.make_mesh_from_height_map(self._heights, **self._box)
        return self._mesh
