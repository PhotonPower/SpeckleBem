---
name: open-surface-pml-findings
description: WP-PML1 open rough surface + CoordinateStretch facts - grid rounding, corner diagonals, C1 test design, Eigen cross link error, PML damping numbers
metadata:
  type: project
---

WP-PML1 (ADR 0009, 2026-10-13): `geometry::make_open_surface_mesh` + `CoordinateStretch`.

- Band boundaries are integer cell counts on the map grid (patch edge = next map grid line,
  widths rounded up); dx == dy and equal parity of n_x - 1, n_y - 1 are required (square bands).
  A map from `generate_gaussian_height_map` with edge L + 2 W_t and h | L/2, W_t needs no rounding.
- Topology: M cells/axis -> 4M boundary edges, Euler 1, 4M - 2 triangles with a boundary edge
  (cells (M-1,0) and (0,M-1) have one triangle with two). The diagonals of those corner cells
  join two *different* outer sides but are interior edges; a test "both endpoints on the outer
  square => boundary" is wrong (check "along one side").
- C1 taper test: first-order one-sided slope jump converges only O(h) with large O(h^2)
  contamination (ratio 0.43-0.66, flaky); second-order one-sided slopes give jump
  ~h^2 |[f''']|/3, ratio 0.23-0.25 for h 20 -> 10 nm (robust window 0.15-0.35).
- `Vec3::cross` without `<Eigen/Geometry>` compiles on GCC 16 but fails at *link* time
  (undefined MatrixBase::cross) - include it in tests.
- Damping one-way at the PML end: exp(-k sigma_max W/(m+1)); W = 2 lambda, sigma 4, m 2:
  k0 -> ~5e-8, 1.055 k0 (Ag SPP) -> ~2e-8, far below the 1e-4 target.

**Why:** WP-PML2..P4 build on this mesh and stretch; the counts and test pitfalls recur.
**How to apply:** reuse the integer-band logic for windows/masks (WP-PML3), use
`make_pml_stretch` for the stretch; see [[ag-null-mode-verdict]] for why Ag needs the PML route.
