---
name: core-fft-findings
description: WP-T3 core::fft2 facts - Eigen kissfft segfaults on length 1, GCC/Clang MSYS2 give identical height-map/autocorrelation digits, pin-first workflow
metadata:
  type: project
---

WP-T3 (2026-10-31) moved the identical row/column Eigen-FFT copies of rough_surface.cpp and
speckle_statistics.cpp (plus next_smooth) into core::fft2 / core::next_smooth (core/fft.hpp).

- Eigen's unsupported FFT (kissfft backend) **segfaults on a length-1 transform** (release build,
  seen with a 1 x 5 matrix). core::fft2 skips length-1 passes (identity) and returns early for
  empty matrices; callers never hit either (noise grids >= 2, padded autocorrelation >= 3).
- The pre-refactor 8 x 8 height map (seed 2026, L 0.7 um, h 0.1 um, Lc 0.25 um) and a 7 x 6
  autocorrelation printed **identical 17-digit values in win-release (GCC) and win-debug (Clang)**
  (both UCRT libm), so one pin table with exact match under _WIN32 suffices; off Windows 1e-14
  of the max. Catch2 test order differs between the two builds (link order), so print with
  separate filters or sort by content.
- Note Lc >= 2 max(dx, dy) is checked strictly: L/(n-1) rounding can make dx slightly above
  the nominal step, so Lc = exactly 2 h is rejected; use some margin.

**How to apply:** for "no behaviour change" refactors, commit the pinned values from the
unmodified code first (both presets), then refactor; a separate TU with the same flags kept the
results bitwise. Another worker's specklebem_profile runs come back-to-back (a new one started
2 s after the last ended), so re-check right before each build/ctest, not only once.
