#pragma once
/// @file fft.hpp
/// Shared 2D discrete Fourier transform of a dense complex matrix (row-column), used by the
/// height-map generator (geometry/rough_surface.cpp, FFT convolution) and the speckle
/// statistics (postprocessing/speckle_statistics.cpp, autocorrelation). WP-T3 moved the two
/// identical anonymous-namespace copies here unchanged, so both callers give bitwise the same
/// numbers as before.
///
/// Backend: Eigen's unsupported FFT module with its default backend (header-only kissfft) and
/// default flags; it stays in fft.cpp. kissfft is fastest for lengths whose only prime factors
/// are 2, 3 and 5 (next_smooth()), but any length >= 1 is transformed exactly.
#include "specklebem/core/types.hpp"

namespace specklebem::core {

/// In-place 2D DFT of a column-major complex matrix a of any shape rows x cols (non-square
/// included; a matrix with no entries is left unchanged, and a dimension of length 1 is not
/// transformed, its DFT being the identity). First every column is transformed (length rows),
/// then every row (length cols), each as a 1D Eigen::FFT<Real> transform through a scratch
/// buffer, and the result overwrites a. With Eigen's default flags and
/// phi = 2 pi (p r / rows + q c / cols):
///   forward (inverse = false): A(p, q) = sum_{r, c} a(r, c) exp(-j phi)  (unscaled);
///   inverse (inverse = true):  a(r, c) = sum_{p, q} A(p, q) exp(+j phi) / (rows cols),
/// where the inverse scaling is applied by the 1D transforms (1 / rows in each column
/// transform, 1 / cols in each row transform), so fft2(a, false) followed by fft2(a, true)
/// returns a up to rounding. The sign of the forward exponent is the usual DFT convention of
/// kissfft, not tied to the exp(+j omega t) time convention: callers that use the transform as
/// a physical spectrum must choose the sign themselves (the two callers only use circular
/// convolution and |A|^2, which do not depend on it). Deterministic: the same input gives
/// bitwise the same output on the same build. Never throws except std::bad_alloc.
void fft2(MatrixXc& a, bool inverse);

/// Smallest m >= max(n, 1) whose only prime factors are 2, 3 and 5 (5-smooth): a padded FFT
/// length with fast kissfft radices. Plain integer search, no overflow check (the callers pad
/// grids of at most a few million points per axis).
[[nodiscard]] Index next_smooth(Index n);

}  // namespace specklebem::core
