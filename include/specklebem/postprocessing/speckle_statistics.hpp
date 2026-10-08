#pragma once
/// @file speckle_statistics.hpp
/// Statistical descriptors of speckle fields (Section 6 of the paper).
#include "specklebem/core/types.hpp"

namespace specklebem::post {

/// Speckle contrast  C = sqrt(<I^2> - <I>^2) / <I>   (Eq. 12)
Real speckle_contrast(const VectorXr& intensity);

/// Angular correlation between two speckle patterns  (Eq. 11)
Real correlation_coefficient(const VectorXr& I1, const VectorXr& I2);

/// Histogram-based probability density of normalised intensity I/<I>  (Eq. 10 check)
struct Histogram {
    VectorXr bin_centers, density;
};
Histogram intensity_pdf(const VectorXr& intensity, int bins = 64);

/// Normalised intensity autocorrelation Gamma_I(dx, dy) on a regular grid and
/// the derived field correlation |mu_A|  (Eq. 13), via FFT.
MatrixXr intensity_autocorrelation(const MatrixXr& intensity);
MatrixXr field_correlation_from_intensity(const MatrixXr& intensity);

/// Speckle size from the FWHM of |mu_A|^2 along one axis.
Real speckle_size(const MatrixXr& mu_A, Real pixel_size);

}  // namespace specklebem::post
