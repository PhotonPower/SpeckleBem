#pragma once
/// @file types.hpp
/// Fundamental scalar / vector / matrix type aliases used throughout SpeckleBem.
///
/// Conventions (see docs/06_conventions.md):
///  * SI units everywhere (lengths in metres, angular frequency in rad/s).
///  * Time dependence exp(+j w t)  =>  lossy media have Im(eps_r) < 0.
///  * Double precision by default; single precision is reserved for GPU kernels.

#include <Eigen/Core>

#include <complex>
#include <cstdint>
#include <utility>

namespace specklebem {

using Real = double;
using Complex = std::complex<Real>;
using Index = std::int64_t;

using Vec3 = Eigen::Matrix<Real, 3, 1>;
using Vec3c = Eigen::Matrix<Complex, 3, 1>;
using Mat3 = Eigen::Matrix<Real, 3, 3>;

using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using VectorXc = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using MatrixXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

/// Row-major integer connectivity (n_triangles x 3).
using Triangles = Eigen::Matrix<Index, Eigen::Dynamic, 3, Eigen::RowMajor>;
/// Row-major vertex coordinates (n_vertices x 3).
using Vertices = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;
/// Row-major edge list (n_edges x 2).
using Edges = Eigen::Matrix<Index, Eigen::Dynamic, 2, Eigen::RowMajor>;

namespace constants {
inline constexpr Real pi = 3.14159265358979323846;
inline constexpr Real c0 = 299792458.0;         // m/s
inline constexpr Real mu0 = 1.25663706212e-6;   // H/m
inline constexpr Real eps0 = 8.8541878128e-12;  // F/m
inline constexpr Real eta0 = 376.730313668;     // Ohm, free-space impedance
}  // namespace constants

}  // namespace specklebem
