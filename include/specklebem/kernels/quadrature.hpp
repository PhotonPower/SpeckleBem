#pragma once
/// @file quadrature.hpp
/// Gauss quadrature rules on the reference triangle (Dunavant) and on the
/// interval (Gauss-Legendre, also used for the MLFMM theta-sampling).
#include "specklebem/core/types.hpp"

#include <vector>

namespace specklebem::kernels {

struct TriangleRule {
    std::vector<Vec3> barycentric;  ///< (l1, l2, l3) per point
    std::vector<Real> weights;      ///< sum = 1 (multiply by triangle area)
};

/// Dunavant rule of the requested polynomial degree (1..20).
[[nodiscard]] const TriangleRule& triangle_rule(int degree);

struct LineRule {
    std::vector<Real> nodes;  ///< in [-1, 1]
    std::vector<Real> weights;
};

/// Gauss-Legendre rule with n points.
[[nodiscard]] LineRule gauss_legendre(int n);

}  // namespace specklebem::kernels
