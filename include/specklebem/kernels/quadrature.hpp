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

/// Dunavant rule of the requested polynomial degree (1..20): integrates every polynomial of
/// total degree <= `degree` exactly (to double precision). Point counts 1, 3, 4, 6, 7, 12, 13,
/// 16, 19, 25, 27, 33, 37, 42, 48, 52, 61, 70, 73, 79. The rules are fully symmetric; degrees
/// 3, 7, 18, 20 have negative weights and degrees 11, 15, 16, 18, 20 have points outside
/// the triangle (details in src/kernels/quadrature.cpp). The rules are built once on
/// first use (thread-safe) and returned by reference; the reference stays valid for the
/// lifetime of the program.
/// @throws std::invalid_argument if `degree` is outside 1..20.
[[nodiscard]] const TriangleRule& triangle_rule(int degree);

/// True iff the rule of this degree has all points strictly inside the triangle and all
/// weights positive (false for degrees 3, 7, 11, 15, 16, 18, 20). Near-field and singular
/// quadrature (ADR 0004) should use only positive-interior degrees: points outside the
/// triangle may approach a neighbouring singularity, and negative weights amplify errors.
/// @throws std::invalid_argument if `degree` is outside 1..20.
[[nodiscard]] bool triangle_rule_is_positive_interior(int degree);

struct LineRule {
    std::vector<Real> nodes;  ///< in (-1, 1), ascending
    std::vector<Real> weights;
};

/// Gauss-Legendre rule with n points on [-1, 1]: nodes ascending and symmetric about 0,
/// positive weights summing to 2, exact for polynomials of degree <= 2n - 1. Computed by
/// Newton iteration on the Legendre recurrence (O(n^2)); nodes and weights accurate to
/// ~1e-16 absolute for n up to several hundred.
/// @throws std::invalid_argument if `n` < 1.
[[nodiscard]] LineRule gauss_legendre(int n);

}  // namespace specklebem::kernels
