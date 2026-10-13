#pragma once
/// @file stretched_detail.hpp
/// Internal interface between kernels/operators.cpp and kernels/stretched_operators.cpp (WP-PML2,
/// ADR 0009): the stretched element blocks and the near/far error model they share. Not installed,
/// not part of the public API.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/kernels/operators.hpp"

#include <array>

namespace specklebem::kernels::detail {

/// log10 of the estimated relative block error of the degree-d double Dunavant rule of a near or
/// far pair at log10(kappa) (the WP7b error model of operators.cpp, margin included).
[[nodiscard]] Real plain_rule_log_error(int degree, Real log_kappa);

/// triangle_rule_is_positive_interior(degree) from a table (no exception for 1..20).
[[nodiscard]] bool positive_interior_degree(int degree);

/// True if triangle t has a vertex in the stretched zone of an axis with sigma_max > 0
/// (|x| > x.start or |y| > y.start): the map is not the identity on t. False for an identity
/// stretch.
[[nodiscard]] bool triangle_stretched(const geometry::CoordinateStretch& stretch,
                                      const geometry::TriangleMesh& mesh, Index t);

/// Element blocks of a pair with at least one stretched triangle (stretched_operators.cpp; the
/// public element_blocks overload dispatches here). Same slots, scaling and exceptions as
/// element_blocks.
void stretched_element_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                              const RegionParams& region, const OperatorOptions& opt,
                              const geometry::CoordinateStretch& stretch,
                              Eigen::Matrix<Complex, 3, 3>& L, Eigen::Matrix<Complex, 3, 3>& K);

/// Both regions of a pair with at least one stretched triangle in one call
/// (element_blocks_both_regions): inactive regions get zero blocks; each active region's blocks
/// are bitwise those of stretched_element_blocks for that region (regions with the same rule share
/// the point loop).
void stretched_element_blocks_both(const basis::RwgSpace& space, Index t_test, Index t_src,
                                   const std::array<RegionParams, 2>& regions,
                                   const std::array<bool, 2>& active, const OperatorOptions& opt,
                                   const geometry::CoordinateStretch& stretch,
                                   std::array<Eigen::Matrix<Complex, 3, 3>, 2>& L,
                                   std::array<Eigen::Matrix<Complex, 3, 3>, 2>& K);

/// Rule information of a stretched pair (stretched_pair_info).
[[nodiscard]] StretchedPairInfo stretched_info(const geometry::TriangleMesh& mesh, Index t_test,
                                               Index t_src, Complex k, const OperatorOptions& opt,
                                               const geometry::CoordinateStretch& stretch);

}  // namespace specklebem::kernels::detail
