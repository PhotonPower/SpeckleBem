#pragma once
/// @file near_field.hpp
/// Near-field part Z_near of the MLFMM (ADR 0008 §1, WP19a).
///
/// A basis pair (m, n) is near iff the leaf boxes of m and n coincide or touch (integer leaf
/// coordinates differ by at most 1 per axis: the box itself and its near_list). Z_near holds the
/// exact Galerkin entries of all four blocks (JJ, JM, MJ, MM) of every near pair, assembled by
/// op::assemble_sparse with the dense assembler's code: Z_near equals the DenseStrategy matrix on
/// its pattern bitwise.
///
/// Layering: the assembly itself is generic (op::assemble_sparse on an op::BasisPattern); this
/// header only turns the octree into the pattern, because mlfmm sits above the operator layer
/// (MlfmmOperator implements op::LinearOperator) and op must not depend on mlfmm.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <cstddef>
#include <memory>

namespace specklebem::mlfmm {

/// Near basis-pair pattern of the leaf level: one group per leaf box; its rows are the leaf's
/// basis functions, its columns the basis functions of the leaf and its near_list, ascending.
/// Memory O(N + sum over leaves of the near-box element counts).
/// @throws std::invalid_argument if the octree was not built on `space` (size of the
///         permutation differs from N, or a basis function's edge midpoint lies outside its leaf
///         box by more than 1e-6 of the box edge).
op::BasisPattern near_pattern(const Octree& tree, const basis::RwgSpace& space);

/// Z_near = op::assemble_sparse(p, near_pattern(tree, *p.space)); logs nnz, memory and time.
/// @throws std::invalid_argument for an invalid Problem (op::validate) or an octree not built
///         on p.space (near_pattern).
std::shared_ptr<op::SparseOperator> assemble_near(const op::Problem& p, const Octree& tree);

/// Bytes of assemble_near(p, tree) (4 blocks x (16 + 8) bytes per near basis pair + 2N + 1 row
/// pointers), computed from the leaf box counts without building the pattern or assembling: the
/// reference of the automatic exact-part budget (MlfmmParams::max_exact_far_bytes). O(leaves).
[[nodiscard]] std::size_t estimate_near_bytes(const Octree& tree);

}  // namespace specklebem::mlfmm
