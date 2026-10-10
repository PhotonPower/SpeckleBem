#pragma once
/// @file near_field.hpp
/// Near-field part Z_near of the MLFMM (ADR 0008 §1, WP19a).
///
/// A basis pair (m, n) is near iff the leaf boxes of m and n coincide or touch (integer leaf
/// coordinates differ by at most 1 per axis: the box itself and its near_list); with elevated
/// basis functions (local leaf rule, octree.hpp "Home levels") iff the ancestors of the two
/// leaves on the coarser home level min(h(m), h(n)) coincide or touch, so an elevated function
/// is near to every function of the 27 boxes around its home box. Z_near holds the
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

/// Near basis-pair pattern: one group per leaf box and home level present among its basis
/// functions (the leaf level first, then coarser ones); the rows of group (leaf X, level h) are
/// X's basis functions with home level h, its columns, ascending, the functions with home level
/// >= h in the 27 boxes around X's ancestor on level h and, for every level l < h, the functions
/// with home level l in the 27 boxes around X's ancestor on level l. Without elevated functions:
/// one group per leaf, columns = the bases of the leaf and its near_list (the WP19a pattern).
/// Memory O(N + sum over groups of the column counts).
/// @throws std::invalid_argument if the octree was not built on `space` (size of the
///         permutation differs from N, or a basis function's edge midpoint lies outside its leaf
///         box by more than 1e-6 of the box edge).
op::BasisPattern near_pattern(const Octree& tree, const basis::RwgSpace& space);

/// Z_near = op::assemble_sparse(p, near_pattern(tree, *p.space)); logs nnz, memory and time.
/// @throws std::invalid_argument for an invalid Problem (op::validate) or an octree not built
///         on p.space (near_pattern).
std::shared_ptr<op::SparseOperator> assemble_near(const op::Problem& p, const Octree& tree);

/// Bytes of assemble_near(p, tree) (4 blocks x (16 + 8) bytes per near basis pair + 2N + 1 row
/// pointers), computed from the box counts (Octree::active_elements, elevated_count) without
/// building the pattern or assembling: the reference of the automatic exact-part budget
/// (MlfmmParams::max_exact_far_bytes). O(N + leaves x levels x 27).
[[nodiscard]] std::size_t estimate_near_bytes(const Octree& tree);

}  // namespace specklebem::mlfmm
