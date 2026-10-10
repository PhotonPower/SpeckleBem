/// @file mlfmm_operator.cpp
/// Full MLFMM operator Z = Z_near + Z_far (WP20b): owns the octree, the exact near field
/// (assemble_near) and the multilevel far operator (MlfmmFarOperator).
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"

#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace specklebem::mlfmm {

namespace {

Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

Real mb(std::size_t bytes) {
    return static_cast<Real>(bytes) / 1048576.0;
}

/// Leaf-size reference of the octree: the wavelength of R1 (ADR 0008 §1), after validation.
Real octree_wavelength(const op::Problem& problem) {
    op::validate(problem);
    const Complex k1 = problem.exterior.wavenumber(problem.omega);
    return 2.0 * constants::pi / k1.real();
}

}  // namespace

OctreeParams leaf_rule_params(const basis::RwgSpace& space, Real wavelength,
                              const MlfmmParams& params) {
    OctreeParams o = params.octree;
    if (!params.automatic_leaf_size)
        return o;
    if (!(params.accuracy_digits > 0.0)) {
        throw std::invalid_argument("leaf_rule_params: accuracy_digits must be positive");
    }
    if (!std::isfinite(wavelength) || !(wavelength > 0.0)) {
        throw std::invalid_argument("leaf_rule_params: wavelength must be positive and finite");
    }
    const Real a_min = params.accuracy_digits <= 3.0 ? kLeafMinLambdaD3 : kLeafMinLambdaD5;
    // r_max / a <= kLeafMaxSupportRatio also with the octree's floor tolerance.
    const Real a_rmax = max_support_radius(space) /
                        (kLeafMaxSupportRatio * (1.0 - kMinBoxSizeTolerance) * wavelength);
    o.min_box_size_lambda = std::max({o.min_box_size_lambda, a_min, a_rmax});
    return o;
}

struct MlfmmOperator::Impl {
    std::unique_ptr<Octree> tree;
    std::unique_ptr<MlfmmFarOperator> far;
    std::shared_ptr<op::SparseOperator> near;
    Real lambda1 = 0;  ///< wavelength of R1, the octree's leaf-size reference
    Real digits = 0;
    Real user_floor = 0, floor = 0;  ///< min_box_size_lambda given / after the leaf rule
    Real tree_s = 0, far_s = 0, near_s = 0;
};

MlfmmOperator::MlfmmOperator(const op::Problem& problem, const MlfmmParams& params)
    : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.lambda1 = octree_wavelength(problem);
    m.digits = params.accuracy_digits;
    auto t0 = std::chrono::steady_clock::now();
    const OctreeParams oparams = leaf_rule_params(*problem.space, m.lambda1, params);
    m.user_floor = params.octree.min_box_size_lambda;
    m.floor = oparams.min_box_size_lambda;
    if (m.floor > m.user_floor) {
        SBEM_INFO(
            "MlfmmOperator: leaf rule raises min_box_size_lambda from {:.3g} to {:.3g} "
            "(d0 = {}, r_max = {:.3g} m)",
            m.user_floor, m.floor, m.digits, max_support_radius(*problem.space));
    }
    m.tree = std::make_unique<Octree>(*problem.space, m.lambda1, oparams);
    m.tree_s = seconds_since(t0);
    // Far part first: its order searches are cheap and reject unusable configurations (lossy
    // regions, coarse meshes) before the expensive near-field assembly.
    t0 = std::chrono::steady_clock::now();
    m.far = std::make_unique<MlfmmFarOperator>(problem, *m.tree, params);
    m.far_s = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    m.near = assemble_near(problem, *m.tree);
    m.near_s = seconds_since(t0);
    SBEM_INFO(
        "MlfmmOperator: 2N = {}, {} levels, leaf edge {:.3g} lambda1, near nnz {} ({:.1f} MB, "
        "{:.2f} s), far {:.1f} MB ({:.2f} s)",
        rows(), m.tree->levels(), m.tree->box_size(m.tree->leaf_level()) / m.lambda1,
        m.near->nonzeros(), mb(m.near->memory_bytes()), m.near_s, mb(m.far->memory_bytes()),
        m.far_s);
}

MlfmmOperator::~MlfmmOperator() = default;

Index MlfmmOperator::rows() const {
    return impl_->far->rows();
}

Index MlfmmOperator::cols() const {
    return impl_->far->cols();
}

void MlfmmOperator::apply(const VectorXc& x, VectorXc& y) const {
    if (x.size() != cols())
        throw std::invalid_argument("MlfmmOperator::apply: x has the wrong size");
    if (&x == &y) {
        const VectorXc copy = x;
        apply(copy, y);
        return;
    }
    VectorXc far;
    impl_->far->apply(x, far);
    impl_->near->apply(x, y);
    y += far;
}

std::string MlfmmOperator::describe() const {
    const Impl& m = *impl_;
    const Octree& t = *m.tree;
    std::ostringstream os;
    os << "MlfmmOperator: 2N = " << rows() << ", d0 = " << m.digits << ", octree " << t.levels()
       << " levels, " << t.boxes_at_level(t.leaf_level()).size() << " leaves of edge "
       << t.box_size(t.leaf_level()) / m.lambda1 << " lambda1 (lambda1 = " << m.lambda1
       << " m), built in " << m.tree_s << " s; leaf floor " << m.floor << " lambda1 (given "
       << m.user_floor << (m.floor > m.user_floor ? ", raised by the leaf rule)" : ")");
    os << "\n  near: " << m.near->describe() << ", assembled in " << m.near_s << " s";
    os << "\n  far: " << m.far_s << " s setup, " << mb(m.far->memory_bytes()) << " MB\n  "
       << m.far->describe();
    os << "\n  total " << mb(memory_bytes()) << " MB (near " << mb(m.near->memory_bytes())
       << ", far " << mb(m.far->memory_bytes()) << ", octree " << mb(octree_bytes()) << ")";
    return os.str();
}

std::size_t MlfmmOperator::octree_bytes() const {
    const Octree& t = *impl_->tree;
    std::size_t b = sizeof(Octree) + t.boxes().size() * sizeof(Box) +
                    2 * t.permutation().size() * sizeof(Index);
    for (const Box& box : t.boxes())
        b += (box.near_list.size() + box.interaction_list.size()) * sizeof(Index);
    return b;
}

std::size_t MlfmmOperator::memory_bytes() const {
    return sizeof(Impl) + impl_->near->memory_bytes() + impl_->far->memory_bytes() + octree_bytes();
}

const Octree& MlfmmOperator::octree() const {
    return *impl_->tree;
}

const op::SparseOperator& MlfmmOperator::near_operator() const {
    return *impl_->near;
}

const MlfmmFarOperator& MlfmmOperator::far_operator() const {
    return *impl_->far;
}

std::shared_ptr<op::LinearOperator> MlfmmStrategy::build(const op::Problem& p) const {
    return std::make_shared<MlfmmOperator>(p, params_);
}

}  // namespace specklebem::mlfmm
