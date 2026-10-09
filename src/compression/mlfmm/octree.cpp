/// @file octree.cpp
/// Uniform-depth octree over RWG edge midpoints: Morton sort, level selection, box ranges,
/// near and interaction lists (see octree.hpp).
#include "specklebem/compression/mlfmm/octree.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace specklebem::mlfmm {

namespace {

constexpr int kMaxLevels = 21;  // finest level 20: 3 x 20 bits of a 64-bit Morton code

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

/// Interleaves the lowest `bits` bits of i, j, k (x lowest).
std::uint64_t encode(const std::array<Index, 3>& ijk, int bits) {
    std::uint64_t code = 0;
    for (int b = 0; b < bits; ++b) {
        for (int a = 0; a < 3; ++a) {
            const auto bit = (static_cast<std::uint64_t>(ijk[sz(a)]) >> b) & 1U;
            code |= bit << (3 * b + a);
        }
    }
    return code;
}

std::array<Index, 3> decode(std::uint64_t code, int bits) {
    std::array<Index, 3> ijk = {0, 0, 0};
    for (int b = 0; b < bits; ++b) {
        for (int a = 0; a < 3; ++a) {
            ijk[sz(a)] |= static_cast<Index>((code >> (3 * b + a)) & 1U) << b;
        }
    }
    return ijk;
}

Index chebyshev(const std::array<Index, 3>& a, const std::array<Index, 3>& b) {
    return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2])});
}

}  // namespace

Octree::Octree(const basis::RwgSpace& space, Real wavelength, const OctreeParams& p)
    : params_(p), wavelength_(wavelength) {
    if (space.size() == 0) {
        throw std::invalid_argument("Octree: the RWG space is empty");
    }
    if (!std::isfinite(wavelength) || !(wavelength > 0.0)) {
        throw std::invalid_argument("Octree: wavelength must be positive and finite");
    }
    if (p.max_elements_per_leaf < 1) {
        throw std::invalid_argument("Octree: max_elements_per_leaf must be >= 1");
    }
    if (p.max_levels < 1 || p.max_levels > kMaxLevels) {
        throw std::invalid_argument("Octree: max_levels must lie in [1, 21]");
    }
    if (!std::isfinite(p.min_box_size_lambda) || p.min_box_size_lambda < 0.0) {
        throw std::invalid_argument("Octree: min_box_size_lambda must be finite and >= 0");
    }
    build(space);
    build_lists();
}

void Octree::check_level(int level) const {
    if (level < 0 || level >= levels_) {
        throw std::out_of_range("Octree: level " + std::to_string(level) + " outside [0, " +
                                std::to_string(levels_) + ")");
    }
}

const std::vector<Index>& Octree::boxes_at_level(int level) const {
    check_level(level);
    return level_boxes_[sz(level)];
}

Real Octree::box_size(int level) const {
    check_level(level);
    return std::ldexp(root_size_, -level);
}

Index Octree::find_box(int level, const std::array<Index, 3>& ijk) const {
    check_level(level);
    const Index n = Index{1} << level;
    for (const Index c : ijk) {
        if (c < 0 || c >= n)
            return -1;
    }
    const auto& codes = level_codes_[sz(level)];
    const std::uint64_t code = encode(ijk, level);
    const auto it = std::lower_bound(codes.begin(), codes.end(), code);
    if (it == codes.end() || *it != code)
        return -1;
    return level_boxes_[sz(level)][static_cast<std::size_t>(it - codes.begin())];
}

void Octree::build(const basis::RwgSpace& space) {
    const auto& vertices = space.mesh().vertices();
    const auto& edges = space.mesh().edges();
    const Index num = space.size();

    std::vector<Vec3> mid(sz(num));
    Vec3 lo = Vec3::Constant(std::numeric_limits<Real>::infinity());
    Vec3 hi = -lo;
    for (Index n = 0; n < num; ++n) {
        const Vec3 m = 0.5 * (vertices.row(edges(n, 0)) + vertices.row(edges(n, 1))).transpose();
        if (!m.allFinite()) {
            throw std::invalid_argument("Octree: non-finite vertex coordinates");
        }
        mid[sz(n)] = m;
        lo = lo.cwiseMin(m);
        hi = hi.cwiseMax(m);
    }
    root_center_ = 0.5 * (lo + hi);
    const Real extent = (hi - lo).maxCoeff();
    root_size_ = extent > 0.0 ? extent * (1.0 + 1e-6) : wavelength_;

    // Deepest admissible level: box edge >= min_box_size_lambda * lambda and < max_levels.
    const Real min_size = params_.min_box_size_lambda * wavelength_;
    int cap = 0;
    while (cap + 1 < params_.max_levels && std::ldexp(root_size_, -(cap + 1)) >= min_size) {
        ++cap;
    }

    // Morton codes on the deepest admissible level, sorted (ties by basis index).
    const Vec3 origin = root_center_ - Vec3::Constant(0.5 * root_size_);
    const Real h = std::ldexp(root_size_, -cap);
    const Index cells = Index{1} << cap;
    std::vector<std::pair<std::uint64_t, Index>> keys(sz(num));
    for (Index n = 0; n < num; ++n) {
        std::array<Index, 3> ijk{};
        for (int a = 0; a < 3; ++a) {
            const Real t = std::floor((mid[sz(n)](a) - origin(a)) / h);
            ijk[sz(a)] = std::clamp(static_cast<Index>(t), Index{0}, cells - 1);
        }
        keys[sz(n)] = {encode(ijk, cap), n};
    }
    std::sort(keys.begin(), keys.end());

    // Leaf level: the first level on which no box exceeds max_elements_per_leaf.
    int depth = cap;
    for (int l = 0; l < cap; ++l) {
        const int shift = 3 * (cap - l);
        Index run = 0, max_run = 0;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            run = (i > 0 && (keys[i].first >> shift) == (keys[i - 1].first >> shift)) ? run + 1 : 1;
            max_run = std::max(max_run, run);
        }
        if (max_run <= params_.max_elements_per_leaf) {
            depth = l;
            break;
        }
    }
    levels_ = depth + 1;

    perm_.resize(sz(num));
    inv_perm_.resize(sz(num));
    for (Index i = 0; i < num; ++i) {
        perm_[sz(i)] = keys[sz(i)].second;
        inv_perm_[sz(keys[sz(i)].second)] = i;
    }

    level_boxes_.assign(sz(levels_), {});
    level_codes_.assign(sz(levels_), {});
    for (int l = 0; l < levels_; ++l) {
        const int shift = 3 * (cap - l);
        const Real s = std::ldexp(root_size_, -l);
        std::size_t parent = l > 0 ? sz(level_boxes_[sz(l - 1)].front()) : 0;
        for (std::size_t i = 0; i < keys.size();) {
            const std::uint64_t code = keys[i].first >> shift;
            std::size_t j = i + 1;
            while (j < keys.size() && (keys[j].first >> shift) == code) ++j;

            Box box;
            box.level = l;
            box.ijk = decode(code, l);
            box.center =
                origin + s * (Vec3(static_cast<Real>(box.ijk[0]), static_cast<Real>(box.ijk[1]),
                                   static_cast<Real>(box.ijk[2])) +
                              Vec3::Constant(0.5));
            box.first_element = static_cast<Index>(i);
            box.num_elements = static_cast<Index>(j - i);
            const auto index = static_cast<Index>(boxes_.size());
            if (l > 0) {
                // Parents of one level are consecutive and in the same order as their children.
                while (boxes_[parent].first_element + boxes_[parent].num_elements <=
                       box.first_element) {
                    ++parent;
                }
                box.parent = static_cast<Index>(parent);
                boxes_[parent].children[code & 7U] = index;
            }
            boxes_.push_back(std::move(box));
            level_boxes_[sz(l)].push_back(index);
            level_codes_[sz(l)].push_back(code);
            i = j;
        }
    }
}

void Octree::build_lists() {
    for (int l = 0; l < levels_; ++l) {
        for (const Index b : level_boxes_[sz(l)]) {
            const std::array<Index, 3> c = boxes_[sz(b)].ijk;
            auto& near = boxes_[sz(b)].near_list;
            for (Index dk = -1; dk <= 1; ++dk) {
                for (Index dj = -1; dj <= 1; ++dj) {
                    for (Index di = -1; di <= 1; ++di) {
                        if (di == 0 && dj == 0 && dk == 0)
                            continue;
                        const Index nb = find_box(l, {c[0] + di, c[1] + dj, c[2] + dk});
                        if (nb >= 0)
                            near.push_back(nb);
                    }
                }
            }
            std::sort(near.begin(), near.end());
        }
    }
    for (int l = 2; l < levels_; ++l) {
        for (const Index b : level_boxes_[sz(l)]) {
            Box& box = boxes_[sz(b)];
            const Box& parent = boxes_[sz(box.parent)];
            auto add_children = [&](const Box& p) {
                for (const Index ch : p.children) {
                    if (ch >= 0 && chebyshev(boxes_[sz(ch)].ijk, box.ijk) > 1) {
                        box.interaction_list.push_back(ch);
                    }
                }
            };
            add_children(parent);
            for (const Index pn : parent.near_list) add_children(boxes_[sz(pn)]);
            std::sort(box.interaction_list.begin(), box.interaction_list.end());
        }
    }
}

std::string Octree::summary() const {
    std::ostringstream os;
    os << std::setprecision(4);
    os << "Octree: " << levels_ << " levels (leaf level " << leaf_level() << "), " << perm_.size()
       << " elements, root edge " << root_size_ << " m = " << root_size_ / wavelength_
       << " lambda (lambda = " << wavelength_ << " m)\n";
    for (int l = 0; l < levels_; ++l) {
        const auto& ids = level_boxes_[sz(l)];
        std::size_t near_sum = 0, near_max = 0, inter_sum = 0, inter_max = 0;
        for (const Index b : ids) {
            const Box& box = boxes_[sz(b)];
            near_sum += box.near_list.size();
            near_max = std::max(near_max, box.near_list.size());
            inter_sum += box.interaction_list.size();
            inter_max = std::max(inter_max, box.interaction_list.size());
        }
        const auto count = static_cast<Real>(ids.size());
        os << "  level " << l << ": " << ids.size() << " boxes, edge " << box_size(l) / wavelength_
           << " lambda, near list mean " << static_cast<Real>(near_sum) / count << " max "
           << near_max << ", interaction list mean " << static_cast<Real>(inter_sum) / count
           << " max " << inter_max << "\n";
    }
    Index lo = std::numeric_limits<Index>::max(), hi = 0;
    const auto& leaves = level_boxes_[sz(leaf_level())];
    for (const Index b : leaves) {
        lo = std::min(lo, boxes_[sz(b)].num_elements);
        hi = std::max(hi, boxes_[sz(b)].num_elements);
    }
    os << "  elements per leaf: min " << lo << ", mean "
       << static_cast<Real>(perm_.size()) / static_cast<Real>(leaves.size()) << ", max " << hi
       << " (limit " << params_.max_elements_per_leaf << ")";
    return os.str();
}

}  // namespace specklebem::mlfmm
