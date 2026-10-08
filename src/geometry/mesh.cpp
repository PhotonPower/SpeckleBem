/// @file mesh.cpp
/// TriangleMesh: input validation, sort-based edge topology, orientation repair,
/// per-triangle geometry cache and quality report.
///
/// The per-triangle loops use plain 3-vectors on the raw row-major arrays instead of
/// Eigen expressions: this keeps the unoptimised sanitizer build (debug preset) fast
/// enough for meshes with 10^5 - 10^6 triangles and costs nothing in release.
#include "specklebem/geometry/mesh.hpp"

#include "specklebem/core/logging.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specklebem::geometry {

namespace {

/// Minimal 3-vector for hot loops.
struct P3 {
    Real x, y, z;
};

P3 load(const Real* xyz, Index i) {
    const Real* p = xyz + 3 * i;
    return {p[0], p[1], p[2]};
}
P3 operator-(const P3& a, const P3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Real dot(const P3& a, const P3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
P3 cross(const P3& a, const P3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// One directed triangle edge, stored with its undirected key (a < b).
struct HalfEdge {
    Index a;       ///< smaller vertex index
    Index b;       ///< larger vertex index
    Index t;       ///< owning triangle
    bool forward;  ///< true if the triangle traverses the edge as a -> b
};

/// All 3F half-edges sorted by the key (a, b) and then by t, so half-edges of the same
/// undirected edge are contiguous. Counting sort on a (O(F + V)) followed by a sort of
/// each bucket by b; buckets hold the edges of one vertex (its valence), so the total cost
/// is O(F + V) for bounded valence and O(F log F) in the worst case. No per-edge map.
std::vector<HalfEdge> sorted_half_edges(const Triangles& triangles, Index num_vertices) {
    const Index nh = 3 * triangles.rows();
    const Index* tri = triangles.data();  // row-major: tri[3 t + k]
    const auto key_a = [&](Index h) {
        const Index next = h - h % 3 + (h % 3 + 1) % 3;
        return std::min(tri[h], tri[next]);
    };

    std::vector<std::size_t> offset(static_cast<std::size_t>(num_vertices) + 1, 0);
    for (Index h = 0; h < nh; ++h) ++offset[static_cast<std::size_t>(key_a(h)) + 1];
    for (std::size_t i = 1; i < offset.size(); ++i) offset[i] += offset[i - 1];

    std::vector<HalfEdge> he(static_cast<std::size_t>(nh));
    std::vector<std::size_t> cursor(offset.begin(), offset.end() - 1);
    for (Index h = 0; h < nh; ++h) {
        const Index next = h - h % 3 + (h % 3 + 1) % 3;
        const Index p = tri[h];
        const Index q = tri[next];
        const Index t = h / 3;
        const HalfEdge e = p < q ? HalfEdge{p, q, t, true} : HalfEdge{q, p, t, false};
        he[cursor[static_cast<std::size_t>(e.a)]++] = e;
    }

    // Within a bucket the half-edges are in increasing t; a stable sort by b keeps that.
    for (std::size_t a = 0; a + 1 < offset.size(); ++a) {
        const std::size_t lo = offset[a];
        const std::size_t hi = offset[a + 1];
        if (hi - lo <= 32) {
            for (std::size_t i = lo + 1; i < hi; ++i) {  // insertion sort (stable)
                const HalfEdge x = he[i];
                std::size_t j = i;
                while (j > lo && he[j - 1].b > x.b) {
                    he[j] = he[j - 1];
                    --j;
                }
                he[j] = x;
            }
        } else {
            const auto first = he.begin() + static_cast<std::ptrdiff_t>(lo);
            const auto last = he.begin() + static_cast<std::ptrdiff_t>(hi);
            std::stable_sort(first, last,
                             [](const HalfEdge& x, const HalfEdge& y) { return x.b < y.b; });
        }
    }
    return he;
}

/// True if triangle t contains the directed edge a -> b (counter-clockwise order).
bool has_directed_edge(const Triangles& triangles, Index t, Index a, Index b) {
    for (int k = 0; k < 3; ++k) {
        if (triangles(t, k) == a && triangles(t, (k + 1) % 3) == b)
            return true;
    }
    return false;
}

}  // namespace

TriangleMesh::TriangleMesh(Vertices vertices, Triangles triangles)
    : vertices_(std::move(vertices)), triangles_(std::move(triangles)) {
    validate();
    build_topology();
    if (!consistently_oriented_) {
        const Index flipped = repair_orientation();
        SBEM_WARN(
            "TriangleMesh: inconsistent triangle orientation repaired ({} of {} triangles "
            "flipped)",
            flipped, num_triangles());
        build_topology();
        if (!consistently_oriented_) {
            throw std::logic_error("TriangleMesh: orientation repair failed");
        }
    }
    compute_geometry();
    label_components();
    orient_closed_components_outward();
    if (is_closed() && euler_characteristic() != 2 * num_components_) {
        SBEM_WARN(
            "TriangleMesh: closed mesh has Euler characteristic {} but {} edge-connected "
            "component(s) (expected {}; {} vertex-connected): genus > 0 or components touching "
            "at a non-manifold vertex",
            euler_characteristic(), num_components_, 2 * num_components_, num_vertex_components_);
    }
}

void TriangleMesh::validate() const {
    const Index nv = vertices_.rows();
    const Index nf = triangles_.rows();
    if (!vertices_.allFinite()) {
        throw std::invalid_argument("TriangleMesh: vertex coordinates must be finite");
    }
    const Real* xyz = vertices_.data();
    const Index* tri = triangles_.data();
    for (Index t = 0; t < nf; ++t) {
        const Index i0 = tri[3 * t];
        const Index i1 = tri[3 * t + 1];
        const Index i2 = tri[3 * t + 2];
        if (i0 < 0 || i0 >= nv || i1 < 0 || i1 >= nv || i2 < 0 || i2 >= nv) {
            throw std::invalid_argument("TriangleMesh: triangle " + std::to_string(t) + " (" +
                                        std::to_string(i0) + ", " + std::to_string(i1) + ", " +
                                        std::to_string(i2) + ") references a vertex outside [0, " +
                                        std::to_string(nv) + ")");
        }
        if (i0 == i1 || i1 == i2 || i0 == i2) {
            throw std::invalid_argument("TriangleMesh: triangle " + std::to_string(t) +
                                        " has repeated vertex indices");
        }
        const P3 v0 = load(xyz, i0);
        const P3 e1 = load(xyz, i1) - v0;
        const P3 e2 = load(xyz, i2) - v0;
        const P3 e3 = e2 - e1;
        const Real lmax2 = std::max({dot(e1, e1), dot(e2, e2), dot(e3, e3)});
        // |e1 x e2|^2 / lmax^4 = (2A / lmax^2)^2, the squared height-to-longest-edge ratio.
        // The threshold (1e-12)^2 lies far above rounding noise (~1e-16) for collinear points.
        const P3 c = cross(e1, e2);
        if (!(dot(c, c) > 1e-24 * lmax2 * lmax2)) {
            throw std::invalid_argument("TriangleMesh: triangle " + std::to_string(t) +
                                        " has (numerically) zero area");
        }
    }
}

void TriangleMesh::build_topology() {
    const std::vector<HalfEdge> he = sorted_half_edges(triangles_, num_vertices());
    const std::size_t nh = he.size();
    const auto same_key = [&](std::size_t i, std::size_t j) {
        return he[i].a == he[j].a && he[i].b == he[j].b;
    };

    // Pass 1: classify edges by the number of adjacent triangles.
    Index n_interior = 0;
    Index n_boundary = 0;
    Index n_nonmanifold = 0;
    for (std::size_t i = 0; i < nh;) {
        std::size_t j = i + 1;
        while (j < nh && same_key(i, j)) ++j;
        const std::size_t count = j - i;
        if (count == 1) {
            ++n_boundary;
        } else if (count == 2) {
            ++n_interior;
        } else {
            ++n_nonmanifold;
        }
        i = j;
    }
    num_nonmanifold_edges_ = n_nonmanifold;
    if (n_nonmanifold > 0) {
        throw std::invalid_argument("TriangleMesh: " + std::to_string(n_nonmanifold) +
                                    " non-manifold edge(s) (more than two adjacent triangles)");
    }

    // Pass 2: fill interior and boundary edges, assign plus / minus triangles.
    edges_.resize(n_interior, 2);
    edge_triangles_.resize(n_interior, 2);
    boundary_edges_.resize(n_boundary, 2);
    Index* edges = edges_.data();
    Index* edge_tris = edge_triangles_.data();
    Index* boundary = boundary_edges_.data();
    boundary_edge_triangle_.assign(static_cast<std::size_t>(n_boundary), -1);
    const Index* tri = triangles_.data();
    const auto opposite_vertex = [&](Index t, Index a, Index b) {
        return tri[3 * t] + tri[3 * t + 1] + tri[3 * t + 2] - a - b;
    };
    consistently_oriented_ = true;
    Index e = 0;
    Index eb = 0;
    for (std::size_t i = 0; i < nh;) {
        if (i + 1 < nh && same_key(i, i + 1)) {
            const HalfEdge& h0 = he[i];
            const HalfEdge& h1 = he[i + 1];
            // Two triangles on one edge with the same opposite vertex have the same vertex
            // set: a duplicate triangle (in any order / orientation).
            if (opposite_vertex(h0.t, h0.a, h0.b) == opposite_vertex(h1.t, h0.a, h0.b)) {
                throw std::invalid_argument("TriangleMesh: triangles " + std::to_string(h0.t) +
                                            " and " + std::to_string(h1.t) +
                                            " are duplicates (same three vertices)");
            }
            edges[2 * e] = h0.a;
            edges[2 * e + 1] = h0.b;
            if (h0.forward == h1.forward) {
                // Both triangles traverse the edge in the same direction. The constructor
                // repairs this; the provisional assignment below is never exposed.
                consistently_oriented_ = false;
            }
            const bool h0_plus = h0.forward || !h1.forward;
            edge_tris[2 * e] = h0_plus ? h0.t : h1.t;
            edge_tris[2 * e + 1] = h0_plus ? h1.t : h0.t;
            ++e;
            i += 2;
        } else {
            boundary[2 * eb] = he[i].a;
            boundary[2 * eb + 1] = he[i].b;
            boundary_edge_triangle_[static_cast<std::size_t>(eb)] = he[i].t;
            ++eb;
            i += 1;
        }
    }
}

Index TriangleMesh::repair_orientation() {
    const Index nf = num_triangles();
    const auto nf_s = static_cast<std::size_t>(nf);

    // Triangle adjacency over interior edges: up to three neighbours per triangle and a
    // flag telling whether the pair currently traverses the shared edge in the same
    // direction (then exactly one of the two must be flipped).
    std::vector<std::array<Index, 3>> nbr(nf_s, {-1, -1, -1});
    std::vector<std::array<bool, 3>> same(nf_s, {false, false, false});
    std::vector<std::size_t> fill(nf_s, 0);
    for (Index e = 0; e < num_edges(); ++e) {
        const Index a = edges_(e, 0);
        const Index b = edges_(e, 1);
        const Index t0 = edge_triangles_(e, 0);
        const Index t1 = edge_triangles_(e, 1);
        const bool s =
            has_directed_edge(triangles_, t0, a, b) == has_directed_edge(triangles_, t1, a, b);
        for (const auto& [t, u] : {std::pair{t0, t1}, std::pair{t1, t0}}) {
            const auto ts = static_cast<std::size_t>(t);
            const std::size_t k = fill[ts]++;
            nbr[ts][k] = u;
            same[ts][k] = s;
        }
    }

    // Breadth-first traversal per connected component; state 0 = keep, 1 = flip.
    std::vector<signed char> state(nf_s, -1);
    std::vector<Index> component;
    Index n_flipped = 0;
    for (Index root = 0; root < nf; ++root) {
        if (state[static_cast<std::size_t>(root)] >= 0)
            continue;
        component.clear();
        component.push_back(root);
        state[static_cast<std::size_t>(root)] = 0;
        for (std::size_t head = 0; head < component.size(); ++head) {
            const auto ts = static_cast<std::size_t>(component[head]);
            for (std::size_t k = 0; k < 3; ++k) {
                const Index u = nbr[ts][k];
                if (u < 0)
                    continue;
                const auto us = static_cast<std::size_t>(u);
                const auto want = static_cast<signed char>(same[ts][k] ? 1 - state[ts] : state[ts]);
                if (state[us] < 0) {
                    state[us] = want;
                    component.push_back(u);
                } else if (state[us] != want) {
                    throw std::invalid_argument(
                        "TriangleMesh: surface is non-orientable (e.g. a Moebius strip); no "
                        "consistent triangle orientation exists");
                }
            }
        }
        // Flip the minority so that the input orientation is kept where possible.
        Index ones = 0;
        for (const Index t : component) ones += state[static_cast<std::size_t>(t)];
        const bool invert = 2 * ones > static_cast<Index>(component.size());
        for (const Index t : component) {
            if ((state[static_cast<std::size_t>(t)] == 1) != invert) {
                std::swap(triangles_(t, 1), triangles_(t, 2));
                ++n_flipped;
            }
        }
    }
    return n_flipped;
}

namespace {

/// Union-find root with path halving.
Index find_root(std::vector<Index>& parent, Index i) {
    while (parent[static_cast<std::size_t>(i)] != i) {
        const auto is = static_cast<std::size_t>(i);
        parent[is] = parent[static_cast<std::size_t>(parent[is])];
        i = parent[is];
    }
    return i;
}

void unite(std::vector<Index>& parent, Index i, Index j) {
    const Index ri = find_root(parent, i);
    const Index rj = find_root(parent, j);
    if (ri != rj)
        parent[static_cast<std::size_t>(std::max(ri, rj))] = std::min(ri, rj);
}

}  // namespace

void TriangleMesh::label_components() {
    const Index nf = num_triangles();
    const Index nv = num_vertices();

    // Edge-connected components of triangles, labelled 0, 1, ... in order of first triangle.
    std::vector<Index> parent(static_cast<std::size_t>(nf));
    for (Index t = 0; t < nf; ++t) parent[static_cast<std::size_t>(t)] = t;
    for (Index e = 0; e < num_edges(); ++e) {
        unite(parent, edge_triangles_(e, 0), edge_triangles_(e, 1));
    }
    triangle_component_.assign(static_cast<std::size_t>(nf), -1);
    std::vector<Index> label_of_root(static_cast<std::size_t>(nf), -1);
    num_components_ = 0;
    for (Index t = 0; t < nf; ++t) {
        const auto r = static_cast<std::size_t>(find_root(parent, t));
        if (label_of_root[r] < 0)
            label_of_root[r] = num_components_++;
        triangle_component_[static_cast<std::size_t>(t)] = label_of_root[r];
    }

    // Vertex-connected components (triangles sharing only a vertex are connected) and the
    // number of referenced vertices.
    std::vector<Index> vparent(static_cast<std::size_t>(nv));
    for (Index v = 0; v < nv; ++v) vparent[static_cast<std::size_t>(v)] = v;
    std::vector<bool> used(static_cast<std::size_t>(nv), false);
    const Index* tri = triangles_.data();
    for (Index t = 0; t < nf; ++t) {
        unite(vparent, tri[3 * t], tri[3 * t + 1]);
        unite(vparent, tri[3 * t], tri[3 * t + 2]);
        for (int k = 0; k < 3; ++k) used[static_cast<std::size_t>(tri[3 * t + k])] = true;
    }
    num_used_vertices_ = 0;
    num_vertex_components_ = 0;
    for (Index v = 0; v < nv; ++v) {
        if (!used[static_cast<std::size_t>(v)])
            continue;
        ++num_used_vertices_;
        if (find_root(vparent, v) == v)
            ++num_vertex_components_;
    }
}

void TriangleMesh::orient_closed_components_outward() {
    const auto nc = static_cast<std::size_t>(num_components_);
    if (nc == 0)
        return;
    const auto comp = [&](Index t) {
        return static_cast<std::size_t>(triangle_component_[static_cast<std::size_t>(t)]);
    };

    std::vector<bool> closed(nc, true);
    for (const Index t : boundary_edge_triangle_) closed[comp(t)] = false;

    // Reference point per component (mean triangle centroid) to limit cancellation; the
    // volume of a closed component does not depend on it.
    const Real* cen = centroids_.data();
    const Real* nrm = normals_.data();
    std::vector<P3> ref(nc, P3{0.0, 0.0, 0.0});
    std::vector<Index> count(nc, 0);
    for (Index t = 0; t < num_triangles(); ++t) {
        const P3 c = load(cen, t);
        P3& r = ref[comp(t)];
        r = P3{r.x + c.x, r.y + c.y, r.z + c.z};
        ++count[comp(t)];
    }
    for (std::size_t c = 0; c < nc; ++c) {
        const auto n = static_cast<Real>(count[c]);
        ref[c] = P3{ref[c].x / n, ref[c].y / n, ref[c].z / n};
    }
    std::vector<Real> volume(nc, 0.0);
    for (Index t = 0; t < num_triangles(); ++t) {
        volume[comp(t)] += dot(load(cen, t) - ref[comp(t)], load(nrm, t)) * areas_(t) / 3.0;
    }

    std::vector<bool> flip(nc, false);
    bool any = false;
    for (std::size_t c = 0; c < nc; ++c) {
        if (closed[c] && volume[c] < 0.0) {
            flip[c] = true;
            any = true;
            SBEM_WARN(
                "TriangleMesh: closed component {} ({} triangles) had inward normals (signed "
                "volume {:.6g} m^3); flipped",
                c, count[c], volume[c]);
        }
    }
    if (!any)
        return;
    for (Index t = 0; t < num_triangles(); ++t) {
        if (!flip[comp(t)])
            continue;
        std::swap(triangles_(t, 1), triangles_(t, 2));
        normals_.row(t) = -normals_.row(t);
    }
    for (Index e = 0; e < num_edges(); ++e) {
        if (flip[comp(edge_triangles_(e, 0))])
            std::swap(edge_triangles_(e, 0), edge_triangles_(e, 1));
    }
}

Index TriangleMesh::euler_characteristic() const {
    return num_used_vertices_ - (num_edges() + num_boundary_edges()) + num_triangles();
}

void TriangleMesh::compute_geometry() {
    const Index nf = num_triangles();
    centroids_.resize(nf, 3);
    normals_.resize(nf, 3);
    areas_.resize(nf);
    const Real* xyz = vertices_.data();
    const Index* tri = triangles_.data();
    Real* cen = centroids_.data();
    Real* nrm = normals_.data();
    Real* area = areas_.data();
    for (Index t = 0; t < nf; ++t) {
        const P3 v0 = load(xyz, tri[3 * t]);
        const P3 v1 = load(xyz, tri[3 * t + 1]);
        const P3 v2 = load(xyz, tri[3 * t + 2]);
        const P3 c = cross(v1 - v0, v2 - v0);
        const Real norm = std::sqrt(dot(c, c));
        cen[3 * t] = (v0.x + v1.x + v2.x) / 3.0;
        cen[3 * t + 1] = (v0.y + v1.y + v2.y) / 3.0;
        cen[3 * t + 2] = (v0.z + v1.z + v2.z) / 3.0;
        nrm[3 * t] = c.x / norm;
        nrm[3 * t + 1] = c.y / norm;
        nrm[3 * t + 2] = c.z / norm;
        area[t] = 0.5 * norm;
    }
}

Vec3 TriangleMesh::centroid(Index t) const {
    return centroids_.row(t).transpose();
}

Vec3 TriangleMesh::normal(Index t) const {
    return normals_.row(t).transpose();
}

Real TriangleMesh::area(Index t) const {
    return areas_(t);
}

Real TriangleMesh::edge_length(Index e) const {
    const P3 d = load(vertices_.data(), edges_(e, 1)) - load(vertices_.data(), edges_(e, 0));
    return std::sqrt(dot(d, d));
}

std::pair<Vec3, Vec3> TriangleMesh::bounding_box() const {
    if (num_vertices() == 0) {
        throw std::logic_error("TriangleMesh::bounding_box: mesh has no vertices");
    }
    return {vertices_.colwise().minCoeff().transpose(), vertices_.colwise().maxCoeff().transpose()};
}

bool TriangleMesh::is_closed() const {
    return num_triangles() > 0 && num_boundary_edges() == 0 && num_nonmanifold_edges_ == 0;
}

Real TriangleMesh::signed_volume() const {
    if (num_triangles() == 0)
        return 0.0;
    const auto [lo, hi] = bounding_box();
    const P3 ref{0.5 * (lo.x() + hi.x()), 0.5 * (lo.y() + hi.y()), 0.5 * (lo.z() + hi.z())};
    const Real* cen = centroids_.data();
    const Real* nrm = normals_.data();
    Real sum = 0.0;
    for (Index t = 0; t < num_triangles(); ++t) {
        sum += dot(load(cen, t) - ref, load(nrm, t)) * areas_(t);
    }
    return sum / 3.0;
}

void TriangleMesh::flip_normals() {
    triangles_.col(1).swap(triangles_.col(2));
    edge_triangles_.col(0).swap(edge_triangles_.col(1));
    normals_ = -normals_;
}

std::string TriangleMesh::quality_report() const {
    const Index nv = num_vertices();
    const Index ne = num_edges();
    const Index nb = num_boundary_edges();
    const Index nf = num_triangles();
    const Real* xyz = vertices_.data();

    Real lmin = std::numeric_limits<Real>::infinity();
    Real lmax = 0.0;
    Real lsum = 0.0;
    const auto accumulate = [&](const Edges& edges) {
        for (Index e = 0; e < edges.rows(); ++e) {
            const P3 d = load(xyz, edges(e, 1)) - load(xyz, edges(e, 0));
            const Real l = std::sqrt(dot(d, d));
            lmin = std::min(lmin, l);
            lmax = std::max(lmax, l);
            lsum += l;
        }
    };
    accumulate(edges_);
    accumulate(boundary_edges_);
    const Index n_all = ne + nb;

    // Aspect ratio R / (2 r) = a b c s / (8 A^2) with s = (a + b + c) / 2; 1 for equilateral.
    Real max_aspect = 0.0;
    const Index* tri = triangles_.data();
    for (Index t = 0; t < nf; ++t) {
        const P3 v0 = load(xyz, tri[3 * t]);
        const P3 v1 = load(xyz, tri[3 * t + 1]);
        const P3 v2 = load(xyz, tri[3 * t + 2]);
        const P3 d01 = v1 - v0;
        const P3 d12 = v2 - v1;
        const P3 d20 = v0 - v2;
        const Real a = std::sqrt(dot(d01, d01));
        const Real b = std::sqrt(dot(d12, d12));
        const Real c = std::sqrt(dot(d20, d20));
        const Real s = 0.5 * (a + b + c);
        const Real area_t = areas_(t);
        max_aspect = std::max(max_aspect, a * b * c * s / (8.0 * area_t * area_t));
    }

    std::ostringstream os;
    os << std::setprecision(6);
    os << "TriangleMesh quality report\n";
    os << "  vertices (V)            : " << nv << "\n";
    os << "  unreferenced vertices   : " << nv - num_used_vertices_ << "\n";
    os << "  interior edges (E)      : " << ne << "\n";
    os << "  triangles (F)           : " << nf << "\n";
    os << "  Euler characteristic    : " << euler_characteristic()
       << "  (V_used - E_all + F, E_all = " << n_all << " incl. boundary edges)\n";
    os << "  components (by edges)   : " << num_components_ << "\n";
    os << "  components (by vertices): " << num_vertex_components_ << "\n";
    os << "  closed                  : " << (is_closed() ? "yes" : "no") << "\n";
    os << "  consistently oriented   : " << (consistently_oriented_ ? "yes" : "no") << "\n";
    os << "  boundary edges          : " << nb << "\n";
    os << "  non-manifold edges      : " << num_nonmanifold_edges_ << "\n";
    if (n_all > 0) {
        os << "  edge length min/max/mean: " << lmin << " / " << lmax << " / "
           << lsum / static_cast<Real>(n_all) << " m\n";
    } else {
        os << "  edge length min/max/mean: n/a\n";
    }
    if (nf > 0) {
        os << "  max aspect ratio        : " << max_aspect << "  (R / 2r, 1 = equilateral)\n";
    } else {
        os << "  max aspect ratio        : n/a\n";
    }
    return os.str();
}

}  // namespace specklebem::geometry
