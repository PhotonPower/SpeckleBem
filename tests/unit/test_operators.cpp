/// Tests of kernels::element_blocks, kernels::jump_block and kernels::validate (WP7).
///
/// References are written directly from the Galerkin definitions with kernels::green,
/// kernels::grad_green, RwgSpace::value and RwgSpace::divergence:
///  * far / near pairs: (composite) Dunavant double quadrature;
///  * touching pairs: outer Dunavant rule, inner integral of the full kernel by a polar
///    (Duffy) transform about the projection of the outer point, with graded Gauss-Legendre
///    panels (independent of the singularity subtraction and of static_integrals).
///
/// Test sets: the regular cases run with ctest and take < 1 s each in the sanitizer build
/// (docs/05): far pairs (2 pairs, 3 materials), near pairs (degree 19 vs plain degree-20
/// brute force), touching pairs (identical, folded shared edge, folded shared vertex; Si;
/// quad_degree_sing in {4, 10, 12}), symmetry per material, static limit, jump block, options,
/// slots, coplanar K, non-finite guard and timing. The full sweep is in hidden test cases
/// tagged [.slow] (not registered by catch_discover_tests): far (6 pairs), near against the
/// composite (4 sub-triangle) brute force, touching (8 pairs incl. coplanar fans and a 90 degree
/// hinge, all three materials, degrees {4, 5, 6, 8, 9, 10, 12}) and the self-convergence of the
/// polar reference. Run them with `build/<preset>/tests/specklebem_unit_tests "[slow]"`.
/// The zero-allocation check lives in its own executable (test_operators_alloc.cpp) because
/// it replaces the global operator new.
///
/// Geometry scale: the work package names make_icosphere(1e-6, 1) (longest edge ~0.62 um).
/// At 500 nm this is k h = 7.8 in vacuum, 33 in Si and 24 in Ag, i.e. 1.2 to 5 wavelengths
/// per triangle edge, far outside any meshing rule (lambda/10 in the material; docs/06). It is
/// used where accuracy does not depend on k h (far pairs: identical rules) and for one vacuum
/// near pair. Near-pair and symmetry tests use an icosphere per material whose radius is
/// scaled with the material wavenumber so that |k| h = 0.78 (h ~ lambda_material / 8); the
/// touching-pair tests use one geometry sized for Si (|k| h = 0.78 in Si, 0.57 in Ag, 0.18 in
/// vacuum) so that the references of all three materials are evaluated in one pass.
///
/// K convention (docs/03): K_mn = int int f_m . (grad' G x f_n) with the source gradient
/// grad' G = -grad_r G = -kernels::grad_green.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/green.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/kernels/singularity.hpp"
#include "specklebem/material/material.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace specklebem;
using basis::RwgSpace;
using geometry::make_icosphere;
using geometry::TriangleMesh;
using kernels::classify;
using kernels::element_blocks;
using kernels::OperatorOptions;
using kernels::Proximity;
using kernels::RegionParams;
using Block = Eigen::Matrix<Complex, 3, 3>;

namespace {

constexpr Real kPi = constants::pi;
constexpr Real kLambda = 500e-9;
const Complex kJ(0.0, 1.0);
/// Icosphere radius giving |k| h = 0.78 in vacuum at 500 nm (h = longest edge).
constexpr Real kVacuumRadius = 1e-7;

struct NamedRegion {
    const char* name;
    RegionParams p;
};

RegionParams region_of(const material::Material& m, Real lambda) {
    const Real omega = 2.0 * kPi * constants::c0 / lambda;
    return {m.wavenumber(omega), m.wave_impedance(omega), omega, constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

std::vector<NamedRegion> all_regions() {
    return {NamedRegion{"vacuum", region_of(material::vacuum(), kLambda)},
            NamedRegion{"Si", region_of(material::silicon_500nm(), kLambda)},
            NamedRegion{"Ag", region_of(material::silver_500nm(), kLambda)}};
}

/// Length scale with the same |k| L for every material: L_vacuum k0 / |k|.
Real scaled(Real length_vacuum, const RegionParams& reg) {
    const Real k0 = 2.0 * kPi / kLambda;
    return length_vacuum * k0 / std::abs(reg.k);
}

Vec3 vertex(const TriangleMesh& m, Index v) {
    return m.vertices().row(v).transpose();
}

std::array<Vec3, 3> corners(const TriangleMesh& m, Index t) {
    return {vertex(m, m.triangles()(t, 0)), vertex(m, m.triangles()(t, 1)),
            vertex(m, m.triangles()(t, 2))};
}

Real longest_edge(const TriangleMesh& m, Index t) {
    const auto v = corners(m, t);
    return std::max({(v[1] - v[0]).norm(), (v[2] - v[1]).norm(), (v[0] - v[2]).norm()});
}

/// Fan of n_rim triangles around a centre vertex at height `height` above the rim circle of
/// radius a (z = 0): coplanar for height 0, a tent otherwise. Open mesh; every triangle
/// carries two RWG functions (the spokes), so slot 2 is unused. All pairs touch (shared edge
/// for neighbours, shared vertex otherwise).
TriangleMesh fan_mesh(Real a, Real height, int n_rim) {
    Vertices v(n_rim + 1, 3);
    v.row(0) << 0.0, 0.0, height;
    for (int i = 0; i < n_rim; ++i) {
        // Slightly irregular rim so that no two triangles are congruent.
        const Real phi = 2.0 * kPi * (i + 0.15 * std::sin(1.7 * i)) / n_rim;
        const Real rad = a * (1.0 + 0.1 * std::cos(2.3 * i));
        v.row(i + 1) << rad * std::cos(phi), rad * std::sin(phi), 0.0;
    }
    Triangles f(n_rim, 3);
    for (int i = 0; i < n_rim; ++i) {
        f.row(i) << 0, i + 1, (i + 1) % n_rim + 1;
    }
    return TriangleMesh(v, f);
}

/// Two triangles hinged at 90 degrees along their shared edge (one RWG function).
TriangleMesh hinge_mesh(Real a) {
    Vertices v(4, 3);
    v << 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.4 * a, 0.8 * a, 0.0, 0.55 * a, 0.0, 0.85 * a;
    Triangles f(2, 3);
    f << 0, 1, 2, 1, 0, 3;
    return TriangleMesh(v, f);
}

/// Quadrature points with weights (area included).
struct QuadSet {
    std::vector<Vec3> r;
    std::vector<Real> w;
};

/// Composite Dunavant rule: triangle split into n_sub^2 congruent sub-triangles.
QuadSet dunavant_points(const TriangleMesh& m, Index t, int degree, int n_sub) {
    const auto v = corners(m, t);
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    const Vec3 a = (v[1] - v[0]) / n_sub;
    const Vec3 b = (v[2] - v[0]) / n_sub;
    const Real sub_area = 0.5 * a.cross(b).norm();
    QuadSet out;
    const auto add = [&](const Vec3& p0, const Vec3& p1, const Vec3& p2) {
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            out.r.push_back(l(0) * p0 + l(1) * p1 + l(2) * p2);
            out.w.push_back(rule.weights[q] * sub_area);
        }
    };
    for (int i = 0; i < n_sub; ++i) {
        for (int j = 0; i + j < n_sub; ++j) {
            const Vec3 p = v[0] + i * a + j * b;
            add(p, p + a, p + b);
            if (i + j < n_sub - 1) {
                add(p + a, p + a + b, p + b);
            }
        }
    }
    return out;
}

struct Blocks {
    Block L = Block::Zero();
    Block K = Block::Zero();
};

/// Raw Galerkin sums written directly from the definitions with kernels::green,
/// kernels::grad_green (K uses the source gradient grad' G = -grad_green, docs/03),
/// RwgSpace::value and RwgSpace::divergence, for several regions at once and the same point
/// sets for every outer point (far and near pairs). The cross product is written out: Eigen's
/// MatrixBase::cross returns conj(a x b) for complex scalars (Eigen/src/Geometry/
/// OrthoMethods.h), which would conjugate K. Scalar inner loop so that the unoptimised
/// sanitizer build stays fast.
std::vector<Blocks> brute_force(const RwgSpace& space, Index t_test, Index t_src,
                                const std::vector<NamedRegion>& regs, const QuadSet& outer,
                                const QuadSet& inner) {
    const RwgSpace::Support st = space.support(t_test);
    const RwgSpace::Support ss = space.support(t_src);
    const auto nt = static_cast<std::size_t>(st.count);
    const auto ns = static_cast<std::size_t>(ss.count);
    std::vector<std::array<Vec3, 3>> fm(outer.r.size());
    std::vector<std::array<Vec3, 3>> fn(inner.r.size());
    for (std::size_t i = 0; i < outer.r.size(); ++i) {
        for (std::size_t a = 0; a < nt; ++a) {
            fm[i][a] = space.value(st.n[a], t_test, outer.r[i]);
        }
    }
    for (std::size_t j = 0; j < inner.r.size(); ++j) {
        for (std::size_t b = 0; b < ns; ++b) {
            fn[j][b] = space.value(ss.n[b], t_src, inner.r[j]);
        }
    }
    std::array<Real, 3> dm{};
    std::array<Real, 3> dn{};
    for (std::size_t a = 0; a < nt; ++a) {
        dm[a] = space.divergence(st.n[a], t_test);
    }
    for (std::size_t b = 0; b < ns; ++b) {
        dn[b] = space.divergence(ss.n[b], t_src);
    }
    std::vector<Blocks> out(regs.size());
    for (std::size_t g = 0; g < regs.size(); ++g) {
        const RegionParams& reg = regs[g].p;
        const Complex c_vec = kJ * reg.omega * reg.mu;
        const Complex c_sca = 1.0 / (kJ * reg.omega * reg.eps);
        for (std::size_t i = 0; i < outer.r.size(); ++i) {
            Complex sum_g{0.0, 0.0};
            std::array<std::array<Complex, 3>, 3> v{};  // int G f_n
            std::array<std::array<Complex, 3>, 3> u{};  // int grad' G x f_n
            for (std::size_t j = 0; j < inner.r.size(); ++j) {
                const Complex gr = inner.w[j] * kernels::green(outer.r[i], inner.r[j], reg.k);
                const Vec3c gp = -inner.w[j] * kernels::grad_green(outer.r[i], inner.r[j], reg.k);
                sum_g += gr;
                for (std::size_t b = 0; b < ns; ++b) {
                    const Vec3& f = fn[j][b];
                    v[b][0] += gr * f(0);
                    v[b][1] += gr * f(1);
                    v[b][2] += gr * f(2);
                    u[b][0] += gp(1) * f(2) - gp(2) * f(1);
                    u[b][1] += gp(2) * f(0) - gp(0) * f(2);
                    u[b][2] += gp(0) * f(1) - gp(1) * f(0);
                }
            }
            for (std::size_t a = 0; a < nt; ++a) {
                const Vec3& f = fm[i][a];
                for (std::size_t b = 0; b < ns; ++b) {
                    const auto ai = static_cast<Eigen::Index>(a);
                    const auto bi = static_cast<Eigen::Index>(b);
                    out[g].L(ai, bi) +=
                        outer.w[i] * (c_vec * (f(0) * v[b][0] + f(1) * v[b][1] + f(2) * v[b][2]) +
                                      c_sca * (dm[a] * dn[b]) * sum_g);
                    out[g].K(ai, bi) +=
                        outer.w[i] * (f(0) * u[b][0] + f(1) * u[b][1] + f(2) * u[b][2]);
                }
            }
        }
    }
    return out;
}

/// Brute force with the same composite Dunavant rule on both triangles.
std::vector<Blocks> brute_force_dunavant(const RwgSpace& space, Index t_test, Index t_src,
                                         const std::vector<NamedRegion>& regs, int degree,
                                         int n_sub) {
    return brute_force(space, t_test, t_src, regs,
                       dunavant_points(space.mesh(), t_test, degree, n_sub),
                       dunavant_points(space.mesh(), t_src, degree, n_sub));
}

/// Composite Gauss-Legendre rule on [0, 1], geometrically graded about `center` (clamped to
/// [0, 1]): breakpoints center +- scale 2^k, plus 0, 1 and center.
std::vector<std::pair<Real, Real>> graded_unit_rule(Real center, Real scale,
                                                    const kernels::LineRule& g) {
    const Real c = std::clamp(center, 0.0, 1.0);
    std::vector<Real> bp = {0.0, 1.0, c};
    for (const Real side : {-1.0, 1.0}) {
        for (Real s = scale; s < 1.0; s *= 2.0) {
            const Real x = c + side * s;
            if (x > 0.0 && x < 1.0) {
                bp.push_back(x);
            }
        }
    }
    std::sort(bp.begin(), bp.end());
    std::vector<std::pair<Real, Real>> rule;
    for (std::size_t k = 0; k + 1 < bp.size(); ++k) {
        const Real a = bp[k];
        const Real b = bp[k + 1];
        if (b - a <= 1e-15) {
            continue;
        }
        for (std::size_t i = 0; i < g.nodes.size(); ++i) {
            rule.emplace_back(a + 0.5 * (b - a) * (g.nodes[i] + 1.0), 0.5 * (b - a) * g.weights[i]);
        }
    }
    return rule;
}

/// Independent inner rule for the source triangle v and observation point r: signed
/// sub-triangles (rho, v_i, v_{i+1}) about the projection rho = r - d n of r onto the plane
/// of the triangle, Duffy map r' = rho + t w(s), w(s) = p + s (q - p) - rho,
/// dS' = 2 A_sub t ds dt (the 1/R and 1/R^2 kernels become bounded), Gauss-Legendre in t on
/// panels graded about t = 0 with scale |d| / |w(s)|, and in s graded about the foot of the
/// perpendicular from rho onto the edge line with scale max(dist, |d|) / |q - p|. For
/// in-plane r (identical or coplanar pairs) one t panel suffices: t / R is analytic in t, and
/// the gradient kernel enters only through f_m . (grad G x f_n), a triple product of in-plane
/// vectors that vanishes pointwise. Calls visit(x, y, z, w) for every point.
template <typename Visit>
void polar_points(const Vec3& r, const std::array<Vec3, 3>& v, const kernels::LineRule& g,
                  Visit&& visit) {
    const Vec3 nn = (v[1] - v[0]).cross(v[2] - v[0]).normalized();
    const Real h = std::max({(v[1] - v[0]).norm(), (v[2] - v[1]).norm(), (v[0] - v[2]).norm()});
    const Real d = (r - v[0]).dot(nn);
    const Vec3 rho = r - d * nn;
    const bool in_plane = std::abs(d) <= 1e-12 * h;
    for (std::size_t e = 0; e < 3; ++e) {
        const Vec3& p = v[e];
        const Vec3& q = v[(e + 1) % 3];
        const Real signed_twice_area = (p - rho).cross(q - rho).dot(nn);
        if (std::abs(signed_twice_area) <= 1e-14 * h * h) {
            continue;  // rho on the line of this edge: empty sub-triangle
        }
        const Vec3 edge = q - p;
        const Real len = edge.norm();
        const Real s0 = (rho - p).dot(edge) / (len * len);
        const Real dist_line = std::abs(signed_twice_area) / len;
        for (const auto& [s, ws] :
             graded_unit_rule(s0, std::max(dist_line, std::abs(d)) / len, g)) {
            const Vec3 w = p + s * edge - rho;
            const Real c = in_plane ? 1.0 : std::max(std::abs(d) / w.norm(), 1e-4);
            const Real w0 = w(0);
            const Real w1 = w(1);
            const Real w2 = w(2);
            const Real rho0 = rho(0);
            const Real rho1 = rho(1);
            const Real rho2 = rho(2);
            for (const auto& [t, wt] : graded_unit_rule(0.0, c, g)) {
                visit(rho0 + t * w0, rho1 + t * w1, rho2 + t * w2, ws * wt * signed_twice_area * t);
            }
        }
    }
}

/// Reference blocks for a touching pair, one ordering: outer Dunavant rule of degree
/// `outer_degree` on t_test, inner integral of the full kernel G = e^{-jkR} / (4 pi R) and
/// grad' G = (1 + jkR) G (r - r') / R^2 with polar_points (n Gauss-Legendre points per panel),
/// for several regions at once. RWG values from space.value at one point per triangle plus
/// the affine structure f(r') = f(r0) + (div / 2)(r' - r0) (scalar code, fast in the
/// sanitizer build); independent of the singularity subtraction and of static_integrals.
std::vector<Blocks> polar_reference_one(const RwgSpace& space, Index t_test, Index t_src,
                                        const std::vector<NamedRegion>& regs, int outer_degree,
                                        int n) {
    const TriangleMesh& mesh = space.mesh();
    const auto src = corners(mesh, t_src);
    const kernels::LineRule g = kernels::gauss_legendre(n);
    const QuadSet outer = dunavant_points(mesh, t_test, outer_degree, 1);
    const RwgSpace::Support st = space.support(t_test);
    const RwgSpace::Support ss = space.support(t_src);
    const auto nt = static_cast<std::size_t>(st.count);
    const auto ns = static_cast<std::size_t>(ss.count);
    const Vec3 c_src = mesh.centroid(t_src);
    std::array<Vec3, 3> f0{};  // f_n at the source centroid
    std::array<Real, 3> dn{};
    std::array<Real, 3> dm{};
    for (std::size_t b = 0; b < ns; ++b) {
        f0[b] = space.value(ss.n[b], t_src, c_src);
        dn[b] = space.divergence(ss.n[b], t_src);
    }
    for (std::size_t a = 0; a < nt; ++a) {
        dm[a] = space.divergence(st.n[a], t_test);
    }
    const std::size_t nr = regs.size();
    std::vector<Complex> mjk(nr);
    for (std::size_t q = 0; q < nr; ++q) {
        mjk[q] = -kJ * regs[q].p.k;
    }
    std::vector<Blocks> out(nr);
    struct Sums {
        Complex g{0.0, 0.0};
        std::array<std::array<Complex, 3>, 3> v{};  // int G f_n
        std::array<std::array<Complex, 3>, 3> u{};  // int grad' G x f_n
    };
    std::vector<Sums> sums(nr);
    for (std::size_t i = 0; i < outer.r.size(); ++i) {
        const Vec3& r = outer.r[i];
        std::fill(sums.begin(), sums.end(), Sums{});
        polar_points(r, src, g, [&](Real x, Real y, Real z, Real w) {
            const Real dx = r(0) - x;  // r - r'
            const Real dy = r(1) - y;
            const Real dz = r(2) - z;
            const Real R = std::sqrt(dx * dx + dy * dy + dz * dz);
            std::array<std::array<Real, 3>, 3> f{};
            for (std::size_t b = 0; b < ns; ++b) {
                const Real hd = 0.5 * dn[b];
                f[b] = {f0[b](0) + hd * (x - c_src(0)), f0[b](1) + hd * (y - c_src(1)),
                        f0[b](2) + hd * (z - c_src(2))};
            }
            for (std::size_t q = 0; q < nr; ++q) {
                const Complex G = w * std::exp(mjk[q] * R) / (4.0 * kPi * R);
                const Complex cg = G * (1.0 - mjk[q] * R) / (R * R);  // grad' G = cg (r - r')
                const Complex gx = cg * dx;
                const Complex gy = cg * dy;
                const Complex gz = cg * dz;
                Sums& sq = sums[q];
                sq.g += G;
                for (std::size_t b = 0; b < ns; ++b) {
                    sq.v[b][0] += G * f[b][0];
                    sq.v[b][1] += G * f[b][1];
                    sq.v[b][2] += G * f[b][2];
                    sq.u[b][0] += gy * f[b][2] - gz * f[b][1];
                    sq.u[b][1] += gz * f[b][0] - gx * f[b][2];
                    sq.u[b][2] += gx * f[b][1] - gy * f[b][0];
                }
            }
        });
        for (std::size_t q = 0; q < nr; ++q) {
            const RegionParams& reg = regs[q].p;
            const Complex c_vec = kJ * reg.omega * reg.mu;
            const Complex c_sca = 1.0 / (kJ * reg.omega * reg.eps);
            const Sums& sq = sums[q];
            for (std::size_t a = 0; a < nt; ++a) {
                const Vec3 fa = space.value(st.n[a], t_test, r);
                for (std::size_t b = 0; b < ns; ++b) {
                    const auto ai = static_cast<Eigen::Index>(a);
                    const auto bi = static_cast<Eigen::Index>(b);
                    out[q].L(ai, bi) +=
                        outer.w[i] *
                        (c_vec * (fa(0) * sq.v[b][0] + fa(1) * sq.v[b][1] + fa(2) * sq.v[b][2]) +
                         c_sca * (dm[a] * dn[b]) * sq.g);
                    out[q].K(ai, bi) +=
                        outer.w[i] * (fa(0) * sq.u[b][0] + fa(1) * sq.u[b][1] + fa(2) * sq.u[b][2]);
                }
            }
        }
    }
    return out;
}

/// Symmetrised reference (B(t1, t2) + B(t2, t1)^T) / 2, matching element_blocks for touching
/// pairs (the outer rule of each ordering then coincides with the one element_blocks uses).
std::vector<Blocks> polar_reference(const RwgSpace& space, Index t1, Index t2,
                                    const std::vector<NamedRegion>& regs, int outer_degree, int n) {
    std::vector<Blocks> a = polar_reference_one(space, t1, t2, regs, outer_degree, n);
    if (t1 == t2) {
        for (Blocks& b : a) {
            b.L = (0.5 * (b.L + b.L.transpose())).eval();
        }
        return a;
    }
    const std::vector<Blocks> b = polar_reference_one(space, t2, t1, regs, outer_degree, n);
    for (std::size_t g = 0; g < a.size(); ++g) {
        a[g].L = 0.5 * (a[g].L + b[g].L.transpose());
        a[g].K = 0.5 * (a[g].K + b[g].K.transpose());
    }
    return a;
}

/// All ordered pairs of a mesh by proximity class.
std::vector<std::pair<Index, Index>> pairs_of_class(const TriangleMesh& m, Proximity cls) {
    std::vector<std::pair<Index, Index>> out;
    for (Index a = 0; a < m.num_triangles(); ++a) {
        for (Index b = 0; b < m.num_triangles(); ++b) {
            if (classify(m, a, b) == cls) {
                out.emplace_back(a, b);
            }
        }
    }
    return out;
}

/// `count` pairs spread evenly over the list (deterministic).
std::vector<std::pair<Index, Index>> spread(const std::vector<std::pair<Index, Index>>& all,
                                            std::size_t count) {
    std::vector<std::pair<Index, Index>> out;
    for (std::size_t i = 0; i < count && i < all.size(); ++i) {
        out.push_back(all[(i * all.size()) / count + (all.size() / count) / 2]);
    }
    return out;
}

Real rel_diff(const Block& a, const Block& ref) {
    return (a - ref).norm() / ref.norm();
}

Blocks blocks(const RwgSpace& space, Index t1, Index t2, const RegionParams& reg,
              const OperatorOptions& opt) {
    Blocks b;
    element_blocks(space, t1, t2, reg, opt, b.L, b.K);
    return b;
}

bool is_coplanar(const TriangleMesh& m, Index t1, Index t2) {
    return m.normal(t1).cross(m.normal(t2)).norm() < 1e-12;
}

const char* class_name(Proximity p) {
    switch (p) {
        case Proximity::identical:
            return "identical";
        case Proximity::shared_edge:
            return "shared_edge";
        case Proximity::shared_vertex:
            return "shared_vertex";
        case Proximity::near:
            return "near";
        case Proximity::far:
            return "far";
    }
    return "?";
}

/// Far pairs on the work package's icosphere(1e-6, 1): with quad_degree_far = 20 both sides
/// use the same points, so the check is independent of k h and covers all three materials.
void check_far(std::size_t n_pairs) {
    const TriangleMesh mesh = make_icosphere(1e-6, 1);
    const RwgSpace space(mesh);
    const std::vector<NamedRegion> regs = all_regions();
    OperatorOptions opt;
    opt.quad_degree_far = 20;
    Real worst = 0.0;
    for (const auto& [t1, t2] : spread(pairs_of_class(mesh, Proximity::far), n_pairs)) {
        REQUIRE((mesh.centroid(t1) - mesh.centroid(t2)).norm() >
                2.0 * std::max(longest_edge(mesh, t1), longest_edge(mesh, t2)));
        const std::vector<Blocks> ref = brute_force_dunavant(space, t1, t2, regs, 20, 1);
        for (std::size_t g = 0; g < regs.size(); ++g) {
            const Blocks b = blocks(space, t1, t2, regs[g].p, opt);
            const Real eL = rel_diff(b.L, ref[g].L);
            const Real eK = rel_diff(b.K, ref[g].K);
            INFO(regs[g].name << " pair " << t1 << "," << t2 << ": L " << eL << ", K " << eK);
            CHECK(eL < 1e-10);
            CHECK(eK < 1e-10);
            worst = std::max({worst, eL, eK});
        }
    }
    WARN("far pairs: worst relative difference to brute force " << worst);
}

/// Near pairs with quad_degree_near = 19 (highest positive-interior rule) against the
/// degree-20 brute force with both triangles split into n_sub^2 sub-triangles. Per-material
/// icospheres with |k| h = 0.78, plus the vacuum case on the work package's icosphere
/// (1e-6, k h = 7.8); `per_mesh` pairs per mesh.
void check_near(int n_sub, std::size_t per_mesh) {
    struct Case {
        TriangleMesh mesh;
        std::vector<NamedRegion> regs;
    };
    std::vector<Case> cases;
    for (const NamedRegion& reg : all_regions()) {
        cases.push_back({make_icosphere(scaled(kVacuumRadius, reg.p), 1), {reg}});
    }
    cases.push_back({make_icosphere(1e-6, 1), {all_regions()[0]}});
    OperatorOptions opt;
    opt.quad_degree_near = 19;
    Real worst = 0.0;
    Real worst_default = 0.0;
    for (const Case& c : cases) {
        const RwgSpace space(c.mesh);
        for (const auto& [t1, t2] : spread(pairs_of_class(c.mesh, Proximity::near), per_mesh)) {
            const std::vector<Blocks> ref = brute_force_dunavant(space, t1, t2, c.regs, 20, n_sub);
            const Blocks b = blocks(space, t1, t2, c.regs[0].p, opt);
            const Blocks b_def = blocks(space, t1, t2, c.regs[0].p, OperatorOptions{});
            const Real eL = rel_diff(b.L, ref[0].L);
            const Real eK = rel_diff(b.K, ref[0].K);
            INFO(c.regs[0].name << " h = " << longest_edge(c.mesh, t1) << " pair " << t1 << ","
                                << t2 << ": L " << eL << ", K " << eK);
            CHECK(eL < 1e-8);
            CHECK(eK < 1e-8);
            worst = std::max({worst, eL, eK});
            worst_default =
                std::max({worst_default, rel_diff(b_def.L, ref[0].L), rel_diff(b_def.K, ref[0].K)});
        }
    }
    WARN("near pairs: worst relative difference (degree 19) "
         << worst << ", default degree 8: " << worst_default);
}

/// Touching-pair geometries, sized for Si: |k| h = 0.78 in Si, 0.57 in Ag, 0.18 in vacuum.
struct TouchingGeometry {
    Real radius;
    TriangleMesh sphere;
    TriangleMesh flat;   ///< coplanar fan
    TriangleMesh tent;   ///< folded fan
    TriangleMesh hinge;  ///< 90-degree hinge
    TouchingGeometry()
        : radius(scaled(kVacuumRadius, all_regions()[1].p)),
          sphere(make_icosphere(radius, 1)),
          flat(fan_mesh(0.6 * radius, 0.0, 6)),
          tent(fan_mesh(0.6 * radius, 0.3 * radius, 6)),
          hinge(hinge_mesh(0.6 * radius)) {}
};

struct TouchingPair {
    const TriangleMesh* mesh;
    Index t1, t2;
};

TouchingPair sphere_pair(const TouchingGeometry& geo, Proximity cls) {
    const auto p = pairs_of_class(geo.sphere, cls);
    return {&geo.sphere, p[3].first, p[3].second};
}

/// Touching pairs against the polar-transform reference. For every quad_degree_sing = d the
/// reference uses the same outer rule (Dunavant degree d on each triangle, symmetrised like
/// element_blocks) and the polar transform for the inner integral of the full kernel, so the
/// difference measures the inner integration by singularity subtraction alone. (Against a
/// single degree-12 reference the difference at d < 12 is dominated by the outer-rule error
/// of the non-smooth outer integrand, which is not monotone in d for the fan pairs: e.g.
/// 4.9e-6 at d = 8 and 4.1e-5 at d = 10 for the coplanar shared-vertex pair.) Acceptance:
/// 1e-6 relative to the block norm at the last degree (12), and a decrease over the degrees
/// with at most one non-monotone step above 1e-7. n_ref Gauss-Legendre points per panel of
/// the reference (6: reference error ~2e-8, regular cases; 8: ~3e-10, slow cases).
void check_touching(const std::vector<TouchingPair>& pairs, const std::vector<NamedRegion>& regs,
                    const std::vector<int>& degrees, int n_ref) {
    constexpr Real kConverged = 1e-7;
    Real worst = 0.0;
    for (const TouchingPair& p : pairs) {
        const RwgSpace space(*p.mesh);
        const Proximity cls = classify(*p.mesh, p.t1, p.t2);
        REQUIRE((cls == Proximity::identical || cls == Proximity::shared_edge ||
                 cls == Proximity::shared_vertex));
        const bool coplanar = is_coplanar(*p.mesh, p.t1, p.t2);
        const Real h = longest_edge(*p.mesh, p.t1);
        std::vector<std::vector<Real>> dL(regs.size());
        std::vector<std::vector<Real>> dK(regs.size());
        for (const int deg : degrees) {
            const std::vector<Blocks> ref = polar_reference(space, p.t1, p.t2, regs, deg, n_ref);
            OperatorOptions opt;
            opt.quad_degree_sing = deg;
            for (std::size_t g = 0; g < regs.size(); ++g) {
                const Blocks b = blocks(space, p.t1, p.t2, regs[g].p, opt);
                dL[g].push_back(rel_diff(b.L, ref[g].L));
                if (coplanar) {
                    // Coplanar: K = 0 exactly; the reference is rounding noise on the natural
                    // scale h^2 of K (in-plane triple product).
                    CHECK(b.K == Block::Zero());
                    CHECK(ref[g].K.norm() <= 1e-12 * h * h);
                } else {
                    dK[g].push_back(rel_diff(b.K, ref[g].K));
                }
            }
        }
        for (std::size_t g = 0; g < regs.size(); ++g) {
            INFO(regs[g].name << " " << class_name(cls) << (coplanar ? " (coplanar)" : "")
                              << " triangles " << p.t1 << "," << p.t2);
            const auto describe = [&](const std::vector<Real>& d) {
                std::string s;
                for (std::size_t i = 0; i < d.size(); ++i) {
                    s += " " + std::to_string(degrees[i]) + ": " + std::to_string(d[i] * 1e9) +
                         "e-9";
                }
                return s;
            };
            INFO("L difference by degree" << describe(dL[g]));
            INFO("K difference by degree" << describe(dK[g]));
            CHECK(dL[g].back() < 1e-6);
            worst = std::max(worst, dL[g].back());
            if (!dK[g].empty()) {
                CHECK(dK[g].back() < 1e-6);
                worst = std::max(worst, dK[g].back());
            }
            // Steps among values below kConverged (a tenth of the acceptance level) are not
            // counted: there the Dunavant rules of neighbouring degrees (not nested) trade
            // places, e.g. 1.2e-8 at d = 9 after 9.5e-9 at d = 8 for a folded shared edge.
            for (const std::vector<Real>* d : {&dL[g], &dK[g]}) {
                int increases = 0;
                for (std::size_t i = 1; i < d->size(); ++i) {
                    increases += ((*d)[i] > (*d)[i - 1] && (*d)[i] > kConverged) ? 1 : 0;
                }
                CHECK(increases <= 1);
                if (d->size() > 1) {
                    CHECK(d->back() < 0.1 * d->front());
                }
            }
        }
    }
    WARN("touching pairs: worst relative difference to the polar reference at degree "
         << degrees.back() << ": " << worst);
}

/// Exchange symmetry for 20 random pairs of each class on the per-material icosphere.
void check_symmetry(const NamedRegion& reg, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Real worst_sep = 0.0;
    Real worst_touch = 0.0;
    const TriangleMesh mesh = make_icosphere(scaled(kVacuumRadius, reg.p), 1);
    const RwgSpace space(mesh);
    for (const Proximity cls : {Proximity::far, Proximity::near, Proximity::identical,
                                Proximity::shared_edge, Proximity::shared_vertex}) {
        const auto all = pairs_of_class(mesh, cls);
        std::uniform_int_distribution<std::size_t> pick(0, all.size() - 1);
        const bool touching = cls != Proximity::far && cls != Proximity::near;
        for (int i = 0; i < 20; ++i) {
            const auto [t1, t2] = all[pick(rng)];
            const Blocks b12 = blocks(space, t1, t2, reg.p, OperatorOptions{});
            const Blocks b21 = blocks(space, t2, t1, reg.p, OperatorOptions{});
            // Slots follow support() of each triangle, so the swapped block is the plain
            // transpose.
            const Real eL = rel_diff(b21.L.transpose(), b12.L);
            const Real eK = b12.K.norm() > 0.0 ? rel_diff(b21.K.transpose(), b12.K) : b21.K.norm();
            INFO(reg.name << " " << class_name(cls) << " " << t1 << "," << t2 << ": L " << eL
                          << ", K " << eK);
            CHECK(eL < (touching ? 1e-8 : 1e-12));
            CHECK(eK < (touching ? 1e-8 : 1e-12));
            Real& worst = touching ? worst_touch : worst_sep;
            worst = std::max({worst, eL, eK});
        }
    }
    WARN(reg.name << " symmetry: worst far/near " << worst_sep << ", touching " << worst_touch);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Regular cases (each < 1 s in the sanitizer build).
// ---------------------------------------------------------------------------------------------

TEST_CASE("element_blocks: far pairs match brute-force double Dunavant quadrature", "[kernels]") {
    check_far(2);
}

TEST_CASE("element_blocks: near pairs match brute-force double Dunavant quadrature", "[kernels]") {
    check_near(1, 1);
}

TEST_CASE("element_blocks: identical pair matches the polar reference (Si)", "[kernels]") {
    const TouchingGeometry geo;
    check_touching({sphere_pair(geo, Proximity::identical)}, {all_regions()[1]}, {4, 10, 12}, 6);
}

// The folded shared edge is the most expensive reference; its degrees {4, 10, 12} are split
// over two cases to keep each below 1 s in the sanitizer build (the slow suite checks the
// whole sequence 4 .. 12 in one run).
TEST_CASE("element_blocks: folded shared-edge pair converges to the polar reference (Si)",
          "[kernels]") {
    const TouchingGeometry geo;
    check_touching({{&geo.tent, 2, 3}}, {all_regions()[1]}, {4, 10}, 6);
}

TEST_CASE("element_blocks: folded shared-edge pair matches the polar reference at degree 12 (Si)",
          "[kernels]") {
    const TouchingGeometry geo;
    check_touching({{&geo.tent, 2, 3}}, {all_regions()[1]}, {12}, 6);
}

TEST_CASE("element_blocks: folded shared-vertex pair matches the polar reference (Si)",
          "[kernels]") {
    const TouchingGeometry geo;
    check_touching({{&geo.tent, 1, 4}}, {all_regions()[1]}, {4, 10, 12}, 6);
}

TEST_CASE("element_blocks: exchange symmetry (vacuum)", "[kernels]") {
    check_symmetry(all_regions()[0], 20261008);
}

TEST_CASE("element_blocks: exchange symmetry (Si)", "[kernels]") {
    check_symmetry(all_regions()[1], 20261009);
}

TEST_CASE("element_blocks: exchange symmetry (Ag)", "[kernels]") {
    check_symmetry(all_regions()[2], 20261010);
}

TEST_CASE("element_blocks: K of coplanar touching pairs is exactly zero", "[kernels]") {
    // Coplanar fan: identical, shared-edge and shared-vertex pairs have K == 0 exactly; the
    // same pairs of the folded fan do not.
    const NamedRegion reg = all_regions()[2];
    const Real a = 0.06e-6;
    const TriangleMesh flat = fan_mesh(a, 0.0, 6);
    const TriangleMesh tent = fan_mesh(a, 0.5 * a, 6);
    const RwgSpace flat_space(flat);
    const RwgSpace tent_space(tent);
    for (const auto& [t1, t2] : {std::pair<Index, Index>{0, 0}, std::pair<Index, Index>{0, 1},
                                 std::pair<Index, Index>{0, 3}, std::pair<Index, Index>{2, 4}}) {
        INFO("triangles " << t1 << "," << t2);
        const Blocks bf = blocks(flat_space, t1, t2, reg.p, OperatorOptions{});
        CHECK(bf.K == Block::Zero());
        CHECK(bf.L.norm() > 0.0);
        const Blocks bt = blocks(tent_space, t1, t2, reg.p, OperatorOptions{});
        if (t1 == t2) {
            CHECK(bt.K == Block::Zero());
        } else {
            CHECK(bt.K.norm() > 0.0);
        }
    }
}

TEST_CASE("element_blocks: intersecting triangles without shared vertex indices throw",
          "[kernels]") {
    // Two copies of the same two-triangle square with distinct vertex indices: the copies of a
    // triangle coincide geometrically, classify() sees a near pair (no shared index), the
    // quadrature points coincide (R = 0) and the non-finite block is rejected.
    Vertices v(8, 3);
    v << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;
    v *= 0.05e-6;
    Triangles f(4, 3);
    f << 0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7;
    const TriangleMesh mesh(v, f);
    const RwgSpace space(mesh);
    REQUIRE(classify(mesh, 0, 2) == Proximity::near);
    Block L;
    Block K;
    CHECK_THROWS_AS(element_blocks(space, 0, 2, all_regions()[0].p, OperatorOptions{}, L, K),
                    std::invalid_argument);
    CHECK_NOTHROW(element_blocks(space, 0, 1, all_regions()[0].p, OperatorOptions{}, L, K));
}

TEST_CASE("element_blocks: static limit of the vector potential matches static_integrals",
          "[kernels]") {
    // k = 1e-3 / m (k h ~ 1e-10), eps huge so that the scalar-potential term vanishes. The
    // reference uses the analytic int f_n / R = f_n(r) I_1R + (D_n / 2) I_rho_R (f_n affine)
    // plus the first dynamic term -jk int f_n (exact for the polynomial f_n); the neglected
    // terms are O((k h)^2) ~ 1e-20 relative.
    RegionParams reg{};
    reg.k = Complex(1e-3, 0.0);
    reg.omega = constants::c0 * reg.k.real();
    reg.mu = Complex(constants::mu0, 0.0);
    reg.eps = Complex(1e200, 0.0);
    reg.eta = Complex(constants::eta0, 0.0);
    const Complex c_vec = kJ * reg.omega * reg.mu / (4.0 * kPi);
    OperatorOptions opt;  // quad_degree_sing = 10
    const TriangleMesh sphere = make_icosphere(kVacuumRadius, 1);
    const Real a = 0.06e-6;
    const TriangleMesh tent = fan_mesh(a, 0.5 * a, 6);
    const TriangleMesh flat = fan_mesh(a, 0.0, 6);
    const auto edge_pairs = pairs_of_class(sphere, Proximity::shared_edge);
    struct Pair {
        const TriangleMesh* mesh;
        Index t1, t2;
    };
    for (const Pair& p : {Pair{&sphere, edge_pairs[0].first, edge_pairs[0].second},
                          Pair{&sphere, edge_pairs[17].first, edge_pairs[17].second},
                          Pair{&tent, 0, 1}, Pair{&flat, 4, 5}}) {
        const RwgSpace space(*p.mesh);
        REQUIRE(classify(*p.mesh, p.t1, p.t2) == Proximity::shared_edge);
        // One ordering: outer rule (degree quad_degree_sing) on t_test, analytic inner.
        const auto one = [&](Index tt, Index ts) {
            const RwgSpace::Support st = space.support(tt);
            const RwgSpace::Support ss = space.support(ts);
            const auto v = corners(*p.mesh, ts);
            const QuadSet outer = dunavant_points(*p.mesh, tt, opt.quad_degree_sing, 1);
            const Vec3 cs = p.mesh->centroid(ts);
            Block A = Block::Zero();
            for (std::size_t i = 0; i < outer.r.size(); ++i) {
                const Vec3& r = outer.r[i];
                const kernels::StaticIntegrals s = kernels::static_integrals(r, v[0], v[1], v[2]);
                for (int m = 0; m < st.count; ++m) {
                    const Vec3 fm = space.value(st.n[m], tt, r);
                    for (int n = 0; n < ss.count; ++n) {
                        const Vec3 fn_r = space.value(ss.n[n], ts, r);  // affine extension
                        const Real dn = space.divergence(ss.n[n], ts);
                        const Vec3 int_f_over_r = fn_r * s.I_1R + 0.5 * dn * s.I_rho_R;
                        const Vec3 int_f = space.value(ss.n[n], ts, cs) * p.mesh->area(ts);
                        A(m, n) += outer.w[i] * (fm.dot(int_f_over_r) - kJ * reg.k * fm.dot(int_f));
                    }
                }
            }
            return Block(c_vec * A);
        };
        const Block ref = 0.5 * (one(p.t1, p.t2) + one(p.t2, p.t1).transpose());
        const Blocks b = blocks(space, p.t1, p.t2, reg, opt);
        INFO("pair " << p.t1 << "," << p.t2 << ": " << rel_diff(b.L, ref));
        CHECK(rel_diff(b.L, ref) < 1e-10);
    }
}

TEST_CASE("jump_block: antisymmetric Gram block of f_m . (n x f_n), exact", "[kernels]") {
    const TriangleMesh mesh = make_icosphere(kVacuumRadius, 1, Vec3(0.3e-7, -0.2e-7, 0.1e-7));
    const TriangleMesh big = make_icosphere(3.0 * kVacuumRadius, 1, Vec3(0.9e-7, -0.6e-7, 0.3e-7));
    const TriangleMesh fan = fan_mesh(0.06e-6, 0.03e-6, 6);
    const RwgSpace space(mesh);
    const RwgSpace space_big(big);
    const RwgSpace space_fan(fan);
    Real worst = 0.0;
    const auto brute = [](const RwgSpace& sp, Index t) {
        const RwgSpace::Support s = sp.support(t);
        const Vec3 n = sp.mesh().normal(t);
        const QuadSet q = dunavant_points(sp.mesh(), t, 6, 1);
        Block I = Block::Zero();
        for (std::size_t i = 0; i < q.r.size(); ++i) {
            for (int a = 0; a < s.count; ++a) {
                for (int b = 0; b < s.count; ++b) {
                    I(a, b) +=
                        q.w[i] *
                        sp.value(s.n[a], t, q.r[i]).dot(n.cross(sp.value(s.n[b], t, q.r[i])));
                }
            }
        }
        return I;
    };
    for (const RwgSpace* sp : {&space, &space_fan}) {
        for (Index t = 0; t < sp->mesh().num_triangles(); ++t) {
            Block I;
            kernels::jump_block(*sp, t, I);
            const Real scale = I.norm();
            REQUIRE(scale > 0.0);
            CHECK((I + I.transpose()).norm() <= 1e-15 * scale);
            CHECK(I.diagonal().norm() <= 1e-15 * scale);
            CHECK(I.imag().norm() == 0.0);
            const Real err = rel_diff(I, brute(*sp, t));
            CHECK(err < 1e-14);
            worst = std::max(worst, err);
            const int count = sp->support(t).count;
            for (int a = count; a < 3; ++a) {
                CHECK(I.row(a).norm() == 0.0);
                CHECK(I.col(a).norm() == 0.0);
            }
        }
    }
    // Scaling: f is scale-invariant, so I scales with the area (factor 9 for 3x the radius).
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        Block I;
        Block I_big;
        kernels::jump_block(space, t, I);
        kernels::jump_block(space_big, t, I_big);
        CHECK(rel_diff(I_big, 9.0 * I) < 1e-13);
    }
    WARN("jump_block: worst relative difference to degree-6 quadrature " << worst);
    Block I;
    CHECK_THROWS_AS(kernels::jump_block(space, -1, I), std::out_of_range);
    CHECK_THROWS_AS(kernels::jump_block(space, mesh.num_triangles(), I), std::out_of_range);
}

TEST_CASE("OperatorOptions validation", "[kernels]") {
    const TriangleMesh mesh = make_icosphere(kVacuumRadius, 1);
    const RwgSpace space(mesh);
    const RegionParams reg = all_regions()[0].p;
    Block L;
    Block K;
    CHECK_NOTHROW(kernels::validate(OperatorOptions{}));
    CHECK_NOTHROW(element_blocks(space, 0, 1, reg, OperatorOptions{}, L, K));
    for (const int deg : {11, 15, 3, 7, 16, 18, 20}) {
        OperatorOptions near;
        near.quad_degree_near = deg;
        OperatorOptions sing;
        sing.quad_degree_sing = deg;
        OperatorOptions far;
        far.quad_degree_far = deg;
        INFO("degree " << deg);
        CHECK_THROWS_AS(kernels::validate(near), std::invalid_argument);
        CHECK_THROWS_AS(kernels::validate(sing), std::invalid_argument);
        CHECK_NOTHROW(kernels::validate(far));
        // element_blocks validates on every call, whatever the class of the pair.
        CHECK_THROWS_AS(element_blocks(space, 0, 0, reg, near, L, K), std::invalid_argument);
        CHECK_THROWS_AS(element_blocks(space, 0, 40, reg, sing, L, K), std::invalid_argument);
    }
    for (const int deg : {1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19}) {
        OperatorOptions o;
        o.quad_degree_near = deg;
        o.quad_degree_sing = deg;
        o.quad_degree_far = deg;
        CHECK_NOTHROW(kernels::validate(o));
    }
    for (const int deg : {0, -1, 21, 100}) {
        OperatorOptions far;
        far.quad_degree_far = deg;
        OperatorOptions near;
        near.quad_degree_near = deg;
        OperatorOptions sing;
        sing.quad_degree_sing = deg;
        CHECK_THROWS_AS(kernels::validate(far), std::invalid_argument);
        CHECK_THROWS_AS(kernels::validate(near), std::invalid_argument);
        CHECK_THROWS_AS(kernels::validate(sing), std::invalid_argument);
    }
    OperatorOptions bad_factor;
    bad_factor.near_distance_factor = -1.0;
    CHECK_THROWS_AS(kernels::validate(bad_factor), std::invalid_argument);
    bad_factor.near_distance_factor = std::nan("");
    CHECK_THROWS_AS(kernels::validate(bad_factor), std::invalid_argument);
    // Invalid triangle indices.
    CHECK_THROWS_AS(element_blocks(space, -1, 0, reg, OperatorOptions{}, L, K), std::out_of_range);
    CHECK_THROWS_AS(element_blocks(space, 0, mesh.num_triangles(), reg, OperatorOptions{}, L, K),
                    std::out_of_range);
}

TEST_CASE("element_blocks: unused slots are zero and slots follow support order", "[kernels]") {
    // Fan: two RWG functions per triangle (slot 2 unused).
    const NamedRegion reg = all_regions()[0];
    const Real a = 0.06e-6;
    const TriangleMesh tent = fan_mesh(a, 0.5 * a, 6);
    const RwgSpace space(tent);
    for (const auto& [t1, t2] : {std::pair<Index, Index>{0, 0}, std::pair<Index, Index>{0, 1},
                                 std::pair<Index, Index>{0, 3}}) {
        REQUIRE(space.support(t1).count == 2);
        const Blocks b = blocks(space, t1, t2, reg.p, OperatorOptions{});
        for (int k = 0; k < 3; ++k) {
            CHECK(b.L(2, k) == Complex(0.0, 0.0));
            CHECK(b.L(k, 2) == Complex(0.0, 0.0));
            CHECK(b.K(2, k) == Complex(0.0, 0.0));
            CHECK(b.K(k, 2) == Complex(0.0, 0.0));
        }
        CHECK(b.L.topLeftCorner<2, 2>().cwiseAbs().minCoeff() > 0.0);
    }
}

TEST_CASE("element_blocks: 10 000 far-pair calls are fast", "[kernels]") {
    // About 0.65 s in the sanitizer build and 0.01 s in release; the bound has a 10x margin
    // over the 2 s target so that a loaded machine does not fail the test.
    const TriangleMesh mesh = make_icosphere(kVacuumRadius, 1);
    const RwgSpace space(mesh);
    const RegionParams reg = all_regions()[2].p;  // Ag, complex k
    const OperatorOptions opt;
    Block L;
    Block K;
    const auto far = pairs_of_class(mesh, Proximity::far);
    Complex sink{0.0, 0.0};
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < 10000; ++i) {
        const auto& [t1, t2] = far[(i * 7919) % far.size()];
        element_blocks(space, t1, t2, reg, opt, L, K);
        sink += L(0, 0) + K(1, 1);
    }
    const Real seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    WARN("10000 far-pair element_blocks calls: " << seconds << " s (target < 2 s)");
    CHECK(std::isfinite(sink.real()));
    CHECK(seconds < 20.0);
}

// ---------------------------------------------------------------------------------------------
// Hidden slow cases ([.slow], not registered with ctest; run with
// `specklebem_unit_tests "[slow]"`, about 40 s in the sanitizer build, < 1 s in release).
// ---------------------------------------------------------------------------------------------

TEST_CASE("element_blocks (full): far pairs, six pairs", "[kernels][.slow]") {
    check_far(6);
}

TEST_CASE("element_blocks (full): near pairs vs composite brute force", "[kernels][.slow]") {
    check_near(2, 2);
}

TEST_CASE("element_blocks (full): touching pairs, all materials and degrees", "[kernels][.slow]") {
    const TouchingGeometry geo;
    check_touching({sphere_pair(geo, Proximity::identical),
                    sphere_pair(geo, Proximity::shared_edge),
                    sphere_pair(geo, Proximity::shared_vertex),
                    {&geo.flat, 0, 1},
                    {&geo.flat, 0, 3},
                    {&geo.tent, 2, 3},
                    {&geo.tent, 1, 4},
                    {&geo.hinge, 0, 1}},
                   all_regions(), {4, 5, 6, 8, 9, 10, 12}, 8);
}

TEST_CASE("polar reference is converged", "[kernels][.slow]") {
    // Self-convergence of the touching-pair reference (n = 8, as used above, vs 12 points per
    // panel) on a folded shared-edge and shared-vertex pair, all three materials: the
    // reference error is far below the 1e-6 acceptance level.
    const std::vector<NamedRegion> regs = all_regions();
    const TouchingGeometry geo;
    const RwgSpace space(geo.tent);
    for (const auto& [t1, t2] : {std::pair<Index, Index>{2, 3}, std::pair<Index, Index>{1, 4}}) {
        const std::vector<Blocks> coarse = polar_reference_one(space, t1, t2, regs, 12, 8);
        const std::vector<Blocks> fine = polar_reference_one(space, t1, t2, regs, 12, 12);
        for (std::size_t g = 0; g < regs.size(); ++g) {
            INFO(regs[g].name << " pair " << t1 << "," << t2 << ": L "
                              << rel_diff(coarse[g].L, fine[g].L) << ", K "
                              << rel_diff(coarse[g].K, fine[g].K));
            CHECK(rel_diff(coarse[g].L, fine[g].L) < 1e-8);
            CHECK(rel_diff(coarse[g].K, fine[g].K) < 1e-8);
        }
    }
}
