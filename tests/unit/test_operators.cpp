/// Tests of kernels::element_blocks, kernels::jump_block and kernels::validate (WP7, WP7b).
///
/// References are written directly from the Galerkin definitions with kernels::green,
/// kernels::grad_green, RwgSpace::value and RwgSpace::divergence:
///  * far / near pairs: (composite) Dunavant double quadrature;
///  * touching pairs, WP7 path (outer_grading_levels = 0): outer Dunavant rule, inner integral of
///    the full kernel by a polar (Duffy) transform about the projection of the outer point, with
///    graded Gauss-Legendre panels (independent of the singularity subtraction and of
///    static_integrals); this checks the inner integration with the SAME outer rule;
///  * touching pairs, WP7b graded path (default): a relative-coordinate reference of
///    Sauter-Schwab type (relative_reference): the full kernel over T x T' in coordinates in
///    which the singular set (diagonal, shared edge, shared vertex) is a corner of the
///    parameter domain and the Jacobian removes the singularity, tensor Gauss-Legendre rules.
///    Independent of the singularity subtraction, of static_integrals and of the graded outer
///    rule; exponentially convergent (n = 16: ~1e-11, n = 20: ~1e-13 of the block norm, slow
///    case "relative-coordinate reference is converged"). It replaces the hp-graded polar
///    reference suggested for WP7b: geometric Gauss-Legendre grading towards the shared edge
///    needs ~30 levels for the log-singular K integrand (the outer integrand of folded pairs)
///    to reach 1e-10, and with the polar inner rule that is ~10^8 kernel evaluations per pair
///    (measured: 90 s in release for 3e-6 with 12 levels).
///
/// Test sets: the regular cases run with ctest and take < 1 s each in the sanitizer build
/// (docs/05): far pairs (2 pairs, 3 materials), near pairs (degree 19 vs plain degree-20
/// brute force), WP7-path touching pairs (identical, folded shared edge, folded shared vertex;
/// Si; quad_degree_sing in {4, 10, 12}), graded touching pairs against the relative-coordinate
/// reference (identical, folded shared edges at 90 and 150 degrees, shared vertices; vacuum,
/// Si, Ag; both orderings; raw asymmetry), convergence in outer_grading_levels, the averaging
/// threshold and determinism, the degree selection (D/h in {2, 3, 5, 10}, |k| h in {0.1, 0.5,
/// 1, 2}, one material per case) and its bounds, symmetry per material, static limit, jump
/// block, options, slots, coplanar K, non-finite guard, WP7 golden blocks (hard-coded blocks of
/// four pairs with the WP7 options, 1e-12) and a 2 000-call far-pair timing smoke check. The
/// full sweeps are hidden test cases tagged [.slow] (not registered by catch_discover_tests):
/// far (6 pairs), near against the composite brute force, WP7 touching (8 pairs, all
/// materials, degrees {4, 5, 6, 8, 9, 10, 12}), polar and relative-coordinate reference
/// convergence, the graded touching sweep (13 pairs, errors by level), sharp folds (60 and 30
/// degrees, asymmetric and skewed hinges, by level), the degree-selection sweep (D/h 1.1 to 15,
/// |k| h 0.1 to 3, all materials), the class-boundary pairs of the n = 4 Mie mesh, 10 000
/// far-pair calls and the cost comparison with the WP7 options. Run them with
/// `build/<preset>/tests/specklebem_unit_tests "[slow]"`. The zero-allocation check lives in
/// its own executable (test_operators_alloc.cpp) because it replaces the global operator new.
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
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
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
/// Default target_accuracy of the near/far degree selection (1e-5).
constexpr Real kDefaultTarget = OperatorOptions{}.target_accuracy;

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

/// Far pairs on the work package's icosphere(1e-6, 1): with quad_degree_far = quad_degree_near =
/// 19 (the selection then has a single choice) both sides use the same points, so the check is
/// independent of k h and covers all three materials.
void check_far(std::size_t n_pairs) {
    const TriangleMesh mesh = make_icosphere(1e-6, 1);
    const RwgSpace space(mesh);
    const std::vector<NamedRegion> regs = all_regions();
    OperatorOptions opt;
    opt.quad_degree_far = 19;
    opt.quad_degree_near = 19;
    Real worst = 0.0;
    for (const auto& [t1, t2] : spread(pairs_of_class(mesh, Proximity::far), n_pairs)) {
        REQUIRE((mesh.centroid(t1) - mesh.centroid(t2)).norm() >
                2.0 * std::max(longest_edge(mesh, t1), longest_edge(mesh, t2)));
        const std::vector<Blocks> ref = brute_force_dunavant(space, t1, t2, regs, 19, 1);
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

/// Near pairs with the fixed degree quad_degree_near = 19 (highest positive-interior rule,
/// target_accuracy = 0) against the degree-20 brute force with both triangles split into
/// n_sub^2 sub-triangles (the default options, with degree selection, are reported). Per-material
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
    opt.target_accuracy = 0.0;
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
         << worst << ", default options: " << worst_default);
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

/// WP7 path (outer_grading_levels = 0): touching pairs against the polar-transform reference.
/// For every quad_degree_sing = d the
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
            opt.outer_grading_levels = 0;  // WP7 path: the reference uses the same outer rule
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

/// Exchange symmetry on the per-material icosphere (|k| h = 0.78, default options) for 20 random
/// pairs of each separated class (far, near) and a sample of 6 pairs of each touching class
/// (identical, shared edge, shared vertex; the graded touching rule dominates the sanitizer run
/// time): far and near pairs use the same rule in both orderings (rounding only); identical L
/// blocks are symmetrised, shared-edge and shared-vertex blocks are not averaged below |k| h = 1
/// (raw asymmetry < 1e-9, WP7b).
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
        // 20 random separated pairs and 6 touching pairs (the graded touching rule is the
        // expensive part in the sanitizer build) per class.
        for (int i = 0; i < (touching ? 6 : 20); ++i) {
            const auto [t1, t2] = all[pick(rng)];
            const Blocks b12 = blocks(space, t1, t2, reg.p, OperatorOptions{});
            const Blocks b21 = blocks(space, t2, t1, reg.p, OperatorOptions{});
            // Slots follow support() of each triangle, so the swapped block is the plain
            // transpose.
            const Real eL = rel_diff(b21.L.transpose(), b12.L);
            const Real eK = b12.K.norm() > 0.0 ? rel_diff(b21.K.transpose(), b12.K) : b21.K.norm();
            INFO(reg.name << " " << class_name(cls) << " " << t1 << "," << t2 << ": L " << eL
                          << ", K " << eK);
            CHECK(eL < (touching ? 1e-9 : 1e-12));
            CHECK(eK < (touching ? 1e-9 : 1e-12));
            Real& worst = touching ? worst_touch : worst_sep;
            worst = std::max({worst, eL, eK});
        }
    }
    WARN(reg.name << " symmetry: worst far/near " << worst_sep << ", touching " << worst_touch);
}

// ---------------------------------------------------------------------------------------------
// WP7b: relative-coordinate reference for touching pairs, degree-selection helpers.
// ---------------------------------------------------------------------------------------------

/// WP7 scheme: one Dunavant outer rule, fixed near/far degrees (8 near, 3 far).
OperatorOptions wp7_options() {
    OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_near = 8;
    return opt;
}

/// Two triangles hinged along their shared edge (0, 1) at dihedral angle `degrees` (180 = flat),
/// one RWG function.
TriangleMesh folded_hinge(Real a, Real degrees) {
    const Real th = degrees * kPi / 180.0;
    Vertices v(4, 3);
    v << 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.4 * a, 0.8 * a, 0.0, 0.55 * a, 0.85 * a * std::cos(th),
        0.85 * a * std::sin(th);
    Triangles f(2, 3);
    f << 0, 1, 2, 1, 0, 3;
    return TriangleMesh(v, f);
}

/// Hinge of two triangles of different shape along the shared edge (0, 1) at dihedral angle
/// `degrees`: apex of the first triangle at (x2, y2) a in its plane, of the second at x3 a along
/// the edge and r3 a from it (folded_hinge: 0.4, 0.8, 0.55, 0.85).
TriangleMesh asymmetric_hinge(Real a, Real degrees, Real x2, Real y2, Real x3, Real r3) {
    const Real th = degrees * kPi / 180.0;
    Vertices v(4, 3);
    v << 0.0, 0.0, 0.0, a, 0.0, 0.0, x2 * a, y2 * a, 0.0, x3 * a, r3 * a * std::cos(th),
        r3 * a * std::sin(th);
    Triangles f(2, 3);
    f << 0, 1, 2, 1, 0, 3;
    return TriangleMesh(v, f);
}

/// RWG data of one triangle for the affine evaluation f(x) = f(x0) + (D / 2)(x - x0).
struct AffineRwg {
    std::size_t count = 0;
    std::array<Vec3, 3> f0{};
    std::array<Real, 3> div{};
    Vec3 x0 = Vec3::Zero();
};

AffineRwg affine_rwg(const RwgSpace& space, Index t) {
    const RwgSpace::Support s = space.support(t);
    AffineRwg out;
    out.count = static_cast<std::size_t>(s.count);
    out.x0 = space.mesh().centroid(t);
    for (std::size_t a = 0; a < out.count; ++a) {
        out.f0[a] = space.value(s.n[a], t, out.x0);
        out.div[a] = space.divergence(s.n[a], t);
    }
    return out;
}

/// Accumulates w times the Galerkin integrands of L and K at the point pair (x, y), for several
/// regions at once: L: j w mu f_m . f_n G + (div f_m)(div f_n) G / (j w eps); K: f_m . (grad' G
/// x f_n) = grad' G . (f_n x f_m), grad' G = (1 + jkR) G (x - y) / R^2. K is skipped for
/// identical triangles (coplanar: the triple product vanishes pointwise).
struct KernelSum {
    const AffineRwg* test;
    const AffineRwg* src;
    const std::vector<NamedRegion>* regs;
    bool with_k;
    std::vector<Blocks> out;

    void add(const Vec3& x, const Vec3& y, Real w) {
        // Scalar arithmetic (no Eigen temporaries) so that the sanitizer build stays fast.
        std::array<std::array<Real, 3>, 3> fm{};
        std::array<std::array<Real, 3>, 3> fn{};
        for (std::size_t a = 0; a < test->count; ++a) {
            for (Eigen::Index c = 0; c < 3; ++c) {
                fm[a][static_cast<std::size_t>(c)] =
                    test->f0[a](c) + 0.5 * test->div[a] * (x(c) - test->x0(c));
            }
        }
        for (std::size_t b = 0; b < src->count; ++b) {
            for (Eigen::Index c = 0; c < 3; ++c) {
                fn[b][static_cast<std::size_t>(c)] =
                    src->f0[b](c) + 0.5 * src->div[b] * (y(c) - src->x0(c));
            }
        }
        const std::array<Real, 3> d = {x(0) - y(0), x(1) - y(1), x(2) - y(2)};
        const Real R = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        // Geometric factors f_m . f_n and d . (f_n x f_m), independent of the region.
        std::array<std::array<Real, 3>, 3> dot{};
        std::array<std::array<Real, 3>, 3> triple{};
        for (std::size_t a = 0; a < test->count; ++a) {
            const auto& m = fm[a];
            for (std::size_t b = 0; b < src->count; ++b) {
                const auto& n = fn[b];
                dot[a][b] = m[0] * n[0] + m[1] * n[1] + m[2] * n[2];
                triple[a][b] = d[0] * (n[1] * m[2] - n[2] * m[1]) +
                               d[1] * (n[2] * m[0] - n[0] * m[2]) +
                               d[2] * (n[0] * m[1] - n[1] * m[0]);
            }
        }
        for (std::size_t g = 0; g < regs->size(); ++g) {
            const RegionParams& reg = (*regs)[g].p;
            const Complex G = w * std::exp(-kJ * reg.k * R) / (4.0 * kPi * R);
            const Complex c_vec = kJ * reg.omega * reg.mu;
            const Complex c_sca = 1.0 / (kJ * reg.omega * reg.eps);
            const Complex cg = G * (1.0 + kJ * reg.k * R) / (R * R);
            for (std::size_t a = 0; a < test->count; ++a) {
                const auto ai = static_cast<Eigen::Index>(a);
                for (std::size_t b = 0; b < src->count; ++b) {
                    const auto bi = static_cast<Eigen::Index>(b);
                    out[g].L(ai, bi) +=
                        c_vec * G * dot[a][b] + c_sca * (test->div[a] * src->div[b]) * G;
                    if (with_k) {
                        out[g].K(ai, bi) += cg * triple[a][b];
                    }
                }
            }
        }
    }
};

/// Gauss-Legendre rule on [0, 1].
std::vector<std::pair<Real, Real>> gauss01(int n) {
    const kernels::LineRule g = kernels::gauss_legendre(n);
    std::vector<std::pair<Real, Real>> r;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        r.emplace_back(0.5 * (g.nodes[i] + 1.0), 0.5 * g.weights[i]);
    }
    return r;
}

/// Identical triangles, x = v0 + s e1 + t e2, y = x + z1 e1 + z2 e2 ((s, t) and (s, t) + z in the
/// unit triangle T^). z ranges over the hexagon T^ - T^, split into the six cones over its
/// edges (the planes z1 = 0, z2 = 0, z1 + z2 = 0), z = rho (P_i + u (P_{i+1} - P_i)), dz = rho
/// drho du (|det(P_i, P_{i+1})| = 1). For fixed z the x-domain T^ n (T^ - z) is the triangle
/// {s >= a, t >= b, s + t <= c}, a = max(0, -z1), b = max(0, -z2), c = 1 - max(0, z1 + z2),
/// on which the integrand is a quadratic polynomial times a constant kernel (x - y depends on z
/// only): degree-4 Dunavant rule. rho G is analytic in rho: Gauss-Legendre in rho and u.
template <typename Sink>
void reference_identical(const std::array<Vec3, 3>& v, int n, Sink& sum) {
    const Vec3 e1 = v[1] - v[0];
    const Vec3 e2 = v[2] - v[0];
    const Real jac = e1.cross(e2).norm();
    const std::array<std::array<Real, 2>, 6> hex = {
        {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {1.0, -1.0}}};
    const auto rule = gauss01(n);
    const kernels::TriangleRule& tri = kernels::triangle_rule(4);
    for (std::size_t i = 0; i < 6; ++i) {
        const auto& p = hex[i];
        const auto& q = hex[(i + 1) % 6];
        for (const auto& [rho, wr] : rule) {
            for (const auto& [u, wu] : rule) {
                const Real z1 = rho * (p[0] + u * (q[0] - p[0]));
                const Real z2 = rho * (p[1] + u * (q[1] - p[1]));
                const Real a = std::max(0.0, -z1);
                const Real b = std::max(0.0, -z2);
                const Real c = 1.0 - std::max(0.0, z1 + z2);
                const Real size = c - a - b;
                const Real w0 = jac * jac * rho * wr * wu * 0.5 * size * size;
                for (std::size_t k = 0; k < tri.weights.size(); ++k) {
                    const Vec3& l = tri.barycentric[k];
                    const Real s = l(0) * a + l(1) * (c - b) + l(2) * a;
                    const Real t = l(0) * b + l(1) * b + l(2) * (c - a);
                    const Vec3 x = v[0] + s * e1 + t * e2;
                    sum.add(x, x + z1 * e1 + z2 * e2, w0 * tri.weights[k]);
                }
            }
        }
    }
}

using P3d = std::array<Real, 3>;

/// Splits a convex polygon by the plane n . p = 0 (through the origin) into its two sides.
std::vector<std::vector<P3d>> split_polygon(const std::vector<P3d>& poly, const P3d& n) {
    const auto dot = [](const P3d& a, const P3d& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    std::vector<P3d> pos;
    std::vector<P3d> neg;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const P3d& a = poly[i];
        const P3d& b = poly[(i + 1) % poly.size()];
        const Real da = dot(n, a);
        const Real db = dot(n, b);
        if (da >= 0.0) {
            pos.push_back(a);
        }
        if (da <= 0.0) {
            neg.push_back(a);
        }
        if ((da > 0.0 && db < 0.0) || (da < 0.0 && db > 0.0)) {
            const Real s = da / (da - db);
            const P3d c = {a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]),
                           a[2] + s * (b[2] - a[2])};
            pos.push_back(c);
            neg.push_back(c);
        }
    }
    std::vector<std::vector<P3d>> out;
    for (std::vector<P3d>* side : {&pos, &neg}) {
        if (side->size() >= 3) {
            out.push_back(*side);
        }
    }
    return out;
}

/// Shared edge AB, x = A + s e + t a (T), y = A + s' e + t' b (T'), e = B - A, a = C - A,
/// b = C' - A. With z = s' - s the kernel depends on (z, t, t') only, and for fixed (z, t, t')
/// the integrand is quadratic in s on [max(0, -z), min(1 - t, 1 - t' - z)] (2-point
/// Gauss-Legendre, exact). (z, t, t') ranges over P = {0 <= t, t' <= 1, t - 1 <= z <= 1 - t'},
/// the union of the cones from the origin (the singular point) over the four facets of P not
/// through it (t = 1, t' = 1, z = 1 - t', z = t - 1), each split by the planes z = 0 and
/// t - t' - z = 0 where the s range changes form, fan-triangulated, and every tetrahedron
/// (0, q0, q1, q2) mapped by p = rho [q0 + u (q1 - q0) + u v (q2 - q1)], dp = |det(q0, q1, q2)|
/// rho^2 u: rho^2 grad' G is analytic in rho.
template <typename Sink>
void reference_shared_edge(const Vec3& A, const Vec3& B, const Vec3& C, const Vec3& Cp, int n,
                           Sink& sum) {
    const Vec3 e = B - A;
    const Vec3 a = C - A;
    const Vec3 b = Cp - A;
    const Real jac = e.cross(a).norm() * e.cross(b).norm();
    std::vector<std::vector<P3d>> facets = {{{0, 1, 0}, {1, 1, 0}, {0, 1, 1}},
                                            {{-1, 0, 1}, {0, 0, 1}, {0, 1, 1}},
                                            {{1, 0, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}},
                                            {{-1, 0, 0}, {0, 1, 0}, {0, 1, 1}, {-1, 0, 1}}};
    for (const P3d& plane : {P3d{1, 0, 0}, P3d{-1, 1, -1}}) {
        std::vector<std::vector<P3d>> next;
        for (const auto& f : facets) {
            for (auto& piece : split_polygon(f, plane)) {
                next.push_back(std::move(piece));
            }
        }
        facets = std::move(next);
    }
    const auto rule = gauss01(n);
    const auto rule_s = gauss01(2);
    for (const auto& f : facets) {
        for (std::size_t i = 1; i + 1 < f.size(); ++i) {
            const P3d& q0 = f[0];
            const P3d& q1 = f[i];
            const P3d& q2 = f[i + 1];
            const Real det = std::abs(q0[0] * (q1[1] * q2[2] - q1[2] * q2[1]) -
                                      q0[1] * (q1[0] * q2[2] - q1[2] * q2[0]) +
                                      q0[2] * (q1[0] * q2[1] - q1[1] * q2[0]));
            for (const auto& [rho, wr] : rule) {
                for (const auto& [u, wu] : rule) {
                    for (const auto& [v, wv] : rule) {
                        P3d p{};
                        for (std::size_t c = 0; c < 3; ++c) {
                            p[c] = rho * (q0[c] + u * (q1[c] - q0[c]) + u * v * (q2[c] - q1[c]));
                        }
                        const Real z = p[0];
                        const Real t = p[1];
                        const Real tp = p[2];
                        const Real lo = std::max(0.0, -z);
                        const Real hi = std::min(1.0 - t, 1.0 - tp - z);
                        if (hi <= lo) {
                            continue;
                        }
                        const Real w0 = jac * det * rho * rho * u * wr * wu * wv * (hi - lo);
                        for (const auto& [ss, ws] : rule_s) {
                            const Real s = lo + (hi - lo) * ss;
                            sum.add(A + s * e + t * a, A + (s + z) * e + tp * b, w0 * ws);
                        }
                    }
                }
            }
        }
    }
}

/// Shared vertex A, x = A + s a1 + t a2, y = A + s' b1 + t' b2 ((s, t), (s', t') in T^). The
/// product T^ x T^ is the union of the cones from its vertex 0 (the singular point) over the
/// facets {s + t = 1} x T^ and T^ x {s' + t' = 1}: (x^, y^) = lambda (edge point, triangle
/// point), Jacobian lambda^3 (lambda^3 grad' G analytic); the triangle point by the collapsed
/// map (mu (1 - nu), mu nu), Jacobian mu.
template <typename Sink>
void reference_shared_vertex(const Vec3& A, const Vec3& a1, const Vec3& a2, const Vec3& b1,
                             const Vec3& b2, int n, Sink& sum) {
    const Vec3 da1 = a1 - A;
    const Vec3 da2 = a2 - A;
    const Vec3 db1 = b1 - A;
    const Vec3 db2 = b2 - A;
    const Real jac = da1.cross(da2).norm() * db1.cross(db2).norm();
    const auto rule = gauss01(n);
    for (const bool edge_on_test : {true, false}) {
        for (const auto& [lam, wl] : rule) {
            for (const auto& [sg, wsg] : rule) {
                for (const auto& [mu, wm] : rule) {
                    for (const auto& [nu, wn] : rule) {
                        const Real w = jac * lam * lam * lam * mu * wl * wsg * wm * wn;
                        const Real t0 = mu * (1.0 - nu);
                        const Real t1 = mu * nu;
                        if (edge_on_test) {
                            sum.add(A + lam * (sg * da1 + (1.0 - sg) * da2),
                                    A + lam * (t0 * db1 + t1 * db2), w);
                        } else {
                            sum.add(A + lam * (t0 * da1 + t1 * da2),
                                    A + lam * (sg * db1 + (1.0 - sg) * db2), w);
                        }
                    }
                }
            }
        }
    }
}

/// Independent reference blocks of a touching pair (one ordering, no averaging): the full
/// kernel (kernels::green / grad_green conventions, no singularity subtraction, no
/// static_integrals) integrated in relative coordinates (Sauter-Schwab type: the singular set
/// is mapped to a corner of the parameter domain and removed by the Jacobian), n Gauss-Legendre
/// points per direction. Exponentially convergent: n = 16 is converged to ~1e-11 (shared edge,
/// K) and ~1e-14 (identical, shared vertex); n = 20 to ~1e-13 (slow test).
std::vector<Blocks> relative_reference(const RwgSpace& space, Index t1, Index t2,
                                       const std::vector<NamedRegion>& regs, int n) {
    const TriangleMesh& m = space.mesh();
    const AffineRwg r1 = affine_rwg(space, t1);
    const AffineRwg r2 = affine_rwg(space, t2);
    KernelSum sum{&r1, &r2, &regs, t1 != t2, std::vector<Blocks>(regs.size())};
    if (t1 == t2) {
        reference_identical(corners(m, t1), n, sum);
        return sum.out;
    }
    const Triangles& f = m.triangles();
    std::vector<Index> shared;
    std::vector<Index> only1;
    std::vector<Index> only2;
    for (Index i = 0; i < 3; ++i) {
        const Index v = f(t1, i);
        (v == f(t2, 0) || v == f(t2, 1) || v == f(t2, 2) ? shared : only1).push_back(v);
        const Index w = f(t2, i);
        if (!(w == f(t1, 0) || w == f(t1, 1) || w == f(t1, 2))) {
            only2.push_back(w);
        }
    }
    REQUIRE((shared.size() == 1 || shared.size() == 2));
    if (shared.size() == 2) {
        reference_shared_edge(vertex(m, shared[0]), vertex(m, shared[1]), vertex(m, only1[0]),
                              vertex(m, only2[0]), n, sum);
    } else {
        reference_shared_vertex(vertex(m, shared[0]), vertex(m, only1[0]), vertex(m, only1[1]),
                                vertex(m, only2[0]), vertex(m, only2[1]), n, sum);
    }
    return sum.out;
}

/// Relative K error; for coplanar pairs (reference K is rounding noise on the scale h^2) the
/// computed K must be exactly zero and 0 is returned.
Real rel_k(const Block& k, const Block& ref, bool coplanar) {
    if (coplanar) {
        CHECK(k == Block::Zero());
        return 0.0;
    }
    return rel_diff(k, ref);
}

/// Touching-pair cases of the WP7b tests (Si-sized geometry: |k| h = 0.78 in Si, 0.57 in Ag,
/// 0.18 in vacuum).
struct GradedGeometry {
    TouchingGeometry base;
    TriangleMesh hinge90;
    TriangleMesh hinge150;
    GradedGeometry()
        : hinge90(folded_hinge(0.6 * base.radius, 90.0)),
          hinge150(folded_hinge(0.6 * base.radius, 150.0)) {}
};

struct GradedCase {
    const TriangleMesh* mesh;
    Index t1, t2;
    std::string name;
};

/// Compares element_blocks with `opt` against the relative-coordinate reference (n points)
/// for both orderings; returns the worst L / K errors and the worst raw asymmetry.
struct GradedResult {
    Real err = 0.0;
    Real asym = 0.0;
};

GradedResult check_graded(const std::vector<GradedCase>& cases,
                          const std::vector<NamedRegion>& regs, const OperatorOptions& opt,
                          int n_ref, Real tol_err, Real tol_asym) {
    GradedResult res;
    for (const GradedCase& c : cases) {
        const RwgSpace space(*c.mesh);
        const bool coplanar = is_coplanar(*c.mesh, c.t1, c.t2);
        const std::vector<Blocks> ref12 = relative_reference(space, c.t1, c.t2, regs, n_ref);
        for (std::size_t g = 0; g < regs.size(); ++g) {
            const Blocks b12 = blocks(space, c.t1, c.t2, regs[g].p, opt);
            const Blocks b21 = blocks(space, c.t2, c.t1, regs[g].p, opt);
            // The reference of the swapped ordering is the transpose (its own swap asymmetry is
            // below 1e-13, slow test "relative-coordinate reference is converged").
            const Block ref21_L = ref12[g].L.transpose();
            const Block ref21_K = ref12[g].K.transpose();
            const Real eL = std::max(rel_diff(b12.L, ref12[g].L), rel_diff(b21.L, ref21_L));
            const Real eK =
                std::max(rel_k(b12.K, ref12[g].K, coplanar), rel_k(b21.K, ref21_K, coplanar));
            const Real aL = rel_diff(b21.L.transpose(), b12.L);
            const Real aK = coplanar ? 0.0 : rel_diff(b21.K.transpose(), b12.K);
            INFO(c.name << " " << regs[g].name << ": L " << eL << ", K " << eK << ", asymmetry L "
                        << aL << ", K " << aK);
            CHECK(eL < tol_err);
            CHECK(eK < tol_err);
            CHECK(aL < tol_asym);
            CHECK(aK < tol_asym);
            res.err = std::max({res.err, eL, eK});
            res.asym = std::max({res.asym, aL, aK});
        }
    }
    return res;
}

/// Reference points per direction of the regular cases: converged to ~1e-11 with n = 16 for
/// shared edges and n = 10 for identical triangles and shared vertices (slow test "relative-
/// coordinate reference is converged").
int regular_n(const GradedCase& c) {
    return classify(*c.mesh, c.t1, c.t2) == Proximity::shared_edge ? 16 : 10;
}

/// Options with a fixed plain-rule degree d for non-touching pairs of class `cls`.
OperatorOptions fixed_degree(int d, Proximity cls) {
    OperatorOptions opt;
    opt.target_accuracy = 0.0;
    opt.quad_degree_far = d;
    opt.quad_degree_near = d;
    if (!kernels::triangle_rule_is_positive_interior(d)) {
        // Only far pairs may use a non-positive-interior degree (the near bound must be PI).
        REQUIRE(cls == Proximity::far);
        opt.quad_degree_near = d + 1;
        while (!kernels::triangle_rule_is_positive_interior(opt.quad_degree_near)) {
            ++opt.quad_degree_near;
        }
    }
    return opt;
}

/// The Dunavant degree element_blocks selected for a near or far pair under `opt`, found by
/// bitwise comparison with the fixed-degree blocks of every degree in the ladder (the plain
/// double-Dunavant code path is the same).
int selected_degree(const RwgSpace& space, Index t1, Index t2, const RegionParams& reg,
                    const OperatorOptions& opt) {
    const Proximity cls = classify(space.mesh(), t1, t2, opt.near_distance_factor);
    REQUIRE((cls == Proximity::near || cls == Proximity::far));
    const Blocks b = blocks(space, t1, t2, reg, opt);
    for (int d = opt.quad_degree_far; d <= opt.quad_degree_near; ++d) {
        if (!(kernels::triangle_rule_is_positive_interior(d) ||
              (cls == Proximity::far && d == opt.quad_degree_far))) {
            continue;
        }
        const Blocks f = blocks(space, t1, t2, reg, fixed_degree(d, cls));
        if ((f.L.array() == b.L.array()).all() && (f.K.array() == b.K.array()).all()) {
            return d;
        }
    }
    FAIL("no degree of the ladder reproduces the blocks bitwise");
    return -1;
}

/// Unit icosphere n = 3 with vertices jittered by up to +-0.25 x 0.1 (deterministic seed): a
/// mesh with irregular triangles, independent of the calibration meshes of the selection rule.
TriangleMesh jittered_icosphere(std::uint64_t seed) {
    const TriangleMesh base = make_icosphere(1.0, 3);
    Vertices v = base.vertices();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> jitter(-0.025, 0.025);
    for (Index i = 0; i < v.rows(); ++i) {
        for (Index c = 0; c < 3; ++c) {
            v(i, c) += jitter(rng);
        }
    }
    return TriangleMesh(v, base.triangles());
}

/// Pair of the (unit-scale) mesh whose centroid distance over the larger longest edge is
/// closest to `ratio` (non-touching pairs only; deterministic: first minimum in index order).
std::pair<Index, Index> pair_at_ratio(const TriangleMesh& m, Real ratio) {
    // Centroids and longest edges once (plain arrays: the sanitizer build stays fast); classify
    // only for candidates that improve the best deviation.
    const auto nt = static_cast<std::size_t>(m.num_triangles());
    std::vector<std::array<Real, 3>> c(nt);
    std::vector<Real> h(nt);
    for (std::size_t t = 0; t < nt; ++t) {
        const Vec3 ct = m.centroid(static_cast<Index>(t));
        c[t] = {ct(0), ct(1), ct(2)};
        h[t] = longest_edge(m, static_cast<Index>(t));
    }
    std::pair<Index, Index> best{0, 0};
    Real best_dev = 1e300;
    for (std::size_t a = 0; a < nt; a += 5) {
        for (std::size_t b = 0; b < nt; ++b) {
            const Real dx = c[a][0] - c[b][0];
            const Real dy = c[a][1] - c[b][1];
            const Real dz = c[a][2] - c[b][2];
            const Real dev =
                std::abs(std::sqrt(dx * dx + dy * dy + dz * dz) / std::max(h[a], h[b]) - ratio);
            if (!(dev < best_dev)) {
                continue;
            }
            const Proximity cls = classify(m, static_cast<Index>(a), static_cast<Index>(b));
            if (cls == Proximity::near || cls == Proximity::far) {
                best_dev = dev;
                best = {static_cast<Index>(a), static_cast<Index>(b)};
            }
        }
    }
    return best;
}

/// The mesh scaled so that |k| h = kh for the pair (h = larger longest edge).
TriangleMesh scaled_mesh(const TriangleMesh& m, Index t1, Index t2, Real kh,
                         const RegionParams& reg) {
    const Real h = std::max(longest_edge(m, t1), longest_edge(m, t2));
    return TriangleMesh(m.vertices() * (kh / (std::abs(reg.k) * h)), m.triangles());
}

/// Degree selection over D/h in `ratios` and |k| h in `khs` on the jittered icosphere: the
/// blocks with the default target (1e-5) and with targets 1e-4, 1e-6 and 1e-8 agree with the
/// degree-20 brute force on n_sub^2 sub-triangles to the target; the selected degree (default
/// target) is non-decreasing in |k| h and non-increasing in D/h. Returns the worst error / target
/// ratio.
Real check_selection(const std::vector<Real>& ratios, const std::vector<Real>& khs,
                     const std::vector<NamedRegion>& regs, int n_sub) {
    const TriangleMesh unit = jittered_icosphere(20261008);
    Real worst = 0.0;
    for (const NamedRegion& reg : regs) {
        std::vector<std::vector<int>> deg(ratios.size(), std::vector<int>(khs.size(), 0));
        for (std::size_t i = 0; i < ratios.size(); ++i) {
            const auto [t1, t2] = pair_at_ratio(unit, ratios[i]);
            for (std::size_t j = 0; j < khs.size(); ++j) {
                const TriangleMesh m = scaled_mesh(unit, t1, t2, khs[j], reg.p);
                const RwgSpace space(m);
                const std::vector<Blocks> ref =
                    brute_force_dunavant(space, t1, t2, {reg}, 20, n_sub);
                for (const Real target : {1e-4, kDefaultTarget, 1e-6, 1e-8}) {
                    OperatorOptions opt;
                    opt.target_accuracy = target;
                    const Blocks b = blocks(space, t1, t2, reg.p, opt);
                    const Real eL = rel_diff(b.L, ref[0].L);
                    const Real eK = rel_diff(b.K, ref[0].K);
                    const int d = selected_degree(space, t1, t2, reg.p, opt);
                    INFO(reg.name << " D/h " << ratios[i] << " (pair " << t1 << "," << t2
                                  << "), |k| h " << khs[j] << ", target " << target << ": degree "
                                  << d << ", L " << eL << ", K " << eK);
                    CHECK(eL <= target);
                    CHECK(eK <= target);
                    worst = std::max(worst, std::max(eL, eK) / target);
                    if (target == kDefaultTarget) {
                        deg[i][j] = d;
                    }
                }
            }
        }
        std::string table;
        for (std::size_t i = 0; i < ratios.size(); ++i) {
            table += " D/h " + std::to_string(ratios[i]).substr(0, 4) + ":";
            for (std::size_t j = 0; j < khs.size(); ++j) {
                table += " " + std::to_string(deg[i][j]);
                if (j > 0) {
                    CHECK(deg[i][j] >= deg[i][j - 1]);  // non-decreasing in |k| h
                }
                if (i > 0) {
                    CHECK(deg[i][j] <= deg[i - 1][j]);  // non-increasing in D/h
                }
            }
        }
        WARN(reg.name << ": selected degrees (default target " << kDefaultTarget
                      << "; rows D/h, columns |k| h):" << table);
    }
    return worst;
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

TEST_CASE("element_blocks: WP7 path, identical pair matches the polar reference (Si)",
          "[kernels]") {
    const TouchingGeometry geo;
    check_touching({sphere_pair(geo, Proximity::identical)}, {all_regions()[1]}, {4, 10, 12}, 6);
}

// The folded shared edge is the most expensive reference; its degrees {4, 10, 12} are split
// over two cases to keep each below 1 s in the sanitizer build (the slow suite checks the
// whole sequence 4 .. 12 in one run).
TEST_CASE(
    "element_blocks: WP7 path, folded shared-edge pair converges to the polar reference "
    "(Si)",
    "[kernels]") {
    const TouchingGeometry geo;
    check_touching({{&geo.tent, 2, 3}}, {all_regions()[1]}, {4, 10}, 6);
}

TEST_CASE(
    "element_blocks: WP7 path, folded shared-edge pair matches the polar reference at "
    "degree 12 (Si)",
    "[kernels]") {
    const TouchingGeometry geo;
    check_touching({{&geo.tent, 2, 3}}, {all_regions()[1]}, {12}, 6);
}

TEST_CASE("element_blocks: WP7 path, folded shared-vertex pair matches the polar reference (Si)",
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
    OperatorOptions opt;  // quad_degree_sing = 10, WP7 path (the reference averages orderings)
    opt.outer_grading_levels = 0;
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
    // Documented defaults (operators.hpp, ADR 0004).
    CHECK(OperatorOptions{}.target_accuracy == 1e-5);
    CHECK(OperatorOptions{}.outer_grading_levels == 4);
    for (const int deg : {11, 15, 3, 7, 16, 18, 20}) {
        OperatorOptions near;
        near.quad_degree_near = deg;
        near.quad_degree_far = 1;
        OperatorOptions sing;
        sing.quad_degree_sing = deg;
        OperatorOptions far;
        far.quad_degree_far = deg;
        INFO("degree " << deg);
        CHECK_THROWS_AS(kernels::validate(near), std::invalid_argument);
        CHECK_THROWS_AS(kernels::validate(sing), std::invalid_argument);
        // Any degree as the lower bound, as long as it does not exceed the upper bound (19).
        if (deg <= 19) {
            CHECK_NOTHROW(kernels::validate(far));
        } else {
            CHECK_THROWS_AS(kernels::validate(far), std::invalid_argument);
        }
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
    // WP7b fields: inconsistent selection bounds, grading level, target, averaging threshold.
    for (const auto& [far_d, near_d] :
         {std::pair<int, int>{9, 8}, std::pair<int, int>{19, 17}, std::pair<int, int>{20, 19}}) {
        OperatorOptions o;
        o.quad_degree_far = far_d;
        o.quad_degree_near = near_d;
        CHECK_THROWS_AS(kernels::validate(o), std::invalid_argument);
        CHECK_THROWS_AS(element_blocks(space, 0, 40, reg, o, L, K), std::invalid_argument);
    }
    for (const int level : {-1, 7, 100}) {
        OperatorOptions o;
        o.outer_grading_levels = level;
        CHECK_THROWS_AS(kernels::validate(o), std::invalid_argument);
    }
    for (const int level : {0, 1, 6}) {
        OperatorOptions o;
        o.outer_grading_levels = level;
        CHECK_NOTHROW(kernels::validate(o));
        CHECK_NOTHROW(element_blocks(space, 0, 0, reg, o, L, K));
    }
    const Real inf = std::numeric_limits<Real>::infinity();
    for (const Real t : {-1e-6, 1.0, 2.0, inf, std::nan("")}) {
        OperatorOptions o;
        o.target_accuracy = t;
        CHECK_THROWS_AS(kernels::validate(o), std::invalid_argument);
    }
    for (const Real t : {0.0, 1e-12, 0.5}) {
        OperatorOptions o;
        o.target_accuracy = t;
        CHECK_NOTHROW(kernels::validate(o));
    }
    for (const Real t : {-1.0, std::nan("")}) {
        OperatorOptions o;
        o.symmetrize_touching_above_kh = t;
        CHECK_THROWS_AS(kernels::validate(o), std::invalid_argument);
    }
    for (const Real t : {0.0, 2.5, inf}) {
        OperatorOptions o;
        o.symmetrize_touching_above_kh = t;
        CHECK_NOTHROW(kernels::validate(o));
    }
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

namespace {

/// Seconds for `calls` element_blocks calls on far pairs of the vacuum-sized icosphere n = 1
/// (Ag region, complex k; default options).
Real far_pair_seconds(std::size_t calls) {
    const TriangleMesh mesh = make_icosphere(kVacuumRadius, 1);
    const RwgSpace space(mesh);
    const RegionParams reg = all_regions()[2].p;  // Ag, complex k
    const OperatorOptions opt;
    Block L;
    Block K;
    const auto far = pairs_of_class(mesh, Proximity::far);
    Complex sink{0.0, 0.0};
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < calls; ++i) {
        const auto& [t1, t2] = far[(i * 7919) % far.size()];
        element_blocks(space, t1, t2, reg, opt, L, K);
        sink += L(0, 0) + K(1, 1);
    }
    const Real seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    CHECK(std::isfinite(sink.real()));
    return seconds;
}

}  // namespace

TEST_CASE("element_blocks: 2 000 far-pair calls (timing smoke)", "[kernels]") {
    // Regular smoke check of the far-pair cost with the default degree selection: about 0.35 s in
    // the sanitizer build, 0.015 s in release. The bound has a large margin so that a loaded
    // machine does not fail the test; a slow run is reported.
    const Real seconds = far_pair_seconds(2000);
    WARN("2000 far-pair element_blocks calls: " << seconds << " s (expected < 1 s)");
    if (seconds > 1.0) {
        WARN("far-pair element_blocks calls slower than expected (machine load or regression)");
    }
    CHECK(seconds < 10.0);
}

TEST_CASE("element_blocks: 10 000 far-pair calls are fast", "[kernels][.slow]") {
    // Hidden since WP7b (the regular case above runs 2 000 calls): with the default degree
    // selection about 1.7 s in the sanitizer build (0.65 s with the fixed WP7 degree 3), 0.07 s
    // in release. The bound has a 10x margin over the 2 s target so that a loaded machine does
    // not fail the test.
    const Real seconds = far_pair_seconds(10000);
    WARN("10000 far-pair element_blocks calls: " << seconds << " s (target < 2 s)");
    CHECK(seconds < 20.0);
}

// ---------------------------------------------------------------------------------------------
// WP7b: graded outer rule for touching pairs and k-aware degree selection (regular cases).
// ---------------------------------------------------------------------------------------------

TEST_CASE("relative-coordinate reference integrates polynomials exactly", "[kernels]") {
    // The cone decompositions must cover T x T' exactly once with the right Jacobians: for a
    // polynomial integrand the rules are exact, compared with tensor Dunavant rules.
    struct PolySink {
        Real total = 0.0;
        static Real p(const Vec3& x, const Vec3& y) {
            return 1.0 + x(0) - 2.0 * y(1) + x(0) * y(0) + 3.0 * x(1) * x(1) * y(2) + y(2) * y(2);
        }
        void add(const Vec3& x, const Vec3& y, Real w) { total += w * p(x, y); }
    };
    const Vec3 o = Vec3::Zero();
    const Vec3 ex(1.0, 0.0, 0.0);
    const Vec3 ey(0.2, 0.9, 0.0);
    const Vec3 c_edge(0.3, -0.2, 0.7);
    const Vec3 b1(-1.0, 0.2, 0.3);
    const Vec3 b2(-0.4, -0.9, 0.1);
    const auto exact = [](const std::array<Vec3, 3>& t, const std::array<Vec3, 3>& s) {
        const kernels::TriangleRule& r = kernels::triangle_rule(10);
        const Real at = 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).norm();
        const Real as = 0.5 * (s[1] - s[0]).cross(s[2] - s[0]).norm();
        Real sum = 0.0;
        for (std::size_t i = 0; i < r.weights.size(); ++i) {
            const Vec3& l = r.barycentric[i];
            const Vec3 x = l(0) * t[0] + l(1) * t[1] + l(2) * t[2];
            for (std::size_t j = 0; j < r.weights.size(); ++j) {
                const Vec3& m = r.barycentric[j];
                const Vec3 y = m(0) * s[0] + m(1) * s[1] + m(2) * s[2];
                sum += r.weights[i] * r.weights[j] * at * as * PolySink::p(x, y);
            }
        }
        return sum;
    };
    PolySink id;
    reference_identical({o, ex, ey}, 6, id);
    PolySink edge;
    reference_shared_edge(o, ex, ey, c_edge, 8, edge);
    PolySink vtx;
    reference_shared_vertex(o, ex, ey, b1, b2, 8, vtx);
    const Real e_id = exact({o, ex, ey}, {o, ex, ey});
    const Real e_edge = exact({o, ex, ey}, {o, ex, c_edge});
    const Real e_vtx = exact({o, ex, ey}, {o, b1, b2});
    INFO("identical " << id.total << " / " << e_id << ", edge " << edge.total << " / " << e_edge
                      << ", vertex " << vtx.total << " / " << e_vtx);
    CHECK(std::abs(id.total - e_id) <= 1e-13 * std::abs(e_id));
    CHECK(std::abs(edge.total - e_edge) <= 1e-13 * std::abs(e_edge));
    CHECK(std::abs(vtx.total - e_vtx) <= 1e-13 * std::abs(e_vtx));
}

// Default options (outer_grading_levels = 4) against the relative-coordinate reference (n = 10
// for identical triangles and shared vertices, n = 16 for shared edges; converged to ~1e-11):
// acceptance 1e-8 of the block norm for L and K, raw (unaveraged, |k| h < 1) swap asymmetry
// below 1e-9. The touching geometry is sized for Si (|k| h = 0.78), so every material uses the
// unaveraged path. A few pairs per test case keep each below 1 s in the sanitizer build.
TEST_CASE("element_blocks: identical pairs agree with the relative-coordinate reference",
          "[kernels]") {
    const GradedGeometry geo;
    const TouchingPair s = sphere_pair(geo.base, Proximity::identical);
    const GradedResult r = check_graded(
        {{s.mesh, s.t1, s.t2, "sphere identical"}, {&geo.base.tent, 2, 2, "tent identical"}},
        all_regions(), OperatorOptions{}, 10, 1e-8, 1e-9);
    WARN("identical pairs (default options): worst error " << r.err << ", asymmetry " << r.asym);
}

TEST_CASE(
    "element_blocks: folded shared-edge pair (90 degrees) agrees with the "
    "relative-coordinate reference",
    "[kernels]") {
    const GradedGeometry geo;
    const GradedResult r = check_graded({{&geo.hinge90, 0, 1, "hinge 90"}}, all_regions(),
                                        OperatorOptions{}, 16, 1e-8, 1e-9);
    WARN("folded shared edge, 90 degrees (default options): worst error " << r.err << ", asymmetry "
                                                                          << r.asym);
}

TEST_CASE(
    "element_blocks: folded shared-edge pair (150 degrees) agrees with the "
    "relative-coordinate reference",
    "[kernels]") {
    const GradedGeometry geo;
    const GradedResult r = check_graded({{&geo.hinge150, 0, 1, "hinge 150"}}, all_regions(),
                                        OperatorOptions{}, 16, 1e-8, 1e-9);
    WARN("folded shared edge, 150 degrees (default options): worst error "
         << r.err << ", asymmetry " << r.asym);
}

TEST_CASE(
    "element_blocks: shared-vertex pair (folded fan) agrees with the relative-coordinate "
    "reference",
    "[kernels]") {
    const GradedGeometry geo;
    const GradedResult r = check_graded({{&geo.base.tent, 1, 4, "tent vertex"}}, all_regions(),
                                        OperatorOptions{}, 10, 1e-8, 1e-9);
    WARN("shared vertex, folded fan (default options): worst error " << r.err << ", asymmetry "
                                                                     << r.asym);
}

TEST_CASE(
    "element_blocks: shared-vertex pair (sphere) agrees with the relative-coordinate "
    "reference",
    "[kernels]") {
    const GradedGeometry geo;
    const TouchingPair s = sphere_pair(geo.base, Proximity::shared_vertex);
    const GradedResult r = check_graded({{s.mesh, s.t1, s.t2, "sphere vertex"}}, all_regions(),
                                        OperatorOptions{}, 10, 1e-8, 1e-9);
    WARN("shared vertex, sphere (default options): worst error " << r.err << ", asymmetry "
                                                                 << r.asym);
}

TEST_CASE("element_blocks: WP7 path (outer_grading_levels = 0) keeps its ~1e-4 accuracy",
          "[kernels]") {
    // Regression of the old path: same blocks as WP7 (one Dunavant outer rule of degree 10,
    // averaged orderings), whose error against the reference is the documented ~1e-4 (L) and
    // ~1e-2 of the small K block of folded shared edges.
    const GradedGeometry geo;
    const std::vector<NamedRegion> si = {all_regions()[1]};
    OperatorOptions opt;
    opt.outer_grading_levels = 0;
    Real worst_l = 0.0;
    Real worst_k = 0.0;
    for (const GradedCase& c :
         {GradedCase{&geo.base.tent, 2, 2, "identical"}, GradedCase{&geo.hinge90, 0, 1, "hinge 90"},
          GradedCase{&geo.base.tent, 1, 4, "vertex"}}) {
        const RwgSpace space(*c.mesh);
        const Blocks ref = relative_reference(space, c.t1, c.t2, si, regular_n(c))[0];
        const Blocks b = blocks(space, c.t1, c.t2, si[0].p, opt);
        INFO(c.name << ": L " << rel_diff(b.L, ref.L) << ", K "
                    << (c.t1 == c.t2 ? 0.0 : rel_diff(b.K, ref.K)));
        CHECK(rel_diff(b.L, ref.L) < 1e-3);
        worst_l = std::max(worst_l, rel_diff(b.L, ref.L));
        if (c.t1 != c.t2) {
            CHECK(rel_diff(b.K, ref.K) < 5e-2);
            worst_k = std::max(worst_k, rel_diff(b.K, ref.K));
        }
    }
    WARN("WP7 path (Si): worst L " << worst_l << ", worst K " << worst_k);
    CHECK(worst_l > 1e-5);  // the old outer-rule error is reproduced, not accidentally fixed
    CHECK(worst_k > 1e-3);
}

namespace {

/// WP7 golden blocks: L then K, row-major, (re, im) per entry.
using GoldenBlocks = std::array<std::array<Real, 2>, 18>;

struct GoldenCase {
    const char* name;
    const TriangleMesh* mesh;
    Index t1, t2;
    std::size_t region;  ///< index into all_regions()
};

/// The four golden cases: folded shared edge (Si), folded shared vertex (Ag), one near pair (Si)
/// and one far pair (Ag) of the Si-sized icosphere n = 1.
std::vector<GoldenCase> golden_cases(const TouchingGeometry& geo) {
    const auto near = pairs_of_class(geo.sphere, Proximity::near)[2];
    const auto far = pairs_of_class(geo.sphere, Proximity::far)[2];
    return {{"folded shared edge (Si)", &geo.tent, 2, 3, 1},
            {"folded shared vertex (Ag)", &geo.tent, 1, 4, 2},
            {"near (Si)", &geo.sphere, near.first, near.second, 1},
            {"far (Ag)", &geo.sphere, far.first, far.second, 2}};
}

}  // namespace

// clang-format off
/// Blocks of golden_cases() with wp7_options(), produced by the WP7b code and printed with 17
/// significant digits.
const std::array<GoldenBlocks, 4> kWp7Golden = {{
    // folded shared edge (Si) 2,3
    {{{-1.5447274398324211e-15, -3.8380301908446726e-15},
    {1.2951680070309858e-15, 3.2005129075399951e-15},
    {0, 0},
    {1.7571054884416683e-15, 4.3669934810766766e-15},
    {-1.412714653575104e-15, -3.5065085355331624e-15},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    {2.9022674555952704e-18, -4.629219102520308e-20},
    {1.01318151464337e-34, 4.4070383806628779e-37},
    {0, 0},
    {-9.6331368838249487e-19, 6.7480530444943917e-22},
    {-2.6497642117550493e-18, 4.3217243231546452e-20},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    }},
    // folded shared vertex (Ag) 1,4
    {{{7.7834542292029316e-17, 1.8600994916665141e-15},
    {-8.8258749009682293e-17, -2.1658258012827201e-15},
    {0, 0},
    {-8.2052922077520674e-17, -2.0147474492081449e-15},
    {9.1792380316196564e-17, 2.2054064383523779e-15},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    {8.7546558919293898e-19, -1.9493496986257011e-21},
    {9.4724221660776706e-20, -2.3935090209343591e-22},
    {0, 0},
    {-1.4982574380188073e-19, 2.9443715704696228e-22},
    {-9.6188417910102623e-19, 2.2154383382890125e-21},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    }},
    // near (Si) 0,10
    {{{8.0587141680702224e-16, 2.4317302411211458e-16},
    {-7.5949044166841465e-16, -2.3139161653419681e-16},
    {9.2327515398657462e-16, 2.7920898295098679e-16},
    {-7.7362865590856084e-16, -2.3544366306792983e-16},
    {8.2502779828788796e-16, 2.4984186838621102e-16},
    {-8.868141900227929e-16, -2.704684177313214e-16},
    {9.0665038560621217e-16, 2.7342161964398734e-16},
    {-9.0212672809560805e-16, -2.7293036135832408e-16},
    {9.9973287356660429e-16, 3.035060848126085e-16},
    {1.7756282039824101e-19, -8.67991661886532e-20},
    {-9.6260496936759527e-21, 3.5811062003348439e-21},
    {-2.8615808122147e-19, 1.0705691926875489e-19},
    {2.6410021188387524e-19, -1.1638044820655796e-19},
    {2.7772376257916799e-19, -9.99752881974433e-20},
    {-6.8428380505587439e-20, 2.3248360980773304e-20},
    {6.4926447738769156e-20, -3.1060003442780055e-20},
    {2.4490089585661793e-19, -1.1233795992413615e-19},
    {1.9897650741105582e-19, -9.1870305835936092e-20},
    }},
    // far (Ag) 0,33
    {{{-2.020171276398854e-17, -3.9713815348261808e-16},
    {2.0126544843759635e-17, 3.9314684868143306e-16},
    {-2.298907761485695e-17, -4.5677984772147509e-16},
    {2.0238755733700724e-17, 3.9904445549656975e-16},
    {-2.034870239089299e-17, -4.0489402182474051e-16},
    {2.269225977232737e-17, 4.4092954998032811e-16},
    {-2.2860446004525918e-17, -4.4992286088516061e-16},
    {2.2821078084136817e-17, 4.4783533315457908e-16},
    {-2.5908677096754513e-17, -5.1184783699425498e-16},
    {-9.0190085642901123e-20, 8.9314258850094389e-22},
    {9.7240991475823922e-21, -5.5776021138959945e-23},
    {9.5204775093187631e-20, -9.574880981920187e-22},
    {-1.0876373279827316e-19, 1.0687675608012735e-21},
    {-7.8921438603878539e-20, 8.1134425778811831e-22},
    {3.5350466654023697e-21, -1.0454851397776598e-22},
    {-2.4249750799074118e-20, 2.2335548000139993e-22},
    {-8.2628612841905029e-20, 8.7703324502316639e-22},
    {-7.9151600855580725e-20, 8.1811003786270057e-22},
    }},
}};
// clang-format on

TEST_CASE("element_blocks: WP7 path reproduces the golden blocks", "[kernels]") {
    // Regression of the shared code of both paths (static integrals, remainder, plain kernel,
    // Dunavant rules, averaging): WP7 options (outer_grading_levels = 0, target_accuracy = 0,
    // quad_degree_near = 8) on a folded shared-edge pair (Si), a folded shared-vertex pair (Ag),
    // a near pair (Si) and a far pair (Ag) reproduce hard-coded blocks to 1e-12 of the block
    // norm (L and K separately), so that shared code cannot drift silently.
    const TouchingGeometry geo;
    const std::vector<GoldenCase> cases = golden_cases(geo);
    REQUIRE(cases.size() == kWp7Golden.size());
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const RwgSpace space(*cases[c].mesh);
        const Blocks b = blocks(space, cases[c].t1, cases[c].t2, all_regions()[cases[c].region].p,
                                wp7_options());
        Block gl;
        Block gk;
        for (std::size_t e = 0; e < 9; ++e) {
            const auto i = static_cast<Eigen::Index>(e / 3);
            const auto j = static_cast<Eigen::Index>(e % 3);
            gl(i, j) = Complex(kWp7Golden[c][e][0], kWp7Golden[c][e][1]);
            gk(i, j) = Complex(kWp7Golden[c][9 + e][0], kWp7Golden[c][9 + e][1]);
        }
        const Real eL = rel_diff(b.L, gl);
        const Real eK = rel_diff(b.K, gk);
        INFO(cases[c].name << " (" << cases[c].t1 << "," << cases[c].t2 << "): L " << eL << ", K "
                           << eK);
        CHECK(eL < 1e-12);
        CHECK(eK < 1e-12);
    }
}

namespace {

/// Convergence in outer_grading_levels 1 to 4 for one touching case of the folded fan / hinge
/// (0: identical, 1: shared edge, 2: shared vertex).
void check_levels(std::size_t which) {
    // (a) Self-convergence |B_l - B_6| / |B_6| (the remainder is the same for every level, so
    // this is the outer-rule error of the analytic part alone), all materials; (b) the error
    // against the reference in vacuum (|k| h = 0.18: remainder error negligible). Each step must
    // decrease unless both values are below 1e-11 (rounding / reference level).
    const GradedGeometry geo;
    const std::vector<NamedRegion> regs = all_regions();
    const auto monotone = [](const std::vector<Real>& e) {
        for (std::size_t i = 1; i < e.size(); ++i) {
            if (!(e[i] < e[i - 1] || std::max(e[i], e[i - 1]) < 1e-11)) {
                return false;
            }
        }
        return true;
    };
    const std::array<GradedCase, 3> cases = {GradedCase{&geo.base.tent, 2, 2, "identical"},
                                             GradedCase{&geo.hinge150, 1, 0, "hinge"},
                                             GradedCase{&geo.base.tent, 1, 4, "vertex"}};
    {
        const GradedCase& c = cases.at(which);
        const RwgSpace space(*c.mesh);
        const bool coplanar = is_coplanar(*c.mesh, c.t1, c.t2);
        const Blocks ref = relative_reference(space, c.t1, c.t2, {regs[0]}, regular_n(c))[0];
        for (std::size_t g = 0; g < regs.size(); ++g) {
            OperatorOptions opt;
            opt.outer_grading_levels = 6;
            const Blocks b6 = blocks(space, c.t1, c.t2, regs[g].p, opt);
            std::vector<Real> self_l;
            std::vector<Real> self_k;
            std::vector<Real> ref_l;
            std::vector<Real> ref_k;
            for (int level = 1; level <= 4; ++level) {
                opt.outer_grading_levels = level;
                const Blocks b = blocks(space, c.t1, c.t2, regs[g].p, opt);
                self_l.push_back(rel_diff(b.L, b6.L));
                self_k.push_back(coplanar ? 0.0 : rel_diff(b.K, b6.K));
                if (g == 0) {
                    ref_l.push_back(rel_diff(b.L, ref.L));
                    ref_k.push_back(coplanar ? 0.0 : rel_diff(b.K, ref.K));
                }
            }
            INFO(c.name << " " << regs[g].name << ": self L " << self_l[0] << " " << self_l[1]
                        << " " << self_l[2] << " " << self_l[3] << ", self K " << self_k[0] << " "
                        << self_k[1] << " " << self_k[2] << " " << self_k[3]);
            CHECK(monotone(self_l));
            CHECK(monotone(self_k));
            if (g == 0) {
                INFO("vacuum vs reference: L " << ref_l[0] << " " << ref_l[1] << " " << ref_l[2]
                                               << " " << ref_l[3] << ", K " << ref_k[0] << " "
                                               << ref_k[1] << " " << ref_k[2] << " " << ref_k[3]);
                CHECK(monotone(ref_l));
                CHECK(monotone(ref_k));
            }
        }
    }
}

}  // namespace

TEST_CASE("element_blocks: graded rule converges monotonically in levels 1 to 4 (identical)",
          "[kernels]") {
    check_levels(0);
}

TEST_CASE("element_blocks: graded rule converges monotonically in levels 1 to 4 (shared edge)",
          "[kernels]") {
    check_levels(1);
}

TEST_CASE("element_blocks: graded rule converges monotonically in levels 1 to 4 (shared vertex)",
          "[kernels]") {
    check_levels(2);
}

TEST_CASE("element_blocks: averaging threshold, determinism", "[kernels]") {
    const GradedGeometry geo;
    const RwgSpace space(geo.hinge90);
    const RegionParams si = all_regions()[1].p;
    // Below the threshold (|k| h < 1): raw blocks; averaging forced with threshold 0 gives
    // bitwise transposed blocks that agree with the raw ones to the raw asymmetry.
    REQUIRE(std::abs(si.k) * longest_edge(geo.hinge90, 0) < 1.0);
    OperatorOptions raw;
    OperatorOptions avg;
    avg.symmetrize_touching_above_kh = 0.0;
    const Blocks r12 = blocks(space, 0, 1, si, raw);
    const Blocks a12 = blocks(space, 0, 1, si, avg);
    const Blocks a21 = blocks(space, 1, 0, si, avg);
    CHECK((a21.L.transpose().array() == a12.L.array()).all());
    CHECK((a21.K.transpose().array() == a12.K.array()).all());
    CHECK(rel_diff(a12.L, r12.L) < 1e-9);
    CHECK(rel_diff(a12.K, r12.K) < 1e-9);
    // Threshold infinity: never averaged, even on a coarse mesh (6x the size: |k| h > 1),
    // where the default threshold averages.
    const TriangleMesh coarse = folded_hinge(3.6 * geo.base.radius, 90.0);
    const RwgSpace coarse_space(coarse);
    REQUIRE(std::abs(si.k) * longest_edge(coarse, 0) > 1.0);
    OperatorOptions never;
    never.symmetrize_touching_above_kh = std::numeric_limits<Real>::infinity();
    const Blocks d12 = blocks(coarse_space, 0, 1, si, OperatorOptions{});
    const Blocks d21 = blocks(coarse_space, 1, 0, si, OperatorOptions{});
    CHECK((d21.L.transpose().array() == d12.L.array()).all());
    const Blocks n12 = blocks(coarse_space, 0, 1, si, never);
    const Blocks n21 = blocks(coarse_space, 1, 0, si, never);
    WARN("coarse folded shared edge (|k| h = "
         << std::abs(si.k) * longest_edge(coarse, 0) << ", Si): raw asymmetry L "
         << rel_diff(n21.L.transpose(), n12.L) << ", K " << rel_diff(n21.K.transpose(), n12.K));
    CHECK(rel_diff(n21.L.transpose(), n12.L) > 0.0);
    // Determinism: repeated calls (with other pairs in between) are bitwise identical, for
    // touching pairs and for near / far pairs with degree selection.
    const TriangleMesh sphere = make_icosphere(kVacuumRadius, 1);
    const RwgSpace sphere_space(sphere);
    std::vector<std::pair<Index, Index>> pairs;
    for (const Proximity cls : {Proximity::identical, Proximity::shared_edge,
                                Proximity::shared_vertex, Proximity::near, Proximity::far}) {
        pairs.push_back(pairs_of_class(sphere, cls)[2]);
    }
    std::vector<Blocks> first;
    for (const auto& [t1, t2] : pairs) {
        first.push_back(blocks(sphere_space, t1, t2, si, OperatorOptions{}));
    }
    for (std::size_t i = pairs.size(); i-- > 0;) {
        const Blocks again =
            blocks(sphere_space, pairs[i].first, pairs[i].second, si, OperatorOptions{});
        CHECK((again.L.array() == first[i].L.array()).all());
        CHECK((again.K.array() == first[i].K.array()).all());
    }
}

TEST_CASE("element_blocks: degree selection reaches the target (vacuum)", "[kernels]") {
    const Real worst = check_selection({2.0, 10.0}, {0.1, 1.0, 2.0}, {all_regions()[0]}, 1);
    WARN("degree selection (vacuum): worst error / target " << worst);
}

TEST_CASE("element_blocks: degree selection reaches the target (Si)", "[kernels]") {
    const Real worst = check_selection({2.0, 10.0}, {0.1, 1.0, 2.0}, {all_regions()[1]}, 1);
    WARN("degree selection (Si): worst error / target " << worst);
}

TEST_CASE("element_blocks: degree selection reaches the target (Ag)", "[kernels]") {
    const Real worst = check_selection({2.0, 10.0}, {0.1, 1.0, 2.0}, {all_regions()[2]}, 1);
    WARN("degree selection (Ag): worst error / target " << worst);
}

TEST_CASE("element_blocks: degree selection bounds and fixed degrees", "[kernels]") {
    const TriangleMesh unit = jittered_icosphere(20261008);
    const auto [t1, t2] = pair_at_ratio(unit, 3.0);
    const RegionParams ag = all_regions()[2].p;
    const TriangleMesh m = scaled_mesh(unit, t1, t2, 1.0, ag);
    const RwgSpace space(m);
    // The selection stays within [quad_degree_far, quad_degree_near].
    OperatorOptions narrow;
    narrow.quad_degree_far = 5;
    narrow.quad_degree_near = 6;
    narrow.target_accuracy = 1e-12;  // unreachable: upper bound
    CHECK(selected_degree(space, t1, t2, ag, narrow) == 6);
    narrow.target_accuracy = 0.5;  // trivial: lower bound
    CHECK(selected_degree(space, t1, t2, ag, narrow) == 5);
    // target_accuracy = 0: the fixed WP7 degrees (far: quad_degree_far even if not PI).
    OperatorOptions fixed;
    fixed.target_accuracy = 0.0;
    REQUIRE(classify(m, t1, t2) == Proximity::far);
    CHECK(selected_degree(space, t1, t2, ag, fixed) == 3);
    const auto [n1, n2] = pair_at_ratio(unit, 1.2);
    const TriangleMesh mn = scaled_mesh(unit, n1, n2, 1.0, ag);
    const RwgSpace near_space(mn);
    REQUIRE(classify(mn, n1, n2) == Proximity::near);
    CHECK(selected_degree(near_space, n1, n2, ag, fixed) == 19);
    // Near pairs never use the non-positive-interior lower bound.
    OperatorOptions low;
    low.target_accuracy = 0.5;
    CHECK(selected_degree(near_space, n1, n2, ag, low) == 4);
    CHECK(selected_degree(space, t1, t2, ag, low) == 3);
}

// ---------------------------------------------------------------------------------------------
// Hidden slow cases ([.slow], not registered with ctest; run with
// `specklebem_unit_tests "[kernels][slow]"`, about 9 s in release; meant for optimised builds).
// ---------------------------------------------------------------------------------------------

TEST_CASE("element_blocks (full): far pairs, six pairs", "[kernels][.slow]") {
    check_far(6);
}

TEST_CASE("element_blocks (full): near pairs vs composite brute force", "[kernels][.slow]") {
    check_near(2, 2);
}

TEST_CASE("element_blocks (full): WP7 path, touching pairs, all materials and degrees",
          "[kernels][.slow]") {
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

TEST_CASE("relative-coordinate reference is converged", "[kernels][.slow]") {
    // Self-convergence against n = 28: n = 16 (regular cases, shared edges) and n = 10 (regular
    // cases, identical triangles and shared vertices, whose integrands are smoother) below
    // 1e-10, n = 20 (sweep) below 1e-12 of the block norm, for every touching case and
    // material. The swapped ordering is the transpose to 1e-12 (the regular cases compare the
    // swapped blocks with the transposed reference).
    const GradedGeometry geo;
    const std::vector<NamedRegion> regs = all_regions();
    const TouchingPair si = sphere_pair(geo.base, Proximity::identical);
    const TouchingPair se = sphere_pair(geo.base, Proximity::shared_edge);
    const TouchingPair sv = sphere_pair(geo.base, Proximity::shared_vertex);
    Real worst_reg = 0.0;
    Real worst20 = 0.0;
    Real worst_swap = 0.0;
    for (const GradedCase& c :
         {GradedCase{si.mesh, si.t1, si.t2, "sphere identical"},
          GradedCase{se.mesh, se.t1, se.t2, "sphere edge"},
          GradedCase{sv.mesh, sv.t1, sv.t2, "sphere vertex"},
          GradedCase{&geo.hinge90, 0, 1, "hinge 90"}, GradedCase{&geo.hinge150, 1, 0, "hinge 150"},
          GradedCase{&geo.base.tent, 1, 4, "tent vertex"},
          GradedCase{&geo.base.flat, 0, 1, "flat edge"}}) {
        const RwgSpace space(*c.mesh);
        const bool coplanar = is_coplanar(*c.mesh, c.t1, c.t2);
        const bool edge = classify(*c.mesh, c.t1, c.t2) == Proximity::shared_edge;
        const int n_reg = edge ? 16 : 10;
        const std::vector<Blocks> rr = relative_reference(space, c.t1, c.t2, regs, n_reg);
        const std::vector<Blocks> r20 = relative_reference(space, c.t1, c.t2, regs, 20);
        const std::vector<Blocks> r28 = relative_reference(space, c.t1, c.t2, regs, 28);
        const std::vector<Blocks> s28 = relative_reference(space, c.t2, c.t1, regs, 28);
        for (std::size_t g = 0; g < regs.size(); ++g) {
            const Real er =
                std::max(rel_diff(rr[g].L, r28[g].L), coplanar ? 0.0 : rel_diff(rr[g].K, r28[g].K));
            const Real e20 = std::max(rel_diff(r20[g].L, r28[g].L),
                                      coplanar ? 0.0 : rel_diff(r20[g].K, r28[g].K));
            const Real es = std::max(rel_diff(s28[g].L.transpose(), r28[g].L),
                                     coplanar ? 0.0 : rel_diff(s28[g].K.transpose(), r28[g].K));
            INFO(c.name << " " << regs[g].name << ": n = " << n_reg << " " << er << ", n = 20 "
                        << e20 << ", swap " << es);
            CHECK(er < 1e-10);
            CHECK(e20 < 1e-12);
            CHECK(es < 1e-12);
            worst_reg = std::max(worst_reg, er);
            worst20 = std::max(worst20, e20);
            worst_swap = std::max(worst_swap, es);
        }
    }
    WARN("relative-coordinate reference vs n = 28: worst n = 16 / 10 "
         << worst_reg << ", n = 20 " << worst20 << ", swap " << worst_swap);
}

TEST_CASE("element_blocks (full): touching pairs vs the relative-coordinate reference",
          "[kernels][.slow]") {
    // All touching geometries, both orderings, three materials, reference n = 20: default
    // options to 1e-8 with raw asymmetry < 1e-9; per level the worst error (WARN table).
    const GradedGeometry geo;
    const std::vector<NamedRegion> regs = all_regions();
    std::vector<GradedCase> cases;
    for (const Proximity cls :
         {Proximity::identical, Proximity::shared_edge, Proximity::shared_vertex}) {
        const auto all = pairs_of_class(geo.base.sphere, cls);
        for (const std::size_t i : {std::size_t{3}, all.size() / 2}) {
            cases.push_back({&geo.base.sphere, all[i].first, all[i].second,
                             std::string("sphere ") + class_name(cls)});
        }
    }
    cases.push_back({&geo.base.tent, 0, 0, "tent identical"});
    cases.push_back({&geo.base.tent, 2, 3, "tent edge"});
    cases.push_back({&geo.base.tent, 1, 4, "tent vertex"});
    cases.push_back({&geo.base.flat, 0, 1, "flat edge"});
    cases.push_back({&geo.base.flat, 0, 3, "flat vertex"});
    cases.push_back({&geo.hinge90, 0, 1, "hinge 90"});
    cases.push_back({&geo.hinge150, 0, 1, "hinge 150"});
    const GradedResult r = check_graded(cases, regs, OperatorOptions{}, 20, 1e-8, 1e-9);
    WARN("touching sweep (default options): worst error " << r.err << ", worst raw asymmetry "
                                                          << r.asym);
    // Error by level (informational).
    std::ostringstream table;
    table << std::scientific << std::setprecision(1);
    for (int level = 0; level <= 6; ++level) {
        OperatorOptions opt;
        opt.outer_grading_levels = level;
        Real worst = 0.0;
        for (const GradedCase& c : cases) {
            const RwgSpace space(*c.mesh);
            const bool coplanar = is_coplanar(*c.mesh, c.t1, c.t2);
            const std::vector<Blocks> ref = relative_reference(space, c.t1, c.t2, regs, 20);
            for (std::size_t g = 0; g < regs.size(); ++g) {
                const Blocks b = blocks(space, c.t1, c.t2, regs[g].p, opt);
                worst = std::max(
                    {worst, rel_diff(b.L, ref[g].L), coplanar ? 0.0 : rel_diff(b.K, ref[g].K)});
            }
        }
        table << " " << level << ": " << worst;
    }
    WARN("worst touching error by outer_grading_levels:" << table.str());
}

TEST_CASE("element_blocks (full): sharp folds (60 and 30 degrees)", "[kernels][.slow]") {
    // Shared edges with a small dihedral angle are near-singular: the far vertex of one triangle
    // comes close to the other triangle, so the outer integrand varies sharply away from the
    // shared edge, where the edge grading does not refine. The 1e-8 of folds >= 90 degrees is
    // then not reached at the default level 4 and the error depends on the triangle shapes
    // (operators.hpp, ADR 0004). Both orderings, three materials, raw asymmetry with the same
    // tolerance; references n = 28 (checked against n = 36 to 1e-10). Measured worst L / K
    // errors (all materials) in the WARN lines; tolerances about 3x to 7x above them:
    //  * asymmetric 60 degree hinge (apexes (0.4, 0.8) / (0.7, 0.6)), level 4: 1e-7;
    //  * skewed 60 degree hinge ((0.3, 0.9) / (0.8, 0.55)), level 4: 1e-6, level 6: 1e-8;
    //  * 30 degree fold (folded_hinge), level 4: 2e-7, level 5: 2e-8;
    //  * skewed 30 degree hinge, level 4: 1e-4, level 6: 1e-5 (documents the limitation: folds
    //    this sharp with skewed triangles are not resolved better than ~1e-5 by the edge grading).
    const GradedGeometry geo;
    const std::vector<NamedRegion> regs = all_regions();
    const Real a = 0.6 * geo.base.radius;
    const TriangleMesh asym60 = asymmetric_hinge(a, 60.0, 0.4, 0.8, 0.7, 0.6);
    const TriangleMesh skew60 = asymmetric_hinge(a, 60.0, 0.3, 0.9, 0.8, 0.55);
    const TriangleMesh fold30 = folded_hinge(a, 30.0);
    const TriangleMesh skew30 = asymmetric_hinge(a, 30.0, 0.3, 0.9, 0.8, 0.55);
    for (const TriangleMesh* m : {&asym60, &skew60, &fold30, &skew30}) {
        const RwgSpace space(*m);
        const std::vector<Blocks> r28 = relative_reference(space, 0, 1, regs, 28);
        const std::vector<Blocks> r36 = relative_reference(space, 0, 1, regs, 36);
        for (std::size_t g = 0; g < regs.size(); ++g) {
            CHECK(rel_diff(r28[g].L, r36[g].L) < 1e-10);
            CHECK(rel_diff(r28[g].K, r36[g].K) < 1e-10);
        }
    }
    struct FoldCase {
        const TriangleMesh* mesh;
        const char* name;
        int level;
        Real tol;
    };
    for (const FoldCase& f :
         {FoldCase{&asym60, "asymmetric hinge 60", 4, 1e-7},
          FoldCase{&skew60, "skewed hinge 60", 4, 1e-6},
          FoldCase{&skew60, "skewed hinge 60", 6, 1e-8}, FoldCase{&fold30, "fold 30", 4, 2e-7},
          FoldCase{&fold30, "fold 30", 5, 2e-8}, FoldCase{&skew30, "skewed hinge 30", 4, 1e-4},
          FoldCase{&skew30, "skewed hinge 30", 6, 1e-5}}) {
        OperatorOptions opt;
        opt.outer_grading_levels = f.level;
        const GradedResult r = check_graded({{f.mesh, 0, 1, f.name}}, regs, opt, 28, f.tol, f.tol);
        WARN(f.name << " degrees, level " << f.level << ": worst error " << r.err
                    << ", raw asymmetry " << r.asym << " (tolerance " << f.tol << ")");
    }
}

TEST_CASE("element_blocks (full): degree selection sweep", "[kernels][.slow]") {
    // D/h from 1.1 to 15 and |k| h from 0.1 to 3, all materials, targets 1e-4, 1e-5 (default),
    // 1e-6, 1e-8.
    const Real worst = check_selection({1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0},
                                       {0.1, 0.5, 1.0, 2.0, 3.0}, all_regions(), 3);
    WARN("degree selection sweep: worst error / target " << worst);
}

TEST_CASE("element_blocks (full): class-boundary pairs of the n = 4 Mie mesh", "[kernels][.slow]") {
    // Sphere d = 1 um, icosphere n = 4 (h ~ lambda / 13 in vacuum), 500 nm; pairs with centroid
    // distance within 10 % of the near/far class boundary 2 h; dielectric n = 1.5 and Ag
    // interiors. The default options must reach their target (1e-5) and target_accuracy = 1e-6
    // must reach 1e-6; the WP7 fixed degrees (3 far / 8 near) are reported for comparison.
    const TriangleMesh mesh = make_icosphere(0.5e-6, 4);
    const RwgSpace space(mesh);
    material::Material n15 = material::vacuum();
    n15.eps_r = Complex(2.25, 0.0);
    const std::vector<NamedRegion> regs = {{"n = 1.5", region_of(n15, kLambda)},
                                           {"Ag", region_of(material::silver_500nm(), kLambda)}};
    std::vector<std::pair<Index, Index>> boundary;
    for (Index a = 0; a < mesh.num_triangles() && boundary.size() < 12; a += 97) {
        for (Index b = 0; b < mesh.num_triangles(); ++b) {
            const Real h = std::max(longest_edge(mesh, a), longest_edge(mesh, b));
            const Real r = (mesh.centroid(a) - mesh.centroid(b)).norm() / h;
            const Proximity cls = classify(mesh, a, b);
            if (r > 1.8 && r < 2.2 && (cls == Proximity::near || cls == Proximity::far)) {
                boundary.emplace_back(a, b);
                break;
            }
        }
    }
    REQUIRE(boundary.size() >= 10);
    for (const NamedRegion& reg : regs) {
        OperatorOptions opt6;
        opt6.target_accuracy = 1e-6;
        Real worst_new = 0.0;
        Real worst_6 = 0.0;
        Real worst_old = 0.0;
        for (const auto& [t1, t2] : boundary) {
            const Blocks ref = brute_force_dunavant(space, t1, t2, {reg}, 20, 2)[0];
            const Blocks b = blocks(space, t1, t2, reg.p, OperatorOptions{});
            const Blocks b6 = blocks(space, t1, t2, reg.p, opt6);
            const Blocks o = blocks(space, t1, t2, reg.p, wp7_options());
            const Real e = std::max(rel_diff(b.L, ref.L), rel_diff(b.K, ref.K));
            const Real e6 = std::max(rel_diff(b6.L, ref.L), rel_diff(b6.K, ref.K));
            INFO(reg.name << " pair " << t1 << "," << t2 << ": default " << e << ", 1e-6 " << e6);
            CHECK(e <= kDefaultTarget);
            CHECK(e6 <= opt6.target_accuracy);
            worst_new = std::max(worst_new, e);
            worst_6 = std::max(worst_6, e6);
            worst_old = std::max(worst_old, std::max(rel_diff(o.L, ref.L), rel_diff(o.K, ref.K)));
        }
        WARN("n = 4 Mie mesh, class boundary, " << reg.name << ": worst error default (1e-5) "
                                                << worst_new << ", target 1e-6 " << worst_6
                                                << ", WP7 " << worst_old);
    }
}

TEST_CASE("element_blocks (full): cost of the WP7b defaults", "[kernels][.slow]") {
    // Informational (release build): mean time per element_blocks call by class on the Mie
    // n = 3 mesh, WP7 options vs the defaults; meaningful only in an optimised build.
    const TriangleMesh mesh = make_icosphere(0.5e-6, 3);
    const RwgSpace space(mesh);
    material::Material n15 = material::vacuum();
    n15.eps_r = Complex(2.25, 0.0);
    const std::vector<NamedRegion> regs = {{"vacuum", region_of(material::vacuum(), kLambda)},
                                           {"n = 1.5", region_of(n15, kLambda)},
                                           {"Ag", region_of(material::silver_500nm(), kLambda)}};
    std::vector<std::vector<std::pair<Index, Index>>> by_class(5);
    for (Index a = 0; a < mesh.num_triangles(); a += 53) {
        for (Index b = 0; b < mesh.num_triangles(); ++b) {
            by_class[static_cast<std::size_t>(classify(mesh, a, b))].emplace_back(a, b);
        }
    }
    Block L;
    Block K;
    Complex sink{0.0, 0.0};
    for (const NamedRegion& reg : regs) {
        for (std::size_t c = 0; c < by_class.size(); ++c) {
            Real us[2] = {0.0, 0.0};
            for (int which = 0; which < 2; ++which) {
                const OperatorOptions opt = which == 0 ? wp7_options() : OperatorOptions{};
                const auto t0 = std::chrono::steady_clock::now();
                for (const auto& [t1, t2] : by_class[c]) {
                    element_blocks(space, t1, t2, reg.p, opt, L, K);
                    sink += L(0, 0);
                }
                us[which] =
                    std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count() *
                    1e6 / static_cast<Real>(by_class[c].size());
            }
            WARN(reg.name << " " << class_name(static_cast<Proximity>(c)) << " ("
                          << by_class[c].size() << " pairs): WP7 " << us[0] << " us, default "
                          << us[1] << " us per call (x" << us[1] / us[0] << ")");
        }
    }
    CHECK(std::isfinite(sink.real()));
}

// ---------------------------------------------------------------------------------------------
// WP7c: dihedral sweep of shared-edge and shared-vertex folds.
// ---------------------------------------------------------------------------------------------

namespace {

/// Closed fan of four triangles around the shared vertex A = 0: T = (A, B, C) in the plane z = 0
/// (y >= 0), T' = (A, D, E) and (A, E, B) in the half-plane at dihedral angle `degrees` about the
/// x axis, (A, C, D) bridging the two; B = (a, 0, 0), C = (x2, y2, 0) a, and D, E at (xd, rd) a and
/// (x3, r3) a in the folded half-plane (along the hinge, distance from it): (x, r) -> (x a, r a
/// cos th, r a sin th). T (0) and T' (2) share only A and carry two RWG functions each.
TriangleMesh vertex_fold(Real a, Real degrees, Real x2, Real y2, Real xd, Real rd, Real x3,
                         Real r3) {
    const Real c = std::cos(degrees * kPi / 180.0);
    const Real s = std::sin(degrees * kPi / 180.0);
    Vertices v(5, 3);
    v << 0.0, 0.0, 0.0, a, 0.0, 0.0, x2 * a, y2 * a, 0.0, xd * a, rd * a * c, rd * a * s, x3 * a,
        r3 * a * c, r3 * a * s;
    Triangles f(4, 3);
    f << 0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1;
    return TriangleMesh(v, f);
}

/// Shapes of the sweep: apex of T at (x2, y2) a; far vertex of T' at x3 a along the hinge and
/// r3 a from it (shared vertex: D at (xd, rd) a). Shared edge: regular (folded_hinge), skewed (the
/// far vertex of T' close to B, small angle at A when folded) and obtuse (obtuse angles at the
/// shared vertex A in both triangles, 106 and 120 degrees). Shared vertex (vertex_fold): regular,
/// skewed (the edge A E at 19 degrees from the hinge, so that its projection onto T runs close to
/// A B) and obtuse (106 and 104 degrees at A).
struct FoldShape {
    const char* name;
    Real x2, y2, x3, r3, xd, rd;
};
constexpr std::array<FoldShape, 5> kEdgeShapes = {{{"regular", 0.4, 0.8, 0.55, 0.85, 0.0, 0.0}, {"obtS", 0.4, 0.8, -0.35, 0.6, 0.0, 0.0}, {"obtB", -0.2, 0.7, -0.35, 0.6, 0.0, 0.0},
                                                    {"skewed", 0.3, 0.9, 0.8, 0.55, 0.0, 0.0},
                                                    {"obtuse", -0.2, 0.7, 0.55, 0.85, 0.0, 0.0}}};
constexpr std::array<FoldShape, 3> kVertexShapes = {
    {{"regular", 0.4, 0.8, 0.3, 0.8, -0.3, 0.6},
     {"skewed", 0.3, 0.9, 0.75, 0.45, -0.1, 0.7},
     {"obtuse", -0.2, 0.7, 0.2, 0.8, -0.6, 0.0}}};

struct FoldPair {
    std::string name;
    TriangleMesh mesh;
    Index t1, t2;
};

/// The fold pair of class `edge` (shared edge / shared vertex), shape and dihedral angle, sized
/// like the other touching geometries (a = 0.6 radius of the Si-sized sphere).
FoldPair fold_pair(bool edge, const FoldShape& s, Real degrees) {
    const Real a = 0.6 * scaled(kVacuumRadius, all_regions()[1].p);
    std::ostringstream name;
    name << (edge ? "edge " : "vertex ") << s.name << " " << degrees;
    if (edge) {
        return {name.str(), asymmetric_hinge(a, degrees, s.x2, s.y2, s.x3, s.r3), 0, 1};
    }
    return {name.str(), vertex_fold(a, degrees, s.x2, s.y2, s.xd, s.rd, s.x3, s.r3), 0, 2};
}

/// Worst relative L / K error over both orderings and all regions against the reference
/// (one ordering, transposed for the other), and the worst raw swap asymmetry.
GradedResult fold_errors(const FoldPair& p, const std::vector<NamedRegion>& regs,
                         const std::vector<Blocks>& ref, const OperatorOptions& opt) {
    const RwgSpace space(p.mesh);
    GradedResult res;
    for (std::size_t g = 0; g < regs.size(); ++g) {
        const Blocks b12 = blocks(space, p.t1, p.t2, regs[g].p, opt);
        const Blocks b21 = blocks(space, p.t2, p.t1, regs[g].p, opt);
        res.err = std::max({res.err, rel_diff(b12.L, ref[g].L), rel_diff(b12.K, ref[g].K),
                            rel_diff(b21.L.transpose(), ref[g].L),
                            rel_diff(b21.K.transpose(), ref[g].K)});
        res.asym = std::max({res.asym, rel_diff(b21.L.transpose(), b12.L),
                             rel_diff(b21.K.transpose(), b12.K)});
    }
    return res;
}

}  // namespace

TEST_CASE("static probe (experiment)", "[kernels][.slow][fold]") {
    if (std::getenv("SBEM_PROBE") == nullptr) {
        return;
    }
    // obtS at 90 degrees: source T = (A, B, C), test T' = (B, A, C').
    const Real deg = std::atof(std::getenv("SBEM_PROBE"));
    const FoldPair p = fold_pair(true, FoldShape{"obtS", 0.4, 0.8, -0.35, 0.6, 0.0, 0.0}, deg);
    const auto src = corners(p.mesh, 0);
    const auto tst = corners(p.mesh, 1);
    const QuadSet fine = dunavant_points(p.mesh, 0, 20, 64);
    std::ostringstream os;
    os << std::scientific << std::setprecision(2);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<Real> U(0.0, 1.0);
    Real worst = 0.0;
    for (int i = 0; i < 400; ++i) {
        Real s = U(rng);
        Real t = U(rng);
        if (s + t > 1.0) {
            s = 1.0 - s;
            t = 1.0 - t;
        }
        const Vec3 r = tst[0] + s * (tst[1] - tst[0]) + t * (tst[2] - tst[0]);
        const kernels::StaticIntegrals si = kernels::static_integrals(r, src[0], src[1], src[2]);
        Vec3 g = Vec3::Zero();
        Real i1 = 0.0;
        Real dmin = 1e300;
        for (std::size_t q = 0; q < fine.r.size(); ++q) {
            const Vec3 d = r - fine.r[q];  // grad'(1/R) = (r - r')/R^3
            const Real R = d.norm();
            g += fine.w[q] * d / (R * R * R);
            i1 += fine.w[q] / R;
            dmin = std::min(dmin, R);
        }
        const Real eg = (si.I_grad - g).norm() / g.norm();
        const Real e1 = std::abs(si.I_1R - i1) / std::abs(i1);
        const Real h = (src[1] - src[0]).norm();
        if (dmin > 0.05 * h && std::max(eg, e1) > 1e-10) {
            os << "\n s " << s << " t " << t << " dmin/h " << dmin / h << " I_grad " << eg
               << " I_1R " << e1;
        }
        if (dmin > 0.05 * h) {
            worst = std::max({worst, eg, e1});
        }
    }
    WARN("static probe worst " << worst << os.str());
}

TEST_CASE("element_blocks (full): fold sweep (experiment)", "[kernels][.slow][fold]") {
    const std::vector<NamedRegion> regs = all_regions();
    std::ostringstream table;
    table << std::scientific << std::setprecision(1);
    for (const bool edge : {true, false}) {
        for (const FoldShape& s : edge ? std::vector<FoldShape>(kEdgeShapes.begin(), kEdgeShapes.end()) : std::vector<FoldShape>(kVertexShapes.begin(), kVertexShapes.end())) {
            for (const Real deg : {30.0, 45.0, 60.0, 75.0, 89.0, 90.0, 120.0, 179.0}) {
                const FoldPair p = fold_pair(edge, s, deg);
                const char* filt = std::getenv("SBEM_SWEEP");
                if (filt != nullptr && p.name.find(filt) == std::string::npos) {
                    continue;
                }
                const RwgSpace space(p.mesh);
                const std::vector<Blocks> ref = relative_reference(space, p.t1, p.t2, regs, 28);
                table << "\n" << p.name << ":";
                if (std::getenv("SBEM_REF") != nullptr) {
                    const std::vector<Blocks> r36 = relative_reference(space, p.t1, p.t2, regs, 36);
                    Real e = 0.0;
                    for (std::size_t g = 0; g < regs.size(); ++g) {
                        e = std::max({e, rel_diff(ref[g].L, r36[g].L), rel_diff(ref[g].K, r36[g].K)});
                    }
                    table << " ref28-36 " << e;
                    const std::vector<Blocks> rs = relative_reference(space, p.t2, p.t1, regs, 36);
                    Real es = 0.0;
                    for (std::size_t g = 0; g < regs.size(); ++g) {
                        es = std::max({es, rel_diff(rs[g].L.transpose(), r36[g].L),
                                       rel_diff(rs[g].K.transpose(), r36[g].K)});
                    }
                    table << " refswap " << es;
                }
                {
                    OperatorOptions old;
                    old.fold_adaptive = false;
                    table << " old " << fold_errors(p, regs, ref, old).err;
                }
                if (std::getenv("SBEM_TIME") != nullptr) {
                    OperatorOptions old;
                    old.fold_adaptive = false;
                    Block L;
                    Block K;
                    Real us[2] = {0.0, 0.0};
                    for (int w = 0; w < 2; ++w) {
                        const OperatorOptions o = w == 0 ? old : OperatorOptions{};
                        const auto t0 = std::chrono::steady_clock::now();
                        for (int it = 0; it < 200; ++it) {
                            element_blocks(space, p.t1, p.t2, regs[1].p, o, L, K);
                            element_blocks(space, p.t2, p.t1, regs[1].p, o, L, K);
                        }
                        us[w] = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0)
                                    .count() *
                                1e6 / 400.0;
                    }
                    table << " t_old " << us[0] << " t_new " << us[1] << " x" << us[1] / us[0];
                }
                if (std::getenv("SBEM_DETAIL") != nullptr) {
                    OperatorOptions dopt;
                    dopt.outer_grading_levels = std::atoi(std::getenv("SBEM_DETAIL"));
                    if (std::getenv("SBEM_DSING") != nullptr) dopt.quad_degree_sing = std::atoi(std::getenv("SBEM_DSING"));
                    if (std::getenv("SBEM_DOLD") != nullptr) dopt.fold_adaptive = false;
                    for (std::size_t g = 0; g < regs.size(); ++g) {
                        const Blocks b12 = blocks(space, p.t1, p.t2, regs[g].p, dopt);
                        const Blocks b21 = blocks(space, p.t2, p.t1, regs[g].p, dopt);
                        table << " [" << regs[g].name << " L " << rel_diff(b12.L, ref[g].L) << "/"
                              << rel_diff(b21.L.transpose(), ref[g].L) << " K "
                              << rel_diff(b12.K, ref[g].K) << "/"
                              << rel_diff(b21.K.transpose(), ref[g].K) << "]";
                    }
                }
                for (const int level : {4, 6}) {
                    OperatorOptions opt;
                    opt.outer_grading_levels = level;
                    const GradedResult r = fold_errors(p, regs, ref, opt);
                    table << "  l" << level << " " << r.err << " (asym " << r.asym << ")";
                }
                {
                    OperatorOptions opt;
                    opt.outer_grading_levels = 6;
                    opt.quad_degree_sing = 19;
                    const GradedResult r = fold_errors(p, regs, ref, opt);
                    table << "  l6s19 " << r.err;
                    opt.fold_adaptive = false;
                    table << "  old-l6s19 " << fold_errors(p, regs, ref, opt).err;
                }
            }
        }
    }
    WARN("fold sweep" << table.str());
}
