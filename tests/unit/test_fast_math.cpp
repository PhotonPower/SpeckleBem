/// @file test_fast_math.cpp
/// Branch-free elementary functions of the plain kernel (kernels/fast_math.hpp, WP-P2) against
/// the C library, and element_blocks with OperatorOptions::fast_plain_kernel against the
/// C-library arithmetic.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/fast_math.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/kernels/singularity.hpp"
#include "specklebem/material/material.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace specklebem;
using kernels::fastmath::fast_exp_nonpositive;
using kernels::fastmath::fast_sincos;

namespace {

constexpr Real kUlp1 = std::numeric_limits<Real>::epsilon();  // 2.2e-16, ulp of 1

kernels::RegionParams region_of(const material::Material& m, Real lambda) {
    const Real omega = 2.0 * constants::pi * constants::c0 / lambda;
    return {m.wavenumber(omega), m.wave_impedance(omega), omega, constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

Real rel_diff(const Eigen::Matrix<Complex, 3, 3>& a, const Eigen::Matrix<Complex, 3, 3>& b) {
    return (a - b).norm() / b.norm();
}

}  // namespace

TEST_CASE("fast_sincos agrees with std::sin and std::cos", "[kernels]") {
    std::mt19937_64 rng(20261009);
    for (const Real range : {0.78, 10.0, 1e3, 1e5, kernels::fastmath::kFastSincosMax}) {
        std::uniform_real_distribution<Real> dist(-range, range);
        Real worst_abs = 0.0;
        for (int i = 0; i < 40000; ++i) {
            const Real x = dist(rng);
            Real s = 0.0;
            Real c = 0.0;
            fast_sincos(x, s, c);
            worst_abs = std::max({worst_abs, std::abs(s - std::sin(x)), std::abs(c - std::cos(x))});
        }
        WARN("|x| <= " << range << ": worst absolute error " << worst_abs);
        // 2 ulp of the unit amplitude (both the reference and the reduction contribute).
        CHECK(worst_abs <= 2.0 * kUlp1);
    }
    // Relative accuracy on the reduced interval (no quadrant change).
    std::uniform_real_distribution<Real> small(-0.78, 0.78);
    Real worst_rel = 0.0;
    for (int i = 0; i < 40000; ++i) {
        const Real x = small(rng);
        Real s = 0.0;
        Real c = 0.0;
        fast_sincos(x, s, c);
        worst_rel = std::max({worst_rel, std::abs(s - std::sin(x)) / std::abs(std::sin(x)),
                              std::abs(c - std::cos(x)) / std::abs(std::cos(x))});
    }
    INFO("|x| <= 0.78: worst relative error " << worst_rel);
    CHECK(worst_rel <= 2.0 * kUlp1);
    // Exact values and quadrant boundaries.
    Real s = 1.0;
    Real c = 0.0;
    fast_sincos(0.0, s, c);
    CHECK(s == 0.0);
    CHECK(c == 1.0);
    for (int q = -8; q <= 8; ++q) {
        const Real x = q * 0.5 * constants::pi;
        fast_sincos(x, s, c);
        CHECK(std::abs(s - std::sin(x)) <= 2.0 * kUlp1);
        CHECK(std::abs(c - std::cos(x)) <= 2.0 * kUlp1);
    }
}

TEST_CASE("fast_exp_nonpositive agrees with std::exp", "[kernels]") {
    std::mt19937_64 rng(20261010);
    for (const Real range : {0.35, 5.0, 100.0, 708.0}) {
        std::uniform_real_distribution<Real> dist(-range, 0.0);
        Real worst = 0.0;
        for (int i = 0; i < 40000; ++i) {
            const Real x = dist(rng);
            const Real ref = std::exp(x);
            worst = std::max(worst, std::abs(fast_exp_nonpositive(x) - ref) / ref);
        }
        WARN("-" << range << " <= x <= 0: worst relative error " << worst);
        CHECK(worst <= 2.0 * kUlp1);
    }
    CHECK(fast_exp_nonpositive(0.0) == 1.0);
    CHECK(fast_exp_nonpositive(-0.0) == 1.0);
    CHECK(fast_exp_nonpositive(-708.5) == 0.0);
    CHECK(fast_exp_nonpositive(-1e300) == 0.0);
    CHECK(fast_exp_nonpositive(-std::numeric_limits<Real>::infinity()) == 0.0);
    CHECK(std::isnan(fast_exp_nonpositive(std::numeric_limits<Real>::quiet_NaN())));
    CHECK(std::abs(fast_exp_nonpositive(-708.0) - std::exp(-708.0)) <=
          2.0 * kUlp1 * std::exp(-708.0));
}

TEST_CASE("element_blocks: fast plain kernel agrees with the C-library arithmetic", "[kernels]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 2);
    const basis::RwgSpace space(mesh);
    const Real lambda = 500e-9;
    material::Material glass;
    glass.eps_r = Complex(2.25, 0.0);
    const std::vector<kernels::RegionParams> regions = {
        region_of(material::vacuum(), lambda), region_of(glass, lambda),
        region_of(material::silicon_500nm(), lambda), region_of(material::silver_500nm(), lambda)};
    kernels::OperatorOptions fast;
    kernels::OperatorOptions libm;
    libm.fast_plain_kernel = false;
    const Index F = mesh.num_triangles();
    Real worst = 0.0;
    std::size_t pairs = 0;
    for (Index t = 0; t < F; t += 7) {
        for (Index s = 0; s < F; s += 5) {
            const kernels::Proximity p = kernels::classify(mesh, t, s);
            if (p != kernels::Proximity::near && p != kernels::Proximity::far)
                continue;
            ++pairs;
            for (const kernels::RegionParams& reg : regions) {
                Eigen::Matrix<Complex, 3, 3> L1;
                Eigen::Matrix<Complex, 3, 3> K1;
                Eigen::Matrix<Complex, 3, 3> L2;
                Eigen::Matrix<Complex, 3, 3> K2;
                kernels::element_blocks(space, t, s, reg, fast, L1, K1);
                kernels::element_blocks(space, t, s, reg, libm, L2, K2);
                worst = std::max({worst, rel_diff(L1, L2), rel_diff(K1, K2)});
            }
        }
    }
    WARN(pairs << " near/far pairs, 4 materials: worst relative difference " << worst);
    CHECK(pairs > 300);
    CHECK(worst <= 1e-14);
    // Touching pairs never use the fast kernel (bitwise).
    Eigen::Matrix<Complex, 3, 3> L1;
    Eigen::Matrix<Complex, 3, 3> K1;
    Eigen::Matrix<Complex, 3, 3> L2;
    Eigen::Matrix<Complex, 3, 3> K2;
    kernels::element_blocks(space, 0, 0, regions[2], fast, L1, K1);
    kernels::element_blocks(space, 0, 0, regions[2], libm, L2, K2);
    CHECK(L1 == L2);
    CHECK(K1 == K2);
}

TEST_CASE("element_blocks: fast plain kernel falls back outside its argument range", "[kernels]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 1);
    const basis::RwgSpace space(mesh);
    const Index F = mesh.num_triangles();
    Index s_far = -1;
    for (Index s = F - 1; s > 0; --s) {
        if (kernels::classify(mesh, 0, s) == kernels::Proximity::far) {
            s_far = s;
            break;
        }
    }
    REQUIRE(s_far > 0);
    kernels::OperatorOptions fast;
    fast.target_accuracy = 0.0;  // fixed degree: the selection does not depend on k here
    kernels::OperatorOptions libm = fast;
    libm.fast_plain_kernel = false;
    const Real omega = 2.0 * constants::pi * constants::c0 / 500e-9;
    const auto region = [&](Complex k) {
        return kernels::RegionParams{k, Complex(constants::eta0, 0.0), omega,
                                     Complex(constants::eps0, 0.0), Complex(constants::mu0, 0.0)};
    };
    // Gain medium (Im k > 0) and |Re k| R beyond 1e6: the C library, bitwise.
    for (const Complex k : {Complex(1.2e7, 3e5), Complex(3e12, -1e6)}) {
        Eigen::Matrix<Complex, 3, 3> L1;
        Eigen::Matrix<Complex, 3, 3> K1;
        Eigen::Matrix<Complex, 3, 3> L2;
        Eigen::Matrix<Complex, 3, 3> K2;
        kernels::element_blocks(space, 0, s_far, region(k), fast, L1, K1);
        kernels::element_blocks(space, 0, s_far, region(k), libm, L2, K2);
        CHECK(L1 == L2);
        CHECK(K1 == K2);
    }
}
