#include "specklebem/kernels/green.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace specklebem;
using Catch::Matchers::WithinRel;

TEST_CASE("Green's function magnitude and outgoing phase", "[kernels]") {
    const Complex k(2 * constants::pi / 500e-9, 0.0);
    const Vec3 r(1e-6, 0, 0), rp(0, 0, 0);
    const Complex G = kernels::green(r, rp, k);
    CHECK_THAT(std::abs(G), WithinRel(1.0 / (4 * constants::pi * 1e-6), 1e-12));
    // exp(-jkR): phase should be -kR modulo 2 pi
    const Real expected = std::fmod(-k.real() * 1e-6, 2 * constants::pi);
    const Real got = std::arg(G);
    CHECK_THAT(std::cos(got - expected), WithinRel(1.0, 1e-9));
}

TEST_CASE("gradient of Green's function matches finite differences", "[kernels]") {
    const Complex k(2 * constants::pi / 500e-9, -1e5);
    const Vec3 r(0.7e-6, -0.2e-6, 0.4e-6), rp(0.1e-6, 0.3e-6, -0.1e-6);
    const Real h = 1e-10;
    const Vec3c g = kernels::grad_green(r, rp, k);
    for (int d = 0; d < 3; ++d) {
        Vec3 e = Vec3::Zero();
        e(d) = h;
        const Complex fd = (kernels::green(r + e, rp, k) - kernels::green(r - e, rp, k)) / (2 * h);
        CHECK(std::abs(g(d) - fd) < 1e-5 * std::abs(fd));
    }
}
