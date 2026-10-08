#include "specklebem/formulation/formulation.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace specklebem;
using Catch::Matchers::WithinAbs;

TEST_CASE("Table 1 combination weights", "[formulation]") {
    const Complex eta1(1.0, 0.0), eta2(0.25, 0.0);

    auto pm = formulation::make_formulation(formulation::Kind::PMCHWT)->weights(eta1, eta2);
    CHECK_THAT(pm.a1.real(), WithinAbs(1.0, 1e-14));
    CHECK_THAT(pm.b2.real(), WithinAbs(4.0, 1e-14));

    auto ic = formulation::make_formulation(formulation::Kind::ICTF)->weights(eta1, eta2);
    CHECK_THAT(ic.a1.real(), WithinAbs(2.0 / 1.25, 1e-14));
    CHECK_THAT(ic.b1.real(), WithinAbs(1.25 / 2.0, 1e-14));
    // a_i * b_i == 1 for ICTF and PMCHWT
    CHECK_THAT((ic.a1 * ic.b1).real(), WithinAbs(1.0, 1e-14));
    CHECK_THAT((pm.a2 * pm.b2).real(), WithinAbs(1.0, 1e-14));

    auto mc = formulation::make_formulation(formulation::Kind::MCTF)->weights(eta1, eta2);
    CHECK_THAT(mc.b1.real(), WithinAbs(0.25, 1e-14));
    CHECK_THAT(mc.b2.real(), WithinAbs(1.0, 1e-14));
}

TEST_CASE("material-based recommendation", "[formulation]") {
    CHECK(formulation::recommend({18.478, -0.606}).diagonal_preconditioner == false);
    CHECK(formulation::recommend({-9.794, -0.313}).diagonal_preconditioner == true);
}
