#include <catch2/catch_test_macros.hpp>

#include "fields_test_support.hpp"

// Validation of the Stratton-Chu post-processing (WP10) with exact Mie surface currents
// projected onto RWG functions (J = n x H, M = -n x E on the R1 side, docs/06): near fields in
// both regions and the bistatic RCS (eps_rr of docs/05) converge to Mie, icosphere n = 3 -> 4.
// The unit test tests/unit/test_fields.cpp covers n = 2 -> 3 (dielectric only).

using namespace fields_test;

TEST_CASE("post validation: projected Mie currents, dielectric n = 1.5, icosphere n = 3, 4",
          "[validation][post]") {
    const material::Material mat = lossless_n15();
    const MieSetup s3(mat, 3);
    const MieSetup s4(mat, 4);
    const ProjectionErrors p3 = near_field_errors(s3);
    const ProjectionErrors p4 = near_field_errors(s4);
    const RcsErrors r3 = rcs_errors(s3);
    const RcsErrors r4 = rcs_errors(s4);
    print_errors("n=1.5", 3, p3, r3);
    print_errors("n=1.5", 4, p4, r4);
    // Measured (WP10): n = 3: E_ext 0.060, H_ext 0.048, E_int 0.065, H_int 0.082, RCS eps_rr
    // 0.020; n = 4: E_ext 0.015, H_ext 0.012, E_int 0.016, H_int 0.021, RCS eps_rr 0.005
    // (O(h^2) convergence).

    CHECK(p4.e_ext < p3.e_ext);
    CHECK(p4.h_ext < p3.h_ext);
    CHECK(p4.e_int < p3.e_int);
    CHECK(p4.h_int < p3.h_int);
    CHECK(p4.e_ext < 0.05);
    CHECK(p4.h_ext < 0.05);
    CHECK(p4.e_int < 0.05);
    CHECK(p4.h_int < 0.05);

    CHECK(r4.rms < r3.rms);
    CHECK(r4.rms < 0.05);
    CHECK(r4.forward < 0.05);
    CHECK(r4.backward < 0.05);
}

TEST_CASE("post validation: projected Mie currents, Ag, icosphere n = 3, 4", "[validation][post]") {
    const material::Material mat = material::silver_500nm();
    const MieSetup s3(mat, 3);
    const MieSetup s4(mat, 4);
    const ProjectionErrors p3 = near_field_errors(s3);
    const ProjectionErrors p4 = near_field_errors(s4);
    const RcsErrors r3 = rcs_errors(s3);
    const RcsErrors r4 = rcs_errors(s4);
    print_errors("Ag", 3, p3, r3);
    print_errors("Ag", 4, p4, r4);

    // Harder than the dielectric (skin depth ~ 25 nm, comparable to the n = 4 edge length);
    // only a monotone decrease is required. Measured (WP10): n = 3: E_ext 0.068, H_ext 0.017,
    // RCS eps_rr 0.014; n = 4: E_ext 0.017, H_ext 0.0042, RCS eps_rr 0.0035 (interior errors
    // ~ 1e-6 because the field at r = a/2 is ~ 10 skin depths deep).
    CHECK(p4.e_ext < p3.e_ext);
    CHECK(p4.h_ext < p3.h_ext);
    CHECK(p4.e_int < p3.e_int);
    CHECK(p4.h_int < p3.h_int);
    CHECK(r4.rms < r3.rms);
    CHECK(r4.forward < r3.forward);
    CHECK(r4.backward < r3.backward);
}
