/// @file test_pml_rough_ag.cpp
/// Unit tests of the 7c / 7d support code (WP-PML4b, tests/support/pml_rough_ag_support.hpp): the
/// docs/05 box-check metric on synthetic far fields (fixed seed), the far-field file round trip,
/// the central crop of the height map that keeps the surface of the L x 1.2 run on the common area,
/// and the sizes and materials of the 7c / 7d cases (Ag, and the rough Si control of 7d). No solve.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

#include "pml_rough_ag_support.hpp"

using namespace pml_rough_ag;
using Catch::Approx;

namespace {

/// Synthetic far field on hemisphere_grid(-1, 8, 16) and the cuts, intensities in [0.1, 1].
FarField synthetic(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(0.1, 1.0);
    FarField f;
    f.tag = "syn";
    f.window = {2.0e-6, 2.5e-6};
    f.n_theta = 8;
    f.n_phi = 16;
    f.hemi.resize(8 * 16);
    for (Index i = 0; i < f.hemi.size(); ++i) f.hemi(i) = u(rng);
    f.cut_xz.resize(box_validity::kCut);
    f.cut_yz.resize(box_validity::kCut);
    for (Index i = 0; i < box_validity::kCut; ++i) {
        f.cut_xz(i) = u(rng);
        f.cut_yz(i) = u(rng);
    }
    f.power = 1.3e-15;
    f.eta1 = 376.73;
    const post::SphereGrid g = post::hemisphere_grid(-1.0, 8, 16);
    f.p_refl = g.weights.dot(f.hemi) / (2.0 * f.eta1);
    return f;
}

}  // namespace

TEST_CASE("pml rough Ag: metric of identical, scaled and locally perturbed far fields",
          "[pml_rough_ag]") {
    const FarField r = synthetic(20261015);
    const Metrics same = compare(r, r);
    CHECK(same.l2 == 0.0);
    CHECK(same.l2_off30 == 0.0);
    CHECK(same.eps_xz == 0.0);
    CHECK(same.eps_yz == 0.0);
    CHECK(same.dR == 0.0);

    // A uniform scaling by 1 + e gives l2 = e, dR = e, eps_rr = e rms(r) / max(r).
    const Real e = 0.01;
    FarField t = r;
    t.hemi *= 1.0 + e;
    t.cut_xz *= 1.0 + e;
    t.cut_yz *= 1.0 + e;
    t.p_refl *= 1.0 + e;
    const Metrics m = compare(t, r);
    CHECK(m.l2 == Approx(e).epsilon(1e-10));
    CHECK(m.l2_off30 == Approx(e).epsilon(1e-10));
    CHECK(m.dR == Approx(e).epsilon(1e-10));
    const Real rms_xz = std::sqrt(r.cut_xz.squaredNorm() / static_cast<Real>(r.cut_xz.size()));
    CHECK(m.eps_xz == Approx(e * rms_xz / r.cut_xz.maxCoeff()).epsilon(1e-10));

    // One direction perturbed: the solid-angle weighted formula; a row below 30 deg does not
    // enter l2_off30, a row above does.
    const post::SphereGrid g = post::hemisphere_grid(-1.0, 8, 16);
    const Real den = g.weights.dot(r.hemi.cwiseProduct(r.hemi));
    for (const Index row : {Index{0}, Index{7 * 16 + 3}}) {
        FarField p = r;
        p.hemi(row) += 0.2;
        const Metrics mp = compare(p, r);
        CHECK(mp.l2 == Approx(std::sqrt(g.weights(row) * 0.04 / den)).epsilon(1e-12));
        if (g.theta(row) < 30.0 * kDeg)
            CHECK(mp.l2_off30 == 0.0);
        else
            CHECK(mp.l2_off30 > mp.l2);
        CHECK(mp.eps_xz == 0.0);
        CHECK(mp.dR == 0.0);  // p_refl is the stored value, not recomputed
    }

    // Invalid comparisons.
    FarField other = r;
    other.n_phi = 32;
    CHECK_THROWS_AS(compare(other, r), std::invalid_argument);
    FarField zero = r;
    zero.hemi.setZero();
    CHECK_THROWS_AS(compare(r, zero), std::invalid_argument);
    CHECK_THROWS_AS(eps_rr(r.cut_xz, VectorXr::Zero(r.cut_xz.size())), std::invalid_argument);

    // DRC on the plane-of-incidence cut: theta = 0 is node 179 of the 0.5 deg cut grid.
    CHECK(drc_xz(r, 0.0) == r.cut_xz(179) / (2.0 * r.eta1 * r.power));
    CHECK(drc_xz(r, -20.0) == r.cut_xz(139) / (2.0 * r.eta1 * r.power));
    CHECK_THROWS_AS(drc_xz(r, 0.3), std::invalid_argument);
    CHECK_THROWS_AS(drc_xz(r, 95.0), std::invalid_argument);
}

TEST_CASE("pml rough Ag: far-field file round trip", "[pml_rough_ag]") {
    const FarField f = synthetic(7);
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::string path = core::path_to_utf8(dir / "specklebem_pml_rough_ag_test.ff");
    write_far_field(path, f);
    const FarField g = read_far_field(path);
    CHECK(g.tag == f.tag);
    CHECK(g.window.inner_half == f.window.inner_half);
    CHECK(g.window.outer_half == f.window.outer_half);
    CHECK(g.n_theta == f.n_theta);
    CHECK(g.n_phi == f.n_phi);
    CHECK(g.power == f.power);
    CHECK(g.eta1 == f.eta1);
    CHECK(g.p_refl == f.p_refl);
    CHECK((g.hemi.array() == f.hemi.array()).all());
    CHECK((g.cut_xz.array() == f.cut_xz.array()).all());
    CHECK((g.cut_yz.array() == f.cut_yz.array()).all());
    {
        std::ofstream bad(dir / "specklebem_pml_rough_ag_test.ff", std::ios::binary);
        bad << "pml_rough_ag_far_field 1\ntag x\nwindow 1 2\ngrid 8 16\npower 1\neta1 1\n";
    }
    CHECK_THROWS_AS(read_far_field(path), std::runtime_error);
    std::filesystem::remove(dir / "specklebem_pml_rough_ag_test.ff");
    CHECK_THROWS_AS(read_far_field(path), std::runtime_error);
}

TEST_CASE("pml rough Ag: central crop of the height map and the 7c / 7d case sizes",
          "[pml_rough_ag]") {
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = 2.0e-6;
    rp.mesh_size = 125e-9;
    rp.rms_roughness = 50e-9;
    rp.correlation_length = 250e-9;
    rp.seed = 3;
    const geometry::HeightMap m = geometry::generate_gaussian_height_map(rp);
    REQUIRE(m.z.rows() == 17);
    const geometry::HeightMap c = crop_map(m, 1.5e-6);
    REQUIRE(c.z.rows() == 13);
    CHECK(c.dx == m.dx);
    CHECK((c.z.array() == m.z.block(2, 2, 13, 13).array()).all());
    CHECK_THROWS_AS(crop_map(m, 1.625e-6), std::invalid_argument);  // odd cell difference
    CHECK_THROWS_AS(crop_map(m, 1.6e-6), std::invalid_argument);    // not whole cells
    CHECK_THROWS_AS(crop_map(m, 2.5e-6), std::invalid_argument);    // larger than the map

    // The 7c / 7d cases: requested zones on the grid, 2N = 6 M^2 - 4 M for M cells per axis
    // (M = 2 outer_half / h), and the L x 1.2 run shares the surface on the common area.
    const RoughCase u = u_case();
    const Size su = size_of(u);
    CHECK(su.unknowns == 6 * 62 * 62 - 4 * 62);
    CHECK(su.half[2] == Approx(2.875e-6).epsilon(1e-12));
    CHECK(size_of(pml_width_case()).unknowns == 6 * 70 * 70 - 4 * 70);
    CHECK(size_of(sigma_case()).unknowns == su.unknowns);
    CHECK(size_of(patch_case()).unknowns == 6 * 68 * 68 - 4 * 68);
    const geometry::HeightMap hu = height_map(u);
    const geometry::HeightMap hl = height_map(patch_case());
    const Index off = (hl.z.rows() - hu.z.rows()) / 2;
    REQUIRE(off == 3);
    CHECK((hu.z.array() == hl.z.block(off, off, hu.z.rows(), hu.z.rows()).array()).all());
    // Flat control (rms 0) on the same grid; the loss diagnostic changes Im(eps_r) only.
    RoughCase flat = u;
    flat.rms = 0.0;
    const geometry::HeightMap hf = height_map(flat);
    CHECK(hf.z.rows() == hu.z.rows());
    CHECK(hf.z.cwiseAbs().maxCoeff() == 0.0);
    CHECK(size_of(flat).unknowns == su.unknowns);
    RoughCase lossy = u;
    lossy.eps_imag = -6.0;
    CHECK(make_config(lossy).object.eps_r == Complex(make_config(u).object.eps_r.real(), -6.0));
    lossy.eps_imag = 0.5;
    CHECK_THROWS_AS(make_config(lossy), std::invalid_argument);
    // 7d Si control: the same zones, map and sizes, only the object material differs.
    const RoughCase si = si_u_case();
    CHECK(si.material == "si");
    CHECK(size_of(si).unknowns == su.unknowns);
    CHECK(size_of(si_patch_case()).unknowns == size_of(patch_case()).unknowns);
    CHECK((height_map(si).z.array() == hu.z.array()).all());
    CHECK(make_config(si).object.eps_r == material::silicon_500nm().eps_r);
    CHECK(make_config(u).object.eps_r == material::silver_500nm().eps_r);
    RoughCase bad_material = u;
    bad_material.material = "au";
    CHECK_THROWS_AS(make_config(bad_material), std::invalid_argument);
    RoughCase off_grid = u;
    off_grid.gap = 0.55e-6;
    CHECK_THROWS_AS(size_of(off_grid), std::invalid_argument);
}
