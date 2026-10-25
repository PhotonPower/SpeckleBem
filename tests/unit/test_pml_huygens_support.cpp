/// @file test_pml_huygens_support.cpp
/// Unit tests of the WP-PML6b support code (tests/support/pml_huygens_support.hpp): the seed
/// parameterisation of the wide-gap cases (zones, tags, the shared height map of the L x 1.2
/// partner, two different surfaces), the flank scaling of a window and the 6b windows, the cone
/// views of the far-field comparison and the acceptance gates on synthetic far fields (fixed
/// seeds). No solve.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "pml_huygens_support.hpp"

using namespace pml_huygens;
using Catch::Approx;

namespace {

/// Synthetic far field on hemisphere_grid(-1, 8, 16) (GL nodes at 1.8 ... 88.2 deg: three in the
/// cone theta <= 30 deg, one in 60-75 deg, two in 75-90 deg) and the cuts, intensities in [0.1, 1].
FarField synthetic(std::uint64_t seed, const std::string& tag = "syn") {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(0.1, 1.0);
    FarField f;
    f.tag = tag;
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

/// f with |F|^2 scaled by (1 + e) on the rows with theta in [lo, hi] deg (p_refl updated).
FarField scaled_band(const FarField& f, Real e, Real lo_deg, Real hi_deg, const std::string& tag) {
    FarField t = f;
    t.tag = tag;
    const post::SphereGrid g = post::hemisphere_grid(-1.0, f.n_theta, f.n_phi);
    const Real deg = constants::pi / 180.0;
    for (Index i = 0; i < t.hemi.size(); ++i)
        if (g.theta(i) >= lo_deg * deg && g.theta(i) <= hi_deg * deg)
            t.hemi(i) *= 1.0 + e;
    t.p_refl = g.weights.dot(t.hemi) / (2.0 * t.eta1);
    return t;
}

}  // namespace

TEST_CASE("pml huygens: wide-gap cases per seed share the zones, the map crop and the tags",
          "[pml_huygens]") {
    REQUIRE(kSeeds.size() == 2);
    CHECK(kSeeds[0] == 1);
    CHECK(kSeeds[1] == 2);
    CHECK(seed_suffix(1).empty());
    CHECK(seed_suffix(2) == "_s2");
    // Seed 1 keeps the WP-PML6 cases (tags of the record, the WP-PML4b surface).
    const RoughCase u1 = wide_gap3_case();
    const RoughCase l1 = wide_gap3_patch_case();
    CHECK(u1.tag == "ugap3");
    CHECK(l1.tag == "l12gap3");
    CHECK(u1.seed == pml_rough_ag::u_case().seed);
    CHECK(wide_gap3_case(1).tag == "ugap3");
    const RoughCase u2 = wide_gap3_case(2);
    const RoughCase l2 = wide_gap3_patch_case(2);
    CHECK(u2.tag == "ugap3_s2");
    CHECK(l2.tag == "l12gap3_s2");
    CHECK(u2.seed == 2);
    CHECK(l2.seed == 2);
    for (const RoughCase* c : {&u1, &u2}) {
        CHECK(c->patch == 3.75e-6);
        CHECK(c->gap == 3.0e-6);
    }
    for (const RoughCase* c : {&l1, &l2}) {
        CHECK(c->patch == Approx(4.5e-6).epsilon(1e-14));
        CHECK(c->gap == 2.625e-6);
    }
    // Everything but tag and seed equals seed 1; pml_start 5.375 um on both partners.
    for (const auto& [a, b] : {std::pair{&u1, &u2}, std::pair{&l1, &l2}}) {
        CHECK(a->map_patch == b->map_patch);
        CHECK(a->taper == b->taper);
        CHECK(a->pml_width == b->pml_width);
        CHECK(a->mesh_size == b->mesh_size);
        CHECK(a->waist == b->waist);
        CHECK(a->rms == b->rms);
        CHECK(a->correlation_length == b->correlation_length);
        CHECK(a->material == b->material);
        CHECK(0.5 * b->patch + b->taper + b->gap == Approx(5.375e-6).epsilon(1e-14));
    }
    // The map: U is the central crop of the L x 1.2 map of the same seed; the seeds differ.
    const geometry::HeightMap mu1 = pml_rough_ag::height_map(u1);
    const geometry::HeightMap ml1 = pml_rough_ag::height_map(l1);
    const geometry::HeightMap mu2 = pml_rough_ag::height_map(u2);
    const geometry::HeightMap ml2 = pml_rough_ag::height_map(l2);
    REQUIRE(mu1.z.rows() == mu2.z.rows());
    REQUIRE(ml1.z.rows() == ml2.z.rows());
    const Index off = (ml1.z.rows() - mu1.z.rows()) / 2;
    REQUIRE(off > 0);
    CHECK((ml1.z.block(off, off, mu1.z.rows(), mu1.z.cols()) - mu1.z).cwiseAbs().maxCoeff() == 0.0);
    CHECK((ml2.z.block(off, off, mu2.z.rows(), mu2.z.cols()) - mu2.z).cwiseAbs().maxCoeff() == 0.0);
    // Two independent surfaces of the same statistics: rms of the difference ~ sqrt(2) sigma.
    const Real n = static_cast<Real>(mu1.z.size());
    const Real rms_diff = std::sqrt((mu1.z - mu2.z).squaredNorm() / n);
    CHECK(rms_diff > 0.5 * u1.rms);
    CHECK(mu2.rms() == Approx(u2.rms).epsilon(0.5));
}

TEST_CASE("pml huygens: flank scaling and the 6b windows of the wide-gap cases", "[pml_huygens]") {
    const post::CurrentWindow w{2.0e-6, 3.0e-6};
    const post::CurrentWindow a = flank_scaled(w, 0.7);
    CHECK(a.inner_half == 2.0e-6);
    CHECK(a.outer_half == Approx(2.7e-6).epsilon(1e-14));
    CHECK(flank_scaled(w, 1.3).outer_half == Approx(3.3e-6).epsilon(1e-14));
    CHECK(flank_scaled(w, 1.0).outer_half == w.outer_half);
    CHECK_THROWS_AS(flank_scaled(w, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(flank_scaled(w, -1.0), std::invalid_argument);
    CHECK_THROWS_AS(flank_scaled(w, std::nan("")), std::invalid_argument);
    CHECK_THROWS_AS(flank_scaled(post::CurrentWindow{2.0e-6, 2.0e-6}, 0.7), std::invalid_argument);

    CHECK(kSixBDelta == 4.0 * kLambda);
    // ugap3: taper end 2.375 um, pml_start 5.375 um (U's own window: no absolute windows).
    const Real te = 2.375e-6;
    const SixBWindows u = six_b_windows({te, 5.375e-6}, te);
    CHECK(u.def.inner_half == te);
    CHECK(u.def.outer_half == 5.375e-6);
    CHECK(u.def_f07.outer_half == Approx(te + 0.7 * 3.0e-6).epsilon(1e-14));
    CHECK(u.d4.inner_half == te);
    CHECK(u.d4.outer_half == Approx(4.375e-6).epsilon(1e-14));
    CHECK(u.d4_f07.inner_half == te);
    CHECK(u.d4_f07.outer_half == Approx(3.775e-6).epsilon(1e-14));
    CHECK(u.d4_f13.outer_half == Approx(4.975e-6).epsilon(1e-14));
    CHECK_FALSE(u.abs.has_value());
    CHECK_FALSE(u.abs_d4.has_value());
    // l12gap3: taper end 2.75 um; x 1.3 ends at 5.35 um < pml_start; U's windows on it.
    const SixBWindows l = six_b_windows({2.75e-6, 5.375e-6}, te);
    CHECK(l.d4.outer_half == Approx(4.75e-6).epsilon(1e-14));
    CHECK(l.d4_f13.outer_half == Approx(5.35e-6).epsilon(1e-14));
    CHECK(l.d4_f13.outer_half < l.def.outer_half);
    REQUIRE(l.abs.has_value());
    REQUIRE(l.abs_d4.has_value());
    CHECK(l.abs->inner_half == te);
    CHECK(l.abs->outer_half == 5.375e-6);
    CHECK(l.abs_d4->outer_half == Approx(4.375e-6).epsilon(1e-14));
    // A gap too narrow for the study (U's 0.5 um gap), an invalid window, U's taper end outside.
    CHECK_THROWS_AS(six_b_windows({te, 2.875e-6}, te), std::invalid_argument);
    CHECK_THROWS_AS(six_b_windows({te, te}, te), std::invalid_argument);
    CHECK_THROWS_AS(six_b_windows({2.75e-6, 5.375e-6}, 6.0e-6), std::invalid_argument);
}

TEST_CASE("pml huygens: cone views of identical, scaled and band-perturbed far fields",
          "[pml_huygens]") {
    const FarField r = synthetic(20261025);
    const ConeMetrics same = cone_compare(r, r);
    CHECK(same.all == 0.0);
    CHECK(same.cone == 0.0);
    CHECK(same.off == 0.0);
    CHECK(same.band_60_75 == 0.0);
    CHECK(same.band_75_90 == 0.0);
    CHECK(same.dR == 0.0);
    // A uniform scaling by 1 + e: every view e.
    const Real e = 3.0e-3;
    const ConeMetrics m = cone_compare(scaled_band(r, e, 0.0, 90.0, "t"), r);
    CHECK(m.all == Approx(e).epsilon(1e-10));
    CHECK(m.cone == Approx(e).epsilon(1e-10));
    CHECK(m.off == Approx(e).epsilon(1e-10));
    CHECK(m.band_60_75 == Approx(e).epsilon(1e-10));
    CHECK(m.band_75_90 == Approx(e).epsilon(1e-10));
    CHECK(m.dR == Approx(e).epsilon(1e-10));
    // A change outside the cone only (theta >= 40 deg): cone 0, off and all > 0; the off view
    // and the hemisphere view are the box-check metric's l2_off30 and l2.
    const FarField t = scaled_band(r, 0.05, 40.0, 90.0, "t40");
    const ConeMetrics o = cone_compare(t, r);
    CHECK(o.cone == 0.0);
    CHECK(o.off > 0.0);
    CHECK(o.all > 0.0);
    CHECK(o.all < o.off);
    const pml_rough_ag::Metrics bm = pml_rough_ag::compare(t, r);
    CHECK(o.all == Approx(bm.l2).epsilon(1e-13));
    CHECK(o.off == Approx(bm.l2_off30).epsilon(1e-13));
    CHECK(o.dR == Approx(bm.dR).epsilon(1e-13));
    CHECK(o.band_60_75 == Approx(0.05).epsilon(1e-10));
    CHECK(o.band_75_90 == Approx(0.05).epsilon(1e-10));
    // A change inside the cone only: off 0, the bands unchanged.
    const ConeMetrics c = cone_compare(scaled_band(r, -0.02, 0.0, 30.0, "t30"), r);
    CHECK(c.cone == Approx(0.02).epsilon(1e-10));
    CHECK(c.off == 0.0);
    CHECK(c.band_60_75 == 0.0);
    CHECK(c.band_75_90 == 0.0);
    // Different grids are refused; the line carries the label and both tags.
    FarField other = r;
    other.n_phi = 8;
    CHECK_THROWS_AS(cone_compare(other, r), std::invalid_argument);
    const std::string line = cone_line("label", t, r, o);
    CHECK(line.rfind("CONE | label", 0) == 0);
    CHECK(line.find("t40") != std::string::npos);
    CHECK(line.find("<=30 0.0000e+00") != std::string::npos);
}

TEST_CASE("pml huygens: 6b gates on synthetic far fields", "[pml_huygens]") {
    // Window sensitivities and 7d in the cone set by scaling a common base on theta <= 30 deg.
    const FarField base = synthetic(20261026);
    const auto make = [&](Real sens_c, Real sens_h, Real d7) {
        std::map<std::string, FarField> far;
        for (const std::string& t : {std::string("u"), std::string("l")}) {
            const Real s7 = t == "l" ? d7 : 0.0;  // L x 1.2 differs from U by d7 in the cone
            for (const char* ref : {".C", ".CD4", ".HD4"})
                far[t + ref] = scaled_band(base, s7, 0.0, 30.0, t + ref);
            far[t + ".CD4f07"] =
                scaled_band(base, (1.0 + s7) * (1.0 + sens_c) - 1.0, 0.0, 30.0, t + ".CD4f07");
            far[t + ".CD4f13"] = scaled_band(base, (1.0 + s7) * (1.0 - 0.5 * sens_c) - 1.0, 0.0,
                                             30.0, t + ".CD4f13");
            far[t + ".HD4f07"] =
                scaled_band(base, (1.0 + s7) * (1.0 - sens_h) - 1.0, 0.0, 30.0, t + ".HD4f07");
            far[t + ".HD4f13"] = scaled_band(base, (1.0 + s7) * (1.0 + 0.5 * sens_h) - 1.0, 0.0,
                                             30.0, t + ".HD4f13");
        }
        return far;
    };
    const SixBGates g = six_b_gates(make(4e-4, 2e-4, 2.3e-2), "u", "l");
    CHECK(g.sens_currents == Approx(4e-4).epsilon(1e-8));
    CHECK(g.sens_huygens == Approx(2e-4).epsilon(1e-8));
    CHECK(g.d7_currents == Approx(2.3e-2).epsilon(1e-8));
    CHECK(g.d7_currents_d4 == Approx(2.3e-2).epsilon(1e-8));
    CHECK(g.d7_huygens == Approx(2.3e-2).epsilon(1e-8));
    CHECK(g.sensitivity_met());
    CHECK(g.ratio_met());
    CHECK(gate_line(2, g).find("seed 2") != std::string::npos);
    // A sensitivity above 1e-3 fails the first gate (and here the ratio, 2.3e-2 < 10 x 2.5e-3).
    const SixBGates h = six_b_gates(make(2.5e-3, 2e-4, 2.3e-2), "u", "l");
    CHECK_FALSE(h.sensitivity_met());
    CHECK_FALSE(h.ratio_met());
    CHECK(gate_line(1, h).find("NOT MET") != std::string::npos);
    // Small sensitivities but a 7d change below 10 x them fails the second gate only.
    const SixBGates k = six_b_gates(make(8e-4, 2e-4, 5e-3), "u", "l");
    CHECK(k.sensitivity_met());
    CHECK_FALSE(k.ratio_met());
    // A missing far field is an error.
    std::map<std::string, FarField> far = make(4e-4, 2e-4, 2.3e-2);
    far.erase("l.HD4");
    CHECK_THROWS_AS(six_b_gates(far, "u", "l"), std::invalid_argument);
}
