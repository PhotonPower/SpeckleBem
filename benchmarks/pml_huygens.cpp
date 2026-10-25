// Validations 7f and 7g of the PML route's Huygens-plane far field (ADR 0010, WP-PML6; docs/05
// "PML route", rows 7f / 7g; record benchmarks/results/pml_huygens.md). Cases and far fields are
// in tests/support/pml_huygens_support.hpp.
//
//  --study 7f  flat Ag and / or Si, 7a geometry at 0 deg: the Huygens far field (defaults z_h =
//              lambda, spacing lambda / 4, the current window) against the windowed currents
//              (L2 <= 1e-3, |dR| <= 1e-4), and its sensitivity to z_h x 1.5 and spacing / 1.5.
//  --study 7g  rough Ag U, L x 1.2 and the wide-gap partner (fast path by default): 7d with the
//              Huygens far field (< 0.5 %), z_h x 1.5, spacing / 1.5 and the plane window x 0.7 /
//              x 1.3 (< 0.1 % each), plus the current-window references and the gap currents.
//  --study si7d  rough Si U and L x 1.2 with the Huygens far field (the 7d control).
// Every solved case also evaluates the diagnostics z_h x 0.5 / x 2 (Hz05, Hz20) and the plane
// field without the PML source triangles (Hnopml, HuygensOptions::pml_sources = false). The
// diagnostic case ugap3 (--cases ugap3: U with a 3 um gap, pml_start 5.375 um) evaluates nested
// plane windows (inner taper_end, outer 2.875 / 3.75 / 4.5 / 5.375 um) of one plane and z_h x 1.5
// at the widest window: does the Huygens far field converge with the plane width? l12gap3 (L x 1.2
// with the gap 2.625 um, the same pml_start) is its 7d partner at the wide plane. On both, the
// diagnostic "extended plane" (pml_huygens.md; delta_windows) evaluates the window transitions
// Delta = 1, 2, 3, 4, 6 lambda from each case's taper end (and from U's on l12gap3): Huygens vs
// currents, z_h x 1.5 and 7d per Delta.
// Every result is one machine-readable line (ROW, PLANE, CMP), also appended to --summary FILE;
// --out DIR stores the far fields (pml_rough_ag::write_far_field, one file per evaluation).
//
// Usage: specklebem_pml_huygens --study 7f|7g|si7d [--material ag|si|both] [--solver dense|fast]
//            [--cases u,l12,ugap,ugap3,l12gap3] [--summary FILE] [--out DIR]
#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_huygens_support.hpp"

namespace {

using namespace pml_huygens;

struct Options {
    std::string study;
    std::string material = "both";
    std::optional<Solver> solver;
    std::vector<std::string> cases;
    std::string summary;
    std::string out;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--study") {
            o.study = value();
        } else if (a == "--material") {
            o.material = value();
        } else if (a == "--solver") {
            const std::string v = value();
            if (v == "dense")
                o.solver = Solver::Dense;
            else if (v == "fast")
                o.solver = Solver::Fast;
            else
                throw std::invalid_argument("--solver must be dense or fast");
        } else if (a == "--cases") {
            std::stringstream ss(value());
            std::string item;
            while (std::getline(ss, item, ',')) o.cases.push_back(item);
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--out") {
            o.out = value();
        } else {
            throw std::invalid_argument("unknown argument " + a);
        }
    }
    if (o.study != "7f" && o.study != "7g" && o.study != "si7d")
        throw std::invalid_argument("--study must be 7f, 7g or si7d");
    return o;
}

void emit(const std::string& summary, const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
    if (summary.empty())
        return;
    std::ofstream f(core::path_from_utf8(summary), std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + summary);
    f << line << '\n';
}

/// The far fields of one study, by tag.
struct Store {
    std::map<std::string, FarField> far;
    std::string out;
    const FarField& add(const FarField& f) {
        if (!out.empty()) {
            std::filesystem::create_directories(core::path_from_utf8(out));
            pml_rough_ag::write_far_field(out + "/" + f.tag + ".txt", f);
        }
        return far[f.tag] = f;
    }
    const FarField& at(const std::string& tag) const {
        const auto it = far.find(tag);
        if (it == far.end())
            throw std::logic_error("missing far field " + tag);
        return it->second;
    }
};

void compare(const Options& o, const Store& st, const std::string& label, const std::string& t,
             const std::string& r) {
    const FarField& ft = st.at(t);
    const FarField& fr = st.at(r);
    emit(o.summary, "CMP | " + label + " | " +
                        pml_rough_ag::compare_line(ft, fr, pml_rough_ag::compare(ft, fr)));
}

/// Huygens evaluation of a solved case: emits a PLANE line, stores the far field.
FarField huygens(const Options& o, Store& st, const Solved& s, const post::HuygensPlaneField& pf,
                 const std::string& tag) {
    const FarField f = radiate(pf, s, tag);
    emit(o.summary, plane_line(tag, pf, f));
    return st.add(f);
}

/// Solves a case and evaluates the standard set: currents (C), Huygens defaults (H), z_h x 1.5
/// (Hz15), spacing / 1.5 (Hsp); returns the solved case for further evaluations.
Solved standard(const Options& o, Store& st, const RoughCase& c, Solver solver) {
    std::printf("SOLVING | %s | %s\n", c.tag.c_str(), solver == Solver::Fast ? "fast" : "dense");
    std::fflush(stdout);
    Solved s = solve(c, solver);
    std::printf("%s\n", s.report.c_str());
    if (!s.converged)
        throw std::runtime_error(c.tag + ": the solve did not converge");
    const FarField& fc = st.add(currents_far_field(s, c.tag + ".C"));
    const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
    const FarField fh = huygens(o, st, s, pf, c.tag + ".H");
    emit(o.summary, row_line(s, fc, fh, pf));
    post::HuygensOptions z15;
    z15.z_h = 1.5 * kLambda;
    (void)huygens(o, st, s, plane_field(s, z15), c.tag + ".Hz15");
    post::HuygensOptions sp;
    sp.spacing = kLambda / 6.0;
    (void)huygens(o, st, s, plane_field(s, sp), c.tag + ".Hsp");
    // Diagnostics: plane distance x 0.5 / x 2 and the plane field without the PML sources.
    post::HuygensOptions z05;
    z05.z_h = 0.5 * kLambda;
    (void)huygens(o, st, s, plane_field(s, z05), c.tag + ".Hz05");
    post::HuygensOptions z20;
    z20.z_h = 2.0 * kLambda;
    (void)huygens(o, st, s, plane_field(s, z20), c.tag + ".Hz20");
    post::HuygensOptions nopml;
    nopml.pml_sources = false;
    (void)huygens(o, st, s, plane_field(s, nopml), c.tag + ".Hnopml");
    return s;
}

int run_7f(const Options& o) {
    Store st{{}, o.out};
    const Solver solver = o.solver.value_or(Solver::Dense);
    std::vector<std::string> mats;
    if (o.material == "both")
        mats = {"ag", "si"};
    else
        mats = {o.material};
    for (const std::string& m : mats) {
        const RoughCase c = flat_case(m);
        const Solved s = standard(o, st, c, solver);
        // Plane quadrature degree 10 instead of 6 (diagnostic).
        post::FieldOptions fo = s.sim->far_field_options(10, FarFieldSource::Huygens);
        (void)huygens(o, st, s, post::huygens_plane_field(s.solution(), fo), c.tag + ".Hdeg10");
        compare(o, st, "7f huygens vs currents", c.tag + ".H", c.tag + ".C");
        compare(o, st, "7f z_h x1.5", c.tag + ".Hz15", c.tag + ".H");
        compare(o, st, "7f spacing /1.5", c.tag + ".Hsp", c.tag + ".H");
        compare(o, st, "7f plane degree 10", c.tag + ".Hdeg10", c.tag + ".H");
        compare(o, st, "7f z_h x1.5 vs currents", c.tag + ".Hz15", c.tag + ".C");
        compare(o, st, "7f spacing /1.5 vs currents", c.tag + ".Hsp", c.tag + ".C");
    }
    return 0;
}

/// Window transitions Delta = d lambda of the diagnostic "extended plane" (pml_huygens.md): the
/// plane window (taper_end, taper_end + Delta) on the wide-gap cases, whose real part reaches
/// to pml_start = 5.375 um (the plane cannot cover the PML zone: exterior_scattered_field takes
/// real observation points over the real part only).
const std::vector<int> kDeltaLambdas = {1, 2, 3, 4, 6};

/// Diagnostic "extended plane" of a wide-gap case: per Delta the Huygens far field (z_h = lambda
/// and 1.5 lambda) and the windowed currents for the window (taper_end, taper_end + Delta), tags
/// <case>.HD<d>, .Hz15D<d>, .CD<d>; with `absolute` also U's absolute window (abs_inner,
/// abs_inner + Delta), tags .HabsD<d>, .CabsD<d>. Windows ending beyond pml_start are skipped.
void delta_windows(const Options& o, Store& st, const Solved& s, Real abs_inner, bool absolute) {
    const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
    post::HuygensOptions z15;
    z15.z_h = 1.5 * kLambda;
    const post::HuygensPlaneField pz = plane_field(s, z15);
    const Real end = pf.plane.window.outer_half;  // pml_start as the Simulation computes it
    const auto fits = [&](Real outer) { return outer <= end * (1.0 + 1e-12); };
    const auto window = [&](Real inner, Real outer) {
        return post::CurrentWindow{inner, std::min(outer, end)};
    };
    for (const int d : kDeltaLambdas) {
        const std::string sfx = "D" + std::to_string(d);
        const Real delta = static_cast<Real>(d) * kLambda;
        const Real a = s.half[1];
        if (fits(a + delta)) {
            const post::CurrentWindow w = window(a, a + delta);
            (void)huygens(o, st, s, post::huygens_reweight(pf, w), s.c.tag + ".H" + sfx);
            (void)huygens(o, st, s, post::huygens_reweight(pz, w), s.c.tag + ".Hz15" + sfx);
            (void)st.add(currents_far_field(s, s.c.tag + ".C" + sfx, w));
        }
        if (absolute && fits(abs_inner + delta)) {
            const post::CurrentWindow w = window(abs_inner, abs_inner + delta);
            (void)huygens(o, st, s, post::huygens_reweight(pf, w), s.c.tag + ".Habs" + sfx);
            (void)st.add(currents_far_field(s, s.c.tag + ".Cabs" + sfx, w));
        }
    }
}

int run_7g(const Options& o, bool si) {
    Store st{{}, o.out};
    const Solver solver = o.solver.value_or(Solver::Fast);
    std::vector<std::string> cases = o.cases;
    if (cases.empty())
        cases = si ? std::vector<std::string>{"si_u", "si_l12"}
                   : std::vector<std::string>{"u", "l12", "ugap"};
    const post::CurrentWindow u_window{pml_rough_ag::size_of(pml_rough_ag::u_case()).half[1],
                                       pml_rough_ag::size_of(pml_rough_ag::u_case()).half[2]};
    for (const std::string& name : cases) {
        RoughCase c;
        if (name == "u")
            c = pml_rough_ag::u_case();
        else if (name == "l12")
            c = pml_rough_ag::patch_case();
        else if (name == "ugap")
            c = wide_gap_case();
        else if (name == "ugap3")
            c = wide_gap3_case();
        else if (name == "l12gap3")
            c = wide_gap3_patch_case();
        else if (name == "si_u")
            c = pml_rough_ag::si_u_case();
        else if (name == "si_l12")
            c = pml_rough_ag::si_patch_case();
        else
            throw std::invalid_argument("unknown case " + name);
        const Solved s = standard(o, st, c, solver);
        if (name == "l12" || name == "si_l12") {
            // U's absolute plane window on the larger patch (a subset of the default grid).
            const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
            (void)huygens(o, st, s, post::huygens_reweight(pf, u_window), c.tag + ".Huwin");
            (void)st.add(currents_far_field(s, c.tag + ".Cuwin", u_window));
        }
        if (name == "ugap") {
            // Plane window x 0.7, x 1, x 1.3 of U's window: one plane at x 1.3, reweighted.
            post::HuygensOptions w13;
            w13.window = scaled(u_window, 1.3);
            const post::HuygensPlaneField pf = plane_field(s, w13);
            (void)huygens(o, st, s, pf, c.tag + ".Hw13");
            (void)huygens(o, st, s, post::huygens_reweight(pf, u_window), c.tag + ".Hw10");
            (void)huygens(o, st, s, post::huygens_reweight(pf, scaled(u_window, 0.7)),
                          c.tag + ".Hw07");
            (void)st.add(currents_far_field(s, c.tag + ".Cuwin", u_window));
        }
        if (name == "ugap3") {
            // Nested plane windows of the default plane (taper_end, pml_start = 5.375 um).
            const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
            const Real a = u_window.inner_half;
            for (const auto& [suffix, outer] : std::vector<std::pair<std::string, Real>>{
                     {".Hw2875", 2.875e-6}, {".Hw3750", 3.75e-6}, {".Hw4500", 4.5e-6}}) {
                (void)huygens(o, st, s, post::huygens_reweight(pf, post::CurrentWindow{a, outer}),
                              c.tag + suffix);
            }
            (void)st.add(currents_far_field(s, c.tag + ".Cuwin", u_window));
            (void)huygens(o, st, s,
                          post::huygens_reweight(
                              pf, post::CurrentWindow{2.75e-6, pf.plane.window.outer_half}),
                          c.tag + ".Hwl3");
        }
        if (name == "l12gap3") {
            // ugap3's absolute plane window (2.375, 5.375 um) on the larger patch.
            const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
            (void)huygens(
                o, st, s,
                post::huygens_reweight(
                    pf, post::CurrentWindow{u_window.inner_half, pf.plane.window.outer_half}),
                c.tag + ".Hwu3");
        }
        if (name == "ugap3" || name == "l12gap3")
            delta_windows(o, st, s, u_window.inner_half, name == "l12gap3");
    }
    const auto has = [&](const std::string& t) { return st.far.count(t) > 0; };
    const std::string u = si ? "si_u" : "u";
    const std::string l = si ? "si_l12" : "l12";
    if (has(u + ".H") && has(l + ".H")) {
        compare(o, st, "7d currents", l + ".C", u + ".C");
        compare(o, st, "7d huygens (7g)", l + ".H", u + ".H");
        compare(o, st, "7d huygens, U window", l + ".Huwin", u + ".H");
        compare(o, st, "7d currents, U window", l + ".Cuwin", u + ".C");
    }
    for (const std::string& t : {u, l, std::string("ugap"), std::string("ugap3")}) {
        if (!has(t + ".H"))
            continue;
        compare(o, st, "huygens vs currents", t + ".H", t + ".C");
        compare(o, st, "z_h x1.5", t + ".Hz15", t + ".H");
        compare(o, st, "spacing /1.5", t + ".Hsp", t + ".H");
        compare(o, st, "z_h x0.5 (diagnostic)", t + ".Hz05", t + ".H");
        compare(o, st, "z_h x2 (diagnostic)", t + ".Hz20", t + ".H");
        compare(o, st, "no PML sources (diagnostic)", t + ".Hnopml", t + ".H");
        compare(o, st, "z_h x0.5 vs currents (diagnostic)", t + ".Hz05", t + ".C");
    }
    if (has("ugap3.H")) {
        compare(o, st, "ugap3 plane outer 2.875 vs 5.375 um", "ugap3.Hw2875", "ugap3.H");
        compare(o, st, "ugap3 plane outer 3.75 vs 5.375 um", "ugap3.Hw3750", "ugap3.H");
        compare(o, st, "ugap3 plane outer 4.5 vs 5.375 um", "ugap3.Hw4500", "ugap3.H");
        compare(o, st, "ugap3 plane outer 3.75 vs 4.5 um", "ugap3.Hw3750", "ugap3.Hw4500");
        if (has("u.H")) {
            compare(o, st, "gap 3 um, huygens, U window", "ugap3.Hw2875", "u.H");
            compare(o, st, "gap 3 um, currents, U window", "ugap3.Cuwin", "u.C");
        }
        if (has("ugap.H"))
            compare(o, st, "gap 3 vs 1.375 um, huygens, ugap window", "ugap3.Hw3750", "ugap.H");
    }
    if (has("ugap3.H") && has("l12gap3.H")) {
        // 7d at the wide plane (pml_start 5.375 um on both patches).
        compare(o, st, "7d wide gap, currents", "l12gap3.C", "ugap3.C");
        compare(o, st, "7d wide gap, huygens", "l12gap3.H", "ugap3.H");
        compare(o, st, "7d wide gap, huygens, window 2.375/5.375", "l12gap3.Hwu3", "ugap3.H");
        compare(o, st, "7d wide gap, huygens, window 2.75/5.375", "l12gap3.H", "ugap3.Hwl3");
    }
    if (has("l12gap3.H")) {
        compare(o, st, "huygens vs currents", "l12gap3.H", "l12gap3.C");
        compare(o, st, "z_h x1.5", "l12gap3.Hz15", "l12gap3.H");
        compare(o, st, "z_h x0.5 (diagnostic)", "l12gap3.Hz05", "l12gap3.H");
        compare(o, st, "z_h x2 (diagnostic)", "l12gap3.Hz20", "l12gap3.H");
        compare(o, st, "no PML sources (diagnostic)", "l12gap3.Hnopml", "l12gap3.H");
    }
    for (const int d : kDeltaLambdas) {
        // Diagnostic "extended plane": per window transition Delta = d lambda.
        const std::string sfx = "D" + std::to_string(d);
        const std::string lab = " (Delta " + std::to_string(d) + " lambda)";
        for (const std::string& t : {std::string("ugap3"), std::string("l12gap3")}) {
            if (!has(t + ".H" + sfx))
                continue;
            compare(o, st, "huygens vs currents" + lab, t + ".H" + sfx, t + ".C" + sfx);
            compare(o, st, "z_h x1.5" + lab, t + ".Hz15" + sfx, t + ".H" + sfx);
        }
        if (has("ugap3.H" + sfx) && has("l12gap3.H" + sfx)) {
            compare(o, st, "7d wide gap, huygens, own windows" + lab, "l12gap3.H" + sfx,
                    "ugap3.H" + sfx);
            compare(o, st, "7d wide gap, currents, own windows" + lab, "l12gap3.C" + sfx,
                    "ugap3.C" + sfx);
        }
        if (has("ugap3.H" + sfx) && has("l12gap3.HabsD" + std::to_string(d))) {
            compare(o, st, "7d wide gap, huygens, U absolute window" + lab, "l12gap3.Habs" + sfx,
                    "ugap3.H" + sfx);
            compare(o, st, "7d wide gap, currents, U absolute window" + lab, "l12gap3.Cabs" + sfx,
                    "ugap3.C" + sfx);
        }
    }
    if (has("ugap.Hw10")) {
        compare(o, st, "window x0.7", "ugap.Hw07", "ugap.Hw10");
        compare(o, st, "window x1.3", "ugap.Hw13", "ugap.Hw10");
        if (has("u.H")) {
            compare(o, st, "gap 1.375 um, huygens, U window", "ugap.Hw10", "u.H");
            compare(o, st, "gap 1.375 um, currents, U window", "ugap.Cuwin", "u.C");
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.study == "7f")
            return run_7f(o);
        return run_7g(o, o.study == "si7d");
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pml_huygens: %s\n", ex.what());
        return 1;
    }
}
