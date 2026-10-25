// MLFMM scaling study (WP22b1, Phase 4 definition of done in docs/01: scaling exponent gamma of
// the time per solve vs N <= 1.5 for Si rough surfaces, <= 2.0 for Ag, over N = 5e4 ... 1e6).
//
// One case per process (a background command is limited in time, and the peak memory of a case
// should be its own): Gaussian rough surface (sigma = 50 nm, Lc = 500 nm, mesh 50 nm, seed 1) on
// an L x L patch with the ADR 0006 closing box of rough_surface_box_params (graded option;
// coarse spacing --box-mesh-size, default 100 nm = lambda1 / 5, 50e-9 = the uniform box that is
// the library default since the ADR 0006 amendment 2026-10-12b; "auto": the automatic rule
// capped at lambda1 / 5; "uncapped": without the cap, the record's "auto" series; Si with the
// fine band 3 delta + 3 sigma), Si or Ag at 500 nm, Gaussian beam at normal incidence, w0 =
// L / --waist-factor, by default the material's limit (beam_waist_fraction: L / 6 for Si, L / 4
// for Ag; the recorded WP22b/WP22d Si rows used L / 4, i.e. --waist-factor 4 --allow-wide-beam)
// (paraxial by default, --beam angular-spectrum for the rigorous beam; check_beam_waist refuses
// a wider waist unless --allow-wide-beam); Simulation with compression "mlfmm" (d0 = 3, automatic
// leaf rule and exact-part budget), formulation per formulation::recommend, full GMRES with
// tolerance 1e-3. The case code is shared with the smoke test
// (tests/support/mlfmm_scaling_support.hpp). Output (stdout): the setup, a summary (2N, octree,
// setup time split, time per matvec, iterations, solve time, memory per component, peak working
// set), a machine-readable ROW line (benchmarks/mlfmm_scaling_fit.py fits the exponents), the
// operator description and the Simulation report. Record: benchmarks/results/mlfmm_scaling.md.
//
// --estimate-only prints the mesh, the octree after the leaf rule, the near-field estimate, the
// estimated Ag exact part and the far patterns and tables (no assembly; seconds to a minute).
// Without --force a case whose setup peak estimate plus 10 GB exceeds the available memory is
// refused. The estimate is computed once per process.
//
// Usage: specklebem_mlfmm_scaling --material si|ag --L 4e-6
//            [--box-mesh-size 100e-9|auto|uncapped] [--fine-band 0|1] [--mesh-size 50e-9]
//            [--seed 1] [--waist-factor 6|4] [--allow-wide-beam] [--beam paraxial|angular-spectrum]
//            [--digits 3] [--leaf-radius-quantile 1] [--tol 1e-3] [--max-iter 3000] [--restart
//            0] [--matvecs 5]
//            [--exact-budget-gb G] [--formulation pmchwt|ictf|mctf] [--jacobi 0|1] [--no-solve]
//            [--estimate-only] [--force] [--sigma 50e-9] [--history] [--far-field 1]
//            [--far-field-degree 6] [--history-file F] [--history-stride 10]
//            [--history-digits 4] [--history-full-file F]
//
// --no-solve stops after the setup and the timed matvecs (no GMRES): the matvec and setup scaling
// at sizes whose full-GMRES solve does not fit the time budget.
//
// Large-area runs (WP22d, scripts/server_run_kit/): --sigma sets the RMS height (default 50 nm,
// the recorded series; the Phase 4 Si 30 x 30 um target uses 100 nm); --history prints one line
// "HIST it=<k> res=<monitored residual> t_s=<seconds since the solve started>" per GMRES
// iteration (flushed, so a log shows the progress of a multi-hour solve); --far-field <dtheta>
// adds the power summary after the solve (reflectance R = P_refl / P_inc on a reflection
// hemisphere grid with polar spacing dtheta [deg] and on the grid with half the nodes,
// absorbed fraction A from the currents; needs --beam angular-spectrum; --far-field-degree sets
// the source quadrature, default 6; cost ~ directions x triangles: L = 2 um at 0.5 deg 80 s on 24
// cores, 1 deg is converged to 6 digits there). R, R_coarse, A and post_s are appended to the ROW
// line. The far-field options are validated before the assembly (validate_power_summary: spacing
// <= 25.7 deg so that the half-node grid has n_phi >= 4, a positive-interior degree); an error in
// the post-processing itself is logged, the ROW line is printed without R / A and with
// post_error=1, and the exit status is 4 (0: done, 1: error, 2: memory guard).
//
// Formulation sweep (WP22b2, record benchmarks/results/formulation_convergence.md): --formulation
// and --jacobi 0|1 override formulation::recommend (Si: ICTF, Ag: ICTF + left Jacobi); the ROW
// line names the solved system (form=, jacobi=, and since the WP-PC1 review precond=
// none|diagonal|block_diagonal). --history-file <path> appends one line
// "HISTORY <case keys> form= jacobi= precond= it= conv= stride= res=r_0,r_s,...,r_it" after the
// solve (--history-stride, default 10; --history-digits significant digits, default 4);
// --history-full-file <path> appends the same record with every iteration at 17 digits (the
// Jacobi identity check). Options are parsed by mlfmm_scaling::parse_options (numeric values
// must be consumed whole; the history options are refused with --no-solve and --estimate-only);
// the history files are probed for appending before the assembly (probe_history_files).
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "mlfmm_scaling_support.hpp"

using namespace mlfmm_scaling;

int main(int argc, char** argv) {
    try {
        const Options o = parse_options(std::vector<std::string>(argv + 1, argv + argc));
        probe_history_files(o);  // an unwritable history path fails before the assembly
        std::printf("%s\n", label(o.c).c_str());
        const Real available = system_memory::available_memory_bytes();
        std::printf("available memory at start: %.1f GB of %.1f GB\n", available / 1e9,
                    system_memory::physical_memory_bytes() / 1e9);
        std::fflush(stdout);
        const Geometry geo = make_geometry(o.c);
        const Estimate e = estimate(geo, o.c);
        if (o.estimate_only) {
            std::printf("ESTIMATE %s\n%s\n", label(o.c).c_str(), summary(e).c_str());
            std::printf(
                "EST material=%s L_um=%g sigma_nm=%g box=%s N2=%lld levels=%d leaf_nm=%g "
                "near_gb=%g "
                "exact_est_gb=%g pattern_gb=%g elevated_pattern_gb=%g far_table_gb=%g "
                "peak_setup_est_gb=%g lq=%g elevated=%lld\n",
                o.c.material.c_str(), o.c.L * 1e6, o.c.sigma * 1e9, box_option(o.c).c_str(),
                static_cast<long long>(e.unknowns), e.levels, e.leaf_edge * 1e9, gb(e.near_bytes),
                gb(e.exact_bytes), gb(e.pattern_bytes), gb(e.elevated_pattern_bytes),
                gb(e.far_table_bytes), gb(e.peak_setup_bytes), o.c.leaf_radius_quantile,
                static_cast<long long>(e.elevated));
            return 0;
        }
        if (!o.force && available > 0.0 && e.peak_setup_bytes + 10e9 > available) {
            std::fprintf(stderr,
                         "mlfmm_scaling: setup peak estimate %.1f GB + 10 GB margin exceeds the "
                         "available %.1f GB; wait or pass --force\n",
                         gb(e.peak_setup_bytes), available / 1e9);
            return 2;
        }
        IterationLog log;
        if (o.history)
            log = [](int k, Real res, Real t) {
                std::printf("HIST it=%d res=%.6e t_s=%.3f\n", k, res, t);
                std::fflush(stdout);
            };
        const Result r = run(geo, o.c, e, log);
        std::printf("RESULT %s\n%s\n%s\n\n%s\n\n%s\n", label(o.c).c_str(), summary(r).c_str(),
                    row(o.c, r).c_str(), r.describe.c_str(), r.report.c_str());
        std::fflush(stdout);
        if (!o.history_file.empty())
            append_line(o.history_file, history_record(o.c, r, o.history_stride, o.history_digits));
        if (!o.history_full_file.empty())
            append_line(o.history_full_file, history_record(o.c, r, 1, 17));
        if (!r.post_error.empty()) {
            // The solve and its ROW line are kept (post_error=1, no R / A); exit 4 flags it.
            std::fprintf(stderr,
                         "mlfmm_scaling: power summary failed after the solve: %s (ROW line "
                         "printed without R / A, post_error=1)\n",
                         r.post_error.c_str());
            return 4;
        }
        return 0;
    } catch (const std::exception& e) {
        // Support-layer messages already start with "mlfmm_scaling"; add the prefix only once.
        const std::string_view msg = e.what();
        std::fprintf(stderr, "%s%s\n",
                     msg.starts_with("mlfmm_scaling") ? "" : "mlfmm_scaling: ", e.what());
        return 1;
    }
}
