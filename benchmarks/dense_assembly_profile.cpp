// Dense assembly profile (WP-P2): where does the time of op::DenseStrategy::build go?
//
// Builds a rough-surface box of the WP15 study (Gaussian surface sigma = 50 nm, Lc = 500 nm,
// h = 50 nm, graded closing box with 400 nm coarse cells; Si depth 5.65 um, Ag 2 um) at a
// reduced patch size, or an icosphere, and
//  * --profile: calls kernels::element_blocks for every (or every --stride-th) test triangle
//    against all source triangles and both regions, exactly as the assembler does, timing every
//    call; reports calls / time per proximity class x region x triangle-size class, the Dunavant
//    degrees of the near / far pairs (kernels::plain_rule_degree), the fold-adaptive statistics
//    of the touching pairs (kernels::touching_rule_info), and the makespan of the assembler's
//    OpenMP schedule estimated from the measured row costs;
//  * --build R: R timed op::DenseStrategy::build calls (the real assembly), in the same process.
//
// Usage: specklebem_dense_assembly_profile [--mesh si|ag|sphere] [--material si|ag|glass|vacuum]
//            [--L 1.6e-6] [--box-mesh-size 4e-7] [--sphere-n 4] [--radius 0.5e-6]
//            [--stride 1] [--profile] [--build R] [--threads T] [--target 1e-5]
//            [--schedule-threads 24]
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/fast_math.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/kernels/singularity.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace specklebem;

namespace {

constexpr Real kLambda = 500e-9;
constexpr Real kSigma = 50e-9;
constexpr Real kCorrelationLength = 500e-9;
constexpr Real kMeshSize = 50e-9;

struct Options {
    std::string mesh = "si";
    std::string material;  // default: the mesh's material (glass for the sphere)
    Real L = 1.6e-6;
    Real box_mesh_size = 400e-9;
    int sphere_n = 4;
    Real radius = 0.5e-6;
    int stride = 1;
    bool profile = false;
    bool micro = false;
    bool no_decay = false;
    int compare = 0;
    int build = 0;
    int threads = 0;
    int schedule_threads = 24;
    Real target = -1.0;  // < 0: OperatorOptions default
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--mesh") {
            o.mesh = value();
        } else if (a == "--material") {
            o.material = value();
        } else if (a == "--L") {
            o.L = std::stod(value());
        } else if (a == "--box-mesh-size") {
            o.box_mesh_size = std::stod(value());
        } else if (a == "--sphere-n") {
            o.sphere_n = std::stoi(value());
        } else if (a == "--radius") {
            o.radius = std::stod(value());
        } else if (a == "--stride") {
            o.stride = std::stoi(value());
        } else if (a == "--no-decay") {
            o.no_decay = true;
        } else if (a == "--compare") {
            o.compare = std::stoi(value());
        } else if (a == "--micro") {
            o.micro = true;
        } else if (a == "--profile") {
            o.profile = true;
        } else if (a == "--build") {
            o.build = std::stoi(value());
        } else if (a == "--threads") {
            o.threads = std::stoi(value());
        } else if (a == "--schedule-threads") {
            o.schedule_threads = std::stoi(value());
        } else if (a == "--target") {
            o.target = std::stod(value());
        } else {
            throw std::invalid_argument("unknown argument " + a);
        }
    }
    if (o.material.empty())
        o.material = o.mesh == "sphere" ? "glass" : o.mesh;
    if (o.stride < 1 || o.schedule_threads < 1)
        throw std::invalid_argument("stride and schedule-threads must be >= 1");
    return o;
}

material::Material material_by_name(const std::string& name) {
    if (name == "si")
        return material::silicon_500nm();
    if (name == "ag")
        return material::silver_500nm();
    if (name == "glass")
        return {Complex(2.25, 0.0), Complex(1.0, 0.0)};
    if (name == "vacuum")
        return material::vacuum();
    throw std::invalid_argument("unknown material " + name);
}

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

Real longest_edge(const geometry::TriangleMesh& m, Index t) {
    const auto v = [&](int i) -> Vec3 { return m.vertices().row(m.triangles()(t, i)).transpose(); };
    return std::sqrt(std::max(
        {(v(1) - v(0)).squaredNorm(), (v(2) - v(1)).squaredNorm(), (v(0) - v(2)).squaredNorm()}));
}

constexpr std::size_t kClasses = 5;
constexpr std::size_t kSizes = 3;
const std::array<const char*, kClasses> kClassName = {"identical", "shared edge", "shared vertex",
                                                      "near", "far"};
const std::array<const char*, kSizes> kSizeName = {"fine", "medium", "coarse"};

struct Cell {
    double calls = 0;
    double seconds = 0;
};

struct Stats {
    // [class][region][size]
    std::array<std::array<std::array<Cell, kSizes>, 2>, kClasses> cell{};
    // degree histogram [region][near/far][degree]
    std::array<std::array<std::array<Cell, 21>, 2>, 2> degree{};
    // touching rule info [class 0..2]
    std::array<double, 3> touch_pairs{};
    std::array<double, 3> touch_adaptive{};
    std::array<double, 3> touch_points{};
    std::array<double, 3> touch_pieces{};
    std::array<double, 3> touch_max_points{};

    void add(const Stats& o) {
        for (std::size_t c = 0; c < kClasses; ++c)
            for (std::size_t r = 0; r < 2; ++r)
                for (std::size_t s = 0; s < kSizes; ++s) {
                    cell[c][r][s].calls += o.cell[c][r][s].calls;
                    cell[c][r][s].seconds += o.cell[c][r][s].seconds;
                }
        for (std::size_t r = 0; r < 2; ++r)
            for (std::size_t p = 0; p < 2; ++p)
                for (std::size_t d = 0; d <= 20; ++d) {
                    degree[r][p][d].calls += o.degree[r][p][d].calls;
                    degree[r][p][d].seconds += o.degree[r][p][d].seconds;
                }
        for (std::size_t c = 0; c < 3; ++c) {
            touch_pairs[c] += o.touch_pairs[c];
            touch_adaptive[c] += o.touch_adaptive[c];
            touch_points[c] += o.touch_points[c];
            touch_pieces[c] += o.touch_pieces[c];
            touch_max_points[c] = std::max(touch_max_points[c], o.touch_max_points[c]);
        }
    }
};

kernels::RegionParams region_params(const material::Material& m, Real omega) {
    return {m.wavenumber(omega), m.wave_impedance(omega), omega, constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

/// Greedy colouring of op::DenseStrategy::build: triangles sharing a basis function differ.
std::vector<std::vector<Index>> colour_groups(const basis::RwgSpace& space) {
    const Index F = space.mesh().num_triangles();
    std::vector<int> colour(static_cast<std::size_t>(F), -1);
    std::vector<std::vector<Index>> groups;
    for (Index t = 0; t < F; ++t) {
        const basis::RwgSpace::Support sup = space.support(t);
        std::array<bool, 4> used{};
        for (int a = 0; a < sup.count; ++a) {
            const Index n = sup.n[a];
            const Index other =
                space.plus_triangle(n) == t ? space.minus_triangle(n) : space.plus_triangle(n);
            const int c = colour[static_cast<std::size_t>(other)];
            if (c >= 0)
                used[static_cast<std::size_t>(c)] = true;
        }
        int c = 0;
        while (used[static_cast<std::size_t>(c)]) ++c;
        colour[static_cast<std::size_t>(t)] = c;
        if (static_cast<std::size_t>(c) >= groups.size())
            groups.resize(static_cast<std::size_t>(c) + 1);
        groups[static_cast<std::size_t>(c)].push_back(t);
    }
    return groups;
}

/// Makespan of list scheduling (each row to the least loaded thread) in the given order.
double list_makespan(const std::vector<Index>& order, const std::vector<double>& row_cost,
                     int threads) {
    std::vector<double> load(static_cast<std::size_t>(threads), 0.0);
    for (const Index t : order) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < load.size(); ++j)
            if (load[j] < load[best])
                best = j;
        load[best] += row_cost[static_cast<std::size_t>(t)];
    }
    double busiest = 0;
    for (const double x : load) busiest = std::max(busiest, x);
    return busiest;
}

/// Makespan of the schedules (colour groups processed in turn) from the measured row costs:
/// OpenMP static (contiguous chunks, the assembler before WP-P2), dynamic in index order, and
/// dynamic in the order of decreasing longest edge (the assembler since WP-P2).
void report_schedule(const basis::RwgSpace& space, const std::vector<double>& row_cost,
                     const std::vector<Real>& edge, int threads) {
    const std::vector<std::vector<Index>> groups = colour_groups(space);
    double total = 0;
    double makespan_static = 0;
    double makespan_dynamic = 0;
    double makespan_largest = 0;
    for (const auto& g : groups) {
        const auto n = static_cast<Index>(g.size());
        const Index chunk = (n + threads - 1) / threads;
        double worst = 0;
        for (Index c0 = 0; c0 < n; c0 += chunk) {
            double s = 0;
            for (Index i = c0; i < std::min(n, c0 + chunk); ++i)
                s += row_cost[static_cast<std::size_t>(g[static_cast<std::size_t>(i)])];
            worst = std::max(worst, s);
        }
        makespan_static += worst;
        for (const Index t : g) total += row_cost[static_cast<std::size_t>(t)];
        makespan_dynamic += list_makespan(g, row_cost, threads);
        std::vector<Index> sorted = g;
        std::stable_sort(sorted.begin(), sorted.end(), [&](Index a, Index b) {
            return edge[static_cast<std::size_t>(a)] > edge[static_cast<std::size_t>(b)];
        });
        makespan_largest += list_makespan(sorted, row_cost, threads);
    }
    const double ideal = total / threads;
    std::printf(
        "SCHEDULE | threads %d | colours %zu | serial %.1f s | ideal %.2f s | static %.2f s "
        "(x%.2f) "
        "| dynamic %.2f s (x%.2f) | dynamic largest first %.2f s (x%.2f)\n",
        threads, groups.size(), total, ideal, makespan_static, makespan_static / ideal,
        makespan_dynamic, makespan_dynamic / ideal, makespan_largest, makespan_largest / ideal);
}

/// The pair loop of op::DenseStrategy::build before WP-P2 (colour groups, OpenMP static
/// schedule in index order, element_blocks for both regions), without the matrix writes: the
/// "before" timing in the same process. Returns the wall time.
double replica_static(const basis::RwgSpace& space,
                      const std::array<kernels::RegionParams, 2>& region,
                      const kernels::OperatorOptions& kopt) {
    const std::vector<std::vector<Index>> groups = colour_groups(space);
    const Index F = space.mesh().num_triangles();
    const double t0 = now_seconds();
    double sink = 0;
    for (const auto& g : groups) {
        const auto ng = static_cast<Index>(g.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(static) reduction(+ : sink)
#endif
        for (Index i = 0; i < ng; ++i) {
            const Index t = g[static_cast<std::size_t>(i)];
            if (space.support(t).count == 0)
                continue;
            Eigen::Matrix<Complex, 3, 3> L;
            Eigen::Matrix<Complex, 3, 3> K;
            for (Index s = 0; s < F; ++s) {
                if (space.support(s).count == 0)
                    continue;
                for (const kernels::RegionParams& r : region) {
                    kernels::element_blocks(space, t, s, r, kopt, L, K);
                    sink += std::abs(L(0, 0)) + std::abs(K(0, 0));
                }
            }
        }
    }
    const double dt = now_seconds() - t0;
    if (!std::isfinite(sink))
        throw std::runtime_error("replica: non-finite blocks");
    return dt;
}

void run(const Options& o) {
    const Real omega = 2.0 * constants::pi * constants::c0 / kLambda;
    const material::Material object = material_by_name(o.material);
    geometry::TriangleMesh mesh;
    Index top = 0;
    if (o.mesh == "sphere") {
        mesh = geometry::make_icosphere(o.radius, o.sphere_n);
        top = mesh.num_triangles();
    } else {
        geometry::RoughSurfaceParams rp;
        rp.edge_length_L = o.L;
        rp.rms_roughness = kSigma;
        rp.correlation_length = kCorrelationLength;
        rp.mesh_size = kMeshSize;
        rp.seed = 1;
        rp.box_depth = o.mesh == "si" ? 5.65e-6 : 2e-6;  // WP15 default_box_depth
        rp.box_mesh_size = o.box_mesh_size;
        const geometry::HeightMap hm = geometry::generate_gaussian_height_map(rp);
        mesh = geometry::make_mesh_from_height_map(hm, rp.box_depth, rp.box_mesh_size);
        top = 2 * (hm.z.rows() - 1) * (hm.z.cols() - 1);
    }
    const basis::RwgSpace space(mesh);
    const Index F = mesh.num_triangles();
    const Index N = space.size();
    const Real h_ref = o.mesh == "sphere" ? longest_edge(mesh, 0) : kMeshSize;
    std::vector<std::size_t> size_class(static_cast<std::size_t>(F));
    std::vector<Real> edge(static_cast<std::size_t>(F));
    std::array<Index, kSizes> size_count{};
    for (Index t = 0; t < F; ++t) {
        edge[static_cast<std::size_t>(t)] = longest_edge(mesh, t);
        const Real e = edge[static_cast<std::size_t>(t)] / h_ref;
        const std::size_t s = e <= 2.5 ? 0 : (e <= 6.0 ? 1 : 2);
        size_class[static_cast<std::size_t>(t)] = s;
        ++size_count[s];
    }
    kernels::OperatorOptions kopt;
    if (o.target >= 0.0)
        kopt.target_accuracy = o.target;
    kopt.decay_aware_target = !o.no_decay;
    const material::Material exterior = material::vacuum();
    const std::array<kernels::RegionParams, 2> region = {region_params(exterior, omega),
                                                         region_params(object, omega)};
    std::printf(
        "SETUP | mesh %s | object %s eps_r %.4g%+.4gj | k2 %.4g%+.4gj /um | triangles %lld (top "
        "%lld) | size classes fine/medium/coarse %lld/%lld/%lld (h_ref %.3g m) | N %lld | 2N %lld "
        "| target %.3g\n",
        o.mesh.c_str(), o.material.c_str(), object.eps_r.real(), object.eps_r.imag(),
        region[1].k.real() * 1e-6, region[1].k.imag() * 1e-6, static_cast<long long>(F),
        static_cast<long long>(top), static_cast<long long>(size_count[0]),
        static_cast<long long>(size_count[1]), static_cast<long long>(size_count[2]), h_ref,
        static_cast<long long>(N), static_cast<long long>(2 * N), kopt.target_accuracy);
    std::fflush(stdout);
#ifdef SPECKLEBEM_HAVE_OPENMP
    if (o.threads > 0)
        omp_set_num_threads(o.threads);
#endif

    if (o.profile) {
        std::vector<Index> rows;
        for (Index t = 0; t < F; t += o.stride) rows.push_back(t);
        std::vector<double> row_cost(static_cast<std::size_t>(F), 0.0);
        const auto nrows = static_cast<Index>(rows.size());
        Stats total;
        const double t0 = now_seconds();
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel
#endif
        {
            Stats local;
            Eigen::Matrix<Complex, 3, 3> L;
            Eigen::Matrix<Complex, 3, 3> K;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
            for (Index i = 0; i < nrows; ++i) {
                const Index t = rows[static_cast<std::size_t>(i)];
                if (space.support(t).count == 0)
                    continue;
                double row = 0;
                for (Index s = 0; s < F; ++s) {
                    if (space.support(s).count == 0)
                        continue;
                    const kernels::Proximity prox = kernels::classify(mesh, t, s, 2.0);
                    const auto c = static_cast<std::size_t>(prox);
                    const std::size_t sz = std::max(size_class[static_cast<std::size_t>(t)],
                                                    size_class[static_cast<std::size_t>(s)]);
                    if (c <= 2) {
                        const kernels::TouchingRuleInfo info =
                            kernels::touching_rule_info(mesh, t, s, kopt);
                        local.touch_pairs[c] += 1;
                        local.touch_adaptive[c] += info.fold_adaptive ? 1 : 0;
                        local.touch_points[c] += static_cast<double>(info.points);
                        local.touch_pieces[c] += static_cast<double>(info.pieces);
                        local.touch_max_points[c] =
                            std::max(local.touch_max_points[c], static_cast<double>(info.points));
                    }
                    for (std::size_t r = 0; r < 2; ++r) {
                        const double a = now_seconds();
                        kernels::element_blocks(space, t, s, region[r], kopt, L, K);
                        const double dt = now_seconds() - a;
                        row += dt;
                        Cell& cell = local.cell[c][r][sz];
                        cell.calls += 1;
                        cell.seconds += dt;
                        if (c >= 3) {
                            const auto d = static_cast<std::size_t>(
                                kernels::plain_rule_degree(mesh, t, s, region[r].k, kopt));
                            Cell& dc = local.degree[r][c - 3][d];
                            dc.calls += 1;
                            dc.seconds += dt;
                        }
                    }
                }
                row_cost[static_cast<std::size_t>(t)] = row;
            }
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp critical(profile_merge)
#endif
            total.add(local);
        }
        const double wall = now_seconds() - t0;
        const double scale = static_cast<double>(o.stride);
        double all = 0;
        for (const auto& c : total.cell)
            for (const auto& r : c)
                for (const auto& s : r) all += s.seconds;
        std::printf(
            "PROFILE | rows %lld of %lld (stride %d) | wall %.1f s | element_blocks %.1f s "
            "(x%d -> full matrix %.1f CPU s)\n",
            static_cast<long long>(nrows), static_cast<long long>(F), o.stride, wall, all, o.stride,
            all * scale);
        std::printf(
            "CLASS | class | region | size | calls(full) | CPU s(full) | share | mean us\n");
        for (std::size_t c = 0; c < kClasses; ++c)
            for (std::size_t r = 0; r < 2; ++r)
                for (std::size_t s = 0; s < kSizes; ++s) {
                    const Cell& x = total.cell[c][r][s];
                    if (x.calls == 0)
                        continue;
                    std::printf("CLASS | %s | %s | %s | %.0f | %.1f | %.1f %% | %.2f\n",
                                kClassName[c], r == 0 ? "ext" : "int", kSizeName[s],
                                x.calls * scale, x.seconds * scale, 100.0 * x.seconds / all,
                                1e6 * x.seconds / x.calls);
                }
        std::printf("DEGREE | region | class | degree | calls(full) | CPU s(full) | mean us\n");
        for (std::size_t r = 0; r < 2; ++r)
            for (std::size_t p = 0; p < 2; ++p)
                for (std::size_t d = 0; d <= 20; ++d) {
                    const Cell& x = total.degree[r][p][d];
                    if (x.calls == 0)
                        continue;
                    std::printf("DEGREE | %s | %s | %zu | %.0f | %.1f | %.2f\n",
                                r == 0 ? "ext" : "int", p == 0 ? "near" : "far", d, x.calls * scale,
                                x.seconds * scale, 1e6 * x.seconds / x.calls);
                }
        for (std::size_t c = 0; c < 3; ++c) {
            if (total.touch_pairs[c] == 0)
                continue;
            std::printf(
                "TOUCH | %s | pairs(sampled) %.0f | fold-adaptive %.1f %% | mean pieces "
                "%.2f | mean points %.1f | max points %.0f\n",
                kClassName[c], total.touch_pairs[c],
                100.0 * total.touch_adaptive[c] / total.touch_pairs[c],
                total.touch_pieces[c] / total.touch_pairs[c],
                total.touch_points[c] / total.touch_pairs[c], total.touch_max_points[c]);
        }
        if (o.stride == 1)
            report_schedule(space, row_cost, edge, o.schedule_threads);
        std::fflush(stdout);
    }

    if (o.micro) {
        // Single-thread cost of one far pair at degree 19 (target_accuracy = 0, fixed degrees)
        // and of the libm calls of its kernel loop.
        kernels::OperatorOptions fixed = kopt;
        fixed.target_accuracy = 0.0;
        fixed.quad_degree_far = 19;
        Index s_far = -1;
        for (Index s = F - 1; s > 0; --s) {
            if (kernels::classify(mesh, 0, s, 2.0) == kernels::Proximity::far &&
                space.support(s).count > 0) {
                s_far = s;
                break;
            }
        }
        Eigen::Matrix<Complex, 3, 3> L;
        Eigen::Matrix<Complex, 3, 3> K;
        constexpr int kReps = 2000;
        const std::array<std::pair<int, int>, 4> degrees = {
            {{1, 1}, {4, 6}, {8, 16}, {19, 73}}};  // (degree, points)
        for (const auto& [degree, points] : degrees) {
            fixed.quad_degree_far = degree;
            fixed.quad_degree_near = std::max(degree, 2);
            for (std::size_t r = 0; r < 2; ++r) {
                for (const bool fast : {false, true}) {
                    fixed.fast_plain_kernel = fast;
                    double best = 1e300;
                    for (int rep = 0; rep < 5; ++rep) {
                        const double a = now_seconds();
                        for (int i = 0; i < kReps; ++i)
                            kernels::element_blocks(space, 0, s_far, region[r], fixed, L, K);
                        best = std::min(best, (now_seconds() - a) / kReps);
                    }
                    std::printf(
                        "MICRO | far pair degree %d (%d x %d points) | %s | %s | %.3f us per "
                        "call | %.2f ns per kernel evaluation\n",
                        degree, points, points, r == 0 ? "ext" : "int", fast ? "fastmath" : "libm",
                        1e6 * best, 1e9 * best / (points * points));
                }
            }
        }
        double sink = 0;
        double best = 1e300;
        constexpr int kN = 5329;
        for (int rep = 0; rep < 5; ++rep) {
            const double a = now_seconds();
            for (int i = 0; i < kReps; ++i)
                for (int q = 0; q < kN; ++q) {
                    const double R = 1e-6 * (1.0 + 1e-4 * q + 1e-9 * i);
                    sink += std::exp(-1e6 * R) * std::cos(5.4e7 * R) +
                            std::sin(5.4e7 * R) / std::sqrt(R * R + 1.0);
                }
            best = std::min(best, (now_seconds() - a) / (kReps * double(kN)));
        }
        std::printf("MICRO | exp + cos + sin + sqrt + div (libm) | %.2f ns | (%g)\n", 1e9 * best,
                    sink);
        // Isolated elementary functions over an array (vectorisable loops).
        std::vector<double> xs(kN);
        std::vector<double> ys(kN);
        std::vector<double> zs(kN);
        for (int q = 0; q < kN; ++q) xs[static_cast<std::size_t>(q)] = -1e-3 * q - 0.1;
        const auto time_loop = [&](const char* name, auto&& body) {
            double t_best = 1e300;
            for (int rep = 0; rep < 5; ++rep) {
                const double a = now_seconds();
                for (int i = 0; i < kReps; ++i) body();
                t_best = std::min(t_best, (now_seconds() - a) / (kReps * double(kN)));
            }
            double s = 0;
            for (int q = 0; q < kN; ++q)
                s += ys[static_cast<std::size_t>(q)] + zs[static_cast<std::size_t>(q)];
            std::printf("MICRO | %s | %.2f ns | (%g)\n", name, 1e9 * t_best, s);
        };
        time_loop("std::exp", [&] {
            for (std::size_t q = 0; q < xs.size(); ++q) ys[q] = std::exp(xs[q]);
        });
        time_loop("fast_exp_nonpositive", [&] {
            for (std::size_t q = 0; q < xs.size(); ++q)
                ys[q] = kernels::fastmath::fast_exp_nonpositive(xs[q]);
        });
        time_loop("std::sin + std::cos", [&] {
            for (std::size_t q = 0; q < xs.size(); ++q) {
                ys[q] = std::sin(30.0 * xs[q]);
                zs[q] = std::cos(30.0 * xs[q]);
            }
        });
        time_loop("fast_sincos", [&] {
            for (std::size_t q = 0; q < xs.size(); ++q)
                kernels::fastmath::fast_sincos(30.0 * xs[q], ys[q], zs[q]);
        });
        std::fflush(stdout);
    }

    if (o.build > 0) {
        const std::unique_ptr<formulation::Formulation> form =
            formulation::make_formulation(formulation::Kind::ICTF);
        op::Problem p;
        p.space = &space;
        p.exterior = exterior;
        p.object = object;
        p.formulation = form.get();
        p.kernel_options = kopt;
        p.omega = omega;
        const op::DenseStrategy dense;
        for (int i = 0; i < o.build; ++i) {
            const double a = now_seconds();
            const std::shared_ptr<op::LinearOperator> Z = dense.build(p);
            const double dt = now_seconds() - a;
#ifdef SPECKLEBEM_HAVE_OPENMP
            const int threads = omp_get_max_threads();
#else
            const int threads = 1;
#endif
            std::printf("BUILD | run %d | threads %d | 2N %lld | %.2f s\n", i + 1, threads,
                        static_cast<long long>(Z->rows()), dt);
            std::fflush(stdout);
        }
    }

    if (o.compare > 0) {
        // Same-process comparison, alternating: (a) the pre-WP-P2 pair loop (static schedule,
        // no decay-aware target), (b) DenseStrategy::build without and (c) with the
        // decay-aware target.
        const std::unique_ptr<formulation::Formulation> form =
            formulation::make_formulation(formulation::Kind::ICTF);
        op::Problem p;
        p.space = &space;
        p.exterior = exterior;
        p.object = object;
        p.formulation = form.get();
        p.omega = omega;
        kernels::OperatorOptions strict = kopt;
        strict.decay_aware_target = false;
        kernels::OperatorOptions relaxed = kopt;
        relaxed.decay_aware_target = true;
        const op::DenseStrategy dense;
        for (int i = 0; i < o.compare; ++i) {
            const double before = replica_static(space, region, strict);
            p.kernel_options = strict;
            double a = now_seconds();
            (void)dense.build(p);
            const double dyn = now_seconds() - a;
            p.kernel_options = relaxed;
            a = now_seconds();
            (void)dense.build(p);
            const double dyn_decay = now_seconds() - a;
            std::printf(
                "COMPARE | round %d | before (static, strict) %.2f s | dynamic, strict %.2f "
                "s (x%.2f) | dynamic, decay-aware %.2f s (x%.2f)\n",
                i + 1, before, dyn, before / dyn, dyn_decay, before / dyn_decay);
            std::fflush(stdout);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
