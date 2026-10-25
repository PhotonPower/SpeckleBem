/// Bindings of the Simulation driver (simulation.hpp), its GMRES result, the system operator
/// and the post-processing (postprocessing/fields.hpp, scattering.hpp).
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"

#include <pybind11/stl.h>

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common.hpp"

namespace specklebem::python {

namespace {

using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;
using formulation::Kind;
using solver::GmresParams;
using solver::GmresResult;
using solver::PreconditionerSide;

/// Interval of the Ctrl-C check of a GMRES solve without a Python callback and of a main
/// thread waiting for a Simulation's lock (each check re-acquires the GIL, which can wait up to
/// sys.getswitchinterval() for a busy thread).
constexpr auto kSignalCheckInterval = std::chrono::milliseconds(50);

/// True on the Python main thread (the only one that handles signals). Needs the GIL.
bool on_main_thread() {
    const py::module_ threading = py::module_::import("threading");
    return threading.attr("current_thread")().is(threading.attr("main_thread")());
}

/// The Simulation bound to Python: the C++ object plus a per-object mutex that serialises calls
/// from several Python threads (the C++ class itself is not thread-safe).
///
/// Lock order: every wrapper releases the GIL first and only then waits for the mutex, so no
/// thread ever waits for the mutex while holding the GIL; the mutex holder may re-acquire the
/// GIL (progress callback, signal check). Waiting is bounded in three ways:
///  - a call on the same Simulation from inside its solve() callback (same thread, mutex
///    already held; owner_ records the holding thread) raises RuntimeError at once;
///  - a thread that already holds some Simulation's mutex (held_ > 0: inside a solve()
///    callback, or a finalizer run there) only try_lock()s another one and raises RuntimeError
///    if it is busy, so two solves calling into each other's Simulation cannot deadlock (calls
///    into an idle Simulation still work);
///  - the main thread waits in slices of kSignalCheckInterval and, between them, re-acquires
///    the GIL for PyErr_CheckSignals(), so Ctrl-C (KeyboardInterrupt) interrupts the wait.
class PySimulation final : public Simulation {
public:
    using Simulation::Simulation;

    /// The excitation passed to the constructor (Simulation keeps it alive but exposes only a
    /// raw pointer through problem()); returned by the Python property `excitation`.
    std::shared_ptr<excitation::Excitation> excitation_ref;

    /// Parameters (after the per-call overrides) and result of the last successful solve();
    /// written and read with the mutex held (properties last_result / last_solve_params).
    struct LastSolve {
        solver::GmresParams params;
        solver::GmresResult result;
    };
    std::optional<LastSolve> last_solve;

    /// Returns f() computed with the GIL released and the mutex held (f must return a value
    /// or void, never a reference into the Simulation). Call with the GIL held.
    template <class F>
    auto locked(F&& f) {
        const bool main_thread = on_main_thread();
        const py::gil_scoped_release release;
        if (owner_.load() == std::this_thread::get_id()) {
            throw std::runtime_error(
                "Simulation: re-entrant call from a solve() callback on the same Simulation");
        }
        std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
        if (held_ > 0) {
            if (!lock.try_lock()) {
                throw std::runtime_error(
                    "Simulation busy: cannot wait for another Simulation's lock from inside a "
                    "solve() callback");
            }
        } else {
            while (!lock.try_lock_for(kSignalCheckInterval)) {
                if (main_thread) {
                    const py::gil_scoped_acquire gil;  // released again before the next wait
                    if (PyErr_CheckSignals() != 0) {
                        throw py::error_already_set();
                    }
                }
            }
        }
        const OwnerGuard owner(owner_);
        return f();
    }

private:
    /// For its lifetime: records the calling thread as the mutex holder (owner_) and counts
    /// the Simulation locks this thread holds (held_).
    struct OwnerGuard {
        explicit OwnerGuard(std::atomic<std::thread::id>& o) : owner(o) {
            owner.store(std::this_thread::get_id());
            ++held_;
        }
        ~OwnerGuard() {
            --held_;
            owner.store(std::thread::id{});
        }
        OwnerGuard(const OwnerGuard&) = delete;
        OwnerGuard& operator=(const OwnerGuard&) = delete;
        std::atomic<std::thread::id>& owner;
    };

    /// Number of Simulation locks (of any Simulation) held by the calling thread.
    static thread_local int held_;

    std::timed_mutex mutex_;
    std::atomic<std::thread::id> owner_{};
};

thread_local int PySimulation::held_ = 0;

template <class T>
T dict_value(const py::handle& v, const std::string& key, const char* what) {
    try {
        return v.cast<T>();
    } catch (const py::cast_error&) {
        throw py::value_error(std::string(what) + "['" + key + "']: invalid value type");
    }
}

/// Dict key as str; non-str keys (int, bytes, ...) are a user error -> ValueError.
std::string dict_key(const py::handle& k, const char* what) {
    if (!py::isinstance<py::str>(k))
        throw py::value_error(std::string(what) + ": keys must be str");
    return py::cast<std::string>(k);
}

PreconditionerSide parse_side(const std::string& side) {
    const std::string s = to_lower(side);
    if (s == "left" || s == "right") {
        return s == "left" ? PreconditionerSide::Left : PreconditionerSide::Right;
    }
    throw py::value_error("side: expected \"left\" or \"right\"");
}

solver::Orthogonalization parse_orthogonalization(const std::string& name) {
    const std::string s = to_lower(name);
    if (s == "cgs2")
        return solver::Orthogonalization::CGS2;
    if (s == "mgs")
        return solver::Orthogonalization::MGS;
    throw py::value_error("gmres['orthogonalization']: expected \"cgs2\" or \"mgs\"");
}

/// gmres=dict(tol, max_iter, restart (None = full GMRES), side, verbose, orthogonalization,
/// keep_hessenberg) onto GmresParams.
void apply_gmres(GmresParams& p, const py::dict& d) {
    for (const auto& [k, v] : d) {
        const auto key = dict_key(k, "gmres");
        if (key == "tol") {
            p.tolerance = dict_value<Real>(v, key, "gmres");
        } else if (key == "max_iter") {
            p.max_iter = dict_value<int>(v, key, "gmres");
        } else if (key == "restart") {
            p.restart = v.is_none() ? 0 : dict_value<int>(v, key, "gmres");
        } else if (key == "side") {
            if (!py::isinstance<py::str>(v))
                throw py::value_error("gmres['side']: expected a str");
            p.side = parse_side(dict_value<std::string>(v, key, "gmres"));
        } else if (key == "verbose") {
            p.verbose = dict_value<bool>(v, key, "gmres");
        } else if (key == "orthogonalization") {
            if (!py::isinstance<py::str>(v))
                throw py::value_error("gmres['orthogonalization']: expected a str");
            p.orthogonalization = parse_orthogonalization(dict_value<std::string>(v, key, "gmres"));
        } else if (key == "keep_hessenberg") {
            p.keep_hessenberg = dict_value<bool>(v, key, "gmres");
        } else {
            throw py::value_error("gmres: unknown key '" + key + "'");
        }
    }
}

/// kernels=dict(...) onto kernels::OperatorOptions (keys = the C++ member names).
void apply_kernels(kernels::OperatorOptions& o, const py::dict& d) {
    for (const auto& [k, v] : d) {
        const auto key = dict_key(k, "kernels");
        if (key == "quad_degree_far") {
            o.quad_degree_far = dict_value<int>(v, key, "kernels");
        } else if (key == "quad_degree_near") {
            o.quad_degree_near = dict_value<int>(v, key, "kernels");
        } else if (key == "quad_degree_sing") {
            o.quad_degree_sing = dict_value<int>(v, key, "kernels");
        } else if (key == "outer_grading_levels") {
            o.outer_grading_levels = dict_value<int>(v, key, "kernels");
        } else if (key == "near_distance_factor") {
            o.near_distance_factor = dict_value<Real>(v, key, "kernels");
        } else if (key == "symmetrize_touching_above_kh") {
            o.symmetrize_touching_above_kh = dict_value<Real>(v, key, "kernels");
        } else if (key == "target_accuracy") {
            o.target_accuracy = dict_value<Real>(v, key, "kernels");
        } else if (key == "quad_degree_rhs") {
            o.quad_degree_rhs = dict_value<int>(v, key, "kernels");
        } else if (key == "fold_adaptive") {
            o.fold_adaptive = dict_value<bool>(v, key, "kernels");
        } else if (key == "decay_aware_target") {
            o.decay_aware_target = dict_value<bool>(v, key, "kernels");
        } else if (key == "fast_plain_kernel") {
            o.fast_plain_kernel = dict_value<bool>(v, key, "kernels");
        } else if (key == "stretched_touching_points") {
            o.stretched_touching_points = dict_value<int>(v, key, "kernels");
        } else {
            throw py::value_error("kernels: unknown key '" + key + "'");
        }
    }
}

/// mlfmm=dict(...) onto mlfmm::MlfmmParams (docs/10): accuracy_digits, max_elements_per_leaf,
/// min_box_size_lambda (the octree keys of mlfmm::OctreeParams), automatic_leaf_size,
/// leaf_radius_quantile, max_exact_far_bytes and the PML ring keys pml_ring_use_aca,
/// pml_ring_aca_epsilon, pml_ring_eta, pml_ring_max_rank, pml_ring_row_clusters,
/// pml_ring_max_far_bytes (mlfmm::PmlRingParams, WP-PML5b).
void apply_mlfmm(mlfmm::MlfmmParams& p, const py::dict& d) {
    for (const auto& [k, v] : d) {
        const auto key = dict_key(k, "mlfmm");
        if (key == "accuracy_digits") {
            p.accuracy_digits = dict_value<Real>(v, key, "mlfmm");
        } else if (key == "max_elements_per_leaf") {
            p.octree.max_elements_per_leaf = dict_value<int>(v, key, "mlfmm");
        } else if (key == "min_box_size_lambda") {
            p.octree.min_box_size_lambda = dict_value<Real>(v, key, "mlfmm");
        } else if (key == "automatic_leaf_size") {
            p.automatic_leaf_size = dict_value<bool>(v, key, "mlfmm");
        } else if (key == "leaf_radius_quantile") {
            p.leaf_radius_quantile = dict_value<Real>(v, key, "mlfmm");
        } else if (key == "max_exact_far_bytes") {
            // Python int >= 0 (a negative value fails the unsigned cast -> ValueError).
            p.max_exact_far_bytes = dict_value<std::size_t>(v, key, "mlfmm");
        } else if (key == "pml_ring_use_aca") {
            p.pml_ring.use_aca = dict_value<bool>(v, key, "mlfmm");
        } else if (key == "pml_ring_aca_epsilon") {
            p.pml_ring.aca_epsilon = dict_value<Real>(v, key, "mlfmm");
        } else if (key == "pml_ring_eta") {
            p.pml_ring.eta = dict_value<Real>(v, key, "mlfmm");
        } else if (key == "pml_ring_max_rank") {
            p.pml_ring.max_rank = dict_value<Index>(v, key, "mlfmm");
        } else if (key == "pml_ring_row_clusters") {
            p.pml_ring.row_clusters = dict_value<bool>(v, key, "mlfmm");
        } else if (key == "pml_ring_max_far_bytes") {
            p.pml_ring.max_far_bytes = dict_value<std::size_t>(v, key, "mlfmm");
        } else {
            throw py::value_error("mlfmm: unknown key '" + key + "'");
        }
    }
}

std::optional<Kind> parse_formulation(const py::object& f) {
    if (py::isinstance<Kind>(f)) {
        return f.cast<Kind>();
    }
    if (py::isinstance<py::str>(f)) {
        const std::string s = to_lower(f.cast<std::string>());
        if (s == "auto") {
            return std::nullopt;
        }
        const std::pair<const char*, Kind> names[] = {{"pmchwt", Kind::PMCHWT},
                                                      {"ictf", Kind::ICTF},
                                                      {"mctf", Kind::MCTF},
                                                      {"jmcfie", Kind::JMCFIE}};
        for (const auto& [name, k] : names) {
            if (s == name) {
                return k;  // JMCFIE: rejected by the Simulation constructor (not implemented)
            }
        }
    }
    throw py::value_error("formulation: expected \"auto\", \"PMCHWT\", \"ICTF\" or \"MCTF\"");
}

/// SimulationConfig from the keyword arguments shared by both Simulation constructors.
SimulationConfig make_config(const std::shared_ptr<excitation::Excitation>& exc,
                             const material::Material& object,
                             const std::optional<material::Material>& exterior,
                             std::optional<Real> wavelength, const py::object& form,
                             const std::string& preconditioner, const std::string& solver_name,
                             const std::string& compression, const std::optional<py::dict>& gmres,
                             const std::optional<py::dict>& kernel_opts,
                             const std::optional<py::dict>& mlfmm_opts) {
    if (!exc) {
        throw py::value_error("excitation: must not be None");
    }
    SimulationConfig c;
    c.wavelength = wavelength.value_or(exc->wavelength());
    c.exterior = exterior.value_or(exc->background());
    c.object = object;
    c.formulation = parse_formulation(form);
    const std::string pre = to_lower(preconditioner);
    if (pre == "diagonal" || pre == "none") {
        c.diagonal_preconditioner = pre == "diagonal";
    } else if (pre != "auto") {
        throw py::value_error("preconditioner: expected \"auto\", \"none\" or \"diagonal\"");
    }
    const std::string sol = to_lower(solver_name);
    if (sol != "gmres" && sol != "direct") {
        throw py::value_error("solver: expected \"gmres\" or \"direct\"");
    }
    c.solver = sol == "direct" ? SolverKind::Direct : SolverKind::Gmres;
    c.compression = compression;
    if (gmres) {
        apply_gmres(c.gmres, *gmres);
    }
    if (kernel_opts) {
        apply_kernels(c.kernels, *kernel_opts);
    }
    if (mlfmm_opts) {
        apply_mlfmm(c.mlfmm, *mlfmm_opts);
    }
    return c;
}

std::unique_ptr<PySimulation> make_simulation(
    const geometry::TriangleMesh& mesh, std::shared_ptr<excitation::Excitation> exc,
    const material::Material& object, const std::optional<material::Material>& exterior,
    std::optional<Real> wavelength, const py::object& form, const std::string& preconditioner,
    const std::string& solver_name, const std::string& compression,
    const std::optional<py::dict>& gmres, const std::optional<py::dict>& kernel_opts,
    const std::optional<py::dict>& mlfmm_opts) {
    if (mlfmm_opts) {  // the PML ring exists on the PML route only
        for (const auto& [k, v] : *mlfmm_opts) {
            (void)v;
            const std::string key = dict_key(k, "mlfmm");
            if (key.rfind("pml_ring_", 0) == 0) {
                throw py::value_error("mlfmm: key '" + key +
                                      "' sets the PML ring of Simulation.with_pml; a closed "
                                      "surface has no PML ring");
            }
        }
    }
    SimulationConfig c = make_config(exc, object, exterior, wavelength, form, preconditioner,
                                     solver_name, compression, gmres, kernel_opts, mlfmm_opts);
    geometry::TriangleMesh copy = mesh;
    std::shared_ptr<excitation::Excitation> keep = exc;
    py::gil_scoped_release release;
    auto sim = std::make_unique<PySimulation>(std::move(copy), std::move(exc), std::move(c));
    sim->excitation_ref = std::move(keep);
    return sim;
}

/// PML route (ADR 0009): open surface + PmlTermination.
std::unique_ptr<PySimulation> make_pml_simulation(
    const geometry::OpenSurfaceMesh& open, std::shared_ptr<excitation::Excitation> exc,
    const std::optional<PmlTermination>& pml, const material::Material& object,
    const std::optional<material::Material>& exterior, std::optional<Real> wavelength,
    const py::object& form, const std::string& preconditioner, const std::string& solver_name,
    const std::string& compression, const std::optional<py::dict>& gmres,
    const std::optional<py::dict>& kernel_opts, const std::optional<py::dict>& mlfmm_opts) {
    SimulationConfig c = make_config(exc, object, exterior, wavelength, form, preconditioner,
                                     solver_name, compression, gmres, kernel_opts, mlfmm_opts);
    geometry::OpenSurfaceMesh copy = open;
    const PmlTermination term = pml.value_or(PmlTermination{});
    std::shared_ptr<excitation::Excitation> keep = exc;
    py::gil_scoped_release release;
    auto sim = std::make_unique<PySimulation>(std::move(copy), term, std::move(exc), std::move(c));
    sim->excitation_ref = std::move(keep);
    return sim;
}

GmresResult solve(PySimulation& sim, std::optional<Real> tol, std::optional<int> max_iter,
                  std::optional<int> restart, const std::optional<std::string>& side,
                  const py::object& callback) {
    GmresParams p = sim.config().gmres;
    p.tolerance = tol.value_or(p.tolerance);
    p.max_iter = max_iter.value_or(p.max_iter);
    p.restart = restart.value_or(p.restart);
    if (side) {
        p.side = parse_side(*side);
    }
    const py::object* fn = nullptr;
    if (!callback.is_none()) {
        if (!PyCallable_Check(callback.ptr())) {
            throw py::type_error("callback: expected a callable f(iteration, residual)");
        }
        fn = &callback;
    }
    // Per-iteration hook with the GIL re-acquired: PyErr_CheckSignals() (Ctrl-C ->
    // KeyboardInterrupt), then the user callback. Without a callback it only runs on the main
    // thread (the only one that handles signals) and at most every kSignalCheckInterval.
    // Holds a pointer only: copies of the std::function never touch Python reference counts
    // without the GIL. A Python exception leaves gmres as py::error_already_set and is restored
    // by pybind11 once the GIL is re-acquired.
    solver::IterationCallback cb;
    if (fn != nullptr || on_main_thread()) {
        using Clock = std::chrono::steady_clock;
        cb = [fn, last = Clock::now()](int iteration, Real residual) mutable {
            if (fn == nullptr) {
                const auto now = Clock::now();
                if (now - last < kSignalCheckInterval) {
                    return;
                }
                last = now;
            }
            const py::gil_scoped_acquire gil;
            if (PyErr_CheckSignals() != 0) {
                throw py::error_already_set();
            }
            if (fn != nullptr) {
                (*fn)(iteration, residual);
            }
        };
    }
    return sim.locked([&] {
        GmresResult r = sim.solve(p, cb);
        sim.last_solve = PySimulation::LastSolve{p, r};
        return r;
    });
}

py::object vector_or_scalar(const VectorXr& v, bool scalar) {
    return scalar ? py::object(py::float_(v(0))) : py::cast(v);
}

py::object bistatic_rcs(PySimulation& sim, const py::object& angles, const py::object& plane) {
    if (sim.has_pml()) {
        // A valid call in the wrong state (std::logic_error -> RuntimeError), as scattered_power.
        throw std::logic_error(
            "bistatic_rcs: not available on the PML route (the far field must be windowed); use "
            "Simulation.far_field(directions), which applies the window across the gap");
    }
    Vec3 normal;
    if (py::isinstance<py::str>(plane)) {
        const std::string s = to_lower(plane.cast<std::string>());
        if (s != "xz" && s != "yz") {
            throw py::value_error("plane: expected \"xz\", \"yz\" or a (3,) normal vector");
        }
        normal = s == "xz" ? Vec3::UnitY() : Vec3(-Vec3::UnitX());
    } else {
        const auto n = real_array(plane, "plane");
        if (n.ndim() != 1 || n.shape(0) != 3) {
            throw py::value_error("plane: expected \"xz\", \"yz\" or a (3,) normal vector");
        }
        normal = Eigen::Map<const Vec3>(n.data());
    }
    const auto a = real_array(angles, "angles");
    if (a.ndim() > 1) {
        throw py::value_error("angles: expected a scalar or a 1-D array");
    }
    const VectorXr th = Eigen::Map<const VectorXr>(a.data(), static_cast<Index>(a.size()));
    require_all_finite(th, "angles");
    const VectorXr sigma =
        sim.locked([&] { return post::bistatic_rcs(sim.solution(), normal, th); });
    return vector_or_scalar(sigma, a.ndim() == 0);
}

py::tuple field(PySimulation& sim, const py::object& points, const std::string& kind,
                int quad_degree, Real min_distance_factor) {
    bool single = false;
    const Vertices p = points_from_array(points, "points", &single);
    require_all_finite(p, "points");
    const std::string k = to_lower(kind);
    if (k != "scattered" && k != "total") {
        throw py::value_error("kind: expected \"scattered\" or \"total\"");
    }
    if (sim.has_pml()) {
        // A valid call in the wrong state (std::logic_error -> RuntimeError), as bistatic_rcs.
        throw std::logic_error(
            "field: near fields are not available on the PML route (they would need the "
            "continued currents of the stretched problem); use Simulation.far_field");
    }
    post::FieldOptions opt;
    opt.quad_degree = quad_degree;
    opt.min_distance_factor = min_distance_factor;
    FieldMatrix E, H;
    sim.locked([&] {
        if (k == "total") {
            post::total_field(sim.solution(), p, E, H, opt);
        } else {
            post::scattered_field(sim.solution(), p, E, H, opt);
        }
    });
    return py::make_tuple(fields_to_array(E, single), fields_to_array(H, single));
}

/// "currents" / "huygens" (case-insensitive) -> FarFieldSource (ValueError otherwise).
FarFieldSource parse_far_field_source(const std::string& name) {
    const std::string s = to_lower(name);
    if (s == "currents")
        return FarFieldSource::Currents;
    if (s == "huygens")
        return FarFieldSource::Huygens;
    throw py::value_error("far-field source: expected \"currents\" or \"huygens\", got \"" + name +
                          "\"");
}

const char* far_field_source_name(FarFieldSource s) {
    return s == FarFieldSource::Huygens ? "huygens" : "currents";
}

py::object far_field(PySimulation& sim, const py::object& directions, int quad_degree,
                     const std::optional<post::CurrentWindow>& window,
                     const std::optional<std::string>& source, const std::optional<Real>& z_h,
                     const std::optional<Real>& spacing) {
    bool single = false;
    const Vertices d = points_from_array(directions, "directions", &single);
    require_all_finite(d, "directions");
    // None: the Simulation's default source (PmlTermination.far_field_source; currents on the
    // box route) and window (the PML route's window across the gap; none for the box).
    const FarFieldSource src = source ? parse_far_field_source(*source) : sim.far_field_source();
    if (src == FarFieldSource::Currents && (z_h || spacing)) {
        throw py::value_error("far_field: z_h and spacing apply to the Huygens source only");
    }
    post::FieldOptions opt = sim.far_field_options(quad_degree, src);
    if (src == FarFieldSource::Huygens) {
        // window: the plane window; z_h / spacing override the termination's Huygens options.
        if (window)
            opt.huygens->window = window;
        if (z_h)
            opt.huygens->z_h = z_h;
        if (spacing)
            opt.huygens->spacing = spacing;
    } else if (window) {
        opt.window = window;
    }
    // Simulation::far_field: on the PML route k_z > 0 rows raise ValueError, as drc.
    const FieldMatrix F = sim.locked([&] { return sim.far_field(d, opt); });
    return fields_to_array(F, single);
}

/// Simulation.drc: D (n,) or a float for one (3,) direction; with co_pol the tuple (co, cross).
py::object drc(PySimulation& sim, const py::object& directions, int quad_degree,
               const std::optional<Vec3>& co_pol) {
    bool single = false;
    const Vertices d = points_from_array(directions, "directions", &single);
    require_all_finite(d, "directions");
    if (co_pol) {
        require_all_finite(*co_pol, "co_pol");
        const post::PolarizedIntensity p =
            sim.locked([&] { return sim.drc_polarized(d, *co_pol, quad_degree); });
        return py::make_tuple(vector_or_scalar(p.co, single), vector_or_scalar(p.cross, single));
    }
    const VectorXr v = sim.locked([&] { return sim.drc(d, quad_degree); });
    return vector_or_scalar(v, single);
}

/// Simulation.radiated_power(directions, weights): (n, 3) directions and (n,) weights [sr].
Real radiated_power(PySimulation& sim, const py::object& directions, const py::object& weights,
                    int quad_degree) {
    const Vertices d = points_from_array(directions, "directions");
    require_all_finite(d, "directions");
    const auto w = real_array(weights, "weights");
    if (w.ndim() != 1) {
        throw py::value_error("weights: expected a 1-D array");
    }
    const VectorXr wv = Eigen::Map<const VectorXr>(w.data(), static_cast<Index>(w.size()));
    return sim.locked([&] { return sim.radiated_power(d, wv, quad_degree); });
}

VectorXc matvec(const op::LinearOperator& A, const py::object& x) {
    const auto a = py::array::ensure(x);
    const char k = a ? a.dtype().kind() : '?';
    const bool shape_ok = a && ((a.ndim() == 1 && a.shape(0) == A.cols()) ||
                                (a.ndim() == 2 && a.shape(0) == A.cols() && a.shape(1) == 1));
    if ((k != 'c' && k != 'f' && k != 'i' && k != 'u') || !shape_ok) {
        throw py::value_error("x: expected a numeric array of shape (n,) or (n, 1), n = " +
                              std::to_string(A.cols()));
    }
    const auto c = py::array_t<Complex, py::array::c_style | py::array::forcecast>::ensure(a);
    const VectorXc v = Eigen::Map<const VectorXc>(c.data(), A.cols());
    require_all_finite(v, "x");
    py::gil_scoped_release release;
    return A * v;
}

/// GmresParams as a dict with the keys of the gmres keyword (restart 0 = full GMRES).
py::dict gmres_dict(const GmresParams& p) {
    py::dict g;
    g["tol"] = p.tolerance;
    g["max_iter"] = p.max_iter;
    g["restart"] = p.restart;
    g["side"] = p.side == PreconditionerSide::Left ? "left" : "right";
    g["verbose"] = p.verbose;
    g["orthogonalization"] =
        p.orthogonalization == solver::Orthogonalization::CGS2 ? "cgs2" : "mgs";
    g["keep_hessenberg"] = p.keep_hessenberg;
    return g;
}

/// Resolved configuration as nested dicts whose keys are those of the constructor keywords
/// (gmres / kernels / mlfmm), for the self-describing HDF5 file (specklebem.io, WP23).
py::dict config_dict(const PySimulation& s) {
    const SimulationConfig& c = s.config();
    py::dict d;
    d["wavelength"] = c.wavelength;
    d["formulation"] = py::cast(s.formulation_kind()).attr("name");
    d["formulation_automatic"] = !c.formulation.has_value();
    d["preconditioner"] = s.diagonal_preconditioner() ? "diagonal" : "none";
    d["preconditioner_automatic"] = !c.diagonal_preconditioner.has_value();
    d["solver"] = c.solver == SolverKind::Direct ? "direct" : "gmres";
    d["compression"] = c.compression;
    d["gmres"] = gmres_dict(c.gmres);
    const kernels::OperatorOptions& o = c.kernels;
    py::dict k;
    k["quad_degree_far"] = o.quad_degree_far;
    k["quad_degree_near"] = o.quad_degree_near;
    k["quad_degree_sing"] = o.quad_degree_sing;
    k["outer_grading_levels"] = o.outer_grading_levels;
    k["near_distance_factor"] = o.near_distance_factor;
    k["symmetrize_touching_above_kh"] = o.symmetrize_touching_above_kh;
    k["target_accuracy"] = o.target_accuracy;
    k["quad_degree_rhs"] = o.quad_degree_rhs;
    k["fold_adaptive"] = o.fold_adaptive;
    k["decay_aware_target"] = o.decay_aware_target;
    k["fast_plain_kernel"] = o.fast_plain_kernel;
    k["stretched_touching_points"] = o.stretched_touching_points;
    d["kernels"] = k;
    const mlfmm::MlfmmParams& p = c.mlfmm;
    py::dict f;
    f["accuracy_digits"] = p.accuracy_digits;
    f["max_elements_per_leaf"] = p.octree.max_elements_per_leaf;
    f["min_box_size_lambda"] = p.octree.min_box_size_lambda;
    f["automatic_leaf_size"] = p.automatic_leaf_size;
    f["leaf_radius_quantile"] = p.leaf_radius_quantile;
    f["max_exact_far_bytes"] = p.max_exact_far_bytes;
    f["pml_ring_use_aca"] = p.pml_ring.use_aca;
    f["pml_ring_aca_epsilon"] = p.pml_ring.aca_epsilon;
    f["pml_ring_eta"] = p.pml_ring.eta;
    f["pml_ring_max_rank"] = p.pml_ring.max_rank;
    f["pml_ring_row_clusters"] = p.pml_ring.row_clusters;
    f["pml_ring_max_far_bytes"] = p.pml_ring.max_far_bytes;
    d["mlfmm"] = f;
    return d;
}

/// Copy of the state that specklebem.io.save_hdf5 stores, taken under one lock acquisition so
/// that a concurrent solve() cannot mix two solves in one file.
struct SaveSnapshot {
    std::optional<PySimulation::LastSolve> last_solve;
    std::optional<VectorXc> currents;  ///< if solved()
    std::optional<VectorXc> rhs;       ///< if assembled()
};

/// dict(result=SolveResult | None, params=dict | None, currents=ndarray | None,
/// rhs=ndarray | None): the last successful solve (result and its effective GMRES parameters),
/// the currents if a solution exists and the right-hand side if assembled, all from one lock.
py::dict save_snapshot(PySimulation& s) {
    SaveSnapshot snap = s.locked([&] {
        SaveSnapshot out;
        out.last_solve = s.last_solve;
        if (s.solved())
            out.currents = s.solution().currents;
        if (s.assembled())
            out.rhs = s.rhs();
        return out;
    });
    py::dict d;
    d["result"] = snap.last_solve ? py::cast(std::move(snap.last_solve->result)) : py::none();
    d["params"] = snap.last_solve ? py::object(gmres_dict(snap.last_solve->params)) : py::none();
    d["currents"] = snap.currents ? py::cast(std::move(*snap.currents)) : py::none();
    d["rhs"] = snap.rhs ? py::cast(std::move(*snap.rhs)) : py::none();
    return d;
}

std::string result_repr(const GmresResult& r) {
    return "<SolveResult iterations=" + std::to_string(r.iterations) +
           " converged=" + (r.converged ? "True" : "False") +
           " true_relative_residual=" + std::to_string(r.true_relative_residual) + ">";
}

}  // namespace

namespace {

/// "uniform" / "graded" (any case) or a BoxGradingMode value; ValueError otherwise.
BoxGradingMode parse_grading(const py::object& obj) {
    if (py::isinstance<py::str>(obj)) {
        std::string s = obj.cast<std::string>();
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s == "uniform")
            return BoxGradingMode::Uniform;
        if (s == "graded")
            return BoxGradingMode::Graded;
    } else if (py::isinstance<BoxGradingMode>(obj)) {
        return obj.cast<BoxGradingMode>();
    }
    throw py::value_error("grading: expected \"uniform\", \"graded\" or a BoxGradingMode value");
}

/// Closing-box defaults of truncated rough patches (ADR 0006 with the amendments 2026-10-10 and
/// 2026-10-12b) and the beam-waist rule.
void bind_box_defaults(py::module_& m) {
    py::enum_<BoxGradingMode>(m, "BoxGradingMode",
                              "Closing box of a rough patch (ADR 0006 amendment 2026-10-12b): "
                              "Uniform (box cells = top-face cells, the default) or Graded (the "
                              "lambda_1 / 5 box of the amendment 2026-10-10, explicit option).")
        .value("Uniform", BoxGradingMode::Uniform)
        .value("Graded", BoxGradingMode::Graded);
    m.def("default_box_depth", &default_box_depth, py::arg("object"), py::arg("wavelength"),
          "Default closing-box depth [m] max(5 delta, 2 um) below z = 0 (ADR 0006); raises "
          "ValueError for a lossless dielectric object.");
    m.def("default_box_fine_depth", &default_box_fine_depth, py::arg("object"),
          py::arg("wavelength"), py::arg("sigma"),
          "Fine band [m] 3 delta + 3 sigma of the graded closing box (ADR 0006).");
    m.def("exterior_wavelength", &exterior_wavelength, py::arg("background"), py::arg("wavelength"),
          "Wavelength lambda_0 / |n_1| [m] in the exterior medium R1 (conservative for a lossy "
          "background); raises ValueError for a metallic background (Re(eps_r) <= 0).");
    m.def("graded_box_mesh_size", &graded_box_mesh_size, py::arg("background"),
          py::arg("wavelength"), py::arg("mesh_size"), R"doc(
Nominal coarse spacing [m] of the graded closing box (BoxGradingMode.Graded, ADR 0006 amendment
2026-10-10): the largest 2^M * mesh_size <= lambda_1 / 5 (M >= 0; M = 0 returns mesh_size, the
uniform box). 500 nm in vacuum with mesh_size = 50 nm gives 100 nm. Renamed from
default_box_mesh_size (WP-B3): the default box is uniform since the amendment 2026-10-12b.
Informational: the generator's grid spacing L / round(L / mesh_size) differs from mesh_size, so
pass exterior_wavelength (as rough_surface_box_params(..., grading="graded") does) rather than
this value as box_mesh_size.
)doc");
    m.def("is_weakly_absorbing", &is_weakly_absorbing, py::arg("object"), py::arg("wavelength"),
          py::arg("mesh_size"), R"doc(
"Weakly absorbing" object of ADR 0006: 3 delta > 2 mesh_size (delta = field_decay_length).
Such objects need the fine band and the beam waist L cos(theta_in) / 6. At 500 nm and
mesh_size = 50 nm: Si True, Ag False. A lossless dielectric counts as weakly absorbing.
)doc");
    m.def("is_plasmonic", &is_plasmonic, py::arg("object"), R"doc(
Plasmonic metal: Re(eps_r) < 0 (Ag at 500 nm). ADR 0006 amendment 2026-10-12b: no quantitative
rough-surface results for such objects until WP-V3; flat reflectance and A_ext stay validated.
)doc");
    m.attr("MAX_BEAM_WAIST_FRACTION") = kMaxBeamWaistFraction;
    m.attr("WEAK_ABSORBER_BEAM_WAIST_FRACTION") = kWeakAbsorberBeamWaistFraction;
    m.def("beam_waist_fraction", &beam_waist_fraction, py::arg("object"), py::arg("wavelength"),
          py::arg("mesh_size"), R"doc(
Largest beam waist relative to L cos(theta_in) for an object (ADR 0006 amendment 2026-10-12b):
1/6 for weakly absorbing objects (is_weakly_absorbing; Si), 1/4 otherwise (Ag). Pass it as
check_beam_waist(..., max_fraction=...).
)doc");
    py::class_<RoughBoxParams>(m, "RoughBoxParams",
                               "Closing-box parameters chosen by rough_surface_box_params "
                               "(lengths in metres).")
        .def_readonly("box_depth", &RoughBoxParams::box_depth)
        .def_readonly("box_fine_depth", &RoughBoxParams::box_fine_depth,
                      "Fine band 3 delta + 3 sigma [m] (None for strongly absorbing objects and "
                      "when the band would reach the bottom plate). With the uniform box the band "
                      "does not change the mesh, so grading='uniform' drops it there silently: a "
                      "caller who takes uniform parameters and then sets a coarse box_mesh_size "
                      "must re-check the fine band (default_box_fine_depth; or request "
                      "grading='graded', which handles this case).")
        .def_readonly("box_mesh_size", &RoughBoxParams::box_mesh_size,
                      "Explicit coarse spacing: mesh_size for the uniform box (the default and "
                      "the graded fallback), None for the graded box (the generator caps its "
                      "automatic rule at lambda_1 / 5 with the actual grid spacing).")
        .def_readonly("exterior_wavelength", &RoughBoxParams::exterior_wavelength,
                      "lambda_1 = wavelength / |n_1| in R1.")
        .def_readonly("uniform_fallback", &RoughBoxParams::uniform_fallback,
                      "Graded only: True if the fine band would reach the bottom plate and the "
                      "uniform box was chosen instead (logged as a warning).")
        .def_readonly("grading", &RoughBoxParams::grading, "The requested BoxGradingMode.")
        .def_readonly("beam_waist_fraction", &RoughBoxParams::beam_waist_fraction,
                      "beam_waist_fraction(object, wavelength, mesh_size): pass it as "
                      "check_beam_waist(..., max_fraction=...).")
        .def(
            "kwargs",
            [](const RoughBoxParams& b) {
                py::dict d;
                d["box_depth"] = b.box_depth;
                d["box_mesh_size"] = py::cast(b.box_mesh_size);
                d["box_fine_depth"] = py::cast(b.box_fine_depth);
                d["exterior_wavelength"] = b.exterior_wavelength;
                return d;
            },
            "The box keywords of RoughSurface / make_rough_surface_mesh / "
            "make_mesh_from_height_map as a dict (None entries select the automatic rules).")
        .def("__repr__", [](const RoughBoxParams& b) {
            std::ostringstream os;
            os << std::setprecision(6);
            const auto opt = [&os](const std::optional<Real>& v) {
                if (v.has_value())
                    os << *v;
                else
                    os << "None";
            };
            os << "<RoughBoxParams grading=" << to_string(b.grading) << " box_depth=" << b.box_depth
               << " box_fine_depth=";
            opt(b.box_fine_depth);
            os << " box_mesh_size=";
            opt(b.box_mesh_size);
            os << " exterior_wavelength=" << b.exterior_wavelength
               << " uniform_fallback=" << (b.uniform_fallback ? "True" : "False")
               << " beam_waist_fraction=" << b.beam_waist_fraction << ">";
            return os.str();
        });
    m.def(
        "rough_surface_box_params",
        [](const material::Material& object, const material::Material& background, Real wavelength,
           Real sigma, Real mesh_size, const py::object& grading) {
            return rough_surface_box_params(object, background, wavelength, sigma, mesh_size,
                                            parse_grading(grading));
        },
        py::arg("object"), py::arg("background"), py::arg("wavelength"), py::arg("sigma"),
        py::arg("mesh_size"), py::kw_only(), py::arg("grading") = "uniform",
        R"doc(
Closing-box parameters of a rough patch from the materials (ADR 0006, amendments 2026-10-10 and
2026-10-12b): box_depth = default_box_depth(object), exterior_wavelength = lambda_1 =
wavelength / |n_1|, the mandatory fine band 3 delta + 3 sigma for weakly absorbing objects
(3 delta > 2 mesh_size; Si yes, Ag no at 500 nm and 50 nm) and beam_waist_fraction (1/6 Si,
1/4 Ag).

grading : {"uniform", "graded"} or BoxGradingMode, keyword-only
    "uniform" (default): box_mesh_size = mesh_size, box cells = top-face cells.
    "graded": box_mesh_size = None, the generator caps its automatic coarse spacing at the
    largest 2^M h <= lambda_1 / 5 with the actual grid spacing h; if the fine band would reach
    the bottom plate, the uniform box is chosen (uniform_fallback = True, warning logged).

A plasmonic object (Re(eps_r) < 0, e.g. Ag) with sigma > 0 logs a warning: no quantitative
rough-surface results until WP-V3 (flat patches, sigma = 0, stay validated). Raises ValueError
for a metallic background (Re(eps_r) <= 0). Use as
``RoughSurface(L, sigma, Lc, mesh_size, seed, **p.kwargs())``.

Returns
-------
RoughBoxParams
)doc");
    m.def(
        "check_beam_waist",
        [](Real patch_length, Real waist, Real incidence_angle, bool allow_wide,
           Real max_fraction) {
            check_beam_waist(patch_length, waist,
                             {.incidence_angle = incidence_angle,
                              .allow_wide = allow_wide,
                              .max_fraction = max_fraction});
        },
        py::arg("patch_length"), py::arg("waist"), py::kw_only(), py::arg("incidence_angle") = 0.0,
        py::arg("allow_wide") = false, py::arg("max_fraction") = kMaxBeamWaistFraction, R"doc(
Beam-waist rule of rough patches (ADR 0006 amendments 2026-10-10, 2026-10-11 and 2026-10-12b):
waist <= max_fraction * patch_length * cos(incidence_angle), incidence_angle [rad] with
|incidence_angle| < pi / 2 (the footprint on z = 0 is 1 / cos longer at oblique incidence).
max_fraction in (0, 1/4]: 1/4 (default, strongly absorbing objects) or
beam_waist_fraction(object, wavelength, mesh_size) (1/6 for weakly absorbing objects such as
Si). Raises ValueError for a wider waist unless allow_wide=True (then a warning is logged), and
for non-positive sizes, an invalid angle or max_fraction. Rough-surface drivers must call it:
Simulation cannot determine the patch size from a mesh.
)doc");
}

}  // namespace

void bind_simulation(py::module_& m) {
    py::class_<GmresResult>(m, "SolveResult", R"doc(
Result of Simulation.solve() (C++ solver::GmresResult).

Attributes
----------
iterations : int
    GMRES iterations (0 for the direct solver).
converged : bool
    Monitored residual <= tolerance (always True for the direct solver).
residual_history : ndarray of float64, shape (iterations + 1,)
    Monitored relative residual per iteration (copy); entry 0 is the initial one. Left
    preconditioning monitors |M^-1 (b - Z x)| / |M^-1 b|, right preconditioning the true one.
true_relative_residual : float
    |b - Z x| / |b| of the returned solution, computed explicitly.
wall_seconds : float
    Solve time [s] (assembly excluded).
orthogonalization_seconds : float
    Part of wall_seconds spent in the GMRES Gram-Schmidt steps (0 for the direct solver).
reorthogonalizations : int
    GMRES steps with a second Gram-Schmidt pass ("cgs2": every step; 0 for the direct solver).
x : ndarray of complex128, shape (2N,)
    The solution [J; M] (read-only view kept alive by this result; J in A/m, M in V/m).
hessenberg : ndarray of complex128, shape (k + 1, k)
    With ``gmres=dict(keep_hessenberg=True)``: the Arnoldi Hessenberg matrix of the last GMRES
    cycle before the Givens rotations (C++ GmresResult::hessenberg; Op V_k = V_{k+1} H with
    Op = M^-1 Z for left, Z M^-1 for right preconditioning), e.g. for Ritz values
    ``np.linalg.eigvals(H[:-1])``; shape (0, 0) otherwise, for the direct solver, and when the
    first step of the last cycle hit a singular Hessenberg (only the last restart cycle is
    kept; a discarded column is never included) (copy).
)doc")
        .def_readonly("iterations", &GmresResult::iterations)
        .def_readonly("converged", &GmresResult::converged)
        .def_readonly("true_relative_residual", &GmresResult::true_relative_residual)
        .def_readonly("wall_seconds", &GmresResult::wall_seconds)
        .def_readonly("orthogonalization_seconds", &GmresResult::orthogonalization_seconds)
        .def_readonly("reorthogonalizations", &GmresResult::reorthogonalizations)
        .def_readonly("x", &GmresResult::x)
        .def_property_readonly("hessenberg",
                               [](const GmresResult& r) { return MatrixXc(r.hessenberg); })
        .def_property_readonly("residual_history",
                               [](const GmresResult& r) {
                                   return py::array_t<double>(
                                       static_cast<py::ssize_t>(r.residual_history.size()),
                                       r.residual_history.data());
                               })
        .def("__repr__", &result_repr);

    py::class_<op::LinearOperator, std::shared_ptr<op::LinearOperator>>(m, "LinearOperator",
                                                                        R"doc(
System operator Z of a Simulation (square, complex; never formed explicitly in general).

Obtained from Simulation.operator(); keeps its Simulation alive. ``matvec(x)`` and ``A @ x``
compute Z x with the GIL released; ``as_scipy()`` wraps it for scipy.sparse.linalg.
)doc")
        .def_property_readonly(
            "shape", [](const op::LinearOperator& A) { return py::make_tuple(A.rows(), A.cols()); })
        .def_property_readonly(
            "dtype", [](const op::LinearOperator&) { return py::dtype::of<Complex>(); },
            "numpy.complex128.")
        .def("matvec", &matvec, py::arg("x"), R"doc(
y = Z x.

Parameters
----------
x : array_like, shape (n,) or (n, 1)
    Finite real or complex vector, n = shape[1]; ValueError otherwise.

Returns
-------
ndarray of complex128, shape (n,)
)doc")
        .def("__matmul__", &matvec, py::arg("x"))
        .def(
            "as_scipy",
            [](const py::object& self) {
                const py::module_ spla = py::module_::import("scipy.sparse.linalg");
                return spla.attr("LinearOperator")(self.attr("shape"),
                                                   py::arg("matvec") = self.attr("matvec"),
                                                   py::arg("dtype") = self.attr("dtype"));
            },
            "scipy.sparse.linalg.LinearOperator calling matvec (imports SciPy lazily; "
            "ImportError if SciPy is not installed).")
        .def("describe", &op::LinearOperator::describe, "Human-readable description.")
        .def_property_readonly("memory_bytes", &op::LinearOperator::memory_bytes,
                               "Approximate memory footprint [bytes].")
        .def("__repr__",
             [](const op::LinearOperator& A) { return "<LinearOperator " + A.describe() + ">"; });

    py::class_<post::CurrentWindow>(m, "CurrentWindow", R"doc(
Raised-cosine window on the surface currents of far_field (ADR 0009 decision 5): w = 1 for
t = max(|x|, |y|) <= inner_half, C^1 fall to 0 at outer_half, 0 beyond [m]. The PML route's
default is (taper_end, pml_start), i.e. across the gap.
)doc")
        .def(py::init([](Real inner_half, Real outer_half) {
                 const post::CurrentWindow w{inner_half, outer_half};
                 (void)post::current_window_value(w, 0.0);  // validates (ValueError)
                 return w;
             }),
             py::arg("inner_half"), py::arg("outer_half"))
        .def_readonly("inner_half", &post::CurrentWindow::inner_half)
        .def_readonly("outer_half", &post::CurrentWindow::outer_half)
        .def(
            "__call__",
            [](const post::CurrentWindow& w, Real t) { return post::current_window_value(w, t); },
            py::arg("t"), "Window value w(t) for t = max(|x|, |y|) [m].")
        .def("__repr__", [](const post::CurrentWindow& w) {
            return "<CurrentWindow inner_half=" + std::to_string(w.inner_half) +
                   " outer_half=" + std::to_string(w.outer_half) + ">";
        });
    const PmlTermination dp;
    py::class_<PmlTermination>(m, "PmlTermination", R"doc(
PML termination of an open surface (ADR 0009; Simulation.with_pml).

sigma_max : profile maximum (> 0; 3-5 recommended), sigma(t) = sigma_max ((t - a_2) / W)^order
order : polynomial order (>= 0; < 1 needs a gap >= the longest mesh edge)
correlation_length : Lc [m] of the height map; required (> 0) for a rough patch, the taper must
    be at least Lc wide; 0 for a flat patch
dark_band_tolerance : largest allowed |E_inc| on the PML-start square relative to its maximum on
    the patch (> 0, default 1e-4)
far_field_source : "currents" (windowed surface currents, ADR 0009 decision 5) or "huygens"
    (windowed equivalent currents of a plane in R1 at huygens_z_h below the surface, ADR 0010);
    the source of far_field, drc, radiated_power, reflected_power and reflectance
huygens_z_h : distance [m] of the Huygens plane below the lowest vertex; None = lambda_1
huygens_spacing : grid spacing [m] of the plane; None = lambda_1 / 4 (must be <= lambda_1 / 2)
huygens_window : CurrentWindow of the plane; None = the current window (taper_end, pml_start)
)doc")
        .def(py::init([](Real sigma_max, Real order, Real correlation_length,
                         Real dark_band_tolerance, const std::string& far_field_source,
                         const std::optional<Real>& huygens_z_h,
                         const std::optional<Real>& huygens_spacing,
                         const std::optional<post::CurrentWindow>& huygens_window) {
                 PmlTermination t;
                 t.sigma_max = sigma_max;
                 t.order = order;
                 t.correlation_length = correlation_length;
                 t.dark_band_tolerance = dark_band_tolerance;
                 t.far_field_source = parse_far_field_source(far_field_source);
                 t.huygens.z_h = huygens_z_h;
                 t.huygens.spacing = huygens_spacing;
                 t.huygens.window = huygens_window;
                 return t;
             }),
             py::kw_only(), py::arg("sigma_max") = dp.sigma_max, py::arg("order") = dp.order,
             py::arg("correlation_length") = dp.correlation_length,
             py::arg("dark_band_tolerance") = dp.dark_band_tolerance,
             py::arg("far_field_source") = far_field_source_name(dp.far_field_source),
             py::arg("huygens_z_h") = py::none(), py::arg("huygens_spacing") = py::none(),
             py::arg("huygens_window") = py::none())
        .def_readwrite("sigma_max", &PmlTermination::sigma_max)
        .def_readwrite("order", &PmlTermination::order)
        .def_readwrite("correlation_length", &PmlTermination::correlation_length)
        .def_readwrite("dark_band_tolerance", &PmlTermination::dark_band_tolerance)
        .def_property(
            "far_field_source",
            [](const PmlTermination& t) { return far_field_source_name(t.far_field_source); },
            [](PmlTermination& t, const std::string& s) {
                t.far_field_source = parse_far_field_source(s);
            },
            "\"currents\" or \"huygens\".")
        .def_property(
            "huygens_z_h", [](const PmlTermination& t) { return t.huygens.z_h; },
            [](PmlTermination& t, const std::optional<Real>& v) { t.huygens.z_h = v; },
            "Huygens plane distance below the lowest vertex [m] or None (lambda_1).")
        .def_property(
            "huygens_spacing", [](const PmlTermination& t) { return t.huygens.spacing; },
            [](PmlTermination& t, const std::optional<Real>& v) { t.huygens.spacing = v; },
            "Huygens plane grid spacing [m] or None (lambda_1 / 4).")
        .def_property(
            "huygens_window", [](const PmlTermination& t) { return t.huygens.window; },
            [](PmlTermination& t, const std::optional<post::CurrentWindow>& v) {
                t.huygens.window = v;
            },
            "CurrentWindow of the Huygens plane or None (the current window).")
        .def("__repr__", [](const PmlTermination& t) {
            return "<PmlTermination sigma_max=" + std::to_string(t.sigma_max) +
                   " order=" + std::to_string(t.order) +
                   " correlation_length=" + std::to_string(t.correlation_length) +
                   " dark_band_tolerance=" + std::to_string(t.dark_band_tolerance) +
                   " far_field_source=" + far_field_source_name(t.far_field_source) + ">";
        });

    py::class_<PySimulation>(m, "Simulation", R"doc(
Dense SIE simulation of one object (C++ specklebem::Simulation): assemble, solve, post-process.

Parameters
----------
mesh : TriangleMesh
    Closed surface, normals out of the object R2 into R1 (copied).
excitation : Excitation
    Incident field in R1 (shared with the caller).
object : Material
    Material of R2 (keyword-only, as all following parameters).
exterior : Material, optional
    Material of R1; None = ``excitation.background`` (the only allowed value).
wavelength : float, optional
    Vacuum wavelength [m]; None = ``excitation.wavelength`` (must agree to 1e-12).
formulation : str or Formulation
    "auto" (formulation.recommend(object.eps_r)), "PMCHWT", "ICTF" or "MCTF" (any case).
preconditioner : str
    "auto" (the recommendation's choice), "none" or "diagonal" (Jacobi); GMRES only.
solver : str
    "gmres" or "direct" (dense LU).
compression : str
    "dense" or "mlfmm" (multilevel fast multipole, GMRES only); "aca" / "hmatrix" and unknown
    names raise ValueError. With "mlfmm", lossy interiors (e.g. silver) use the ADR 0008 §6
    policy (far pairs truncated by the decay bound or evaluated exactly); ``assemble()`` raises
    RuntimeError when a region has neither a usable expansion nor enough decay, or when the
    exactly evaluated far part would exceed ``max_exact_far_bytes``.
gmres : dict, optional
    Keys ``tol`` (1e-3), ``max_iter`` (2000), ``restart`` (None = full GMRES), ``side``
    ("left" / "right"), ``verbose`` (True: progress via the C++ log) and
    ``orthogonalization`` ("cgs2": parallel classical Gram-Schmidt with re-orthogonalisation,
    default; "mgs": serial modified Gram-Schmidt), ``keep_hessenberg`` (False; True returns
    the Arnoldi Hessenberg matrix in SolveResult.hessenberg, a diagnostic).
kernels : dict, optional
    Quadrature options (kernels::OperatorOptions member names): quad_degree_far,
    quad_degree_near, quad_degree_sing, outer_grading_levels, near_distance_factor,
    symmetrize_touching_above_kh, target_accuracy, quad_degree_rhs, fold_adaptive,
    decay_aware_target (True: relaxed near/far target in lossy regions, ADR 0004),
    fast_plain_kernel (True: vectorised sin/cos/exp in near/far pairs; False reproduces the
    pre-WP-P2 C-library arithmetic bitwise), stretched_touching_points (4..16, touching pairs
    with a PML triangle; Simulation.with_pml only).
mlfmm : dict, optional
    MLFMM parameters (used with compression="mlfmm"): ``accuracy_digits`` (3; d0 in (0, 5]),
    ``max_elements_per_leaf`` (100), ``min_box_size_lambda`` (0.25; leaf-edge floor in
    exterior wavelengths, a lower bound of the automatic leaf rule),
    ``automatic_leaf_size`` (True: leaf edge >= max(lambda/4 for d0 <= 3, lambda/2 for d0 > 3,
    r_q / 0.6 for d0 <= 3, r_q / 0.3 for d0 > 3); False uses min_box_size_lambda as given),
    ``leaf_radius_quantile`` (1.0 = the global rule with r_max; < 1: local leaf rule, r_q is
    this quantile of the RWG support radii and larger functions live on coarser octree levels;
    opt-in, accuracy_digits <= 3 only, otherwise ValueError)
    and ``max_exact_far_bytes`` (0 = automatic: max(2 x the near-field bytes, 1 GiB), at most
    the dense matrix bytes 16 (2N)^2; budget of the exactly evaluated far interactions of
    lossy regions, checked before allocation); PML route only (Simulation.with_pml): the
    pml_ring_* keys of the PML ring (use_aca, aca_epsilon, eta, max_rank, row_clusters,
    max_far_bytes; docs/10; ValueError for Simulation()).

Raises
------
ValueError
    For unknown strings or dictionary keys and everything the C++ constructor rejects (open
    or inward-oriented mesh, wavelength or background mismatch, invalid options).

Notes
-----
Thread-safe, with limits: calls on one Simulation from several threads are serialised by a
per-object lock (taken with the GIL released), so they run one after another, not in
parallel (use one Simulation per thread for parallel work). The methods that take the lock
are assemble, solve, report, currents, operator, rhs, field, far_field, bistatic_rcs, drc,
radiated_power, reflected_power, reflectance and scattered_power.
From inside a solve() callback they raise RuntimeError instead of waiting: always on the
Simulation being solved, and on another Simulation while that one is busy (an idle one
works). A main thread waiting for the lock stays interruptible with Ctrl-C
(KeyboardInterrupt).
)doc")
        .def(py::init(&make_simulation), py::arg("mesh"), py::arg("excitation"), py::kw_only(),
             py::arg("object"), py::arg("exterior") = py::none(),
             py::arg("wavelength") = py::none(), py::arg("formulation") = "auto",
             py::arg("preconditioner") = "auto", py::arg("solver") = "gmres",
             py::arg("compression") = "dense", py::arg("gmres") = py::none(),
             py::arg("kernels") = py::none(), py::arg("mlfmm") = py::none())
        .def_static("with_pml", &make_pml_simulation, py::arg("open_mesh"), py::arg("excitation"),
                    py::kw_only(), py::arg("pml") = py::none(), py::arg("object"),
                    py::arg("exterior") = py::none(), py::arg("wavelength") = py::none(),
                    py::arg("formulation") = "auto", py::arg("preconditioner") = "auto",
                    py::arg("solver") = "gmres", py::arg("compression") = "dense",
                    py::arg("gmres") = py::none(), py::arg("kernels") = py::none(),
                    py::arg("mlfmm") = py::none(), R"doc(
Simulation of an open surface terminated by a PML (ADR 0009; dense, or "mlfmm" since WP-PML5a).

Parameters
----------
open_mesh : OpenSurfaceMesh
    From make_open_surface_mesh (copied); needs a gap (gap_width > 0).
excitation : Excitation
    Incident field in R1 with a dark band at the PML start (e.g. an AngularSpectrumBeam with
    waist <~ pml_start / 3.5 whose region_radius covers the gap corners).
pml : PmlTermination, optional
    None = PmlTermination() (sigma_max 4, order 2, flat patch, dark band 1e-4).
object, exterior, wavelength, formulation, preconditioner, solver, gmres, kernels, mlfmm :
    As for Simulation(); compression "dense" or "mlfmm" (GMRES only: MLFMM on the functions
    outside the PML plus the PML ring: near pairs exact, far pairs by ACA since WP-PML5b). The
    mlfmm keys pml_ring_use_aca (True; False: the exact far ring of WP-PML5a),
    pml_ring_aca_epsilon (0 = 10^-(accuracy_digits + 1)), pml_ring_eta (admissibility, 2.0),
    pml_ring_max_rank (0 = break-even), pml_ring_row_clusters (True) and
    pml_ring_max_far_bytes (0 = automatic) set the ring (docs/10).

The right-hand side is zero on test functions touching the PML; far_field uses the window
across the gap (far_field_window) by default. Near fields (field()) are not available.

Raises
------
ValueError
    For everything the C++ constructor rejects (orientation, gap, taper < Lc, dark band,
    controlled region, compression, options).
)doc")
        .def(
            "assemble", [](PySimulation& s) { s.locked([&] { s.assemble(); }); },
            "Assemble Z, the right-hand side and (Jacobi GMRES) the diagonal; idempotent.")
        .def("solve", &solve, py::arg("tol") = py::none(), py::arg("max_iter") = py::none(),
             py::arg("restart") = py::none(), py::arg("side") = py::none(),
             py::arg("callback") = py::none(), R"doc(
Assemble if needed and solve; the solution replaces any previous one.

Parameters
----------
tol, max_iter, restart, side : optional
    Per-call GMRES overrides of the ``gmres`` dict (restart 0 = full GMRES); None keeps it.
callback : callable, optional
    ``callback(iteration, residual)`` once per GMRES iteration (iteration = 1, 2, ...),
    called with the GIL held; an exception it raises aborts the solve and propagates. Calls
    from it into locking methods of this Simulation, or of another busy one, raise
    RuntimeError (see the class notes).

Notes
-----
A GMRES solve on the main thread is interruptible with Ctrl-C (KeyboardInterrupt; signals
are checked every iteration with a callback, otherwise every 50 ms). Assembly and the direct
solver are not interruptible. An aborted solve keeps the previous solution.

Returns
-------
SolveResult
)doc")
        .def(
            "report", [](PySimulation& s) { return s.locked([&] { return s.report(); }); },
            "Multi-line text report (formulation, timings, ...).")
        // Unlocked: fixed at construction.
        .def_property_readonly("formulation", &Simulation::formulation_kind,
                               "Formulation used (resolved).")
        .def_property_readonly(
            "preconditioner",
            [](const PySimulation& s) { return s.diagonal_preconditioner() ? "diagonal" : "none"; },
            "Preconditioner of the GMRES solve (resolved): \"diagonal\" or \"none\".")
        .def_property_readonly(
            "solver",
            [](const PySimulation& s) {
                return s.config().solver == SolverKind::Direct ? "direct" : "gmres";
            },
            "\"gmres\" or \"direct\".")
        .def_property_readonly(
            "wavelength", [](const PySimulation& s) { return s.config().wavelength; },
            "Vacuum wavelength [m].")
        .def_property_readonly("num_unknowns", &Simulation::num_unknowns,
                               "Number of unknowns 2N (N interior edges).")
        .def_property_readonly("has_pml", &Simulation::has_pml,
                               "True for Simulation.with_pml (open surface with a PML).")
        .def_property_readonly(
            "far_field_window",
            [](const PySimulation& s) -> std::optional<post::CurrentWindow> {
                return s.far_field_window();
            },
            "Default CurrentWindow of far_field (PML route: (taper_end, pml_start)); None for "
            "the box route.")
        .def_property_readonly("dark_band_ratio", &Simulation::dark_band_ratio,
                               "PML route: measured max |E_inc| on the PML-start square over its "
                               "maximum on the patch; RuntimeError for the box route.")
        .def_property_readonly(
            "zone_half_widths",
            [](const PySimulation& s) {
                const std::array<Real, 4> a = s.zone_half_widths();
                return py::make_tuple(a[0], a[1], a[2], a[3]);
            },
            "PML route: (patch_half, taper_end, pml_start, outer_half) [m]; RuntimeError for the "
            "box route.")
        .def_property_readonly(
            "mesh", [](const PySimulation& s) { return s.mesh(); },
            "The mesh actually used (TriangleMesh copy; triangles as re-oriented by the "
            "constructor, interior edges = RWG functions in the order of the currents).")
        .def_property_readonly(
            "excitation", [](const PySimulation& s) { return s.excitation_ref; },
            "The excitation (shared with the caller).")
        .def_property_readonly(
            "object", [](const PySimulation& s) { return s.config().object; },
            "Material of R2 (copy).")
        .def_property_readonly(
            "exterior", [](const PySimulation& s) { return s.config().exterior; },
            "Material of R1 (copy).")
        .def_property_readonly(
            "zones",
            [](const PySimulation& s) -> py::object {
                if (!s.has_pml())
                    return py::none();
                const std::vector<geometry::SurfaceZone>& z = s.zones();
                py::array_t<std::int64_t> a(static_cast<py::ssize_t>(z.size()));
                auto v = a.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < a.shape(0); ++i)
                    v(i) = static_cast<std::int64_t>(z[static_cast<std::size_t>(i)]);
                return a;
            },
            "PML route: zone per triangle as int64 (SurfaceZone values: 0 Patch, 1 Taper, 2 Gap, "
            "3 Pml); None for the box route.")
        .def_property_readonly(
            "pml_termination",
            [](const PySimulation& s) -> std::optional<PmlTermination> {
                if (!s.has_pml())
                    return std::nullopt;
                return s.pml_termination();
            },
            "PML route: the PmlTermination (copy); None for the box route.")
        .def_property_readonly("config", &config_dict, R"doc(
Resolved configuration (a new dict): wavelength [m], formulation (name), formulation_automatic,
preconditioner ("diagonal" / "none"), preconditioner_automatic, solver, compression, and the
dicts gmres, kernels and mlfmm with every key of the constructor keywords (restart 0 = full
GMRES). Used by specklebem.io.save_hdf5.
)doc")
        .def_property_readonly(
            "last_result",
            [](PySimulation& s) -> std::optional<GmresResult> {
                return s.locked([&]() -> std::optional<GmresResult> {
                    if (!s.last_solve)
                        return std::nullopt;
                    return s.last_solve->result;
                });
            },
            "SolveResult of the last successful solve() (a copy); None before.")
        .def_property_readonly(
            "last_solve_params",
            [](PySimulation& s) -> py::object {
                const std::optional<GmresParams> p = s.locked([&]() -> std::optional<GmresParams> {
                    if (!s.last_solve)
                        return std::nullopt;
                    return s.last_solve->params;
                });
                return p ? py::object(gmres_dict(*p)) : py::none();
            },
            "GMRES parameters of the last successful solve() after its per-call overrides, as a "
            "dict with the keys of the gmres keyword (the direct solver validates but ignores "
            "them); None before.")
        .def_property_readonly(
            "currents",
            [](PySimulation& s) {
                return s.locked([&] { return VectorXc(s.solution().currents); });
            },
            "Solution [J; M] (copy; J in A/m, M in V/m). RuntimeError before solve().")
        .def(
            "operator",
            [](PySimulation& s) {
                return s.locked([&] {
                    return std::const_pointer_cast<op::LinearOperator>(s.system_operator());
                });
            },
            py::keep_alive<0, 1>(),
            "System operator Z (LinearOperator; immutable, matvec is not serialised); "
            "RuntimeError before assemble().")
        .def(
            "rhs", [](PySimulation& s) { return s.locked([&] { return VectorXc(s.rhs()); }); },
            "Right-hand side b (copy); RuntimeError before assemble().")
        .def_property_readonly(
            "assembled", [](PySimulation& s) { return s.locked([&] { return s.assembled(); }); },
            "True once assemble() has completed (operator() and rhs() do not raise).")
        .def_property_readonly(
            "solved", [](PySimulation& s) { return s.locked([&] { return s.solved(); }); },
            "True once a solve() has completed, also a non-converged GMRES solve (currents "
            "does not raise).")
        .def("_save_snapshot", &save_snapshot,
             "Private (specklebem.io.save_hdf5): dict with result (last_result), params "
             "(last_solve_params), currents (None before solve()) and rhs (None before "
             "assemble()), copied under one lock acquisition.")
        .def("field", &field, py::arg("points"), py::arg("kind") = "scattered",
             py::arg("quad_degree") = 6, py::arg("min_distance_factor") = 1e-3, R"doc(
E [V/m] and H [A/m] at points [m] from the solved currents.

Parameters
----------
points : array_like, shape (n, 3) or (3,)
    Finite points; their region (R1 or R2) is found by the winding number of the mesh.
kind : str
    "scattered" (R1: scattered field, R2: total field) or "total" (adds the incident field
    in R1).
quad_degree, min_distance_factor :
    post::FieldOptions; points closer than min_distance_factor * h to the surface raise
    ValueError.

Returns
-------
(E, H) : ndarrays of complex128, shape (n, 3) or (3,)

Raises
------
RuntimeError
    Before solve() and on the PML route (no near fields there; use far_field).
)doc")
        .def("far_field", &far_field, py::arg("directions"), py::arg("quad_degree") = 6,
             py::arg("window") = py::none(), py::kw_only(), py::arg("source") = py::none(),
             py::arg("z_h") = py::none(), py::arg("spacing") = py::none(),
             "Far-field amplitude F [V] with E_s ~ F exp(-jkr) / r for directions (n, 3) "
             "(normalised internally); returns (n, 3) complex128. source: \"currents\" or "
             "\"huygens\" (PML route only, ADR 0010); None = far_field_source. window: "
             "CurrentWindow on the currents (or, for \"huygens\", on the plane); None = "
             "far_field_window (the PML route's window across the gap, none for the box route) "
             "or the termination's huygens_window. z_h, spacing: Huygens plane distance and "
             "grid spacing [m] for this call (None = the termination's; ValueError with "
             "\"currents\"). On the PML route the directions must lie in the reflection "
             "hemisphere k_z <= 0 (ValueError otherwise; R1 is the half space below the "
             "surface). RuntimeError before solve() and for \"huygens\" on the box route.")
        .def_property_readonly(
            "far_field_source",
            [](const PySimulation& s) { return far_field_source_name(s.far_field_source()); },
            "Source of far_field / drc / radiated_power / reflected_power: \"currents\" or "
            "\"huygens\" (PmlTermination.far_field_source; \"currents\" on the box route).")
        .def("bistatic_rcs", &bistatic_rcs, py::arg("angles"), py::arg("plane") = "xz", R"doc(
Bistatic radar cross section sigma(theta) [m^2] (incidence along +z).

Parameters
----------
angles : float or array_like, shape (n,)
    Finite scattering angles theta [rad] from +z (0 = forward, pi = backward).
plane : str or array_like, shape (3,)
    "xz" (normal +y: phi = 0), "yz" (normal -x: phi = pi / 2) or a plane normal
    perpendicular to z (k_hat = rotation of +z about it by theta).

Returns
-------
float or ndarray of float64, shape (n,)

Raises
------
RuntimeError
    Before solve() and on the PML route (use far_field or drc there).
)doc")
        // Unlocked: depends only on the excitation, fixed at construction.
        .def("incident_power", &Simulation::incident_power, R"doc(
Incident power P_inc [W] of the excitation (no solve needed).

AngularSpectrumBeam: its power (through any plane normal to the beam axis); GaussianBeam:
pi w0^2 / (4 eta_1) (paraxial). The normalisation of drc and of the reflectance
reflected_power() / incident_power().

Raises
------
RuntimeError
    For a plane wave (infinite power: normalise by |E_0|^2 / (2 eta_1), i.e. cross sections).
)doc")
        .def("drc", &drc, py::arg("directions"), py::arg("quad_degree") = 6,
             py::arg("co_pol") = py::none(), R"doc(
Absolute differential reflection coefficient D = |F|^2 / (2 eta_1 P_inc) [1/sr].

Parameters
----------
directions : array_like, shape (n, 3) or (3,)
    Finite, non-zero far-field directions (normalised internally). On the PML route they must
    lie in the reflection hemisphere (k_z <= 0), otherwise ValueError.
quad_degree : int
    Dunavant degree of the source quadrature (positive-interior rule).
co_pol : array_like, shape (3,), optional
    Fixed Cartesian co-polarisation vector (e.g. the incident polarisation); then the tuple
    (co, cross) with co = |F . e_co|^2 / (2 eta_1 P_inc) and cross = D - co is returned.

Returns
-------
float or ndarray of float64, shape (n,), or a tuple (co, cross) of them
    Its integral over the reflection hemisphere (weights of hemisphere_grid(-1, ...)) is the
    reflectance. The PML route's current window is applied automatically.

Raises
------
RuntimeError
    Before solve() and for a plane wave (no finite incident power).
)doc")
        .def("radiated_power", &radiated_power, py::arg("directions"), py::arg("weights"),
             py::arg("quad_degree") = 6, R"doc(
Power [W] radiated into the solid angle of a quadrature rule: (1 / 2 eta_1) sum w |F|^2.

Parameters
----------
directions : array_like, shape (n, 3)
    Far-field directions (e.g. SphereGrid.directions; k_z <= 0 on the PML route).
weights : array_like, shape (n,)
    Solid-angle weights [sr] (finite, >= 0; e.g. SphereGrid.weights).
quad_degree : int
    Dunavant degree of the source quadrature.

Returns
-------
float
    Works for a plane wave too (no incident power needed). RuntimeError before solve().
)doc")
        .def(
            "reflected_power",
            [](PySimulation& s, int n_theta, int n_phi, int quad_degree) {
                return s.locked([&] { return s.reflected_power(n_theta, n_phi, quad_degree); });
            },
            py::arg("n_theta"), py::arg("n_phi"), py::arg("quad_degree") = 6, R"doc(
Power [W] into the reflection hemisphere k_z < 0 over hemisphere_grid(-1, n_theta, n_phi).

reflected_power(90, 180) / incident_power() is the reflectance on a 1 deg grid. Needs
n_theta >= 1 and n_phi >= 4 (ValueError otherwise); RuntimeError before solve().
)doc")
        .def(
            "scattered_power",
            [](PySimulation& s, int n_theta, int n_phi, int quad_degree) {
                return s.locked([&] { return s.scattered_power(n_theta, n_phi, quad_degree); });
            },
            py::arg("n_theta"), py::arg("n_phi"), py::arg("quad_degree") = 6, R"doc(
Total scattered power [W] over the full sphere (sphere_grid(n_theta, n_phi)).

For a plane wave the scattering cross section is scattered_power / (|E_0|^2 / (2 eta_1)).
RuntimeError before solve() and on the PML route (the forward hemisphere lies in R2 there).
)doc")
        .def(
            "reflectance",
            [](PySimulation& s, int n_theta, int n_phi, int quad_degree) {
                const Real p_inc = s.incident_power();  // fixed at construction: no lock
                return s.locked([&] { return s.reflected_power(n_theta, n_phi, quad_degree); }) /
                       p_inc;
            },
            py::arg("n_theta"), py::arg("n_phi"), py::arg("quad_degree") = 6, R"doc(
Reflectance R = reflected_power(n_theta, n_phi, quad_degree) / incident_power() (dimensionless).

The integral of drc over the reflection hemisphere k_z < 0 (1 deg grid: n_theta = 90,
n_phi = 180). RuntimeError before solve() and for a plane wave (no finite incident power);
ValueError for n_theta < 1 or n_phi < 4.
)doc");

    py::class_<post::SphereGrid>(m, "SphereGrid", R"doc(
Direction grid with solid-angle weights (hemisphere_grid, sphere_grid).

Gauss-Legendre in the polar angle theta from the grid's pole times the trapezoid rule in phi,
row i * n_phi + j. The weights sum to 2 pi (hemisphere) or 4 pi (sphere) to rounding from
n_theta >= 7 / 8 (Gauss-Legendre rule for int sin(theta)).
)doc")
        .def_property_readonly(
            "directions", [](const post::SphereGrid& g) -> const Vertices& { return g.dirs; },
            py::return_value_policy::reference_internal, "(n, 3) unit directions (read-only view).")
        .def_property_readonly(
            "weights", [](const post::SphereGrid& g) -> const VectorXr& { return g.weights; },
            py::return_value_policy::reference_internal, "(n,) solid-angle weights [sr].")
        .def_property_readonly(
            "theta", [](const post::SphereGrid& g) -> const VectorXr& { return g.theta; },
            py::return_value_policy::reference_internal,
            "(n,) polar angle from the grid's pole [rad].")
        .def("__len__", [](const post::SphereGrid& g) { return g.dirs.rows(); })
        .def("__repr__", [](const post::SphereGrid& g) {
            std::ostringstream o;
            o << "SphereGrid(" << g.dirs.rows() << " directions, sum of weights " << g.weights.sum()
              << " sr)";
            return o.str();
        });
    m.def("hemisphere_grid", &post::hemisphere_grid, py::arg("z_sign"), py::arg("n_theta"),
          py::arg("n_phi"), R"doc(
Hemisphere grid around the pole z_sign * z (z_sign = -1: the reflection hemisphere k_z < 0).

n_theta Gauss-Legendre nodes in theta in (0, pi / 2) times n_phi azimuths phi_j = 2 pi j / n_phi;
weights (pi / 4) w_GL sin(theta) (2 pi / n_phi) [sr]. 1 deg grid: n_theta = 90, n_phi = 180.
ValueError for z_sign not +1 / -1, n_theta < 1 or n_phi < 4.

Returns
-------
SphereGrid
)doc");
    m.def("sphere_grid", &post::sphere_grid, py::arg("n_theta"), py::arg("n_phi"), R"doc(
Full-sphere grid: n_theta Gauss-Legendre nodes in theta in (0, pi) from +z times n_phi azimuths.

Weights (pi / 2) w_GL sin(theta) (2 pi / n_phi) [sr]. ValueError for n_theta < 1 or n_phi < 4.

Returns
-------
SphereGrid
)doc");

    m.def(
        "plane_grid",
        [](const Vec3& origin, const Vec3& u, const Vec3& v, int nu, int nv) {
            require_all_finite(origin, "origin");
            require_all_finite(u, "u");
            require_all_finite(v, "v");
            return post::plane_grid(origin, u, v, nu, nv);
        },
        py::arg("origin"), py::arg("u"), py::arg("v"), py::arg("nu"), py::arg("nv"),
        "(nu * nv, 3) points origin + i / (nu - 1) u + j / (nv - 1) v [m], row i * nv + j "
        "(u, v are the full extents).");
    m.def("cylinder_grid", &post::cylinder_grid, py::arg("radius"), py::arg("theta_min"),
          py::arg("theta_max"), py::arg("n_theta"), py::arg("y_min"), py::arg("y_max"),
          py::arg("n_y"),
          "(n_theta * n_y, 3) points (r sin theta, y, -r cos theta) [m] on a cylinder around "
          "the y-axis, theta [rad] from -z, row k * n_y + l.");
    m.def(
        "polarized_intensity",
        [](const py::object& E, const Vec3& co_pol) {
            const FieldMatrix f = fields_from_array(E, "E");
            const post::PolarizedIntensity p = post::polarized_intensity(f, co_pol);
            return py::make_tuple(p.co, p.cross);
        },
        py::arg("E"), py::arg("co_pol"),
        "(co, cross) intensities [V^2/m^2] of E (n, 3): co = |E . e_co|^2, cross = |E|^2 - co.");
    m.def(
        "intensity",
        [](const py::object& E) {
            bool single = false;
            const FieldMatrix f = fields_from_array(E, "E", &single);
            return vector_or_scalar(f.rowwise().squaredNorm(), single);
        },
        py::arg("E"), "|E|^2 [V^2/m^2] per row of E (n, 3) (a float for one (3,) vector).");
    m.def(
        "differential_reflection_coefficient",
        [](const py::object& E, int n_theta, int n_y, const std::optional<Real>& radius,
           const std::optional<Real>& incident_power,
           const std::optional<material::Material>& background) {
            const FieldMatrix f = fields_from_array(E, "E_cyl");
            if (radius.has_value() != incident_power.has_value()) {
                throw py::value_error(
                    "differential_reflection_coefficient: give both radius and incident_power "
                    "(absolute DRC) or neither (relative DRC)");
            }
            if (!radius) {
                if (background) {
                    throw py::value_error(
                        "differential_reflection_coefficient: background is used only by the "
                        "absolute DRC (with radius and incident_power)");
                }
                return post::differential_reflection_coefficient(f, n_theta, n_y);
            }
            return post::differential_reflection_coefficient(
                f, n_theta, n_y, *radius, *incident_power, background.value_or(material::vacuum()));
        },
        py::arg("E_cyl"), py::arg("n_theta"), py::arg("n_y"), py::kw_only(),
        py::arg("radius") = py::none(), py::arg("incident_power") = py::none(),
        py::arg("background") = py::none(), R"doc(
DRC from fields on cylinder_grid (row k * n_y + l), averaged over the n_y axial samples.

Parameters
----------
E_cyl : array_like, shape (n_theta * n_y, 3)
    Scattered field [V/m] on the cylinder.
n_theta, n_y : int
    Grid counts of cylinder_grid.
radius, incident_power : float, optional
    Cylinder radius [m] and incident power [W] (e.g. Simulation.incident_power()). Without
    them: the relative DRC, mean |E|^2 [V^2/m^2]. With both: the absolute DRC [1/sr],
    r^2 / (2 eta_1 P_inc) mean |E|^2, which approximates the far-field drc only in the far
    zone (deviation ~ 1 / r; r >> L^2 / lambda).
background : Material, optional
    Lossless medium R1 for eta_1 (absolute DRC only; default vacuum).

Returns
-------
ndarray of float64, shape (n_theta,)
)doc");
    bind_box_defaults(m);
}

}  // namespace specklebem::python
