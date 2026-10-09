/// Zero-allocation check of kernels::element_blocks and kernels::jump_block (WP7, WP7b: graded
/// outer rules of every level, forced averaging, degree selection).
///
/// Separate executable (specklebem_alloc_tests): it replaces the global allocation functions
/// with counting versions that forward to malloc / aligned_alloc / free (_aligned_malloc /
/// _aligned_free on Windows, which has no std::aligned_alloc). Linking them into the
/// main unit-test binary would disable AddressSanitizer's new/delete mismatch detection for
/// every other test.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/kernels/singularity.hpp"
#include "specklebem/material/material.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <new>
#include <utility>
#include <vector>

namespace {
std::atomic<long long> g_allocations{0};

void* counted_malloc(std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void* counted_aligned(std::size_t size, std::align_val_t align) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    const auto a = static_cast<std::size_t>(align);
    const std::size_t rounded = ((size == 0 ? 1 : size) + a - 1) / a * a;
#ifdef _WIN32
    return _aligned_malloc(rounded, a);
#else
    return std::aligned_alloc(a, rounded);
#endif
}

// Memory from counted_aligned: _aligned_malloc memory must be released with _aligned_free.
void aligned_free(void* p) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void* checked(void* p) {
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}
}  // namespace

void* operator new(std::size_t size) {
    return checked(counted_malloc(size));
}
void* operator new[](std::size_t size) {
    return checked(counted_malloc(size));
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return counted_malloc(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return counted_malloc(size);
}
void* operator new(std::size_t size, std::align_val_t align) {
    return checked(counted_aligned(size, align));
}
void* operator new[](std::size_t size, std::align_val_t align) {
    return checked(counted_aligned(size, align));
}
void* operator new(std::size_t size, std::align_val_t align, const std::nothrow_t&) noexcept {
    return counted_aligned(size, align);
}
void* operator new[](std::size_t size, std::align_val_t align, const std::nothrow_t&) noexcept {
    return counted_aligned(size, align);
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
    std::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    aligned_free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    aligned_free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    aligned_free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    aligned_free(p);
}
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
    aligned_free(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
    aligned_free(p);
}

using namespace specklebem;
using kernels::Proximity;

TEST_CASE("element_blocks and jump_block are allocation-free after the first call", "[kernels]") {
    const Real omega = 2.0 * constants::pi * constants::c0 / 500e-9;
    const material::Material ag = material::silver_500nm();
    const kernels::RegionParams reg{ag.wavenumber(omega), ag.wave_impedance(omega), omega,
                                    constants::eps0 * ag.eps_r, constants::mu0 * ag.mu_r};
    const geometry::TriangleMesh mesh = geometry::make_icosphere(1e-7, 1);
    const basis::RwgSpace space(mesh);
    // Two pairs of every proximity class (default options), the first of each class for the
    // other option sets (keeps the sanitizer build below 1 s).
    std::vector<std::pair<Index, Index>> pairs;
    for (const Proximity cls : {Proximity::far, Proximity::near, Proximity::identical,
                                Proximity::shared_edge, Proximity::shared_vertex}) {
        int found = 0;
        for (Index a = 0; a < mesh.num_triangles() && found < 2; a += 7) {
            for (Index b = 0; b < mesh.num_triangles() && found < 2; ++b) {
                if (kernels::classify(mesh, a, b) == cls) {
                    pairs.emplace_back(a, b);
                    ++found;
                }
            }
        }
        REQUIRE(found == 2);
    }
    // Option sets: defaults (graded outer rule level 4, degree selection; at |k| h ~ 2 touching
    // blocks are averaged over both orderings), the highest grading level (the hot path is the
    // same for every level, only the cached table differs), the unaveraged graded path, another
    // selection target, the WP7 scheme (level 0, fixed degrees).
    std::vector<kernels::OperatorOptions> options(1);
    for (const int level : {0, 6}) {
        kernels::OperatorOptions o;
        o.outer_grading_levels = level;
        options.push_back(o);
    }
    {
        kernels::OperatorOptions o;
        o.symmetrize_touching_above_kh = std::numeric_limits<Real>::infinity();
        options.push_back(o);
        o.target_accuracy = 1e-10;
        options.push_back(o);
        o.target_accuracy = 0.0;
        o.quad_degree_near = 8;
        options.push_back(o);
    }
    Eigen::Matrix<Complex, 3, 3> L;
    Eigen::Matrix<Complex, 3, 3> K;
    Eigen::Matrix<Complex, 3, 3> J;
    const auto run_all = [&]() {
        for (std::size_t o = 0; o < options.size(); ++o) {
            for (std::size_t p = 0; p < pairs.size(); p += (o == 0 ? 1 : 2)) {
                kernels::element_blocks(space, pairs[p].first, pairs[p].second, reg, options[o], L,
                                        K);
            }
        }
    };
    // First calls build the rule cache, the graded tables and the validation table.
    run_all();
    kernels::jump_block(space, 0, J);
    const long long before = g_allocations.load();
    run_all();
    kernels::jump_block(space, 5, J);
    const long long after = g_allocations.load();
    CHECK(after - before == 0);
    CHECK(L.allFinite());
    CHECK(K.allFinite());
}
