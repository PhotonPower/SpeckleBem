#pragma once
/// @file timer.hpp
/// Scoped wall-clock timer used for the performance reports (assembly time,
/// iteration time, MLFMM level timings).
#include <chrono>
#include <string>

namespace specklebem {

class ScopedTimer {
public:
    explicit ScopedTimer(std::string label);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    [[nodiscard]] double elapsed_seconds() const;

private:
    std::string label_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace specklebem
