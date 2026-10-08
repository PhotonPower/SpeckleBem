#pragma once
/// @file logging.hpp
/// Thin wrapper around spdlog so the rest of the code does not depend on the
/// logging backend directly (allows silencing / redirecting from Python).
#include <spdlog/spdlog.h>

namespace specklebem::log {
enum class Level { trace, debug, info, warn, error, off };
void set_level(Level level);
}  // namespace specklebem::log

#define SBEM_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define SBEM_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define SBEM_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define SBEM_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define SBEM_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
