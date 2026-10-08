#include "specklebem/core/logging.hpp"

namespace specklebem::log {

void set_level(Level level) {
    switch (level) {
        case Level::trace:
            spdlog::set_level(spdlog::level::trace);
            break;
        case Level::debug:
            spdlog::set_level(spdlog::level::debug);
            break;
        case Level::info:
            spdlog::set_level(spdlog::level::info);
            break;
        case Level::warn:
            spdlog::set_level(spdlog::level::warn);
            break;
        case Level::error:
            spdlog::set_level(spdlog::level::err);
            break;
        case Level::off:
            spdlog::set_level(spdlog::level::off);
            break;
    }
}

}  // namespace specklebem::log
