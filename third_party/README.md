# Third-party code

No vendored sources. Dependencies are resolved by `cmake/Dependencies.cmake`
(system packages first, FetchContent fallback). This directory exists for
small, licence-compatible single-file dependencies that cannot be fetched
(e.g. a Dunavant quadrature table), each with its licence file next to it.
