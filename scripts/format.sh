#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
find include src tests examples python/src benchmarks -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' | xargs clang-format -i
command -v ruff >/dev/null && ruff format python tests/python examples/python || true
