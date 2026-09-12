#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# The smoke test intentionally uses a fixed tiny model in src/smoke_test.py.
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" -m lazy_onerec.src.smoke_test "$@"
