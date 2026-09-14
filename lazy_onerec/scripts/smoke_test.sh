#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# The smoke test intentionally uses a fixed tiny model in src/smoke_test.py.
python_bin="${python_bin:-python3}"

exec "${python_bin}" -m lazy_onerec.src.smoke_test
