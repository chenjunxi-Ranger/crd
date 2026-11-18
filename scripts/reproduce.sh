#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3 || command -v python)}"
"$PY" code/verify_release.py
