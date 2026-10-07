#!/usr/bin/env bash
set -euo pipefail
R2_OUT_DIR=$(cd "$(dirname "$0")/.." && pwd)
R2_BASE_DIR=$(cd "${1:-$R2_OUT_DIR/../app}" && pwd)
R2_PYTHON=${R2_PYTHON:-python3}
"$R2_PYTHON" -B "$R2_OUT_DIR/tests/check_hashes.py" "$R2_OUT_DIR/SHA256SUMS.base" "$R2_BASE_DIR"
R2_PATCHED_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r861-r2-test.XXXXXX")
trap 'rm -rf "$R2_PATCHED_DIR"' EXIT
# Include unchanged modules used by real request models and template harness.
cp -R "$R2_BASE_DIR/." "$R2_PATCHED_DIR/"
patch -p1 --fuzz=0 --forward --dry-run -d "$R2_PATCHED_DIR" < "$R2_OUT_DIR/app.patch"
patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -d "$R2_PATCHED_DIR" < "$R2_OUT_DIR/app.patch"
"$R2_PYTHON" -B "$R2_OUT_DIR/tests/check_hashes.py" "$R2_OUT_DIR/SHA256SUMS.patched" "$R2_PATCHED_DIR"
R861_APP="$R2_PATCHED_DIR" R861_BASE="$R2_BASE_DIR" R2_PYTHON="$R2_PYTHON" bash "$R2_OUT_DIR/tests/run_applied.sh"
