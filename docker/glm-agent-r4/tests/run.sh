#!/usr/bin/env bash
set -euo pipefail
R4_PACKAGE=$(cd "$(dirname "$0")/.." && pwd)
R4_INPUT=$(cd "${1:?Usage: tests/run.sh /path/to/stock-or-r2/app [stock|r2]}" && pwd)
R4_KIND=${2:-stock}
R4_PYTHON=${R4_PYTHON:-python3}
R4_WORK=$(mktemp -d "${TMPDIR:-/tmp}/glm-r4-test.XXXXXX")
trap 'rm -rf "$R4_WORK"' EXIT
mkdir -p "$R4_WORK/app" "$R4_WORK/base"
cp -R "$R4_INPUT/." "$R4_WORK/app/"
if [[ $R4_KIND == r2 ]]; then
    "$R4_PYTHON" -B "$R4_PACKAGE/tests/check_hashes.py" "$R4_PACKAGE/SHA256SUMS.base" "$R4_WORK/app"
    patch -p1 --fuzz=0 --reverse --batch --dry-run -d "$R4_WORK/app" < "$R4_PACKAGE/parent-r861.patch"
    patch -p1 --fuzz=0 --reverse --batch --no-backup-if-mismatch -d "$R4_WORK/app" < "$R4_PACKAGE/parent-r861.patch"
elif [[ $R4_KIND != stock ]]; then
    echo 'Input kind must be stock or r2' >&2; exit 2
fi
"$R4_PYTHON" -B "$R4_PACKAGE/tests/check_hashes.py" "$R4_PACKAGE/SHA256SUMS.stock" "$R4_WORK/app"
cp -R "$R4_WORK/app/." "$R4_WORK/base/"
test ! -e "$R4_WORK/app/common/glm_tag_safety.py"
patch -p1 --fuzz=0 --forward --batch --dry-run -d "$R4_WORK/app" < "$R4_PACKAGE/app.patch"
patch -p1 --fuzz=0 --forward --batch --no-backup-if-mismatch -d "$R4_WORK/app" < "$R4_PACKAGE/app.patch"
R861_APP="$R4_WORK/app" R861_BASE="$R4_WORK/base" R4_PYTHON="$R4_PYTHON" bash "$R4_PACKAGE/tests/run_applied.sh"
