#!/usr/bin/env bash
set -euo pipefail
R2_TEST_DIR=$(cd "$(dirname "$0")" && pwd)
R2_PYTHON=${R2_PYTHON:-python3}
: "${R861_APP:?Set R861_APP to patched app}"
: "${R861_BASE:?Set R861_BASE to pristine app}"
export TOOLFIX_APP="$R861_APP" TOOLFIX_BASE="$R861_BASE"
export TOOLFIX_TEMPLATE="$R2_TEST_DIR/../glm53-chat_template.jinja"
# Make environment inheritance deterministic; individual tests explicitly opt in.
unset TABBY_STREAM_TOOLCALLS TABBY_SSE_KEEPALIVE_S TABBY_GLM_TOOL_FIXES TABBY_GLM_FORCING
"$R2_PYTHON" -B "$R2_TEST_DIR/landing.py"
"$R2_PYTHON" -B "$R2_TEST_DIR/landing_fixes.py"
for R2_SUITE in test_overlay test_review test_fixes test_r2 test_sse_runtime test_sse_send_disconnect test_lifecycle; do
    "$R2_PYTHON" -B "$R2_TEST_DIR/$R2_SUITE.py"
done
