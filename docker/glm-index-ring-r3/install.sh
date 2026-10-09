#!/usr/bin/env bash
# Apply from the real installed package root: GNU patch will reject a symlinked -d root.
set -euo pipefail
packet_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
package_dir=$(PYTHONPATH=/app python - "$@" <<'PY'
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(importlib.util.find_spec('exllamav3').origin).parent
print(root.resolve(strict=True))
PY
)
(cd "$package_dir" && sha256sum -c "$packet_dir/SOURCE-SHA256SUMS")
patch --batch --forward --fuzz=0 --no-backup-if-mismatch -p2 -d "$package_dir" < "$packet_dir/ring3.patch"
(cd "$package_dir" && sha256sum -c "$packet_dir/PATCHED-SHA256SUMS")
PYTHONPATH=/app python - "$package_dir" <<'PY'
import ast
from pathlib import Path
import sys
root = Path(sys.argv[1])
for path in root.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
print('RING3 Python syntax PASS')
PY
# Generated from the actual base/src native diff, rather than unconditionally rebuilding.
if [[ -s "$packet_dir/NATIVE-CHANGED.txt" ]]; then
    (cd /app && PYTHONPATH=/app TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 python "$packet_dir/rebuild_extension.py")
fi
(cd /app && PYTHONPATH=/app python - <<'PY'
import torch  # libtorch must be loaded before the native extension.
import exllamav3_ext
from exllamav3.cache.mla_index_ring import INDEX_RING, INDEX_RING_SHADOW
assert not INDEX_RING and not INDEX_RING_SHADOW
assert hasattr(exllamav3_ext, 'BC_MLAttention')
assert hasattr(exllamav3_ext, 'dsa_topk_shadow')
import exllamav3.model.moe_split_check  # splitdev2 must still be present.
print('RING3 landing PASS:', torch.__version__, exllamav3_ext.__file__)
PY
)
for test in test_cpu test_state test_shadow test_graph_fixture; do
    (cd /app && PYTHONPATH=/app python "$packet_dir/tests/$test.py" --source "$package_dir")
done
for test in test_checker test_r929 test_unit_offline; do
    (cd /app && PYTHONPATH=/app python "$packet_dir/tests/$test.py")
done
