#!/usr/bin/env python3
"""Check exact source fingerprint before overlay application or after landing."""
import argparse
import hashlib
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('package_parent',type=Path,help='directory containing exllamav3/')
p.add_argument('--landed',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parent
reference=root/('src' if a.landed else 'base')
for expected in sorted(reference.rglob('*.py')):
    relative=expected.relative_to(reference)
    target=a.package_parent/relative
    if not target.is_file(): raise SystemExit(f'missing: {target}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(target)!=sha(expected): raise SystemExit(f'fingerprint mismatch: {target}')
if a.landed:
    hook=(a.package_parent/'exllamav3/modules/block_sparse_mlp.py').read_text()
    assert hook.index('record(self, selected_experts')<hook.index('self.cpu_split_submit(y, bsz')
    assert 'REVISION = 1' in (a.package_parent/'exllamav3/route_trace.py').read_text()
print('route-trace landed assert OK' if a.landed else 'route-trace base assert OK')
