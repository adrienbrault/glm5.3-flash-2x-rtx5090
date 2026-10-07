#!/usr/bin/env python3
"""Docker build helper: locate the installed package and require the exact served sources."""
import hashlib, importlib.util, json, os, subprocess
from pathlib import Path
packet=Path(__file__).resolve().parent
spec=importlib.util.find_spec('exllamav3')
assert spec and spec.submodule_search_locations,'installed exllamav3 not found'
root=Path(next(iter(spec.submodule_search_locations)))
for rel,expected in json.loads((packet/'source_hashes.json').read_text()).items():
    actual=hashlib.sha256((root/rel).read_bytes()).hexdigest()
    assert actual==expected,f'base image source mismatch: {root/rel}; expected exact served tree'
subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-d',str(root),'-i',str(packet/'cheapswap.patch')],check=True)
patch=(packet/'cheapswap.patch').read_text()
native=any(line.startswith('+++ b/exllamav3_ext/') for line in patch.splitlines())
if native:
    # Re-enter the package's own JIT build with the updated sources. Never reuse old .so.
    precompiled=importlib.util.find_spec('exllamav3_ext')
    if precompiled and precompiled.origin: Path(precompiled.origin).unlink()
    env=dict(os.environ,TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='4')
    subprocess.run(['python','-c','import exllamav3.ext'],env=env,check=True)
else:
    print('Native sources untouched; preserving the base native extension.')
subprocess.run(['python',str(packet/'landing_assert.py'),'--root',str(root)],check=True)
