#!/usr/bin/env python3
"""Build-time apply of mtp-fast.patch onto the installed exllamav3 package: strict baseline digest,
fuzz=0, landing digest, compile, CPU test of _join_host. Imports no CUDA extension."""
import hashlib, importlib.util, py_compile, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = "modules/block_sparse_mlp_cpu.py"
BEFORE = "89af8491e4aa422f782191e825501171f30bf8eb696e34010cc41b7b23fcbfe6"   # cheapswap-r3 landing
AFTER = "e6144c43ee514f00a0133b12e5ce384ccf3842733fb4538eb885c23d9491aa83"

spec = importlib.util.find_spec("exllamav3")
assert spec is not None and spec.origin is not None, "installed exllamav3 package not found"
pkg = Path(spec.origin).resolve().parent
path = pkg / REL
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(path) == BEFORE, f"wrong baseline {path}: {digest(path)}"
assert shutil.which("patch"), "patch utility missing in base image"
subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-d", str(pkg),
                "-i", str(HERE / "mtp-fast.patch")], check = True)
assert digest(path) == AFTER, f"landing digest mismatch {path}: {digest(path)}"
py_compile.compile(str(path), doraise = True)
subprocess.run([sys.executable, "-I", str(HERE / "test_join_host.py"), str(path)], check = True)
print("mtp-fast r1: exact baseline, fuzz=0, landing digest, compile, join-host tests PASS", flush = True)
