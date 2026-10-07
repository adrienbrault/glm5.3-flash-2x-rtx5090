"""Portable equivalent of sha256sum -c for the local CPU runner."""
import hashlib
from pathlib import Path
import sys

manifest, root = Path(sys.argv[1]), Path(sys.argv[2])
for line in manifest.read_text().splitlines():
    digest, relative = line.split('  ', 1)
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f'Hash mismatch: {root / relative}: expected {digest}, got {actual}')
print(f'Verified {manifest.name} against {root}')
