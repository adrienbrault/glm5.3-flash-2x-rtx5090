#!/usr/bin/env python3
"""Rebuild the R3 patch/hashes from authoritative implementation files, without git."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
packet=Path(__file__).resolve().parent
patch=[];source={};patched={}
for f in sorted((packet/'implementation').rglob('*.py')):
    rel=f.relative_to(packet/'implementation');original=a.base/rel
    if original.exists(): source[str(rel)]=hashlib.sha256(original.read_bytes()).hexdigest()
    patched[str(rel)]=hashlib.sha256(f.read_bytes()).hexdigest()
    patch.extend(difflib.unified_diff(original.read_text().splitlines(True) if original.exists() else [],
        f.read_text().splitlines(True),fromfile='a/'+str(rel) if original.exists() else '/dev/null',tofile='b/'+str(rel)))
(packet/'cheapswap.patch').write_text(''.join(patch))
(packet/'source_hashes.json').write_text(json.dumps(source,indent=2)+'\n')
(packet/'patched_hashes.json').write_text(json.dumps(patched,indent=2)+'\n')
