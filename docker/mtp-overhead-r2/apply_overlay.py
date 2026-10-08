"""CPU-only image landing; originals and results both verified, no fuzzy/offset patches."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

bundle=Path(__file__).resolve().parent
manifest=json.loads((bundle/'SHA256.json').read_text())
exl=Path(importlib.util.find_spec('exllamav3').submodule_search_locations[0])
roots={'exllamav3':exl,'tabbyapi':Path('/app')}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def real(rel):
    group,*tail=Path(rel).parts
    return roots[group].joinpath(*tail)
with tempfile.TemporaryDirectory(prefix='mtp-overlay-') as tmp:
    stage=Path(tmp)
    for rel,hashes in manifest.items():
        original=real(rel);target=stage/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if hashes['before'] is None:
            assert not original.exists(), f'new landing path already exists: {original}'
        else:
            assert original.is_file() and digest(original)==hashes['before'], f'original SHA256 mismatch: {original}'
            shutil.copyfile(original,target)
    r=subprocess.run(['patch','--batch','--forward','-p1','--fuzz=0','-i',str(bundle/'mtp-overhead.patch')],cwd=stage,text=True,capture_output=True)
    print(r.stdout,end='')
    assert r.returncode==0,r.stderr
    assert 'offset' not in r.stdout.lower() and 'fuzz' not in r.stdout.lower(),r.stdout
    for rel,hashes in manifest.items():
        patched=stage/rel
        assert digest(patched)==hashes['after'],f'patched SHA256 mismatch: {rel}'
        ast.parse(patched.read_bytes(),str(patched))
    # Install only after all source checks and the entire patch succeed.
    for rel,hashes in manifest.items():
        dest=real(rel);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(stage/rel,dest)
        assert digest(dest)==hashes['after']
        shutil.rmtree(dest.parent/'__pycache__',ignore_errors=True)
assert '[MTP-OVERHEAD] landed' in (exl/'generator/generator.py').read_text()
assert 'class PhaseProfiler' in (exl/'util/mtp_phase.py').read_text()
print(f'[MTP-OVERHEAD] landing SHA256 verified: {len(manifest)} files; patch fuzz=0, offsets=0',flush=True)
