from pathlib import Path
import hashlib
import json
import subprocess
root=Path(__file__).resolve().parent.parent
out=root/'out'
base=root/'base';src=root/'src'
changed=[]
for p in sorted(src.rglob('*')):
 if p.is_file():
  rel=p.relative_to(src);before=base/rel
  if not before.exists() or before.read_bytes()!=p.read_bytes():changed.append(rel)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={str(rel): {'before':digest(base/rel) if (base/rel).exists() else None,'after':digest(src/rel)} for rel in changed}
(out/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
r=subprocess.run(['diff','-ruN','base','src'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert r.returncode==1,r.stderr
(out/'mtp-overhead.patch').write_bytes(r.stdout)
print('packaged',len(changed),'paths',len(r.stdout),'patch bytes')
