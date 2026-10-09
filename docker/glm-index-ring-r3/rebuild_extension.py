"""Force the package's own JIT build, even when the base image has a precompiled extension."""
import importlib.util
from pathlib import Path
import shutil
import os
import torch  # Required before any exllamav3_ext load (libtorch symbols/DSOs).

assert os.environ['TORCH_CUDA_ARCH_LIST']=='12.0'
assert os.environ['MAX_JOBS']=='4'
original_find=importlib.util.find_spec
old=original_find('exllamav3_ext')
old_path=Path(old.origin) if old and old.origin else None
importlib.util.find_spec=lambda name,*args,**kw: None if name=='exllamav3_ext' else original_find(name,*args,**kw)
try:
    from exllamav3.ext import exllamav3_ext
finally:
    importlib.util.find_spec=original_find
built=Path(exllamav3_ext.__file__).resolve()
assert built.suffix=='.so',built
assert '/index-ring-extension/' in str(built),built
if old_path and old_path.suffix=='.so' and old_path.resolve()!=built:
    shutil.copy2(built,old_path)
else:
    # Make the rebuilt extension visible even if the base used JIT rather than a wheel.
    import sysconfig
    shutil.copy2(built,Path(sysconfig.get_paths()['platlib'])/built.name)
assert hasattr(exllamav3_ext,'BC_MLAttention')
assert hasattr(exllamav3_ext,'dsa_topk_shadow')
print('Rebuilt extension:',built,flush=True)
