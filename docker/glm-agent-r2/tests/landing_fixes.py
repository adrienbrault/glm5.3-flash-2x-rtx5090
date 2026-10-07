"""Exact landed-source hashes, syntax and independent flag assertions; no GPU imports."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

OUT=Path(__file__).resolve().parents[1]
APP=Path(os.environ.get('TOOLFIX_APP','/app'))

def hashes(manifest,root):
    for line in manifest.read_text().splitlines():
        digest,relative=line.split(maxsplit=1)
        actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
        assert actual==digest,(relative,actual,digest)

hashes(OUT/'SHA256SUMS.patched',APP)
for line in (OUT/'SHA256SUMS.patched').read_text().splitlines():
    _,relative=line.split(maxsplit=1)
    if relative.endswith('.py'):ast.parse((APP/relative).read_text(),filename=relative)
spec=importlib.util.spec_from_file_location('landing_fixes',APP/'endpoints/OAI/utils/glm_tool_fixes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'0','TABBY_STREAM_TOOLCALLS':'1'}):
    assert not m.enabled('glm4_5')
with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1','TABBY_STREAM_TOOLCALLS':'0'}):
    assert m.enabled('glm4_5') and not m.enabled('qwen3_coder')
    assert m.typed_value('123',{'type':'string'})=='123'
    assert m.typed_value('null',{'type':['integer','null']}) is None
with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1','TABBY_GLM_FORCING':'0'}):
    assert not m.forcing_enabled('glm4_5')
with patch.dict(os.environ,{'TABBY_GLM_TOOL_FIXES':'1','TABBY_GLM_FORCING':'1'}):
    assert m.forcing_enabled('glm4_5')
print('landing: exact source hashes, syntax and independent GLM flag passed')
