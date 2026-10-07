import ast,os
from pathlib import Path
app=Path(os.environ['R861_APP'])
s=(app/'common/glm_tag_safety.py').read_text();b=(app/'backends/exllamav3/model.py').read_text()
for n in ('_plain_codec','trace_prompt','LiteralWindowTrace','TABBY_GLM_LITERAL_ENCODING'):
    assert n in s
assert "'runs'" in s
assert 'literal_window_trace.feed(chunk, token_id_list, xlogger, request_id,' in b
assert '"job_input", request_id' in b
assert 'TABBY_GLM_CACHE_VERIFY' in b
for name in ('common/glm_tag_safety.py','backends/exllamav3/model.py'):ast.parse((app/name).read_text())
print('r4b landing: contiguous ordinary runs, correlated IDs, cache comparison, literal windows')
