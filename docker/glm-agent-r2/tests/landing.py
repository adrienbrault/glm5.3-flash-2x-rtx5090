"""Assert that the source landed and both endpoint paths use the idle wrapper."""
import ast
import os
from pathlib import Path

app = Path(os.environ.get('R861_APP', '/app'))
router = ast.parse((app / 'endpoints/OAI/router.py').read_text())
for endpoint in ('completion_request', 'chat_completion_request'):
    function = next(n for n in router.body if isinstance(n, ast.AsyncFunctionDef) and n.name == endpoint)
    response_calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == 'keepalive_response']
    assert len(response_calls) == 1, endpoint
    assert len(response_calls[0].args) == 2, endpoint
    assert response_calls[0].args[1].id == 'disconnect_handler', endpoint
for relative in ('endpoints/OAI/router.py', 'endpoints/OAI/utils/chat_completion.py',
                 'endpoints/OAI/utils/sse_keepalive.py',
                 'endpoints/OAI/utils/toolcall_formats/glm4_5_stream.py'):
    compile((app / relative).read_text(), str(app / relative), 'exec')
    assert not (app / (relative + '.rej')).exists()
    assert not (app / (relative + '.orig')).exists()
print('r861 landing assert passed')
