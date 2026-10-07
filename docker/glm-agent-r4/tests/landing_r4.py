"""Assert the r4 prompt and production token-provenance paths landed."""
import ast
import os
from pathlib import Path
app=Path(os.environ['R861_APP'])
safety=(app/'common/glm_tag_safety.py').read_text()
assert 'Tokenizer encoded a literal GLM tag as a control ID' not in safety
assert 'requires a text-only model' not in safety
for filename in ('common/glm_tag_safety.py','backends/exllamav3/model.py',
                 'endpoints/OAI/utils/stream_parser.py','endpoints/OAI/utils/chat_completion.py'):
    ast.parse((app/filename).read_text(),filename=filename)
backend=(app/'backends/exllamav3/model.py').read_text()
collector=(app/'endpoints/OAI/utils/chat_completion.py').read_text()
assert 'glm_tag_safety.ReasoningControls(self.tokenizer)' in backend
assert 'generation["glm_reasoning_controls"] = controls' in backend
assert 'parser.feed_tokenized(text, generation["glm_reasoning_controls"])' in collector
assert 'TABBY_GLM_TAG_TRACE' in safety
assert not list(app.rglob('*.rej'))
print('r4 landing assert: safe literal encoding and ID-aware reasoning path passed')
