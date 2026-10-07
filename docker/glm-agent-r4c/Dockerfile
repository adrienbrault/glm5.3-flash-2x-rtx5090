# glm-agent-r4c = r4b + `import os` in backends/exllamav3/model.py. r4b's _encode_prompt reads TABBY_GLM_CACHE_VERIFY via
# os.getenv, but the module never imports os: every chat request died with NameError (R881d). No other change.
FROM tabbyapi:r861-glm-agent-r4b
RUN f=/app/backends/exllamav3/model.py \
 && ! grep -q '^import os$' "$f" \
 && sed -i '0,/^import /s//import os\nimport /' "$f" \
 && grep -q '^import os$' "$f" \
 && python -c "import ast,sys; ast.parse(open('$f').read())"
LABEL local.r861.overlay="glm-agent-r4c"
