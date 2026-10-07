FROM tabbyapi:cheapswap-r3
COPY . /opt/r861/
RUN cd /opt/r861 && sha256sum -c SHA256SUMS \
    && cd /app && sha256sum -c /opt/r861/SHA256SUMS.base \
    && mkdir -p /opt/r861/base \
    && cp -a /app/. /opt/r861/base/ \
    && patch -p1 --fuzz=0 --forward --dry-run -d /app < /opt/r861/app.patch \
    && patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -d /app < /opt/r861/app.patch \
    && sha256sum -c /opt/r861/SHA256SUMS.patched \
    && find /app/endpoints/OAI -name __pycache__ -type d -prune -exec rm -rf {} + \
    && R861_APP=/app R861_BASE=/opt/r861/base bash /opt/r861/tests/run_applied.sh \
    && rm -rf /opt/r861/base
# All behavior changes remain opt-in. Forcing additionally requires tokenizer validation.
ENV TABBY_STREAM_TOOLCALLS=0 TABBY_SSE_KEEPALIVE_S=0 TABBY_GLM_TOOL_FIXES=0 TABBY_GLM_FORCING=0
LABEL local.r861.overlay="glm-agent-r2" local.r882.base="tabbyapi:cheapswap-r3"
