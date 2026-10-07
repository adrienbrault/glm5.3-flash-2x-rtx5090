FROM tabbyapi:r861-glm-agent-r2
COPY . /opt/r4b/
RUN cd /opt/r4b && sha256sum -c SHA256SUMS \
    && cd /app && sha256sum -c /opt/r4b/SHA256SUMS.base \
    && test ! -e /app/common/glm_tag_safety.py \
    && patch -p1 --fuzz=0 --reverse --batch --dry-run -d /app < /opt/r4b/parent-r861.patch \
    && patch -p1 --fuzz=0 --reverse --batch --no-backup-if-mismatch -d /app < /opt/r4b/parent-r861.patch \
    && cd /app && sha256sum -c /opt/r4b/SHA256SUMS.stock \
    && mkdir -p /opt/r4b/base && cp -a /app/. /opt/r4b/base/ \
    && patch -p1 --fuzz=0 --forward --batch --dry-run -d /app < /opt/r4b/app.patch \
    && patch -p1 --fuzz=0 --forward --batch --no-backup-if-mismatch -d /app < /opt/r4b/app.patch \
    && cd /app && sha256sum -c /opt/r4b/SHA256SUMS.patched \
    && find /app/common /app/backends /app/endpoints/OAI -name __pycache__ -type d -prune -exec rm -rf {} + \
    && R861_APP=/app R861_BASE=/opt/r4b/base bash /opt/r4b/tests/run_applied.sh \
    && rm -rf /opt/r4b/base
ENV TABBY_STREAM_TOOLCALLS=0 TABBY_SSE_KEEPALIVE_S=0 TABBY_GLM_TOOL_FIXES=0 TABBY_GLM_FORCING=0 TABBY_GLM_TAG_TRACE=0 TABBY_GLM_LITERAL_ENCODING=runs TABBY_GLM_CACHE_VERIFY=0
LABEL local.r861.overlay="glm-agent-r4b"
