# R859 route-trace overlay r1 (codex prefetch round 1): Python-only, default off (EXL3_ROUTE_TRACE unset).
# Parent = the served Flash-Next image tabbyapi:r828-prompt-lookup-r3, pinned by ID.
FROM sha256:dfaed2cba56f353a99589549b6ccb971fe137f5e7d11cbeed522f207fb8b8724
COPY . /opt/route-trace-r1
RUN set -e; cd /opt/route-trace-r1; sha256sum -c SHA256SUMS >/dev/null; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    python3 landing_assert.py "$SP"; \
    cd base/.. && patch --fuzz=0 -p1 -d "$SP" < route-trace.patch; \
    python3 /opt/route-trace-r1/landing_assert.py "$SP" --landed
LABEL r859.overlay="route-trace-r1" r859.parent="sha256:dfaed2cba56f353a99589549b6ccb971fe137f5e7d11cbeed522f207fb8b8724"
