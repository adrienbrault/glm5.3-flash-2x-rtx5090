FROM tabbyapi:cheapswap-r3-agent-r2
COPY mtp-fast.patch apply.py test_join_host.py /opt/mtp-fast-r1/
# Python-only, strict baseline digest, fuzz=0, landing digest. No native rebuild.
RUN python3 /opt/mtp-fast-r1/apply.py
# Nothing changes unless the container runs with EXL3_MTP_FAST=1 (launcher must pass it through
# its EXL3_* sanitizer). Boot-log landing check: exactly one "CPU MoE worker started: 43 layers" line
# and exactly one "EXL3_MTP_FAST: 'text' split layers join the 'mtp' CPU worker" line; no "1 layers" line.
