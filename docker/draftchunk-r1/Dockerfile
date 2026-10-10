ARG BASE
FROM ${BASE}
# Target path verified against the served image's /app layout.
COPY draftchunk.py /tmp/r959c-draftchunk.py
RUN python3 /tmp/r959c-draftchunk.py /app/backends/exllamav3/model.py \
    && grep -Fq 'max_chunk_size=self.chunk_size,' /app/backends/exllamav3/model.py \
    && python3 -c "from pathlib import Path; p=Path('/app/backends/exllamav3/model.py'); s=p.read_text(); compile(s, str(p), 'exec'); exec(Path('/tmp/r959c-draftchunk.py').read_text(), ns := {'__name__': 'verify'}); assert s.count(ns['PATCHED']) == 1" \
    && rm /tmp/r959c-draftchunk.py
