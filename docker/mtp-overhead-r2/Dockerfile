FROM tabbyapi:cheapswap-r3-agent-r2_mtpfast1
COPY mtp-overhead.patch SHA256.json apply_overlay.py gpu_probe.py repair_merge_probe.py /opt/mtp-overhead/
RUN python /opt/mtp-overhead/apply_overlay.py
RUN python -c "import torch; from exllamav3.generator import Generator; from exllamav3.util import mtp_phase; assert not mtp_phase.ENABLED; assert 'num_draft_tokens_by_batch' in __import__('inspect').signature(Generator.__init__).parameters; print('[MTP-OVERHEAD] CPU import landing PASS (all new optimization flags default off)')"
