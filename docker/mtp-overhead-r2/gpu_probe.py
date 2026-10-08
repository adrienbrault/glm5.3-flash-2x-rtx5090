#!/usr/bin/env python3
"""Run original Tabby entrypoint with exact sampled-ID taps and optional short Kineto trace.
Use inside the served image; this script changes observability only.
"""
import json
import hashlib
import os
from pathlib import Path
import runpy
import sys

# Both settings must be set before exllamav3 is imported (including its phase module).
os.environ.setdefault('EXL3_MTP_PHASE_PROF', '1')
os.environ.setdefault('EXL3_MTP_PHASE_TRACE', '1')
import torch
from exllamav3.generator import Generator, Job

_original_sample = Job.receive_sample

def tapped_sample(self, *args, **kwargs):
    if not hasattr(self, '_mtp_probe_prompt_key'):
        prompt_ids = self.sequences[0].input_ids.torch().reshape(-1).tolist()
        self._mtp_probe_prompt_key = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
    eos, token, rq = result = _original_sample(self, *args, **kwargs)
    ids = getattr(self, '_mtp_probe_ids', None)
    if ids is None:
        ids = self._mtp_probe_ids = []
    ids.extend(token.reshape(-1).tolist())  # original receive_sample has already read this to CPU
    if eos or rq:
        print('[MTP-EXACT] '+json.dumps({'prompt_key': self._mtp_probe_prompt_key, 'serial': self.serial_number, 'identifier': self.identifier,
              'requeue': bool(rq), 'eos': bool(eos), 'ids': ids}), flush=True)
    return result
Job.receive_sample = tapped_sample
_original_requeue = Job.prepare_for_requeue
def tapped_requeue(self, *args, **kwargs):
    new = _original_requeue(self, *args, **kwargs)
    if hasattr(self, '_mtp_probe_prompt_key'):
        new._mtp_probe_prompt_key = self._mtp_probe_prompt_key
    return new
Job.prepare_for_requeue = tapped_requeue

if os.environ.get('EXL3_MTP_REPAIR_MERGE_PROBE', '0') == '1':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import repair_merge_probe
    repair_merge_probe.install()

_original_iterate = Generator.iterate
_prof = None
_count = 0
_skip = int(os.environ.get('EXL3_MTP_TRACE_SKIP', '128'))
_steps = int(os.environ.get('EXL3_MTP_TRACE_STEPS', '32'))
_trace = Path(os.environ.get('EXL3_MTP_TRACE_FILE', '/tmp/mtp-trace.json'))

def traced_iterate(self, *args, **kwargs):
    global _prof, _count
    ready = any(job.is_prefill_done() for job in self.active_jobs)
    # The trace run is a diagnostic arm. Disable with TRACE_STEPS=0 for token-ID-only gates.
    if ready and _steps > 0 and _count == _skip:
        _prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
             torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=False)
        _prof.__enter__()
    try:
        return _original_iterate(self, *args, **kwargs)
    finally:
        if ready:
            _count += 1
            if _prof is not None and _count >= _skip+_steps:
                _prof.__exit__(None, None, None)
                _trace.parent.mkdir(parents=True, exist_ok=True)
                _prof.export_chrome_trace(str(_trace))
                print('[MTP-PROBE] trace='+str(_trace), flush=True)
                _prof = None
Generator.iterate = traced_iterate

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: gpu_probe.py /app/main.py [original Tabby arguments]')
    main = Path(sys.argv[1]).resolve()
    sys.argv = [str(main), *sys.argv[2:]]
    sys.path.insert(0, str(main.parent))
    runpy.run_path(str(main), run_name='__main__')
