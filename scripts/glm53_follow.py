#!/usr/bin/env python3
"""Follow logs without leaving a Docker CLI holding the GPU flock after parent exit."""
import os
from pathlib import Path
import signal
import subprocess
import sys

p = subprocess.Popen(['sudo','-n','docker','logs','-f','glm53'], stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, start_new_session=True)
def stop(sig, frame):
    raise SystemExit(128 + sig)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    with Path(sys.argv[1]).open('w', buffering=1) as f:
        for line in p.stdout:
            f.write(line)
            print(line, end='', flush=True)
finally:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()
