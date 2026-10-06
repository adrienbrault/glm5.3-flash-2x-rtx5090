#!/usr/bin/env python3
"""Stream/archive resources; fail closed before host RAM exhaustion."""
import datetime
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(sys.argv[1])
def stop(sig, frame): raise SystemExit(128+sig)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
with (root/'resources.log').open('a',buffering=1) as f:
    while True:
        pieces=[datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n']
        for cmd in (['free','-b'], ['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw','--format=csv'],
                    ['sudo','-n','docker','stats','--no-stream','--format','{{json .}}','glm53']):
            try: pieces.append(subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=15).stdout)
            except subprocess.TimeoutExpired: pieces.append('RESOURCE SAMPLE TIMEOUT '+repr(cmd)+'\n')
        text=''.join(pieces)
        f.write(text); print(text,end='',flush=True)
        avail=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if avail < 2*(1<<30):
            (root/'host-guard-failed').touch()
            text='HOST RAM GUARD: MemAvailable below 2 GiB; terminating candidate\n'
            f.write(text); print(text,end='',flush=True)
            subprocess.run(['sudo','-n','docker','rm','-f','glm53'],timeout=30)
            break
        time.sleep(2)
