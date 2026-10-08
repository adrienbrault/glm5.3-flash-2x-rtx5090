#!/usr/bin/env python3
"""Count graph launch/capture/runtime events in Kineto Chrome JSON; inspect trace for attribution."""
import collections
import json
import sys
from pathlib import Path
for name in sys.argv[1:]:
    data=json.loads(Path(name).read_text());events=data.get('traceEvents',[])
    counts=collections.Counter(e.get('name','') for e in events if 'cudaGraph' in e.get('name',''))
    phases=[(e['name'],e.get('dur',0)) for e in events if e.get('name','').startswith('MTP/') and e.get('ph')=='X']
    print(json.dumps({'file':name,'graph_runtime_counts':dict(counts),'mtp_ranges':len(phases),'note':'Follow CUDA correlation IDs in Perfetto; a configured slot is not proof of replay.'},sort_keys=True))
