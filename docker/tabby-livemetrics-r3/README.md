# TabbyAPI live token counters (r3)

Metrics-only overlay on the served image. TabbyAPI books `tabby_generated_tokens_total` and `tabby_prompt_tokens_total` when a request finishes, so a rate computed from them stays flat during long generations. This overlay adds:

- `tabby_generated_tokens_live_total`: completed tokens plus the accepted tokens of requests still generating (rejected MTP drafts are not counted, rewinds subtract, aborted requests settle to their emitted count).
- `tabby_prompt_tokens_live_total`: uncached prompt tokens processed, including in-flight prefill.
- `tabby_generated_tokens_live_drift`: live minus completion total when no request is running (0 when correct).
- `tabby_requests_generating`.

The generation loop does one integer update per step; the counters are read at scrape time.

Build from the repository root (the Dockerfile checks the base files' hashes and the result's):

```sh
docker build --pull=false --network=none -f docker/tabby-livemetrics-r3/Dockerfile \
  -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2 .
```

Measured on 2026-10-10 (R961b, ABAB against the image without the overlay, results `2026-10-10-r961b-glm53-livemetrics-030149`): c4 server ms/token +0.8 % after a linear fit of a drift common to both images, within the 0.7-3 % run-to-run spread; final live total equal to the completion total (21,824 tokens) at the end of each measured window, drift 0.
