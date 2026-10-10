"""
Prometheus metrics for TabbyAPI.

A small in-process registry plus a text exposition writer (Prometheus format
0.0.4), so the monitoring stack can scrape GET /metrics without adding
prometheus_client to the image; the container build has no package-install
step for it.

Wiring:
  - common.gen_logging.log_metrics calls record_completion() once per
    completed generation job (the existing per-request completion hook).
  - backends.exllamav3.model observes raw results with a request-local live
    tracker; scrapes read accepted engine counts and cleanup drops unbooked work.
  - endpoints.core.router serves GET /metrics from render_metrics(),
    unauthenticated like /health.

Conventions enforced here:
  - Every recorded series carries a `model` label with the served model id
    (the container's model_dir.name, the same id /v1/model reports). Nothing
    request-shaped (prompts, request ids, context lengths) is ever a label
    value; distributions go into histogram buckets.
  - Rates are exported as pairs of counters (tokens and seconds) so a PromQL
    rate() ratio stays time-weighted. No precomputed average-rate gauges.
  - record_completion() runs on the request completion path and never raises;
    a failure is logged once and swallowed.

This module imports only stdlib at module level. It is pulled in by
common.gen_logging, which is itself imported while the common.model ->
backends.exllamav3.model import chain may still be running, so the container
is reached through sys.modules and the logger through a deferred import.
"""

import bisect
import logging
import math
import sys
import threading
from typing import Optional
from os import getenv

# Histogram bucket bounds, as le= values.

# Generation length: resolves the 2.7k-4k region where slow tails were seen.
GENERATED_TOKEN_BUCKETS = (16, 64, 256, 512, 1024, 2048, 3072, 4096, 8192, 16384, 32768)
# Context depth.
PROMPT_TOKEN_BUCKETS = (1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
# Per-request decode rate, tokens per second.
DECODE_RATE_BUCKETS = (10, 20, 40, 60, 80, 120, 160, 200, 250, 300, 400)
# Queue wait and time to first token.
SECONDS_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class _Counter:
    """Cumulative monotonic counter, keyed by model label (None = unlabeled)."""

    type = "counter"

    def __init__(self, name, doc):
        self.name = name
        self.doc = doc
        self.values = {}

    def add(self, label, amount):
        self.values[label] = self.values.get(label, 0.0) + amount


class _Histogram:
    """Fixed-bucket histogram, keyed by model label (None = unlabeled)."""

    type = "histogram"

    def __init__(self, name, doc, buckets):
        self.name = name
        self.doc = doc
        self.buckets = tuple(buckets)
        self.values = {}

    def observe(self, label, value):
        entry = self.values.get(label)
        if entry is None:
            entry = self.values[label] = {
                "buckets": [0] * (len(self.buckets) + 1),
                "sum": 0.0,
                "count": 0,
            }
        entry["buckets"][bisect.bisect_left(self.buckets, value)] += 1
        entry["sum"] += value
        entry["count"] += 1


# Metric families. Declaration order is the order in the exposition.

_REQUESTS = _Counter(
    "tabby_requests_total",
    "Completed generation jobs. The completion hook only sees jobs that "
    "reached an EOS result; cancelled and errored requests never reach it.",
)
_PROMPT_TOKENS = _Counter("tabby_prompt_tokens_total", "Prompt tokens processed.")
_GENERATED_TOKENS = _Counter("tabby_generated_tokens_total", "Tokens generated.")
_CACHED_TOKENS = _Counter(
    "tabby_cached_tokens_total",
    "Prompt tokens served from the prefix cache instead of being processed.",
)
_PROMPT_SECONDS = _Counter("tabby_prompt_seconds_total", "Seconds spent in prefill.")
_GENERATE_SECONDS = _Counter("tabby_generate_seconds_total", "Seconds spent generating.")
_QUEUE_SECONDS = _Counter("tabby_queue_seconds_total", "Seconds jobs spent queued.")
_RECORD_ERRORS = _Counter(
    "tabby_metrics_record_errors_total",
    "Completion recordings that failed inside the metrics code and were swallowed.",
)

_COUNTERS = [
    _REQUESTS,
    _PROMPT_TOKENS,
    _GENERATED_TOKENS,
    _CACHED_TOKENS,
    _PROMPT_SECONDS,
    _GENERATE_SECONDS,
    _QUEUE_SECONDS,
    _RECORD_ERRORS,
]

_GEN_LEN = _Histogram(
    "tabby_generated_tokens", "Generated tokens per request.", GENERATED_TOKEN_BUCKETS
)
_PROMPT_LEN = _Histogram(
    "tabby_prompt_tokens", "Prompt tokens per request.", PROMPT_TOKEN_BUCKETS
)
_DECODE_RATE = _Histogram(
    "tabby_decode_tokens_per_second",
    "Per-request decode rate in tokens per second.",
    DECODE_RATE_BUCKETS,
)
_TTFT = _Histogram(
    "tabby_time_to_first_token_seconds",
    "Queue plus prefill time per request, as the client experiences it.",
    SECONDS_BUCKETS,
)
_QUEUE_HIST = _Histogram("tabby_queue_seconds", "Queue wait per request.", SECONDS_BUCKETS)

_HISTOGRAMS = [_GEN_LEN, _PROMPT_LEN, _DECODE_RATE, _TTFT, _QUEUE_HIST]

# Gauges are not stored state; their values are read live at render time.
# (name, doc, container attribute)
_REQUESTS_ACTIVE = (
    "tabby_requests_active",
    "Requests inside stream_generate: queued or generating.",
)
_CONTAINER_GAUGES = [
    ("tabby_cache_size_tokens", "Configured KV/prefix cache size in tokens.", "cache_size"),
    ("tabby_max_batch_size", "Configured maximum batch size.", "max_batch_size"),
    ("tabby_max_seq_len", "Configured maximum sequence length in tokens.", "max_seq_len"),
]

_LOCK = threading.Lock()

# Last model label seen, so a request that outlives a model unload still lands
# on the series it was counted under.
_last_model_label = "unknown"
_warned = False


def _num(value) -> Optional[float]:
    """Coerce a payload value to a finite float; None when it isn't a number."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _nonneg(value) -> float:
    """Finite float clamped at zero; counters must stay monotonic."""

    result = _num(value)
    if result is None or result < 0:
        return 0.0
    return result


def _container():
    """The live model container, or None. Never triggers an import."""

    model_mod = sys.modules.get("common.model")
    return getattr(model_mod, "container", None)


def _model_label() -> str:
    global _last_model_label

    container = _container()
    name = getattr(getattr(container, "model_dir", None), "name", None)
    if name:
        _last_model_label = str(name)
    return _last_model_label


# Live state belongs to the asyncio generation consumer. Updates never await,
# and /metrics runs on that same event loop; no additional hot-path lock is
# needed. The legacy completion registry and its lock remain unchanged.
LIVE_METRICS_ENABLED = getenv("TABBY_METRICS_LIVE_ENABLED", "true").lower() not in (
    "0", "false", "off", "no",
)
_GENERATED_TOKENS_LIVE = _Counter(
    "tabby_generated_tokens_live_total",
    "Booked completion tokens plus accepted tokens in flight. Rewinds and aborts can decrease this value.",
)
_PROMPT_TOKENS_LIVE = _Counter(
    "tabby_prompt_tokens_live_total", "Uncached prompt tokens processed, including in-flight prefill.",
)
# Exact completion reconciliation permits rollback (rewind, abort). A gauge
# is required even though the historical live series name ends in _total.
_GENERATED_TOKENS_LIVE.type = "gauge"
_COUNTERS += [_PROMPT_TOKENS_LIVE]
_REQUESTS_GENERATING = (
    "tabby_requests_generating", "Jobs currently decoding, excluding queue and prefill.",
)
_generating = {}
_live_jobs = set()
_LIVE_DRIFT = (
    "tabby_generated_tokens_live_drift",
    "Settled live tokens minus booked completion tokens; must be zero. Excludes in-flight work.",
)
_live_warned = False


def _note_live_error(exc):
    global _live_warned

    if not _live_warned:
        _live_warned = True
        try:
            logging.getLogger(__name__).warning("metrics: live recording failed: %r", exc)
        except Exception:
            pass


def start_generation(context_len, model_label=None, job=None, page_size=None):
    """Create a request-local tracker, or do nothing when live metrics are off."""

    if not LIVE_METRICS_ENABLED:
        return None
    try:
        return _LiveGeneration(model_label or _model_label(), context_len, job, page_size)
    except Exception as exc:
        _note_live_error(exc)
        return None


class _LiveGeneration:
    def __init__(self, model_label, context_len, job=None, page_size=None):
        self.model_label = model_label
        self.prompt_tokens = _nonneg(context_len)
        self.cached_tokens = None
        self.job = job
        self.page_size = page_size
        self.generated = 0
        self.eos_total = None
        self.prompt_booked = 0.0
        self.prefill_done = False
        self.decoding = False
        self.closed = False
        _GENERATED_TOKENS_LIVE.add(model_label, 0)
        _PROMPT_TOKENS_LIVE.add(model_label, 0)
        _generating.setdefault(model_label, 0)
        _live_jobs.add(self)

    def _prompt_progress(self, result, progress=None):
        if result.get("prompt_tokens") is not None:
            self.prompt_tokens = _nonneg(result["prompt_tokens"])
        job = result.get("job") or self.job
        if job is not None:
            self.job = job
        if result.get("cached_tokens") is not None:
            # EOS carries the authoritative, already normalized cache total.
            self.cached_tokens = _nonneg(result["cached_tokens"])
        elif job is not None:
            # Match Job.emit's EOS formula, including requeues and CFG. These
            # are Python scalars, not tensors; PAGE_SIZE comes from the engine.
            cached = getattr(job, "rq_cached", None)
            if cached is not None:
                self.cached_tokens = _nonneg(cached[1])
            else:
                pages = getattr(job, "cached_pages", None)
                tokens = getattr(job, "cached_tokens", None)
                if pages is not None and tokens is not None:
                    if pages and self.page_size is None:
                        return  # Unknown page size: defer booking to EOS.
                    self.cached_tokens = _nonneg(
                        (pages * (self.page_size or 0) + tokens) // len(job.sequences)
                    )
        if self.cached_tokens is None:
            return  # Missing cache data must never book the whole prompt.
        processed = max(self.prompt_tokens - self.cached_tokens, 0)
        if progress is not None:
            # Engine progress sums KV positions across sequences and includes
            # cache hits. Clamp to this event's prefill extent, then normalize.
            maximum = _num(result.get("max_progress"))
            if maximum is not None:
                progress = min(progress, max(maximum, 0))
            progress = progress // len(job.sequences) if job is not None else progress
            processed = min(processed, max(progress - self.cached_tokens, 0))
        delta = max(processed - self.prompt_booked, 0)
        _PROMPT_TOKENS_LIVE.add(self.model_label, delta)
        self.prompt_booked += delta

    def _prefilled(self, result):
        self._prompt_progress(result)
        self.prefill_done = True
        self.decoding = True
        _generating[self.model_label] += 1

    def _generated_now(self):
        # EOS is an immutable snapshot; Job can advance ahead of its consumer.
        if self.eos_total is not None:
            return self.eos_total
        if self.job is not None:
            # Exactly Job.emit's logical-request formula. new_tokens advances
            # only for sampled/accepted tokens, decreases on checkpoint rewind,
            # starts at -1 for healing, and excludes unaccepted draft positions.
            return max(getattr(self.job, "rq_new_tokens", 0) + self.job.new_tokens, 0)
        return self.generated  # Compatibility for engines without a Job handle.

    def _stop_decoding(self):
        if self.decoding:
            _generating[self.model_label] -= 1
            self.decoding = False

    def observe(self, result):
        """Observe one raw backend result before merging, yielding or polling."""

        if self.closed:
            return
        try:
            stage = result.get("stage")
            if stage == "prefill" and not self.prefill_done:
                progress = _num(result.get("curr_progress"))
                if progress is not None:
                    self._prompt_progress(result, max(progress, 0))
            if stage != "streaming" and not result.get("eos"):
                return
            if not self.prefill_done:
                self._prefilled(result)
            elif result.get("eos"):
                self._prompt_progress(result)
            if result.get("eos"):
                self.eos_total = _nonneg(result.get("new_tokens"))
                self._stop_decoding()
            elif self.job is None:
                total = result.get("new_tokens")
                if total is not None:
                    self.generated = _nonneg(total)
                else:
                    tokens = result.get("token_ids")
                    if isinstance(tokens, tuple):
                        tokens = tokens[0]
                    shape = getattr(tokens, "shape", None)
                    self.generated += shape[-1] if shape is not None else len(tokens or [])
        except Exception as exc:
            _note_live_error(exc)

    def finish(self, final=None):
        """Settle only a booked completion; abort/error contributes zero tokens."""
        if self.closed:
            return
        try:
            if final is not None:
                total = _nonneg(final.get("gen_tokens"))
                if self.prefill_done or total > 0:
                    self._prompt_progress(final)
                _GENERATED_TOKENS_LIVE.add(self.model_label, total)
        except Exception as exc:
            _note_live_error(exc)
        finally:
            try:
                self._stop_decoding()
            except Exception as exc:
                _note_live_error(exc)
            _live_jobs.discard(self)
            self.closed = True


def record_completion(label, metrics, context_len=None, max_seq_len=None):
    """
    Record one completed generation. Called from common.gen_logging.log_metrics
    on the request completion path; swallows every failure.
    """

    try:
        _record(metrics if isinstance(metrics, dict) else {}, context_len)
    except Exception as exc:
        _note_record_error(label, exc)


def _record(metrics, context_len):
    model_label = _model_label()

    prompt_tokens = _nonneg(_num(metrics.get("prompt_tokens")) or _num(context_len))
    cached_tokens = _nonneg(metrics.get("cached_tokens"))
    gen_tokens = _nonneg(metrics.get("gen_tokens"))
    queue_time = _nonneg(metrics.get("queue_time"))
    prompt_time = _nonneg(metrics.get("prompt_time"))
    gen_time = _nonneg(metrics.get("gen_time"))

    with _LOCK:
        _REQUESTS.add(model_label, 1)
        _PROMPT_TOKENS.add(model_label, prompt_tokens)
        _GENERATED_TOKENS.add(model_label, gen_tokens)
        _CACHED_TOKENS.add(model_label, cached_tokens)
        _PROMPT_SECONDS.add(model_label, prompt_time)
        _GENERATE_SECONDS.add(model_label, gen_time)
        _QUEUE_SECONDS.add(model_label, queue_time)

        _GEN_LEN.observe(model_label, gen_tokens)
        _PROMPT_LEN.observe(model_label, prompt_tokens)
        _QUEUE_HIST.observe(model_label, queue_time)
        _TTFT.observe(model_label, queue_time + prompt_time)
        # A request that produced no tokens or measured no decode time has no
        # meaningful rate; the token and seconds counters still capture it.
        if gen_time > 0:
            _DECODE_RATE.observe(model_label, gen_tokens / gen_time)


def _note_record_error(label, exc):
    global _warned

    try:
        with _LOCK:
            _RECORD_ERRORS.add(None, 1)
    except Exception:
        pass

    if _warned:
        return
    _warned = True
    message = (
        f"metrics: recording a completion failed ({label}); further failures will be silent"
    )
    try:
        from common.logger import xlogger

        xlogger.warning(message, {"exception": repr(exc)})
    except Exception:
        logging.getLogger(__name__).warning("%s: %r", message, exc)


def _live_gauges():
    """Current gauge values, as {metric name: {model label: value}}."""

    container = _container()
    if container is None:
        return {}

    label = _model_label()
    values = {}
    jobs = getattr(container, "active_job_ids", None)
    if jobs is not None:
        try:
            values[_REQUESTS_ACTIVE[0]] = len(jobs)
        except Exception:
            pass
    for name, _doc, attr in _CONTAINER_GAUGES:
        value = _num(getattr(container, attr, None))
        if value is not None:
            values[name] = value

    return {name: {label: value} for name, value in values.items()}


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt_num(value) -> str:
    value = float(value)
    if math.isnan(value):
        return "Nan"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _fmt_bound(bound) -> str:
    return "+Inf" if bound == math.inf else _fmt_num(bound)


def _line(name, labels, value) -> str:
    if labels:
        inner = ",".join(f'{key}="{_esc(str(val))}"' for key, val in labels.items())
        return f"{name}{{{inner}}} {_fmt_num(value)}"
    return f"{name} {_fmt_num(value)}"


def _series_labels(label):
    return {"model": label} if label else {}


def render_metrics() -> str:
    """
    Serialize the registry to Prometheus text exposition format 0.0.4.

    Copies all accumulated state under the lock, then formats outside it, so a
    scrape never blocks the completion path beyond a dict copy.
    """

    with _LOCK:
        counter_data = {fam.name: dict(fam.values) for fam in _COUNTERS}
        counter_data[_GENERATED_TOKENS_LIVE.name] = dict(_GENERATED_TOKENS_LIVE.values)
        histogram_data = {
            fam.name: {
                label: {
                    "buckets": list(entry["buckets"]),
                    "sum": entry["sum"],
                    "count": entry["count"],
                }
                for label, entry in fam.values.items()
            }
            for fam in _HISTOGRAMS
        }

    # Same asyncio loop as generation: no await or engine/tensor work. Only
    # scrapes sum active job scalars; decoding never updates the registry.
    settled = counter_data[_GENERATED_TOKENS_LIVE.name]
    live_data = dict(settled)
    for job in tuple(_live_jobs):
        live_data[job.model_label] = live_data.get(job.model_label, 0) + job._generated_now()
    counter_data[_GENERATED_TOKENS_LIVE.name] = live_data
    drift_data = {label: value - counter_data[_GENERATED_TOKENS.name].get(label, 0)
                  for label, value in settled.items()}
    gauge_data = _live_gauges()

    lines = []
    for fam in _COUNTERS:
        lines.append(f"# HELP {fam.name} {fam.doc}")
        lines.append(f"# TYPE {fam.name} {fam.type}")
        for label, value in counter_data[fam.name].items():
            lines.append(_line(fam.name, _series_labels(label), value))

    fam = _GENERATED_TOKENS_LIVE
    lines.append(f"# HELP {fam.name} {fam.doc}")
    lines.append(f"# TYPE {fam.name} {fam.type}")
    for label, value in live_data.items():
        lines.append(_line(fam.name, _series_labels(label), value))

    for fam in _HISTOGRAMS:
        lines.append(f"# HELP {fam.name} {fam.doc}")
        lines.append(f"# TYPE {fam.name} {fam.type}")
        for label, entry in histogram_data[fam.name].items():
            labels = _series_labels(label)
            cumulative = 0
            for bound, count in zip(fam.buckets + (math.inf,), entry["buckets"]):
                cumulative += count
                lines.append(
                    _line(
                        fam.name + "_bucket",
                        {**labels, "le": _fmt_bound(bound)},
                        cumulative,
                    )
                )
            lines.append(_line(fam.name + "_sum", labels, entry["sum"]))
            lines.append(_line(fam.name + "_count", labels, entry["count"]))

    gauge_docs = {name: doc for name, doc, _attr in _CONTAINER_GAUGES}
    gauge_docs[_REQUESTS_ACTIVE[0]] = _REQUESTS_ACTIVE[1]
    gauge_order = [_REQUESTS_ACTIVE[0]] + [name for name, _doc, _a in _CONTAINER_GAUGES]
    for name in gauge_order:
        lines.append(f"# HELP {name} {gauge_docs[name]}")
        lines.append(f"# TYPE {name} gauge")
        for label, value in gauge_data.get(name, {}).items():
            lines.append(_line(name, _series_labels(label), value))

    name, doc = _REQUESTS_GENERATING
    lines.append(f"# HELP {name} {doc}")
    lines.append(f"# TYPE {name} gauge")
    for label, value in dict(_generating).items():
        lines.append(_line(name, _series_labels(label), value))

    name, doc = _LIVE_DRIFT
    lines.append(f"# HELP {name} {doc}")
    lines.append(f"# TYPE {name} gauge")
    for label, value in drift_data.items():
        lines.append(_line(name, _series_labels(label), value))

    return "\n".join(lines) + "\n"


def reset():
    """Drop all accumulated state. Exists for tests; the server never calls it."""

    global _warned, _last_model_label, _live_warned

    with _LOCK:
        for fam in _COUNTERS:
            fam.values.clear()
        for fam in _HISTOGRAMS:
            fam.values.clear()
    _GENERATED_TOKENS_LIVE.values.clear()
    _generating.clear()
    _live_jobs.clear()
    _live_warned = False
    _warned = False
    _last_model_label = "unknown"
