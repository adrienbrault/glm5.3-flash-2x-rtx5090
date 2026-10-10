"""Live counters must move before EOS, while legacy booking stays unchanged."""

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common import metrics
from test_metrics_record import FakeContainer, parse_exposition, sample


PAGE_SIZE = 256


def engine_job(cached_tokens=0, cached_pages=0, sequences=1, rq_cached=None):
    return SimpleNamespace(new_tokens=0, cached_tokens=cached_tokens,
                           cached_pages=cached_pages, rq_cached=rq_cached,
                           sequences=[object()] * sequences)


class LiveMetricsTests(unittest.TestCase):
    def setUp(self):
        metrics.reset()
        self.model_patch = patch.dict(sys.modules, {
            "common.model": SimpleNamespace(container=FakeContainer()),
        })
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop()
        metrics.reset()

    def value(self, name, model="test-model"):
        return sample(parse_exposition(metrics.render_metrics()), name, model=model)

    def test_counters_advance_before_completion_and_reconcile_once(self):
        live = metrics.start_generation(100)
        job = engine_job(cached_tokens=80)
        live.observe({"stage": "started", "job": job})
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        job.new_tokens = 2
        live.observe({"stage": "streaming", "job": job, "token_ids": [4, 5]})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 2)
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)
        self.assertEqual(self.value("tabby_requests_generating"), 1)
        self.assertNotIn("tabby_generated_tokens_total{", metrics.render_metrics())
        job.new_tokens = 5
        live.observe({"stage": "streaming", "token_ids": [6, 7, 8]})
        final = {"gen_tokens": 6, "prompt_tokens": 100, "cached_tokens": 80}
        live.finish(final)
        live.finish(final)
        metrics.record_completion("t", final)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 6)
        self.assertEqual(self.value("tabby_generated_tokens_total"), 6)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    def test_real_started_and_streaming_events_exclude_cached_tokens(self):
        live = metrics.start_generation(100)
        job = engine_job(cached_tokens=80)
        live.observe({"stage": "started", "job": job, "eos": False})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 0)
        live.observe({"stage": "streaming", "job": job, "token_ids": [1], "eos": False})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)
        live.observe({"stage": "streaming", "job": job, "token_ids": [], "eos": True,
                      "new_tokens": 2, "prompt_tokens": 100, "cached_tokens": 80})
        live.finish({"gen_tokens": 2, "prompt_tokens": 100, "cached_tokens": 80})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)

    def test_job_cache_formula_matches_eos_for_pages_sequences_and_requeues(self):
        for job in (engine_job(cached_pages=2, cached_tokens=8),
                    engine_job(cached_pages=4, cached_tokens=16, sequences=2),
                    engine_job(cached_pages=9, cached_tokens=99, rq_cached=(2, 520)),
                    engine_job(cached_pages=9, cached_tokens=99, rq_cached=(0, 0))):
            with self.subTest(job=job):
                metrics.reset()
                live = metrics.start_generation(1000, job=job, page_size=PAGE_SIZE)
                live.observe({"stage": "started", "job": job})
                live.observe({"stage": "streaming", "job": job, "token_ids": [1]})
                cached = job.rq_cached[1] if job.rq_cached is not None else 520
                self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 1000 - cached)

    def test_unknown_job_cache_defers_prompt_booking_until_authoritative_eos(self):
        live = metrics.start_generation(100)
        live.observe({"stage": "started"})
        live.observe({"stage": "prefill", "curr_progress": 90, "max_progress": 99})
        live.observe({"stage": "streaming", "token_ids": [1]})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 0)
        live.observe({"stage": "streaming", "eos": True, "new_tokens": 2,
                      "prompt_tokens": 100, "cached_tokens": 80})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)

    def test_prefill_chunks_are_live_and_reconcile_without_overbooking(self):
        job = engine_job(cached_pages=2, cached_tokens=8)
        live = metrics.start_generation(1000, job=job, page_size=PAGE_SIZE)
        live.observe({"stage": "started", "job": job})
        for progress, expected in ((500, 0), (700, 180), (700, 180), (600, 180), (999, 479)):
            live.observe({"stage": "prefill", "job": job, "curr_progress": progress,
                          "max_progress": 999, "eos": False})
            self.assertEqual(self.value("tabby_prompt_tokens_live_total"), expected)
            self.assertEqual(self.value("tabby_requests_generating"), 0)
        live.observe({"stage": "streaming", "job": job, "token_ids": [1]})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 480)
        live.observe({"stage": "streaming", "job": job, "eos": True, "new_tokens": 2,
                      "prompt_tokens": 1000, "cached_tokens": 520})
        live.finish({"gen_tokens": 2, "prompt_tokens": 1000, "cached_tokens": 520})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 480)

    def test_prefill_subtracts_new_partial_page_hits_and_normalizes_sequences(self):
        job = engine_job(cached_pages=4, sequences=2)
        live = metrics.start_generation(1000, job=job, page_size=PAGE_SIZE)
        live.observe({"stage": "prefill", "job": job, "curr_progress": 1400,
                      "max_progress": 1998})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 188)
        job.cached_tokens = 16
        live.observe({"stage": "prefill", "job": job, "curr_progress": 1800,
                      "max_progress": 1998})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 380)
        live.finish({"gen_tokens": 0})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 380)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    def test_prefill_progress_is_capped_and_requeue_does_not_book_prompts_again(self):
        job = engine_job(cached_tokens=80)
        live = metrics.start_generation(100, job=job, page_size=PAGE_SIZE)
        live.observe({"stage": "prefill", "job": job, "curr_progress": 1000,
                      "max_progress": 99})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 19)
        live.observe({"stage": "streaming", "job": job, "token_ids": [1]})
        live.observe({"stage": "started", "job": engine_job(rq_cached=(0, 80))})
        live.observe({"stage": "prefill", "job": job, "curr_progress": 5000,
                      "max_progress": 5000})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)

    def test_100k_prompt_advances_during_prefill_and_ends_at_20k_uncached(self):
        job = engine_job(cached_pages=312, cached_tokens=128)
        live = metrics.start_generation(100000, job=job, page_size=PAGE_SIZE)
        for progress, expected in ((81920, 1920), (98304, 18304), (99999, 19999)):
            live.observe({"stage": "prefill", "job": job, "curr_progress": progress,
                          "max_progress": 99999, "eos": False})
            self.assertEqual(self.value("tabby_prompt_tokens_live_total"), expected)
            self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
            self.assertEqual(self.value("tabby_requests_generating"), 0)
        live.observe({"stage": "streaming", "job": job, "token_ids": [1]})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20000)
        live.finish({"gen_tokens": 2, "prompt_tokens": 100000, "cached_tokens": 80000})
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20000)

    def test_finish_reconciles_buffered_decode_without_starting_the_gauge(self):
        job = engine_job(cached_tokens=80)
        live = metrics.start_generation(100, job=job, page_size=PAGE_SIZE)
        with patch.object(live, "_prefilled", side_effect=AssertionError("not streaming")):
            live.finish({"gen_tokens": 2})
        self.assertTrue(live.closed)
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 2)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    def test_prefilled_only_runs_for_the_first_streaming_result(self):
        live = metrics.start_generation(100)
        job = engine_job(cached_tokens=80)
        with patch.object(live, "_prefilled", wraps=live._prefilled) as prefilled:
            for tokens in ([1], [2], []):
                live.observe({"stage": "streaming", "job": job, "token_ids": tokens})
            live.observe({"stage": "streaming", "job": job, "eos": True,
                          "new_tokens": 3, "prompt_tokens": 100, "cached_tokens": 80})
            self.assertEqual(prefilled.call_count, 1)
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)

    def test_finish_swallows_stop_decoding_errors_and_still_closes(self):
        live = metrics.start_generation(10)
        live.observe({"stage": "streaming", "job": engine_job(), "token_ids": [1]})
        metrics.reset()  # Removes the active model gauge entry.
        with self.assertLogs(metrics.__name__, level="WARNING"):
            live.finish({"gen_tokens": 1})
        self.assertTrue(live.closed)

    def test_new_names_and_types_are_exposed(self):
        text = metrics.render_metrics()
        for name, kind in (("tabby_generated_tokens_live_total", "gauge"),
                           ("tabby_prompt_tokens_live_total", "counter"),
                           ("tabby_requests_generating", "gauge"),
                           ("tabby_generated_tokens_live_drift", "gauge")):
            self.assertIn(f"# HELP {name} ", text)
            self.assertIn(f"# TYPE {name} {kind}\n", text)

    def test_model_label_is_captured_for_the_request(self):
        live = metrics.start_generation(10)
        sys.modules["common.model"].container.model_dir = pathlib.Path("other-model")
        live.observe({"stage": "streaming", "new_tokens": 2})
        live.finish({"gen_tokens": 3})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 3)

    def test_disabled_live_metrics_do_no_work(self):
        with patch.object(metrics, "LIVE_METRICS_ENABLED", False):
            with patch.object(metrics, "_model_label", side_effect=AssertionError("called")):
                self.assertIsNone(metrics.start_generation(10))


    def test_cached_prompts_and_repeated_cumulative_results(self):
        live = metrics.start_generation(100)
        job = engine_job(cached_tokens=100)
        live.observe({"stage": "started", "job": job})
        for total in (3, 3, 7):
            live.observe({"stage": "streaming", "job": job, "new_tokens": total})
        live.finish({"gen_tokens": 8, "prompt_tokens": 100, "cached_tokens": 100})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 8)
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 0)

    def test_tensor_and_cfg_tuple_fallback_count_without_token_copies(self):
        live = metrics.start_generation(10)
        for tokens in (FakeTensor([1, 2, 3]), (FakeTensor([4, 5]), FakeTensor([6, 7])), [8]):
            live.observe({"stage": "streaming", "token_ids": tokens})
        live.finish({"gen_tokens": 7})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 7)

    def test_gauge_counts_concurrent_decode_but_excludes_queued_and_prefill(self):
        first = metrics.start_generation(100)
        second = metrics.start_generation(100)
        second.observe({"stage": "started", "job": engine_job(cached_tokens=80)})
        second.observe({"stage": "prefill", "curr_progress": 100})
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        first.observe({"stage": "streaming", "new_tokens": 2})
        self.assertEqual(self.value("tabby_requests_generating"), 1)
        second.observe({"stage": "streaming", "new_tokens": 3})
        self.assertEqual(self.value("tabby_requests_generating"), 2)
        first.finish({"gen_tokens": 4})
        self.assertEqual(self.value("tabby_requests_generating"), 1)
        second.finish({"gen_tokens": 5})
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 9)

    def test_decode_never_updates_a_counter_or_touches_token_tensors(self):
        live = metrics.start_generation(10, job=engine_job())
        live.observe({"stage": "streaming", "token_ids": [1]})
        with patch.object(metrics._GENERATED_TOKENS_LIVE, "add", side_effect=AssertionError("counter called")):
            for total in (2, 3):
                live.job.new_tokens = total
                live.observe({"stage": "streaming", "token_ids": object()})
                self.assertEqual(self.value("tabby_generated_tokens_live_total"), total)
        live.finish({"gen_tokens": 3})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 3)

    def test_settlement_failure_is_swallowed_and_exposed_as_drift(self):
        live = metrics.start_generation(10)
        metrics.record_completion("t", {"gen_tokens": 3})
        with self.assertLogs(metrics.__name__, level="WARNING"):
            with patch.object(metrics._GENERATED_TOKENS_LIVE, "add", side_effect=RuntimeError("boom")):
                live.finish({"gen_tokens": 3})
        self.assertTrue(live.closed)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), -3)


class R3AccountingTests(unittest.TestCase):
    setUp = LiveMetricsTests.setUp
    tearDown = LiveMetricsTests.tearDown
    value = LiveMetricsTests.value
    # Engine counts sampled/accepted tokens, not verifier/draft KV positions.
    def test_mtp_rewind_stop_thinking_and_requeue_use_engine_booking(self):
        job = engine_job()
        job.rq_new_tokens = 0
        live = metrics.start_generation(100, job=job, page_size=PAGE_SIZE)
        live.observe({"stage": "streaming", "job": job, "token_ids": [1]})
        for accepted, rejected, verify, total in ((1, 0, 3, 2), (1, 2, 8, 3), (2, 2, 10, 5)):
            job.accepted_draft_tokens = accepted
            job.rejected_draft_tokens = rejected
            job.verify_position = verify
            job.new_tokens = total
            # No observe call: progress must be visible at scrape time.
            self.assertEqual(self.value("tabby_generated_tokens_live_total"), total)
        live.observe({"stage": "streaming", "job": job,
                      "text": "<think>reasoning</think><tool_call>{}</tool_call>",
                      "token_ids": [11, 12, 13, 14, 15, 16]})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 5)
        job.new_tokens = 3  # banned-string checkpoint rewind
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 3)
        job.rq_new_tokens, job.new_tokens = 3, 2  # same Job, reinitialized
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 5)
        # Thinking/tool tokens count when the engine books them. Text trimming
        # does not subtract tokens; healed prefix and rejected drafts do not count.
        live.observe({"stage": "streaming", "eos": True, "job": job,
                      "token_ids": [90, 91, 92], "text": "", "new_tokens": 4,
                      "eos_reason": "stop_string", "eos_triggering_string": "STOP",
                      "prompt_tokens": 100, "cached_tokens": 0})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 4)
        final = {"gen_tokens": 4}
        metrics.record_completion("t", final)
        live.finish(final)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 4)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)

    def test_abort_removes_partial_work_and_returns_to_zero_drift(self):
        job = engine_job()
        live = metrics.start_generation(100, job=job)
        live.observe({"stage": "streaming", "job": job, "token_ids": [1]})
        job.new_tokens = 12
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 12)
        live.finish()
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)

    def test_healed_prefix_and_replayed_chunks_cannot_overcount_engine_total(self):
        job = engine_job()
        job.new_tokens = -1
        live = metrics.start_generation(10, job=job)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        event = {"stage": "streaming", "job": job, "token_ids": [7, 8]}
        job.new_tokens = 1  # healed prefix + one new token emitted
        live.observe(event)
        live.observe(event)  # duplicate consumer event
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 1)
        metrics.record_completion("t", {"gen_tokens": 1})
        live.finish({"gen_tokens": 1})
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 1)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)

    def test_many_overlapping_requests_reconcile_exactly(self):
        import random
        rng = random.Random(961)
        settled = 0
        active = []
        for _ in range(100):
            job = engine_job()
            live = metrics.start_generation(100, job=job)
            job.new_tokens = rng.randrange(1, 100)
            active.append((live, job))
            if len(active) > 4:
                previous, previous_job = active.pop(0)
                if rng.randrange(4):
                    # Authoritative final may be smaller after rewind/trim.
                    total = rng.randrange(previous_job.new_tokens + 1)
                    metrics.record_completion("t", {"gen_tokens": total})
                    previous.finish({"gen_tokens": total})
                    settled += total
                else:
                    previous.finish()
            self.assertEqual(self.value("tabby_generated_tokens_live_total"),
                             settled + sum(j.new_tokens for _, j in active))
            self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        for live, job in active:
            metrics.record_completion("t", {"gen_tokens": job.new_tokens})
            live.finish({"gen_tokens": job.new_tokens})
            settled += job.new_tokens
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), settled)
        self.assertEqual(self.value("tabby_generated_tokens_total"), settled)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)

    def test_r961_window_excess_is_inflight_not_settled_overcount(self):
        import json
        root = APP.parents[3] / "ref/r961-live/B"
        if not root.exists():
            self.skipTest("recorded probe is workspace-only")
        rows = []
        for line in (root / "metrics.tsv").read_text().splitlines():
            fields = line.split()
            rows.append(dict(zip(fields[1::2], map(float, fields[2::2]))))
        first, last = rows[0], rows[-1]
        self.assertEqual(first["tabby_requests_active"], 0)
        self.assertEqual(first["tabby_generated_tokens_live_total"], 1344)
        self.assertEqual(last["tabby_requests_generating"], 4)
        live_delta = last["tabby_generated_tokens_live_total"] - 1344
        booked_delta = last["tabby_generated_tokens_total"] - 1344
        self.assertEqual((live_delta, booked_delta, live_delta - booked_delta), (20409, 16384, 4025))
        usage = [json.loads(line) for line in (root / "dec.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["usage"]["completion_tokens"] for row in usage
                             if row.get("phase") == "decode-stream" and row["run"] == 1), 4096)
        self.assertAlmostEqual(4025 / 16384 * 100, 24.566650390625)

# Load the real backend methods with deferred annotations, avoiding torch,
# exllamav3 CUDA initialization and the app's third-party import graph. Engine
# objects are faked via sys.modules, just like the existing metrics tests.
import ast
import asyncio
import contextlib
import io
from unittest.mock import AsyncMock, Mock

APP = pathlib.Path(__file__).resolve().parents[1]


def load_functions(path, namespace, methods=None):
    parsed = ast.parse(path.read_text(), filename=str(path))
    if methods is None:
        body = [node for node in parsed.body
                if not isinstance(node, (ast.Import, ast.ImportFrom))]
    else:
        container = next(node for node in parsed.body
                         if isinstance(node, ast.ClassDef) and node.name == "ExllamaV3Container")
        body = [node for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name == "_merge_stream_results"]
        body.append(ast.ClassDef(name="ExllamaV3Container", bases=[], keywords=[],
                                 decorator_list=[], body=[node for node in container.body
                                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                     and node.name in methods]))
    body.insert(0, ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace


class FakeTensor:
    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))

    def size(self, dim=-1):
        return self.shape[dim]

    def flatten(self):
        return self

    def tolist(self):
        return list(self.tokens)


class FakeJob:
    def __init__(self, results, buffered=()):
        self.results = iter(results)
        self.queue = asyncio.Queue()
        self.cancelled = False
        self.job = engine_job(cached_tokens=80)
        self.buffered = buffered

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            result = next(self.results)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(result, BaseException):
            raise result
        self.record_result(result)
        if result.get("stage") == "streaming" and self.buffered:
            for pending in self.buffered:
                self.queue.put_nowait(pending)
                if isinstance(pending, dict):
                    self.record_result(pending)
            self.buffered = ()
        return result

    def record_result(self, result):
        result["job"] = self.job
        if result.get("new_tokens") is not None:
            self.job.new_tokens = result["new_tokens"]
        elif result.get("stage") == "streaming":
            self.job.new_tokens += result["token_ids"].shape[-1]

    async def cancel(self):
        self.cancelled = True


def stream_result(total, text="", eos=False, count=None):
    result = {"stage": "streaming", "text": text,
              "token_ids": FakeTensor(list(range(len(text) if count is None else count))),
              "eos": eos}
    # Real engines put prompt/cache totals and cumulative new_tokens only on EOS.
    if eos:
        result.update(new_tokens=total, prompt_tokens=100, cached_tokens=80,
                      time_prefill=0.2, time_generate=1.0, time_enqueued=0.1,
                      eos_reason="stop_token")
    return result


def scripted_job(tail=None, buffered=()):
    return FakeJob([
        {"stage": "started", "eos": False},
        {"stage": "prefill", "eos": False, "curr_progress": 99, "max_progress": 99},
        stream_result(2, "ab"),
        *(tail if tail is not None else [stream_result(4, "cd"), stream_result(5, eos=True)]),
    ], buffered)


class BackendLiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        metrics.reset()
        self.logger = Mock()
        self.config = SimpleNamespace(logging=SimpleNamespace(
            log_prompt=False, log_generation_params=False))
        logging_ns = load_functions(APP / "common/gen_logging.py", {
            "xlogger": self.logger, "config": self.config,
            "record_completion": metrics.record_completion,
        })
        self.snapshots = []

        async def poll():
            self.snapshots.append(parse_exposition(metrics.render_metrics()))

        self.disconnect = SimpleNamespace(poll=poll, add_cleanup_task=AsyncMock(),
                                          finish=AsyncMock())
        status = SimpleNamespace(started=Mock(), prefill=Mock(), generated=Mock())
        self.job = scripted_job()
        ns = load_functions(APP / "backends/exllamav3/model.py", {
            "asyncio": asyncio, "CancelledError": asyncio.CancelledError,
            "aclosing": contextlib.aclosing,
            "torch": SimpleNamespace(Tensor=FakeTensor, cat=lambda tensors, dim:
                FakeTensor([token for tensor in tensors for token in tensor.tokens])),
            "unwrap": lambda value, default: default if value is None else value,
            "AsyncJob": lambda *args, **kwargs: self.job,
            "ExllamaV3SamplerBuilder": SimpleNamespace(from_params=lambda *args:
                SimpleNamespace(build=lambda *args: None, settings=[])),
            "ExLlamaV3Grammar": lambda: SimpleNamespace(filters=[]),
            "validate_context_requirements": lambda *args: None,
            "start_generation": metrics.start_generation, "PAGE_SIZE": PAGE_SIZE,
            "xlogger": self.logger,
            "status_display": SimpleNamespace(add_job=lambda *args: status, remove_job=Mock()),
            "ContextLengthExceededError": ValueError,
            **{name: logging_ns[name] for name in (
                "format_settings", "log_generation_params", "log_metrics", "log_prompt",
                "log_request_start")},
        }, methods={"generate", "stream_generate", "generate_gen", "handle_finish_chunk"})
        self.backend_namespace = ns
        self.container = ns["ExllamaV3Container"]()
        self.container.model_dir = pathlib.Path("test-model")
        self.container.active_job_ids = {}
        self.container.loaded = True
        self.container.load_condition = asyncio.Condition()
        self.container.load_lock = asyncio.Lock()
        self.container.tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=0, bos_token="")
        self.container.hf_model = SimpleNamespace(add_bos_token=lambda: False, eos_tokens=lambda: [0])
        self.container.config = SimpleNamespace(eos_token_id_list=[])
        self.container.generator = SimpleNamespace(generator=SimpleNamespace(recurrent_cache=None))
        self.container.max_seq_len = 2048
        self.container.cache = SimpleNamespace(max_num_tokens=4096)
        self.container.job_max_rq_tokens = lambda *args: 0
        self.container._encode_prompt = AsyncMock(return_value=FakeTensor([1] * 100))

        async def recover(exc, job):
            await job.cancel()

        self.container._recover_from_generation_error = AsyncMock(side_effect=recover)
        self.params = SimpleNamespace(temperature=0, stop=[], add_bos_token=False,
            max_tokens=10, min_tokens=0, json_schema=None, regex_pattern=None,
            grammar_string=None, banned_strings=[], token_healing=False,
            logprobs=0, top_logprobs=0, param_source=lambda *args: "default",
            get_stop_on_loop=lambda: None, model_dump=lambda **kwargs: {})
        self.modules_patch = patch.dict(sys.modules, {
            "common.model": SimpleNamespace(container=self.container),
            "exllamav3": SimpleNamespace(cache_trace=SimpleNamespace(
                render=lambda *args, **kwargs: None, attach=Mock(), finished=Mock())),
            "exllamav3.generator.prompt_lookup": SimpleNamespace(COUNTERS=()),
        })
        self.modules_patch.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.modules_patch.stop()
        metrics.reset()

    def value(self, name):
        return sample(parse_exposition(metrics.render_metrics()), name)

    def stream(self, label="chat/completions", request_id="r"):
        return self.container.stream_generate(request_id, "prompt", self.params,
                                              self.disconnect, label=label)

    async def collect(self, label="chat/completions"):
        return [chunk async for chunk in self.stream(label)]

    async def test_streaming_chat_and_completions_are_live(self):
        for label in ("chat/completions", "completions"):
            with self.subTest(label=label):
                metrics.reset()
                self.job = scripted_job()
                iterator = self.stream(label)
                first = await anext(iterator)
                self.assertEqual(first["text"], "ab")
                self.assertEqual(self.value("tabby_generated_tokens_live_total"), 2)
                self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)
                self.assertEqual(self.value("tabby_requests_generating"), 1)
                self.assertNotIn("tabby_generated_tokens_total{", metrics.render_metrics())
                await anext(iterator)
                await anext(iterator)
                self.assertEqual(self.value("tabby_generated_tokens_live_total"), 5)
                self.assertEqual(self.value("tabby_requests_generating"), 0)
                with self.assertRaises(StopAsyncIteration):
                    await anext(iterator)
                self.assertEqual(self.value("tabby_generated_tokens_total"), 5)
                self.assertEqual(self.value("tabby_prompt_tokens_total"), 100)
                self.assertEqual(self.value("tabby_cached_tokens_total"), 80)
                self.assertEqual(self.container.active_job_ids, {})

    async def test_non_streaming_generate_updates_before_returning(self):
        result = await self.container.generate("r", "prompt", self.params, self.disconnect)
        self.assertEqual(result["text"], "abcd")
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 5)
        self.assertEqual(self.value("tabby_generated_tokens_total"), 5)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertTrue(any(sample(snapshot, "tabby_generated_tokens_live_total") == 2
            and not any(key[0] == "tabby_generated_tokens_total" for key in snapshot)
            for snapshot in self.snapshots))

    async def test_disconnect_reconciles_work_produced_before_poll(self):
        self.disconnect.poll = AsyncMock(side_effect=[None, None, None, asyncio.CancelledError()])
        chunks = await self.collect()
        self.assertEqual(len(chunks), 1)
        self.assertTrue(self.job.cancelled)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertNotIn("tabby_requests_total{", metrics.render_metrics())

    async def test_closing_stream_cleans_up_immediately_and_reconciles_backend(self):
        iterator = self.stream()
        await anext(iterator)
        self.job.job.new_tokens = 7  # Work buffered in the engine, not yet consumed.
        await iterator.aclose()
        self.assertTrue(self.job.cancelled)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertEqual(self.container.active_job_ids, {})
        self.assertNotIn("tabby_requests_total{", metrics.render_metrics())

    async def test_error_after_partial_generation_discards_unbooked_counts(self):
        self.job = scripted_job(tail=[RuntimeError("engine failed")])
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            await self.collect()
        self.assertTrue(self.job.cancelled)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertNotIn("tabby_requests_total{", metrics.render_metrics())

    async def test_drained_span_updates_counters_before_the_next_result_or_finish(self):
        self.job = scripted_job(tail=[stream_result(5, eos=True)],
                                buffered=[stream_result(4, "cd")])
        iterator = self.stream()
        try:
            first = await anext(iterator)
            self.assertEqual(first["text"], "abcd")
            self.assertEqual(self.value("tabby_generated_tokens_live_total"), 4)
            self.assertEqual(self.value("tabby_requests_generating"), 1)
            self.assertNotIn("tabby_generated_tokens_total{", metrics.render_metrics())
        finally:
            await iterator.aclose()

    async def test_closing_after_the_finish_chunk_does_not_cancel_a_finished_job(self):
        iterator = self.stream()
        await anext(iterator)
        await anext(iterator)
        finish = await anext(iterator)
        self.assertEqual(finish["gen_tokens"], 5)
        await iterator.aclose()
        self.assertFalse(self.job.cancelled)
        self.assertEqual(self.value("tabby_generated_tokens_total"), 5)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    async def test_stop_decoding_failure_cannot_skip_legacy_completion_logging(self):
        with patch.object(metrics._LiveGeneration, "_stop_decoding", side_effect=KeyError("gauge")):
            with self.assertLogs(metrics.__name__, level="WARNING"):
                await self.collect()
        self.assertEqual(self.value("tabby_generated_tokens_total"), 5)

    async def test_chunked_prefill_is_visible_before_any_generated_token(self):
        self.job = FakeJob([
            {"stage": "started", "eos": False},
            {"stage": "prefill", "curr_progress": 85, "max_progress": 99, "eos": False},
            {"stage": "prefill", "curr_progress": 95, "max_progress": 99, "eos": False},
            stream_result(2, "ab"), stream_result(5, eos=True),
        ])
        await self.collect()
        self.assertEqual(sample(self.snapshots[1], "tabby_prompt_tokens_live_total"), 5)
        self.assertEqual(sample(self.snapshots[2], "tabby_prompt_tokens_live_total"), 15)
        self.assertEqual(sample(self.snapshots[2], "tabby_generated_tokens_live_total"), 0)
        self.assertEqual(sample(self.snapshots[2], "tabby_requests_generating"), 0)
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 20)

    async def test_cancel_during_prefill_retains_only_processed_uncached_tokens(self):
        self.job = FakeJob([
            {"stage": "started", "eos": False},
            {"stage": "prefill", "curr_progress": 90, "max_progress": 99, "eos": False},
            asyncio.CancelledError(),
        ])
        await self.collect()
        self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 10)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    async def test_drained_chunks_and_eos_are_counted_once(self):
        self.job = scripted_job(tail=[], buffered=[stream_result(4, "cd"), stream_result(5, eos=True)])
        chunks = await self.collect()
        self.assertEqual(chunks[0]["text"], "abcd")
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 5)
        self.assertEqual(self.value("tabby_generated_tokens_total"), 5)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    async def test_drained_eos_freezes_authoritative_count_before_cleanup(self):
        self.job = scripted_job(tail=[], buffered=[stream_result(4, "cd"), stream_result(5, eos=True)])
        iterator = self.stream()
        try:
            await anext(iterator)
            # Drained EOS must already be observed even before the finish chunk.
            self.job.job.new_tokens = 999
            self.assertEqual(self.value("tabby_generated_tokens_live_total"), 5)
            self.assertEqual(self.value("tabby_requests_generating"), 0)
            await anext(iterator)
        finally:
            await iterator.aclose()
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)

    async def test_status_cleanup_error_drops_unbooked_work(self):
        self.backend_namespace["status_display"].remove_job.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            await self.collect()
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertFalse(metrics._live_jobs)

    async def test_error_during_drain_discards_unbooked_results(self):
        self.job = scripted_job(tail=[], buffered=[stream_result(4, "cd"), RuntimeError("drain failed")])
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            await self.collect()
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    async def test_cancel_during_drain_discards_unbooked_results(self):
        self.job = scripted_job(tail=[], buffered=[stream_result(4, "cd"), None])
        self.assertEqual(await self.collect(), [])
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)

    async def test_cancel_and_error_before_prefill_do_not_count_prompts(self):
        for reason in (asyncio.CancelledError(), RuntimeError("queued failed")):
            with self.subTest(reason=type(reason).__name__):
                metrics.reset()
                self.job = FakeJob([{"stage": "started", "eos": False}, reason])
                if isinstance(reason, RuntimeError):
                    with self.assertRaisesRegex(RuntimeError, "queued failed"):
                        await self.collect()
                else:
                    await self.collect()
                self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
                self.assertEqual(self.value("tabby_prompt_tokens_live_total"), 0)
                self.assertEqual(self.value("tabby_requests_generating"), 0)

    async def test_disabled_metrics_leave_generation_and_legacy_booking_intact(self):
        with patch.object(metrics, "LIVE_METRICS_ENABLED", False):
            with patch.object(metrics, "_LiveGeneration", side_effect=AssertionError("allocated")):
                await self.collect()
        self.assertEqual(self.value("tabby_generated_tokens_total"), 5)
        self.assertNotIn("tabby_generated_tokens_live_total{", metrics.render_metrics())

    async def test_empty_text_tokens_and_multiple_token_chunks_are_counted(self):
        self.job = scripted_job(tail=[stream_result(4, count=2), stream_result(5, eos=True)])
        chunks = await self.collect()
        self.assertEqual(chunks[0]["text"], "ab")
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 5)
        self.assertEqual(self.value("tabby_generated_tokens_total"), 5)

    async def test_finish_formatting_error_discards_unbooked_eos_count(self):
        self.container.handle_finish_chunk = Mock(side_effect=RuntimeError("finish failed"))
        with self.assertRaisesRegex(RuntimeError, "finish failed"):
            await self.collect()
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertNotIn("tabby_requests_total{", metrics.render_metrics())

    async def test_logging_error_cannot_leak_the_decoding_gauge(self):
        self.backend_namespace["log_metrics"] = Mock(side_effect=RuntimeError("logging failed"))
        with self.assertRaisesRegex(RuntimeError, "logging failed"):
            await self.collect()
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertEqual(self.container.active_job_ids, {})

    async def test_task_cancellation_during_decode_runs_cleanup(self):
        class BlockingJob(FakeJob):
            def __init__(self):
                super().__init__([stream_result(2, "ab")])
                self.waiting = asyncio.Event()
                self.first = True

            async def __anext__(self):
                if self.first:
                    self.first = False
                    return await super().__anext__()
                self.waiting.set()
                await asyncio.Event().wait()

        self.job = BlockingJob()
        task = asyncio.create_task(self.collect())
        await asyncio.wait_for(self.job.waiting.wait(), timeout=1)
        self.assertEqual(self.value("tabby_requests_generating"), 1)
        task.cancel()
        await task
        self.assertTrue(self.job.cancelled)
        self.assertEqual(self.value("tabby_generated_tokens_live_total"), 0)
        self.assertEqual(self.value("tabby_generated_tokens_live_drift"), 0)
        self.assertEqual(self.value("tabby_requests_generating"), 0)
        self.assertEqual(self.container.active_job_ids, {})


if __name__ == "__main__":
    unittest.main()
