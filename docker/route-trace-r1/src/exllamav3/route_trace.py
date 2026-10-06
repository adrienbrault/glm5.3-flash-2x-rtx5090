"""Default-off route trace v1: independent, atomically committed NPZ records.

No extension dependency. Copies are ordered BEFORE split mutates routing buffers.
The background writer alone waits for CUDA completion and serializes to disk.
"""
import os

ENABLED = bool(os.environ.get("EXL3_ROUTE_TRACE"))
REVISION = 1
_writer = None
if ENABLED:
    import itertools
    import threading
    _calls = itertools.count()
    _init_lock = threading.Lock()


def begin(params, input_ids, component="text", phase="forward"):
    if not ENABLED:
        return
    shape = tuple(input_ids.shape)
    batch, q = (shape[0], shape[-1]) if len(shape) > 1 else (1, shape[-1])
    starts = params.get("cache_seqlens")
    if starts is None and isinstance(params.get("past_len"), int):
        starts = (params["past_len"],) * batch
    # Snapshot host staging buffers now; generator reuses these on subsequent calls.
    # GPU-resident starts are copied with the route record on its producer stream.
    if starts is not None and hasattr(starts, "clone") and getattr(starts.device, "type", "cpu") == "cpu":
        starts = starts.clone()
    params["_route_trace_context"] = dict(
        call=f"{os.getpid()}:{next(_calls)}", component=str(component),
        phase=params.get("_route_trace_phase", phase), batch=batch, q=q,
        keys=tuple(str(k) for k in params.get("_route_trace_keys", [""] * batch)),
        starts=starts,
    )


def metadata(ctx, rows):
    """Pure CPU row metadata. -1 / empty key means unavailable, never a guess."""
    import numpy as np
    batch, q = ctx["batch"], ctx["q"]
    if rows != batch * q:
        raise ValueError(f"route row shape {rows} != {batch}*{q}; unsupported row pruning")
    starts = ctx.get("starts")
    positions = np.full(rows, -1, dtype=np.int64)
    if starts is not None:
        starts = np.asarray(starts).reshape(-1)
        if len(starts) != batch:
            raise ValueError("cache_seqlens must have one start per batch sequence")
        positions = (starts[:, None] + np.arange(q)[None, :]).reshape(-1).astype(np.int64)
    keys = ctx["keys"]
    if len(keys) != batch:
        raise ValueError("request keys must have one entry per batch sequence")
    return positions, np.repeat(np.asarray(keys, dtype=str), q), np.repeat(np.arange(batch), q)


class Writer:
    """Bounded queue, no silent loss; saturation/error fails the audit explicitly."""
    def __init__(self, path, capacity=256):
        import pathlib
        import queue
        import threading
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(capacity)
        self.error = None
        self.closed = False
        self.written = 0
        self.thread = threading.Thread(target=self._run, name="exl3-route-writer", daemon=True)
        self.thread.start()

    def submit(self, packet):
        if self.error is not None:
            raise RuntimeError("route trace writer failed") from self.error
        if self.closed:
            raise RuntimeError("route trace writer closed")
        try:
            self.queue.put_nowait(packet)
        except Exception as exc:
            raise RuntimeError("route trace queue full: use a faster output volume/shorter audit") from exc

    def _run(self):
        import json
        import numpy as np
        import threading
        seq = 0
        while True:
            packet = self.queue.get()
            try:
                if packet is None:
                    return
                if self.error is not None:
                    continue
                ctx, layer, device, instance, permutation, ids, weights, event = packet
                if event is not None:
                    event.synchronize()  # writer thread only; producer stream never waits here
                def array(t):
                    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)
                ctx = dict(ctx)
                if ctx["starts"] is not None:
                    ctx["starts"] = array(ctx["starts"])
                ids = array(ids).astype(np.int32, copy=False)
                weights = array(weights)  # preserve fp16 values/bits from router
                positions, keys, batch_rows = metadata(ctx, len(ids))
                # static hot ordering permutes router columns; restore checkpoint IDs.
                if permutation is not None:
                    ids = np.asarray(permutation, dtype=np.int32)[ids]
                info = {k: v for k, v in ctx.items() if k not in ("starts", "keys")}
                info.update(version=REVISION, kind="actual", layer=layer, device=device,
                            layer_instance=instance, expert_space="checkpoint")
                dest = self.path / f"route-{os.getpid()}-{threading.get_ident()}-{seq:09d}.npz"
                tmp = dest.with_suffix(".tmp")
                with tmp.open("wb", buffering=1024*1024) as f:
                    np.savez(f, meta=np.asarray(json.dumps(info)), ids=ids,
                             weights=weights, positions=positions, request_keys=keys,
                             batch_rows=batch_rows)
                os.replace(tmp, dest)
                seq += 1
                self.written = seq
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
            self.thread.join()
        if self.error is not None:
            raise RuntimeError("route trace writer failed") from self.error
        import json
        marker = self.path / f"complete-{os.getpid()}-{self.thread.ident}.json"
        marker.write_text(json.dumps(dict(version=REVISION, records=self.written, clean_close=True))+"\n")


def record(module, ids, weights, params):
    if not ENABLED:
        return
    if params.get("autosplit_measure") or params.get("tp_warmup"):
        return
    import atexit
    import torch
    global _writer
    with _init_lock:
        if _writer is None:
            _writer = Writer(os.environ["EXL3_ROUTE_TRACE"])
            atexit.register(_writer.close)
    if ids.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EXL3_ROUTE_TRACE requires Python MoE hooks outside outer CUDA capture")
    ctx = dict(params.get("_route_trace_context", {}))
    if not ctx:
        # Direct module drivers have no model/generator call metadata.
        ctx = dict(call=f"direct:{os.getpid()}:{id(params)}", component="unknown",
                   phase="unknown", batch=len(ids), q=1, keys=("",)*len(ids), starts=None)
    event = None
    if ids.is_cuda:
        # Same stream D2H protects reusable routing statics and precedes map translation.
        with torch.cuda.device(ids.device):
            def stage(t):
                if t is None:
                    return None
                if not hasattr(t, "is_cuda"):
                    import numpy as np
                    return np.asarray(t).copy()
                if not t.is_cuda:
                    return t.clone()
                h = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                h.copy_(t, non_blocking=True)
                return h
            hi, hw = stage(ids), stage(weights)
            ctx["starts"] = stage(ctx["starts"])
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(ids.device))
    else:
        hi, hw = ids.detach().clone(), weights.detach().clone()
    _writer.submit((ctx, str(module.key), str(ids.device), params.get("layer_instance", 0),
                    getattr(module, "_split_perm", None), hi, hw, event))
