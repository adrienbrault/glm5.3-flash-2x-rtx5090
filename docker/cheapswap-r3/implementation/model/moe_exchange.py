"""Opt-in, in-place CPU/GPU expert exchange. No checkpoint access or native ABI changes."""
import os

# Only populated if CUDA itself refuses to drain an interrupted transfer. Retaining the
# tensor exporters prevents scratch reuse/unmapping until the operator restarts the process.
_failed_transfers = []


def poison(host):
    host.exchange_failed = True
    abort = getattr(host, "v_abort", None)
    if abort is not None:
        abort[0] = 1


def require_healthy(host):
    if getattr(host, "exchange_failed", False):
        raise RuntimeError("previous expert exchange failed; reload model")
    abort = getattr(host, "v_abort", None)
    if abort is not None and int(abort[0]):
        raise RuntimeError("CPU MoE worker aborted; reload model")
    proc = getattr(host, "proc", None)
    if proc is not None and not proc.is_alive():
        raise RuntimeError("CPU MoE worker died; reload model")


def enabled():
    return os.environ.get("EXL3_MOE_CPU_SWAP_MODE", "checkpoint") == "exchange"


def native_view(t, swizzled):
    """View physical band order as native logical tiles; K8 is never swizzled."""
    if not swizzled or t.shape[2] // 16 == 8:
        return t
    tk, tn, ps = t.shape
    return t.view(tn // 8, tk, 8, ps).permute(1, 0, 2, 3).reshape(t.shape)


def band_view(t, swizzled):
    if not swizzled or t.shape[2] // 16 == 8:
        return t
    tk, tn, ps = t.shape
    return t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).reshape(t.shape)


def compatible(gpu, host):
    return gpu is not None and host is not None and gpu.shape == host.shape and gpu.dtype == host.dtype


def exchange_tensors(pairs, stream, swizzled=False, verify=False):
    """pairs=(GPU tensor, shared arena tensor, is_trellis, optional GPU aux mirror).

    Caller fences all previous readers of this layer and forbids new submissions until return.
    GPU scratch only: no permanent duplicate host expert. All arena DMA operands contiguous.
    Return after D2H completion, so raw worker pointers immediately see committed bytes.
    A failure is fatal: caller must not continue inference with a partly exchanged expert.
    """
    import torch
    if not all(compatible(g, h) and g.is_contiguous() and h.is_contiguous()
               and (mirror is None or compatible(mirror, h)) for g,h,_,mirror in pairs):
        return False
    # Preallocate every temporary before the first write; an OOM cannot leave a half swap.
    scratch = [(torch.empty_like(g), torch.empty_like(g)) for g,_,_,_ in pairs]
    try:
        with torch.cuda.stream(stream):
            for (g,h,tr,mirror),(incoming,outgoing) in zip(pairs,scratch):
                outgoing.copy_(g)
                incoming.copy_(h, non_blocking=True)
                if tr:
                    tk, tn, ps = g.shape
                    if swizzled and ps // 16 != 8:
                        g.view(tk, tn//8, 8, ps).copy_(
                            incoming.view(tn//8, tk, 8, ps).permute(1,0,2,3))
                        incoming.view(tn//8, tk, 8, ps).copy_(
                            outgoing.view(tk, tn//8, 8, ps).permute(1,0,2,3))
                    else:
                        g.copy_(incoming)
                        incoming.copy_(outgoing)
                else:
                    g.copy_(incoming)
                    incoming.copy_(outgoing)
                if mirror is not None:
                    mirror.copy_(outgoing)
                h.copy_(incoming, non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
        # Keeps scratch and pinned host mappings alive through both DMA directions.
        done.synchronize()
    except BaseException:
        # Allocations belong to the producer stream, but copies use the side stream.
        # A failed enqueue must drain that stream before these Python references die.
        try:
            stream.synchronize()
        except BaseException:
            _failed_transfers.append((stream, pairs, scratch))
        raise
    if verify:
        for (g,h,tr,mirror),(incoming,outgoing) in zip(pairs,scratch):
            expected = band_view(outgoing,swizzled) if tr else outgoing
            assert torch.equal(h,expected.cpu()), "demotion bytes differ"
            if mirror is not None:
                assert torch.equal(mirror,outgoing), "streamed auxiliary bytes differ"
    return True


def fence_layer(module):
    """Stream-tail events are the reader references for this layer's fixed-address slots.

    Each recorded stream includes the layer's collect wait/readback. Thus waiting releases
    all old GPU and worker readers of this layer, without draining other devices' rings.
    The served engine has one serialized Python inference producer; no work is submitted
    while this synchronous sweep holds control. Prefill copy streams are fenced too.
    """
    import torch
    streams = dict(getattr(module, "_exchange_read_streams", {}))
    current = torch.cuda.current_stream(module.device)
    streams[current.cuda_stream] = current
    for st in (getattr(module.cpu_host, "sstate", None) or {}).values():
        s = st.get("copy_stream")
        if s is not None and s.device == current.device:
            streams[s.cuda_stream] = s
    events=[]
    for s in streams.values():
        with torch.cuda.device(s.device):
            e=torch.cuda.Event(); e.record(s); events.append(e)
    for e in events: e.synchronize()


def exchange_experts(module, r_cold, r_hot, mp):
    import torch
    host=module.cpu_host
    require_healthy(host)
    if not host.pinned:
        raise RuntimeError("exchange requires EXL3_MOE_PINNED_ARENA=1")
    slot=int(mp[r_cold]); local=int(mp[r_hot])-module.cpu_split_first
    desc=host.exchange_views[module.cpu_layer_idx][local]
    names=(["g"] if module.gated else [])+["u","d"]
    gpu_lists=([module.gates] if module.gated else [])+[module.ups,module.downs]
    if len(desc) != len(names):
        raise RuntimeError("exchange projection descriptor count differs")
    pairs=[]
    for name,gs,projection in zip(names,gpu_lists,desc):
        if len(projection) != 4:
            raise RuntimeError("exchange field descriptor count differs")
        live=gs[slot].inner
        for field,d in zip(("trellis","suh","svh","bias"),projection):
            g=getattr(live,field,None)
            if d is None:
                if g is not None: return False
                continue
            h=arena_tensor(host,d)
            mirror=None
            if field!='trellis':
                aux=host.aux.get(module.cpu_layer_idx,{}).get(field+'_'+name)
                if aux is not None: mirror=aux[local]
            pairs.append((g,h,field=='trellis',mirror))
    if os.environ.get("EXL3_MOE_CPU_SWAP_DEBUG"):
        print(f" -- exchange bytes: layer={module.key} bytes={sum(h.numel()*h.element_size() for _,h,_,_ in pairs)} "
              f"shapes={[tuple(h.shape) for _,h,tr,_ in pairs if tr]}",flush=True)
    swz=host.exchange_swizzled
    stream=getattr(module,"_exchange_copy_stream",None)
    if stream is None:
        stream=module._exchange_copy_stream=torch.cuda.Stream(device=module.device)
    try:
        if not exchange_tensors(pairs,stream,swz,bool(os.environ.get("EXL3_MOE_CPU_SWAP_VERIFY"))):
            return False
        # Unfolded reconstruction owns stacked copies, not pointer tables. Update both
        # resident and streamed rows in place, before the router map can be published.
        with torch.cuda.stream(stream):
            changed=refresh_recon_scales(module,slot,local)
            if changed:
                ready=torch.cuda.Event(); ready.record(stream)
        if changed:
            ready.synchronize()
        mp[r_cold],mp[r_hot]=int(mp[r_hot]),slot
        return True
    except BaseException:
        poison(host)
        try:
            stream.synchronize()
        except BaseException:
            pass
        raise


def arena_descriptor(t, chunks):
    """Exact actual allocation, including aux tensors spilling into a later chunk."""
    if t is None:
        return None
    import ctypes
    ptr=t.data_ptr()
    nbytes=t.numel()*t.element_size()
    if not t.is_contiguous() or nbytes <= 0:
        raise RuntimeError("exchange requires a nonempty contiguous arena tensor")
    for ci,chunk in enumerate(chunks):
        start=ctypes.addressof(ctypes.c_char.from_buffer(chunk))
        if start <= ptr and ptr+nbytes <= start+len(chunk):
            return (ci,ptr-start,tuple(t.shape),str(t.dtype).split(".")[-1])
    raise RuntimeError("exchange tensor is outside shared arena")


def arena_tensor(host, descriptor):
    import math
    import torch
    ci,offset,shape,dtype=descriptor
    if not isinstance(ci,int) or not 0 <= ci < len(host.arena_views):
        raise RuntimeError("invalid exchange arena chunk")
    if dtype not in ("int16","float16"):
        raise RuntimeError("invalid exchange arena dtype")
    if not shape or any(not isinstance(x,int) or x <= 0 for x in shape):
        raise RuntimeError("invalid exchange arena shape")
    dt=getattr(torch,dtype)
    raw=host.arena_views[ci].view(torch.uint8)
    size=2  # all served trellises and auxiliaries are 16-bit
    nbytes=math.prod(shape)*size
    if not isinstance(offset,int) or offset < 0 or offset % size or offset+nbytes > raw.numel():
        raise RuntimeError("invalid exchange arena offset/extent")
    return raw[offset:offset+nbytes].view(dt).view(shape)


def refresh_recon_scales(module, slot, local):
    """Preserve graph-visible addresses while refreshing the unfolded copied scales."""
    changed=False
    resident=getattr(module,"batch_recon",None)
    if resident and not resident.folded:
        changed=True
        for name,ls in (("g",module.gates) if module.gated else (None,None),
                        ("u",module.ups),("d",module.downs)):
            if name is None: continue
            suh,svh=resident.scales[name]
            suh[slot].copy_(ls[slot].inner.suh)
            svh[slot].copy_(ls[slot].inner.svh)
    host=module.cpu_host
    seen=set()
    for state in list((getattr(host,"_dev_bufs",None) or {}).values()) + list((getattr(host,"sstate",None) or {}).values()):
        recon=state.get("recon",{}).get(module.cpu_layer_idx)
        if recon is None or recon.folded or id(recon) in seen: continue
        seen.add(id(recon))
        changed=True
        aux=host.aux[module.cpu_layer_idx]
        for name,(suh,svh) in recon.scales.items():
            suh[local].copy_(aux["suh_"+name][local])
            svh[local].copy_(aux["svh_"+name][local])
    return changed


def run_sweep(ip, reg):
    import torch
    if torch.cuda.is_current_stream_capturing():
        ip.moe_cpu_swap_pending=True
        return
    # Include every affected host: earlier layers may already have committed before a
    # later layer fails. No retry is allowed against any part of this registry.
    hosts={id(m.cpu_host):m.cpu_host for m in reg}.values()
    try:
        for host in hosts: require_healthy(host)
        budget=int(os.environ.get("EXL3_MOE_CPU_SWAP_MAX",64))
        scope=os.environ.get("EXL3_MOE_CPU_SWAP_BUDGET_SCOPE","global")
        if scope not in ("global","layer"): raise ValueError("swap budget scope must be global or layer")
        reg[0]._split_sweep_layer_reset()
        import time
        started=time.perf_counter(); fence_s=0.; total=0
        with torch.inference_mode():
            for m in reg:
                b=budget if scope=='layer' else budget-total
                if b<=0: break
                before=time.perf_counter(); fence_layer(m); fence_s+=time.perf_counter()-before
                # Timeout/watchdog abort can release GPU waits without worker completion.
                require_healthy(m.cpu_host)
                with torch.cuda.device(m.device):
                    total+=m._split_sweep_layer(b)
                    committed=torch.cuda.Event(); committed.record(torch.cuda.current_stream(m.device))
                    committed.synchronize()
                require_healthy(m.cpu_host)
        if os.environ.get("EXL3_MOE_CPU_SWAP_DEBUG"):
            print(f" -- exchange sweep: {total} swaps wall_ms={(time.perf_counter()-started)*1000:.3f} fence_ms={fence_s*1000:.3f}",flush=True)
    except BaseException:
        for host in hosts: poison(host)
        raise
