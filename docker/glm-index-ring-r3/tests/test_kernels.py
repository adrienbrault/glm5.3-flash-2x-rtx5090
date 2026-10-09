#!/usr/bin/env python3
"""GPU: same append/pool arithmetic with paged vs ring addressing; fail on any bit difference."""
import argparse
import random
from types import SimpleNamespace
import torch
from exllamav3.cache.mla_index_ring import MLAIndexRingState
from exllamav3.modules.attention_fn.mla_triton import _mla_plane_update_kernel
from exllamav3.modules.attention_fn.dsa_triton import _dsa_pool_update_kernel
from exllamav3.modules.attention_fn.bc_attn import _compile_kernel
from exllamav3.modules.mla_attn import MLAttention

P, D, PAGE = 4, 128, 256


def bit_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def run(slots, chunk, device):
    rng = random.Random(234 + slots + chunk)
    torch.manual_seed(17)
    module = SimpleNamespace(index_kpool=P, idx_plane_dim=2*D)
    state = MLAIndexRingState(module, slots, 16, 0, max_chunk_size=chunk)
    state.alloc(device)
    # Enough physical pages to wrap the ring many times with unrelated mappings.
    capacity = max(16384, 8*state.rows)
    npr = -(-capacity // PAGE)
    permutation = list(range(slots*npr)); rng.shuffle(permutation)
    bt = torch.tensor(permutation, dtype=torch.int32, device=device).view(slots, npr)
    old = torch.zeros((slots*npr, PAGE, 2*D), dtype=torch.half, device=device)
    pools_old = torch.zeros((slots*npr, PAGE//P, D), dtype=torch.half, device=device)
    pools_new = pools_old.clone()
    ape = torch.randn((P, D), dtype=torch.float32, device=device)
    pos = [0]*slots
    app_sig = {"rows_new":"*fp16", "plane_cache":"*fp16", "block_table":"*i32",
               "cache_seqlens":"*i32", "num_pages_per_seq":"i32", "append_len":"i32"}
    app_sig.update({k:"constexpr" for k in ("page_size","D","DST_D","DST_OFF","RING_ROWS")})
    pool_sig = {"plane":"*fp16", "pool_plane":"*fp16", "ape":"*fp32", "block_table":"*i32",
                "cache_seqlens":"*i32", "num_pages_per_row":"i32", "append_len":"i32"}
    pool_sig.update({k:"constexpr" for k in ("page_size","P","D","MAXPOOLS","RING_ROWS")})
    # AOT compilation is also a probe of the native launch ABI, without a model.
    app_aot = {off:_compile_kernel(torch.device(device), _mla_plane_update_kernel, app_sig,
               dict(page_size=PAGE,D=D,DST_D=2*D,DST_OFF=off,RING_ROWS=state.rows),2,2)
               for off in (0,D)}
    pool_aot = _compile_kernel(torch.device(device),_dsa_pool_update_kernel,pool_sig,
               dict(page_size=PAGE,P=P,D=D,MAXPOOLS=1,RING_ROWS=state.rows),2,1)
    assert app_aot and pool_aot

    def advance(slot, length):
        start = pos[slot]
        keys = torch.randn((length,D),dtype=torch.half,device=device)
        gates = torch.randn_like(keys)
        seq = torch.tensor([start],dtype=torch.int32,device=device)
        row_bt = bt[slot:slot+1]
        for data, off in ((keys,0),(gates,D)):
            for plane, nr in ((old,0),(state.ring[slot:slot+1],state.rows)):
                _mla_plane_update_kernel[(length,)](data,plane,row_bt,seq,npr,length,
                    page_size=PAGE,D=D,DST_D=2*D,DST_OFF=off,RING_ROWS=nr,num_warps=2,num_stages=2)
        for plane, pool, nr in ((old,pools_old,0),(state.ring[slot:slot+1],pools_new,state.rows)):
            _dsa_pool_update_kernel[(1,length//P+1)](plane,pool,ape,row_bt,seq,npr,length,
                page_size=PAGE,P=P,D=D,MAXPOOLS=1,RING_ROWS=nr,num_warps=2,num_stages=1)
        torch.cuda.synchronize(device)
        # Includes partial pools, exactly as the native kernel writes them.
        if not bit_equal(pools_old,pools_new):
            raise AssertionError(f"pool mismatch slot={slot} pos={start} append={length}")
        pos[slot] += length

    # Different positions per slot; maximum chunks, arbitrary tails and wrap boundaries.
    for round_no in range(28):
        order = list(range(slots)); rng.shuffle(order)
        for slot in order:
            length = min(chunk, (1,2,3,4,7,16,chunk)[(round_no+slot)%7])
            advance(slot,length)
    # Depth-1 MTP: two-row verify, one rejected row, then rewrite the rejected position.
    for slot in range(slots):
        for _ in range(8):
            advance(slot,2)
            state.rewind(slot,1,1)
            pos[slot]-=1
            advance(slot,1)
    # Checkpoint survives after its original ring has been overwritten.
    for remainder in (1,2,3):
        slot = remainder % slots
        advance(slot, (remainder-pos[slot])%P or P)
        saved_pos = pos[slot]
        tail = state.stash(slot,saved_pos)
        moved = MLAIndexRingState(module, slots+1, 16, 0, max_chunk_size=chunk)
        moved.alloc(device)
        moved.unstash(slots,tail,saved_pos)
        copied = moved.stash(slots,saved_pos)
        assert bit_equal(copied[0],tail[0]), "checkpoint slot migration mismatch"
        moved.free()
        old_image, pool_image = old.clone(), pools_old.clone()
        for _ in range(-(-state.rows//chunk)+1): advance(slot,chunk)
        state.unstash(slot,tail,saved_pos)
        pos[slot] = saved_pos
        old.copy_(old_image); pools_old.copy_(pool_image); pools_new.copy_(pool_image)
        advance(slot,4-remainder)
    # Python pooling: exercise the actual _update_pool_plane gather on both layouts.
    class Layer:
        def __init__(self, plane, pool, ring): self.plane,self.pool,self.index_ring=plane,pool,ring
        def get_idx(self): return self.plane
        def update_pool_direct(self,seqlens,table,keys):
            t = torch.arange(keys.shape[1],device=device)+seqlens[0]
            ids = table[0,t//(PAGE//P)].long()*(PAGE//P)+t%(PAGE//P)
            self.pool.view(-1,D)[ids] = keys[0]
    reference, candidate = torch.zeros_like(pools_old),torch.zeros_like(pools_old)
    obj = SimpleNamespace(index_kpool=P,index_head_dim=D,idx_kpool_ape=ape)
    # A small append at each slot's present position shares the same complete member rows.
    starts = [p-1 for p in pos]
    MLAttention._update_pool_plane(obj,Layer(old,reference,None),bt,starts,1)
    MLAttention._update_pool_plane(obj,Layer(state.ring,candidate,state),bt,starts,1,tuple(range(slots)))
    assert bit_equal(reference,candidate), "Python pooling gather mismatch"
    print(f"PASS slots={slots} chunk={chunk} ring_rows={state.rows}",flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--chunk',type=int,default=2048)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('GPU test requires CUDA; no skip counts as a pass')
    with torch.cuda.device(a.device):
        for slots in (1,4,8): run(slots,a.chunk,a.device)

if __name__=='__main__': main()
