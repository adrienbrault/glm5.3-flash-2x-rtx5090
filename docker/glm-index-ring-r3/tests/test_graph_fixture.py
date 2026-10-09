#!/usr/bin/env python3
"""CPU replay of the graph probe's real fixture and append/pool kernel bodies.

Checks history/addressing, not CUDA compilation or GPU floating-point behavior.
Uses real PyTorch in Docker, the explicit NumPy adapter on the authoring host.
"""
import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
try:
    import torch
except ModuleNotFoundError:
    if os.environ.get('RING2_NUMPY_TESTS') != '1':
        raise
    from numpy_torchshim import torch
    print('CPU NumPy adapter: real graph fixture/kernel address replay; GPU rerun required', flush=True)
import test_cpu as cpu

ROOT = Path(__file__).resolve().parents[2] / 'src/exllamav3'


class Launch:
    def __init__(self, file, name):
        self.tl = cpu.TL()
        self.fn = cpu.kernel(file, name, self.tl)

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in ('num_warps', 'num_stages')}
            args = [cpu.Ptr(a.numpy().reshape(-1)) if hasattr(a, 'numpy') else a for a in args]
            for x in range(grid[0]):
                for y in range(grid[1] if len(grid) > 1 else 1):
                    self.tl.pid = (x, y)
                    self.fn(*args, **kwargs)
        return run


class GraphFixtureTests(unittest.TestCase):
    def test_warmup_then_restore_and_append_matches_all_scored_history(self):
        cpu.ROOT = ROOT
        app = Launch('modules/attention_fn/mla_triton.py', '_mla_plane_update_kernel')
        pool_kernel = Launch('modules/attention_fn/dsa_triton.py', '_dsa_pool_update_kernel')
        tree = ast.parse((Path(__file__).with_name('test_graph_shadow.py')).read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_history')
        ns = dict(torch=torch, _mla_plane_update_kernel=app, _dsa_pool_update_kernel=pool_kernel)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), 'graph fixture', 'exec'), ns)
        torch.manual_seed(919)
        P, D, PAGE = 4, 8, 256
        module = SimpleNamespace(index_kpool=P, index_head_dim=D,
                                 idx_kpool_ape=torch.randn((P, D)))
        ring = SimpleNamespace(rows=2304, max_chunk_size=2048,
                               ring=torch.zeros((4, 2304, 2 * D), dtype=torch.half))
        old = torch.zeros((16, PAGE, 2 * D), dtype=torch.half)
        pool = torch.zeros((16, PAGE // P, D), dtype=torch.half)
        shadow = SimpleNamespace(pool=torch.zeros_like(pool))
        table = torch.tensor([[9, 2, 15, 0, 7, 5, 1, 13, 4, 12, 6, 10, 3, 8, 11, 14]], dtype=torch.int32)
        seq = torch.tensor([0], dtype=torch.int32)
        prefix = torch.randn((1, 3007, 2 * D), dtype=torch.half)
        slot, length = 3, 2

        def append(plane, destination, nr, data):
            app[(length,)](data, plane, table, seq, 16, length, page_size=PAGE,
                           D=2 * D, DST_D=0, DST_OFF=0, RING_ROWS=nr)
            pool_kernel[(1, length // P + 1)](plane, destination, module.idx_kpool_ape, table, seq,
                16, length, page_size=PAGE, P=P, D=D, MAXPOOLS=1, RING_ROWS=nr)

        # Exact R919 trigger: warmup/capture writes only served pool 0 at seq=0.
        append(ring.ring[slot:slot+1], pool, ring.rows, prefix[:, :length].contiguous())
        self.assertFalse(torch.equal(pool.view(torch.uint8), shadow.pool.view(torch.uint8)))
        for start in (250, 251, 252, 253, 254, 255, 256, 2303, 2305, 3003, 3004, 3005):
            ns['prepare_history'](start, prefix, ring, old, pool, shadow, module, table, seq, slot)
            data = prefix[:, start:start+length].contiguous()
            append(ring.ring[slot:slot+1], pool, ring.rows, data)
            append(old, shadow.pool, 0, data)
            ids = torch.arange((start + length) // P)
            physical = table[0, ids // (PAGE // P)].long() * (PAGE // P) + ids % (PAGE // P)
            served = pool.view(-1, D)[physical]
            oracle = shadow.pool.view(-1, D)[physical]
            self.assertTrue(torch.equal(served.contiguous().view(torch.uint8),
                                        oracle.contiguous().view(torch.uint8)),
                            f'inconsistent scored history at start={start}, including logical pool 0')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, default=ROOT)
    a, left = p.parse_known_args(); ROOT = a.source
    unittest.main(argv=['test_graph_fixture.py'] + left)
