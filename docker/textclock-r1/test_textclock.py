#!/usr/bin/env python3
"""Execute the complete patched Python modules with only native/CUDA dependencies stubbed.
No GPU math/transfer correctness claims. Mutation variants exist only in memory.
"""
import contextlib
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CPU = (ROOT / 'textclock/modules/block_sparse_mlp_cpu.py').read_text()
EXCHANGE = (ROOT / 'textclock/model/moe_exchange.py').read_text()
ACTIVE_CPU, ACTIVE_EXCHANGE = CPU, EXCHANGE
VALID = dict(EXL3_MOE_CPU_SWAP_MODE='exchange', EXL3_MOE_CPU_SWAP_POLICY='histogram',
             EXL3_MOE_CPU_SWAP_CADENCE='exact', EXL3_MOE_CPU_SWAP_INTERVAL='64')


def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


class Harness:
    def __init__(self, flag='1', debug=False, env=None, cpu=None, exchange=None):
        self.capture = False
        self.output = io.StringIO()
        self.env = dict(VALID, EXL3_MOE_CPU_SWAP_DEBUG='1' if debug else '')
        if flag is not None:
            self.env['EXL3_MOE_CPU_SWAP_TEXT_CLOCK'] = flag
        self.env.update(env or {})
        class Mode(contextlib.ContextDecorator):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        class Event:
            def record(self, *a): pass
            def synchronize(self): pass
        torch = mod('torch', inference_mode=Mode, device=lambda d: Mode(),
                    arange=lambda *a, **k: object(), zeros=lambda *a, **k: object(),
                    long=0, float=1)
        torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: self.capture,
                                          device=lambda d: Mode(), Event=Event,
                                          current_stream=lambda d: None)
        modules = {n: mod(n, __path__=[]) for n in ('exllamav3', 'exllamav3.modules',
                                                  'exllamav3.model', 'exllamav3.util')}
        modules.update({'torch': torch, 'typing_extensions': mod('typing_extensions', override=lambda f: f),
                        'exllamav3.ext': mod('exllamav3.ext', exllamav3_ext=object()),
                        'exllamav3.model.moe_split_check': mod('exllamav3.model.moe_split_check', ENABLED=False),
                        'exllamav3.model.moe_score': mod('exllamav3.model.moe_score', initial_map=lambda x: x, settings=lambda: None),
                        'exllamav3.util.mtp_phase': mod('exllamav3.util.mtp_phase', instrument=lambda *a, **k: lambda f: f)})
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.dict(os.environ, self.env, clear=True))
        self.stack.enter_context(patch.dict(sys.modules, modules))
        self.stack.enter_context(contextlib.redirect_stdout(self.output))
        try:
            self.cpu = mod('exllamav3.modules.block_sparse_mlp_cpu')
            sys.modules[self.cpu.__name__] = self.cpu
            exec(compile(cpu if cpu is not None else ACTIVE_CPU, 'block_sparse_mlp_cpu.py', 'exec'), self.cpu.__dict__)
            self.exchange = mod('exllamav3.model.moe_exchange')
            sys.modules[self.exchange.__name__] = self.exchange
            exec(compile(exchange if exchange is not None else ACTIVE_EXCHANGE, 'moe_exchange.py', 'exec'), self.exchange.__dict__)
            self.exchange.fence_layer = lambda m: None
        except BaseException:
            self.stack.close()
            raise
        self.ip = types.SimpleNamespace(moe_cpu_swap_modules=[], moe_cpu_swap_pending=False)
        self.sweep_calls = []

    def __enter__(self): return self
    def __exit__(self, *a): self.stack.close()
    def layer(self, component, key=None):
        m = self.cpu.BlockSparseMLP_CPU()
        m.config = types.SimpleNamespace(infer_params=self.ip)
        m.cpu_component, m.key = component, key or component
        m._swap_tick_count = 0
        m._score_policy = False
        m.device, m.num_experts = 0, 256
        m.cpu_split_first, m._split_perm, m._split_dynamic = 128, None, True
        m.cpu_host = types.SimpleNamespace(exchange_failed=False)
        m._split_sweep_layer = lambda budget: self.sweep_calls.append((m.key, budget)) or 0
        m.cpu_post_load()
        return m
    def calls(self, layer, n):
        for _ in range(n): layer._split_swap_tick()


class ClockTests(unittest.TestCase):
    def test_mtp_first_on(self):
        with Harness() as h:
            mtp, text = h.layer('mtp'), h.layer('text')
            h.calls(mtp, 100)
            self.assertEqual(mtp._swap_tick_count, 0)
            h.calls(text, 6000)
            self.assertEqual(len(h.sweep_calls), 93 * 2)
            self.assertEqual(text._swap_tick_count, 48)
            self.assertIs(h.cpu.swap_clock_owner(h.ip.moe_cpu_swap_modules), text)

    def test_text_first_on_and_off(self):
        for flag in ('0', '1'):
            with Harness(flag) as h:
                text, mtp, text2 = h.layer('text', 'first'), h.layer('mtp'), h.layer('text', 'second')
                h.calls(mtp, 80); h.calls(text2, 80); h.calls(text, 129)
                self.assertEqual(len(h.sweep_calls), 6)
                self.assertEqual(text._swap_tick_count, 1)
                self.assertEqual(text2._swap_tick_count, 0)

    def test_mtp_first_off_and_default(self):
        for flag in ('0', None):
            with Harness(flag, debug=True) as h:
                mtp, text = h.layer('mtp'), h.layer('text')
                h.calls(text, 6000)
                self.assertEqual(len(h.sweep_calls), 0)
                h.calls(mtp, 129)
                self.assertEqual(len(h.sweep_calls), 4)
                self.assertEqual(mtp._swap_tick_count, 1)
                self.assertNotIn('[SWAP-', h.output.getvalue())
                self.assertFalse(hasattr(h.ip, '_text_clock_sweeps'))

    def test_reset_same_owner_not_every_call_after_64(self):
        with Harness() as h:
            mtp, text = h.layer('mtp'), h.layer('text')
            h.calls(text, 64)
            self.assertEqual(len(h.sweep_calls), 0)
            h.calls(text, 1)
            self.assertEqual(len(h.sweep_calls), 2)
            self.assertEqual(text._swap_tick_count, 1)
            h.calls(text, 63)
            self.assertEqual(len(h.sweep_calls), 2)
            h.calls(text, 1)
            self.assertEqual(len(h.sweep_calls), 4)
            self.assertEqual(mtp._swap_tick_count, 0)

    def test_queue_drained_reset_same_owner(self):
        with Harness() as h:
            mtp, text = h.layer('mtp'), h.layer('text')
            h.calls(text, 64)
            h.ip.moe_cpu_swap_pending = True
            h.cpu.run_pending_swap_sweeps(h.ip)
            self.assertEqual(text._swap_tick_count, 0)
            h.calls(text, 64)
            self.assertEqual(len(h.sweep_calls), 2)

    def test_draft_fallback_current_registry_and_rollback(self):
        with Harness() as h:
            mtp = h.layer('mtp'); h.calls(mtp, 65)
            self.assertEqual(mtp._swap_tick_count, 1)
            text = h.layer('text'); h.calls(text, 65)
            self.assertEqual(text._swap_tick_count, 1)
            self.assertEqual(mtp._swap_tick_count, 1)
            h.ip.moe_cpu_swap_modules.remove(text)
            self.assertIs(h.cpu.swap_clock_owner(h.ip.moe_cpu_swap_modules), mtp)
            text2 = h.layer('text', 'replacement')
            self.assertIs(h.cpu.swap_clock_owner(h.ip.moe_cpu_swap_modules), text2)

    def test_capture_does_not_tick_or_reset(self):
        with Harness() as h:
            h.layer('mtp'); text = h.layer('text'); h.calls(text, 64)
            h.capture = True; h.calls(text, 100)
            h.exchange.run_sweep(h.ip, h.ip.moe_cpu_swap_modules)
            self.assertEqual(text._swap_tick_count, 64)
            self.assertEqual(len(h.sweep_calls), 0)
            self.assertTrue(h.ip.moe_cpu_swap_pending)

    def test_validation_rejects_each_unsupported_setting(self):
        for key, values in (('MODE', ('checkpoint', 'bogus')), ('POLICY', ('score', 'bogus')),
                            ('CADENCE', ('served', 'bogus'))):
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, 'TEXT_CLOCK=1 requires'):
                        with Harness(env={'EXL3_MOE_CPU_SWAP_' + key: value}): pass
        with self.assertRaisesRegex(ValueError, 'TEXT_CLOCK=1 requires'):
            with Harness(env={'EXL3_MOE_CPU_SWAP_MODE': 'checkpoint'}): pass

    def test_validation_off_accepts_legacy_settings(self):
        with Harness('0', env={'EXL3_MOE_CPU_SWAP_MODE': 'checkpoint', 'EXL3_MOE_CPU_SWAP_POLICY': 'score',
                               'EXL3_MOE_CPU_SWAP_CADENCE': 'served'}): pass
        with Harness(env={'EXL3_MOE_CPU_SWAP_POLICY': 'histogram', 'EXL3_MOE_CPU_SWAP_CADENCE': 'exact'}): pass

    def test_debug_counters_and_startup(self):
        with Harness(debug=True) as h:
            h.layer('mtp'); text = h.layer('text'); h.calls(text, 129)
            self.assertEqual(h.ip._text_clock_ticks, 129)
            self.assertEqual(h.ip._text_clock_sweeps, 2)
            self.assertIn('[SWAP-OWNER] owner=text component=text modules=2', h.output.getvalue())
            self.assertIn('owner_tick=0 ticks=128 pending=0 sweeps=2 event=sweep', h.output.getvalue())
        with Harness(debug=False) as h:
            text = h.layer('text'); h.calls(text, 129)
            self.assertNotIn('[SWAP-CLOCK]', h.output.getvalue())
            self.assertIn('[SWAP-OWNER]', h.output.getvalue())

    def test_flag_off_identical_trace_and_log_to_daily(self):
        base_cpu = (ROOT.parent / 'src/exllamav3/modules/block_sparse_mlp_cpu.py').read_text()
        base_ex = (ROOT.parent / 'src/exllamav3/model/moe_exchange.py').read_text()
        for order in (('mtp', 'text'), ('text', 'mtp')):
            traces = []
            for c, e in ((base_cpu, base_ex), (ACTIVE_CPU, ACTIVE_EXCHANGE)):
                with Harness('0', cpu=c, exchange=e) as h:
                    # Suppress timing jitter in the legacy DEBUG line; DEBUG itself is off.
                    a, b = [h.layer(comp) for comp in order]
                    h.calls(b, 70); h.calls(a, 129)
                    traces.append((h.sweep_calls, a._swap_tick_count, b._swap_tick_count,
                                   h.ip.__dict__.copy(), h.output.getvalue()))
                    traces[-1][3].pop('moe_cpu_swap_modules')
            self.assertEqual(traces[0], traces[1])


def suite(): return unittest.defaultTestLoader.loadTestsFromTestCase(ClockTests)


def main():
    global ACTIVE_CPU, ACTIVE_EXCHANGE
    print('CPU module tests: complete modules, fake Torch/native dependencies; no GPU executed.', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite())
    if not result.wasSuccessful(): return 1
    mutations = [
        ('gate uses registry zero', CPU.replace('if swap_clock_owner(reg) is not self:', 'if reg[0] is not self:'), EXCHANGE),
        ('reset uses registry zero', CPU, EXCHANGE.replace('swap_clock_owner(reg)._split_sweep_layer_reset()', 'reg[0]._split_sweep_layer_reset()')),
        ('last text instead of first', CPU.replace('m for m in reg if m.cpu_component', 'm for m in reversed(reg) if m.cpu_component'), EXCHANGE),
        ('no draft-only fallback', CPU.replace(', reg[0])', ', None)'), EXCHANGE),
        ('flag ignored when off', CPU.replace('if _text_clock_enabled:\n        return next', 'if True:\n        return next'), EXCHANGE),
        ('default flag on', CPU.replace('"EXL3_MOE_CPU_SWAP_TEXT_CLOCK", "0"', '"EXL3_MOE_CPU_SWAP_TEXT_CLOCK", "1"'), EXCHANGE),
        ('no configuration validation', CPU.replace('if _clock_config != {', 'if False and _clock_config != {'), EXCHANGE),
        ('startup line missing', CPU.replace('[SWAP-OWNER]', '[OWNER]'), EXCHANGE),
        ('debug sweep counter missing', CPU, EXCHANGE.replace('ip._text_clock_sweeps = getattr(ip, "_text_clock_sweeps", 0) + 1', 'ip._text_clock_sweeps = 0')),
        ('capture tick allowed', CPU.replace('if torch.cuda.is_current_stream_capturing():', 'if False:'), EXCHANGE),
    ]
    killed = 0
    for name, c, e in mutations:
        assert c != CPU or e != EXCHANGE, name
        ACTIVE_CPU, ACTIVE_EXCHANGE = c, e
        stream = io.StringIO()
        r = unittest.TextTestRunner(stream=stream).run(suite())
        caught = not r.wasSuccessful()
        killed += caught
        print(f'MUTATION {"CAUGHT" if caught else "SURVIVED"}: {name}; failures={len(r.failures)} errors={len(r.errors)}')
        if not caught: print(stream.getvalue())
    ACTIVE_CPU, ACTIVE_EXCHANGE = CPU, EXCHANGE
    print(f'Mutation result: {killed}/{len(mutations)} caught')
    return 0 if killed == len(mutations) else 1


if __name__ == '__main__': sys.exit(main())
