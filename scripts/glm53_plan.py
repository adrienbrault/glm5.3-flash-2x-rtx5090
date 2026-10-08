#!/usr/bin/env python3
"""CPU-only header accounting and deterministic launch configuration."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import struct

G = 1 << 30
PACKS = {'A': 'glm53-flash-exl3-2.05bpw-turboderp',
         'B': 'glm53-flash-exl3-2.25bpw-r0b0tlab'}
EXPERT = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$')

def template(path):
    p = Path(path)
    tok = json.loads((p / 'tokenizer_config.json').read_text())
    t = (p / 'chat_template.jinja').read_text() if (p / 'chat_template.jinja').exists() else tok.get('chat_template', '')
    if not isinstance(t, str):
        raise ValueError('non-string chat_template; inspect named templates before audition')
    if not all(x in t for x in ('<think>', '</think>')):
        raise ValueError('template does not establish <think>/</think>; run the template probe and adapt before audition')
    if 'enable_thinking' in t:
        return {'enable_thinking': True, 'reasoning_start': '<think>', 'reasoning_end': '</think>', 'template': t}
    # GLM-5.3-Flash (2026-10-07): no thinking-off switch; the generation prompt always opens <think> and only
    # reasoning_effort ('low' | 'high', default 'max') is a template variable. The probe runs effort low/high arms.
    if 'reasoning_effort' in t:
        return {'enable_thinking': False, 'always_think': True, 'effort_var': 'reasoning_effort',
                'reasoning_start': '<think>', 'reasoning_end': '</think>', 'template': t}
    raise ValueError('template has neither enable_thinking nor reasoning_effort; run the template probe and adapt before audition')

def inspect(path):
    p = Path(path)
    cfg = json.loads((p / 'config.json').read_text())
    if cfg['architectures'] != ['Glm5NextForConditionalGeneration']:
        raise ValueError('wrong architecture')
    tc = cfg['text_config']
    if (tc['hidden_size'], tc['moe_intermediate_size'], tc['n_routed_experts'], tc['num_hidden_layers']) != (4096, 2048, 288, 45):
        raise ValueError('pack geometry differs from brief; review memory analysis')
    for name in ('processor_config.json', 'tokenizer_config.json'):
        if not (p / name).is_file():
            raise ValueError(f'missing {name}')
    if not any((p / name).is_file() for name in ('tokenizer.json', 'tokenizer.model')):
        raise ValueError('missing tokenizer data')
    index = json.loads((p / 'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    for f in sorted(p.glob('*.safetensors')):
        with f.open('rb') as fp:
            prefix = fp.read(8)
            if len(prefix) != 8:
                raise ValueError(f'incomplete {f}')
            n = struct.unpack('<Q', prefix)[0]
            if not 2 <= n <= 128 * 1024 * 1024:
                raise ValueError(f'bad header size {f}')
            h = json.loads(fp.read(n))
        for k, v in h.items():
            if k == '__metadata__':
                continue
            a, b = v['data_offsets']
            if not 0 <= a <= b <= f.stat().st_size - 8 - n:
                raise ValueError(f'truncated shard {f}: {k}')
            if k in headers:
                raise ValueError(f'duplicate tensor {k}')
            width = {'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U64':8,'U32':4,'U16':2,'U8':1,'BOOL':1,'F8_E4M3':1,'F8_E5M2':1}.get(v['dtype'])
            if width is not None and math.prod(v['shape']) * width != b-a:
                raise ValueError(f'wrong payload byte count {f}: {k}')
            headers[k] = {**v, 'file': f.name, 'bytes': b - a}
    for k, f in index.items():
        if k not in headers or headers[k]['file'] != f:
            raise ValueError(f'missing/wrong indexed tensor {k} in {f}')
    projections = {}
    for k, h in headers.items():
        m = EXPERT.match(k)
        if m:
            l, e, proj, suffix = m.groups()
            projections.setdefault((int(l), int(e)), {}).setdefault(proj, {})[suffix] = h
    for l in {l for l,_ in projections}:
        for proj in ('gate_proj','up_proj','down_proj'):
            bias = ['bias' in ps[proj] for (layer,_),ps in projections.items() if layer==l]
            if any(bias) and not all(bias):
                raise ValueError(f'ineligible mixed expert biases L{l} {proj}')
    experts = []
    for (l, e), ps in sorted(projections.items()):
        cpu = gpu = trellis = 0
        blocks = []
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            hs = ps[proj]
            for required in ('trellis', 'suh', 'svh', 'mul1'):
                if required not in hs:
                    raise ValueError(f'ineligible expert L{l} E{e} {proj}: no {required}')
            expected_shape = [128, 256] if proj == 'down_proj' else [256, 128]
            if hs['trellis']['shape'][:2] != expected_shape:
                raise ValueError(f'ineligible dimensions L{l} E{e} {proj}')
            if hs['trellis']['shape'][-1] / 16 > 8:
                raise ValueError(f'ineligible expert L{l} E{e}: K > 8')
            sizes = []
            for suffix in ('trellis', 'suh', 'svh', 'bias'):
                if suffix not in hs:
                    continue
                h = hs[suffix]
                nb = h['bytes']
                if suffix != 'trellis' and h['dtype'] == 'F32':
                    nb = math.prod(h['shape']) * 2
                sizes.append(nb)
            tr = hs['trellis']['bytes']
            trellis += tr
            cpu += sum(sizes)
            # GPU retains tiny codebook metadata; subtract only bytes proven to move.
            gpu += tr  # aux is also retained on GPU for streamed prefill
            blocks.append(sizes)
        experts.append({'layer': l, 'expert': e, 'cpu': cpu, 'gpu': gpu,
                        'trellis': trellis, 'aux_sizes': [n for s in blocks for n in s[1:]]})
    layers = sorted({e['layer'] for e in experts if e['layer'] < 45})
    if layers != list(range(3, 45)):
        raise ValueError('expected 42 routed text layers (3..44)')
    for l in layers + [45]:
        if sorted(e['expert'] for e in experts if e['layer'] == l) != list(range(288)):
            raise ValueError(f'layer {l}: expected all 288 experts (including MTP)')
    # Include all disk tensors as a conservative resident base; fp32 aux becomes fp16.
    weights = sum(h['bytes'] for h in headers.values())
    draft = sum(h['bytes'] for k, h in headers.items() if k.startswith('model.language_model.layers.45.'))
    return {'path': str(p), 'weights': weights, 'draft_weights': draft,
            'quantization': cfg.get('quantization_config'), 'experts': experts,
            'eligible_layers': layers, 'tensor_count': len(headers)}

def fixture(pack):
    bits, disk = (2.05, 80e9) if pack == 'A' else (2.32, 98.5e9)
    exp = 3 * 4096 * 2048
    experts = []
    for l in range(3, 46):
        b = bits if l < 45 else (2 if pack == 'A' else 4)
        tr = int(exp * b / 8)
        for e in range(288):
            experts.append({'layer': l, 'expert': e, 'cpu': tr + 36864,
                            'gpu': tr, 'trellis': tr,
                            'aux_sizes': [8192, 4096, 8192, 4096, 4096, 8192]})
    return {'weights': int(disk), 'draft_weights': int(exp * 288 * (2 if pack == 'A' else 4) / 8),
            'experts': experts, 'eligible_layers': list(range(3, 45)), 'fixture': True}

def arena_capacity(es):
    cap = off = 0
    for e in es:
        tr = e['trellis']
        if cap == 0 or off + ((tr + 63) // 64 * 64) > G:
            cap += G
            off = 0
        off += (tr + 63) // 64 * 64
        for n in e['aux_sizes']:
            aligned = (n + 63) // 64 * 64
            if off + aligned > G:
                cap += G
                off = 0
            off += aligned
    return cap

def candidate(data, mode, n, draft, cache):
    chosen = [e for e in data['experts'] if (draft or e['layer'] < 45) and
              ((mode == 'layers' and e['layer'] in data['eligible_layers'][:n]) or
               (mode == 'split' and e['expert'] >= 288 - n))]
    workers = [[e for e in chosen if e['layer'] < 45], [e for e in chosen if e['layer'] >= 45]]
    workers = [es for es in workers if es]
    arena = sum(arena_capacity(es) for es in workers)
    incoming = max((sum(e['cpu'] for e in chosen if e['layer'] == l) for l in range(3, 46)), default=0)
    moved = sum(e['gpu'] for e in chosen)
    resident = data['weights'] - moved - (0 if draft else data['draft_weights'])
    kv = cache * (11 + draft) * 864
    return {'mode': mode, 'n': n, 'draft': draft, 'cache': cache,
            'resident': resident, 'offload_bytes': sum(e['cpu'] for e in chosen),
            'arena_capacity': arena, 'incoming_slice': incoming,
            'host_peak': arena + incoming + len(workers) * (72 << 20) + 3 * G,
            'gpu_estimate': resident + kv + 2 * G}

def ladder(data, available, cache, split, draft):
    accepted, excluded = [], []
    for mode, counts in [('split', range(8, 288, 8)), ('layers', range(1, 43))]:
        for n in counts:
            for d in ([1, 0] if draft == 'auto' else [int(draft)]):
                c = candidate(data, mode, n, d, cache)
                if c['gpu_estimate'] > sum(split) * G:
                    continue
                if c['host_peak'] > available:
                    excluded.append({**c, 'reason': 'host load budget exceeds MemAvailable'})
                else:
                    accepted.append(c)
    accepted.sort(key=lambda c: (-c['resident'], -c['draft'], c['mode'] != 'split', c['n']))
    return {'available_bytes': available, 'gpu_split_gib': split,
            'candidates': accepted, 'excluded': excluded}

def settings():
    pack = os.environ.get('PACK', 'A')
    if pack in PACKS:
        pack = PACKS[pack]
    if pack not in PACKS.values():
        raise ValueError('PACK must be A, B, or one of the two confirmed directory names')
    def integer(key, default, low, high):
        raw = os.environ.get(key, str(default))
        if not re.fullmatch(r'[0-9]+', raw):
            raise ValueError(f'{key} must be an unsigned integer')
        value = int(raw)
        if not low <= value <= high:
            raise ValueError(f'{key} out of range {low}..{high}')
        return value
    if os.environ.get('PORT', '8029') != '8029':
        raise ValueError('PORT must be 8029; 8022 is reserved for the daily')
    if os.environ.get('TP', 'false') != 'false':
        raise ValueError('tensor parallel is forbidden in this audition')
    if os.environ.get('EXTRA_ENV') or os.environ.get('EXTRA_ENV_ADD'):
        raise ValueError('EXTRA_ENV passthrough forbidden; use the documented knobs')
    mode = os.environ.get('OFFLOAD_MODE', 'split')
    if mode not in ('split', 'layers'):
        raise ValueError('OFFLOAD_MODE must be split or layers')
    n = integer('OFFLOAD_N', 80 if mode == 'split' else 12, 1, 287 if mode == 'split' else 42)
    cache = integer('CACHE_TOKENS', 65536, 65536, 1048576)
    if cache % 256:
        raise ValueError('CACHE_TOKENS must be a multiple of 256')
    cm = os.environ.get('CACHE_MODE', '8,8')
    if not re.fullmatch(r'[2-8],[2-8]', cm):
        raise ValueError('CACHE_MODE must be K,V bits in 2..8')
    split = [float(s.strip()) for s in os.environ.get('GPU_SPLIT', '31,31').split(',')]
    # R865: up to 31.8 GiB (a 5090 has 31.84 GiB). R864 at split 31,31 left card 1 with ~2 GiB unused.
    if len(split) != 2 or not all(math.isfinite(v) and 0 < v <= 31.8 for v in split):
        raise ValueError('GPU_SPLIT must be two finite GiB values in (0,31.8]')
    draft = integer('DRAFT', 1, 0, 1)
    threads = integer('CPU_THREADS', 8, 1, 16)
    # R859 (2026-10-07): long-context c1 runs. MAX_SEQ defaults to the R858 audition cap; SYSMEM_RC_MB must be > 0:
    # with 0 the generator still builds a RecurrentCache (GLM has KDA layers) and the first prefill stash raises
    # KeyError('dictionary is empty') in recurrent.py put() (R858 A-r0 prefill-8192).
    max_seq = integer('MAX_SEQ', min(49152, cache), 4096, cache)
    sysmem_rc = integer('SYSMEM_RC_MB', 1024, 64, 16384)
    max_batch = integer('MAX_BATCH', 4, 1, 8)
    chunk = integer('CHUNK', 512, 256, 8192)
    # R860: MTP depth (GLM-5.3 has one MTP layer; depth > 1 chains it) and confidence-calibrated draft truncation
    draft_n = integer('DRAFT_N', 1, 1, 4)
    dyn_draft = integer('DYN_DRAFT', 0, 0, 1)
    # R864: VISION=0 drops the vision tower (DeepSeek Harness sends text only) to free VRAM for a lower N;
    # PLOOKUP=1 sets EXL3_PROMPT_LOOKUP=1 (adaptive prompt lookup beside MTP, R828), which needs fixed-depth MTP.
    vision = integer('VISION', 1, 0, 1)
    # R901 (2026-10-08): VISION_OFFLOAD=1 sets TabbyAPI's vision_offload (infer_params.vision_pinned): the ViT weights
    # (0.494 GiB) stay in pinned system RAM and are streamed to the GPU per image, freeing VRAM for GPU experts.
    vision_offload = integer('VISION_OFFLOAD', 0, 0, 1)
    # R866: expert placement for the CPU split. 'dynamic' is the served default (EXL3_MOE_CPU_SWAP=1); SWAP_INTERVAL /
    # SWAP_FLOOR (0 = engine default 128 / 8) tune its sweeps. 'static' = EXL3_MOE_CPU_SWAP=0 plus a per-layer counts
    # file (SPLIT_STATS, host path) that orders experts hot-to-cold once at load (R860b: dynamic placement left the CPU
    # share at 0.347 vs 0.361 uniform; sim_placement.py).
    placement = os.environ.get('PLACEMENT', 'dynamic')
    if placement not in ('dynamic', 'static'):
        raise ValueError('PLACEMENT must be dynamic or static')
    swap_interval = integer('SWAP_INTERVAL', 0, 0, 4096)
    swap_floor = integer('SWAP_FLOOR', 0, 0, 64)
    if placement == 'static' and (swap_interval or swap_floor):
        raise ValueError('SWAP_INTERVAL/SWAP_FLOOR apply to dynamic placement only')
    plookup = integer('PLOOKUP', 0, 0, 1)
    if plookup and (not draft or dyn_draft):
        raise ValueError('PLOOKUP=1 needs DRAFT=1 and DYN_DRAFT=0 (prompt lookup r3 requires fixed-depth MTP)')
    # R884 (2026-10-08): KEEP_THINKING=1 (default) serves templates/glm53-keep-thinking.jinja, the stock template with
    # clear_thinking defaulting to false: earlier turns keep their reasoning, so a new user message no longer rewrites
    # the previous tool loop and the prefix cache keeps matching (TensorFold's default). Requests can still send
    # template_vars/chat_template_kwargs {"clear_thinking": true}. KEEP_THINKING=0 = templates/glm53-stock.jinja, the model's own
    # template. Both default a missing/unknown reasoning_effort to high, not max (user 2026-10-08: "yes high default, not max";
    # max loops or ends at the length limit with no answer in three other recipes and in our 53-minute run); explicit max stays max.
    keep_thinking = integer('KEEP_THINKING', 1, 0, 1)
    return dict(keep_thinking=keep_thinking, pack=pack, mode=mode, n=n, cache=cache, cm=cm, split=split, draft=draft, threads=threads,
                max_seq=max_seq, sysmem_rc=sysmem_rc, max_batch=max_batch, chunk=chunk, draft_n=draft_n,
                dyn_draft=dyn_draft, vision=vision, vision_offload=vision_offload, plookup=plookup,
                placement=placement, swap_interval=swap_interval, swap_floor=swap_floor)

def config(s):
    text = f'''model:
  model_dir: /models
  model_name: {s['pack']}
  backend: exllamav3
  max_seq_len: {s['max_seq']}
  cache_size: {s['cache']}
  cache_mode: {s['cm']}
  max_batch_size: {s['max_batch']}
  tensor_parallel: false
  gpu_split: {json.dumps(s['split'])}
  gpu_split_auto: false
  cpu_moe_offload_layers: {s['n'] if s['mode'] == 'layers' else 0}
  cpu_moe_split_experts: {s['n'] if s['mode'] == 'split' else 0}
  chunk_size: {s['chunk']}
  output_chunking: true
  vision: {'true' if s['vision'] else 'false'}
  vision_offload: {'true' if s.get('vision_offload') else 'false'}
  # GLM-5.3 emits <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>; without tool_format
  # TabbyAPI returns it as plain content (2026-10-07, DeepSeek Harness bring-up). glm4_5 parses that layout.
  tool_format: glm4_5
  reasoning: true
  reasoning_start_token: <think>
  reasoning_end_token: </think>
  start_in_reasoning: auto
{"  prompt_template: " + ('glm53-keep-thinking' if s.get('keep_thinking') else 'glm53-stock') + chr(10)}draft_model:
  draft_mode: {'mtp' if s['draft'] else 'disabled'}
'''
    if s['draft']:
        text += (f"  draft_num_tokens: {s['draft_n']}\n  draft_cache_mode: Q8\n"
                 f"  dynamic_draft: {'true' if s['dyn_draft'] else 'false'}\n")
    text += f"memory:\n  sysmem_recurrent_cache: {s['sysmem_rc']}\n  sysmem_kv_cache: 0\n"
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['inspect', 'template', 'ladder', 'config', 'check'])
    ap.add_argument('path', nargs='?')
    ap.add_argument('--fixture', choices=['A', 'B'])
    ap.add_argument('--available', type=int, default=40 * G)
    ap.add_argument('--cache', type=int, default=65536)
    ap.add_argument('--split', default='31,31')
    ap.add_argument('--draft', choices=['auto', '0', '1'], default='auto')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.command in ('config', 'check'):
        s = settings()
        result = config(s) if a.command == 'config' else json.dumps(s)
    elif a.command == 'template':
        result = json.dumps(template(a.path), indent=2)
    else:
        data = fixture(a.fixture) if a.fixture else inspect(a.path)
        if a.command == 'inspect':
            data = {k: v for k, v in data.items() if k != 'experts'} | {
                'routed_text_cpu_bytes': sum(e['cpu'] for e in data['experts'] if e['layer'] < 45)}
        else:
            data = ladder(data, a.available, a.cache, [float(x) for x in a.split.split(',')], a.draft)
        result = json.dumps(data, indent=2)
    if a.out:
        Path(a.out).write_text(result + '\n')
    else:
        print(result)

if __name__ == '__main__':
    main()
