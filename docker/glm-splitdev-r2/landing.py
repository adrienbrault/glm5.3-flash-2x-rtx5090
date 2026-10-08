#!/usr/bin/env python3
"""Image landing assert. Torch must load its shared libraries before the extension."""
import torch
import exllamav3_ext
import inspect
from exllamav3.model.moe_cpu_host import MoeCpuHost
from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU
from exllamav3.modules.moe_batch_recon import BatchReconLayer
from exllamav3.model import moe_split_check

for symbol in ('exl3_moe_cpu_make_layer', 'moe_split_map', 'moe_split_issue'):
    assert hasattr(exllamav3_ext, symbol), symbol
assert 'EXL3_MOE_CPU_SPLIT_BY_DEVICE' in inspect.getsource(BlockSparseMLP_CPU.cpu_maybe_split_load)
assert 'old["up_keys"] == up_keys' in inspect.getsource(MoeCpuHost.register_layer)
assert 'split_check.index_range' in inspect.getsource(MoeCpuHost.submit_prefill)
assert 'layer_key' in inspect.signature(BatchReconLayer).parameters
assert callable(moe_split_check.module_layout)
print(f'SPLITDEV2 LANDED torch={torch.__version__} ext={exllamav3_ext.__file__} CHECK={int(moe_split_check.ENABLED)}')
