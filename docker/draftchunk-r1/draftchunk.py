#!/usr/bin/env python3
"""Fail-closed, single-call patch for the daily TabbyAPI MTP draft loader."""
from pathlib import Path
import sys

TARGET = '''            for value in self.draft_model.load_gen(
                reserve_per_device=self.autosplit_reserve,
                use_per_device=self.draft_gpu_split or None,
                callback=progress_callback,
            ):'''
PATCHED = TARGET.replace('                callback=progress_callback,',
                         '                callback=progress_callback,\n                max_chunk_size=self.chunk_size,')


def patch_source(source):
    if source.count('self.draft_model.load_gen(') != 1 or source.count(TARGET) != 1:
        raise ValueError('draft load_gen target must match exactly once, unpatched')
    result = source.replace(TARGET, PATCHED, 1)
    assert result.count(PATCHED) == 1
    compile(result, '/app/backends/exllamav3/model.py', 'exec')
    return result


if __name__ == '__main__':
    path = Path(sys.argv[1])
    path.write_text(patch_source(path.read_text()))
