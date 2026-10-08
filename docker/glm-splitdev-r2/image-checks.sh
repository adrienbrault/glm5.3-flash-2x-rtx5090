#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 EXL3_MOE_CPU_SPLIT_CHECK=0
python3 /opt/splitdev/landing.py
P=$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("exllamav3").origin).parent.parent)')
python3 /opt/splitdev/test_splitdev.py --tree "$P" --base-tree /opt/splitdev/base --torch
