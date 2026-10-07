#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -B tests/check_hashes.py SHA256SUMS .
docker image inspect tabbyapi:r828-prompt-lookup-r3 --format '{{.Id}}'
docker build --pull=false --network=none -t tabbyapi:r861-glm-agent-r2 .
docker image inspect tabbyapi:r861-glm-agent-r2 --format '{{.Id}}'
