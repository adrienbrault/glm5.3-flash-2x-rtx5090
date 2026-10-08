#!/usr/bin/env bash
# Install the GLM-5.3-Flash daily definition on flan (2026-10-08). One file is the daily: /srv/qwen5090/glm-daily.env
# (repo: flan/r858/glm-daily.env), launched with the tools in /srv/qwen5090/glm-daily-tools (r885c-tools + the repo's
# current launcher/plan/probe). glm_arms.sh units take it as their experiment base at start and re-read it when they
# restore the daily, so a promotion mid-chain is honored by whichever unit restores last.
# Run from the repo:  rsync -a flan/r858/{glm-daily.env,launch-glm53.sh,glm53_plan.py,glm53_probe.py,mtp_steps.py} flan:/tmp/glm-daily-src/
#                     ssh flan 'bash -s' < flan/r858/install-glm-daily.sh
set -euo pipefail
S=/tmp/glm-daily-src
T=/srv/qwen5090/glm-daily-tools
rm -rf "$T.new"; cp -a /srv/qwen5090/r885c-tools "$T.new"
cp "$S/launch-glm53.sh" "$S/glm53_plan.py" "$S/glm53_probe.py" "$S/mtp_steps.py" "$T.new/"
bash -n "$T.new/launch-glm53.sh"
[[ -f "$T.new/split-stats-broad-r869.json" ]]
rm -rf "$T.old"; [[ -d "$T" ]] && mv "$T" "$T.old"; mv "$T.new" "$T"
cp "$S/glm-daily.env" /srv/qwen5090/glm-daily.env.new && mv /srv/qwen5090/glm-daily.env.new /srv/qwen5090/glm-daily.env
echo "installed: $(cat /srv/qwen5090/glm-daily.env)"
