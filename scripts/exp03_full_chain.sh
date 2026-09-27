#!/bin/bash
# exp03 full run on the one GPU, after the smoke tests wrote their decision files. Every step skips work whose
# outputs exist, so the chain can be restarted. Weights downloaded for exp03 are deleted once a model is done
# (disk); Qwen3.8-27B (judge) and Qwen3-32B stay.
set -euo pipefail
cd "$(dirname "$0")"
VLLM=./vllm_python.sh
PY=/venv/main/bin/python
LOG=../log/exp03
step() { echo "== $* $(date -u +%FT%TZ)"; }
repo_dir() { $PY -c "import ladder_config as b; s=b.ALL_SUBJECTS['$1']; print(b.hf_repo_dir(s['repo']))"; }
download() { $PY -c "import ladder_config as b; s=b.ALL_SUBJECTS['$1']; print(s['repo'], s['revision'])" | \
             { read -r repo rev; /venv/main/bin/hf download "$repo" --revision "$rev" > /dev/null; }; }
delete_weights() { local d; d=$(repo_dir "$1"); step "delete $d"; rm -rf "$d"; }

[ -f /workspace/eval-awareness/results/exp03_vea_ladder_v2/manifest.json ] || $PY exp03_manifest.py

step "Qwen3.8-27B stage-2 rollouts 30-49"
$VLLM exp03_generate.py stage2 > $LOG/stage2_Qwen3.8-27B.log 2>&1

for m in Qwen3.8-27B Qwen3.5-27B Qwen3.5-9B Qwen3-14B; do
    step "$m interventions"
    $VLLM exp03_generate.py interventions --models $m > $LOG/interventions_$m.log 2>&1
    step "$m logit v2"
    $VLLM exp03_logit.py run --model $m --kinds format disguise > $LOG/logit_v2_$m.log 2>&1
    if [ "$m" != "Qwen3.8-27B" ]; then delete_weights $m; fi
done

step "Qwen3-32B logit v2"
$VLLM exp03_logit.py run --model Qwen3-32B --kinds format disguise > $LOG/logit_v2_Qwen3-32B.log 2>&1

for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B; do
    step "$m download + logit v2"
    download $m
    $VLLM exp03_logit.py run --model $m --kinds format disguise > $LOG/logit_v2_$m.log 2>&1
    delete_weights $m
done

step "judge: relabel all rollouts"
$VLLM exp03_judge.py label > $LOG/judge_label.log 2>&1
step "judge: intervention answers"
$VLLM exp03_judge.py interventions > $LOG/judge_interventions.log 2>&1
step "analysis"
$PY exp03_analysis.py --out-name final > $LOG/analysis_final.log 2>&1
step "done"
