#!/bin/bash
# exp03 main chain, per-model order (replaces exp03_full_chain_resume2.sh, stopped 2026-09-27 ~22:07 UTC): each model
# is finished end to end (interventions, logit v2, relabel, judged intervention answers) before the next starts, so
# the live figures (exp03_live.py) gain one complete model at a time. Same work as before; the judge engine is loaded
# once per model instead of twice in total. Every step skips work whose outputs exist.
# Appends to full_chain.log so exp03_insertion_extension.sh sees "== done".
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
logit_v2() { step "$1 logit v2"; $VLLM exp03_logit.py run --model $1 --kinds format disguise >> $LOG/logit_v2_$1.log 2>&1; }
judge_model() { step "$1 judge: relabel + intervention answers"; $VLLM exp03_judge.py model --models $1 >> $LOG/judge_$1.log 2>&1; }

step "per-model chain start (judge decision: $(cat ../results/exp03_vea_ladder_v2/judge_engine_decision.json | tr -d '\n '))"

# Qwen3.8-27B: stage 2, interventions and logit v2 are done; its weights are the judge's, so they stay.
judge_model Qwen3.8-27B

for m in Qwen3.5-27B Qwen3.5-9B Qwen3-14B; do
    step "$m interventions"
    $VLLM exp03_generate.py interventions --models $m >> $LOG/interventions_$m.log 2>&1
    logit_v2 $m
    if [ "$m" != "Qwen3.5-9B" ]; then delete_weights $m; fi
    judge_model $m
done

logit_v2 Qwen3-32B  # logit only: no exp03 rollouts to relabel

for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B; do
    step "$m download"
    download $m
    logit_v2 $m
    delete_weights $m
    judge_model $m
done

step "analysis"
$PY exp03_analysis.py --out-name final > $LOG/analysis_final.log 2>&1
step "done"
