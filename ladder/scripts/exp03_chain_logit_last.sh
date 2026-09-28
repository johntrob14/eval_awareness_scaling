#!/bin/bash
# exp03, second part of the run (after exp03_full_chain.sh and the Qwen3.8-27B relabel; see README.md): the inputs of
# F1-F5 for every model first, the logit monitor (F6) last. One model at a time:
#   1. Qwen3.5-27B, Qwen3.5-9B, Qwen3-14B: interventions, then the judge (relabel + intervention answers)
#   2. the 7 small models: download, interventions, delete weights, judge
#   3. Qwen3-32B: stage 1 + stage 2 (exp02's code and design), relabel, interventions (bases from the relabel), judge
#   4. logit v2 for every model without it (the 27B/14B/32B weights are kept for this), then the final analysis
# Every step skips work whose outputs exist. Start it with exp03_launch.sh, which also starts the live watcher.
set -euo pipefail
cd "$(dirname "$0")"
VLLM=./vllm_python.sh
PY=/venv/main/bin/python
LOG=../log/exp03
SMALL_MODELS="Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B"
step() { echo "== $* $(date -u +%FT%TZ)"; }
repo_dir() { $PY -c "import ladder_config as b; s=b.ALL_SUBJECTS['$1']; print(b.hf_repo_dir(s['repo']))"; }
download() { step "$1 download"; $PY -c "import ladder_config as b; s=b.ALL_SUBJECTS['$1']; print(s['repo'], s['revision'])" | \
             { read -r repo rev; /venv/main/bin/hf download "$repo" --revision "$rev" > /dev/null; }; }
delete_weights() { local d; d=$(repo_dir "$1"); step "delete $d"; rm -rf "$d"; }
interventions() { step "$1 interventions"; $VLLM exp03_generate.py interventions --models $1 >> $LOG/interventions_$1.log 2>&1; }
judge_model() { step "$1 judge: relabel + intervention answers"; $VLLM exp03_judge.py model --models $1 >> $LOG/judge_$1.log 2>&1; }
logit_v2() { step "$1 logit v2"; $VLLM exp03_logit.py run --model $1 --kinds format disguise >> $LOG/logit_v2_$1.log 2>&1; }

step "chain start, logit monitor last"
# One GPU job at a time: wait for any vLLM job that is still running.
while pgrep -f "^/workspace/.venv-vllm/bin/python|^VLLM::EngineCore" > /dev/null; do sleep 30; done

for m in Qwen3.5-27B Qwen3.5-9B Qwen3-14B; do
    interventions $m
    judge_model $m
done

for m in $SMALL_MODELS; do
    download $m
    interventions $m
    delete_weights $m
    judge_model $m
done

step "Qwen3-32B stage 1 + stage 2"
$VLLM exp03_generate.py ladder --models Qwen3-32B >> $LOG/ladder_Qwen3-32B.log 2>&1
step "Qwen3-32B judge: relabel"
$VLLM exp03_judge.py label --models Qwen3-32B >> $LOG/judge_Qwen3-32B.log 2>&1
interventions Qwen3-32B
step "Qwen3-32B judge: intervention answers"
$VLLM exp03_judge.py interventions --models Qwen3-32B >> $LOG/judge_Qwen3-32B.log 2>&1

for m in Qwen3.5-27B Qwen3-14B Qwen3-32B; do
    logit_v2 $m
    delete_weights $m
done
for m in $SMALL_MODELS; do
    download $m
    logit_v2 $m
    delete_weights $m
done

step "final analysis (all models)"
$PY exp03_analysis.py --out-name final > $LOG/analysis_final.log 2>&1
step "done"
