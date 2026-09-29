#!/bin/bash
# Full unified chain after the ladder cache is in place (docs/UNIFIED_PROTOCOL.md):
#   1. judge the ladder's 12 models (vea_cal, refusal, category)            scripts/run_unified_ladder.sh judge
#   2. convert + judge the ladder's intervention continuations               ... iconvert ijudge (needs 1)
#   3. generate + judge the ladder-design interventions for my 4 models      scripts/run_unified_interventions.sh
#   4. analysis: --final when complete, otherwise a report marked PARTIAL
# A failed step 1 skips step 2 (it needs the VEA labels); step 3 is independent and runs regardless.
# Milestones go to logs/unified_v2_all.log.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/home/jtr2875/.cache/huggingface} HF_HUB_CACHE=${HF_HUB_CACHE:-/home/jtr2875/.cache/huggingface/hub}
L=logs/unified_v2_all.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
rc=0
say "step 1: judge ladder models"
if STEPS=judge scripts/run_unified_ladder.sh >> logs/unified_v2_step1.log 2>&1; then
  say "step 1 done"
  say "step 2: convert + judge ladder continuations"
  if STEPS="iconvert ijudge" scripts/run_unified_ladder.sh >> logs/unified_v2_step2.log 2>&1; then say "step 2 done"
  else say "step 2 FAILED (logs/unified_v2_step2.log)"; rc=1; fi
else
  say "step 1 FAILED (logs/unified_v2_step1.log); skipping step 2"; rc=1
fi
say "step 3: interventions for my models"
if STEPS="gen judge" scripts/run_unified_interventions.sh >> logs/unified_v2_step3.log 2>&1; then say "step 3 done"
else say "step 3 FAILED (logs/unified_v2_step3.log)"; rc=1; fi
say "step 4: analysis"
if uv run python -m eval_awareness.unified analyze --final >> logs/unified_v2_step4.log 2>&1 \
   && uv run python -m eval_awareness.intervene_unified analyze --final >> logs/unified_v2_step4.log 2>&1; then
  say "step 4 done: FINAL report in results/unified/v2/REPORT.md"
else
  uv run python -m eval_awareness.unified analyze >> logs/unified_v2_step4.log 2>&1
  uv run python -m eval_awareness.intervene_unified analyze >> logs/unified_v2_step4.log 2>&1
  say "step 4: not final (see logs/unified_v2_step4.log); PARTIAL report written"; rc=1
fi
say "chain finished (exit $rc)"
exit $rc
