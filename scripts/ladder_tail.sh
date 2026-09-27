#!/usr/bin/env bash
# Finishes exp02 with nobody present. Every step is attempted even if an earlier one failed, and every
# step skips outputs that already exist, so this also retries whatever ladder_chain.sh left undone.
#
#   setsid nohup scripts/ladder_tail.sh <pid of ladder_chain.sh> > log/exp02/tail.log 2>&1 &
#
# 1. wait for ladder_chain.sh; if it is still running at T+20, write an interim analysis + report
# 2. retry: generation of any subject not finished (weights still on disk), judging of any generated
#    model without labels, the two validation runs
# 3. analysis + report of the validated main ladder (CPU, in the background while the GPU runs step 4)
# 4. Qwen3-32B add-on, no time cutoff: logit monitor and stage 1 in one engine load, then its judging;
#    then delete the Qwen3-32B and Qwen3.8-27B weights (the latter are the judge until here)
# 5. final analysis (+ labeling sheet 2), the auto-drafted report, git commit (no push)
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
LOG_DIR=log/exp02
MAIN_PY=/venv/main/bin/python
VLLM_PY=scripts/vllm_python.sh
STATUS=("$MAIN_PY" scripts/ladder_status.py)
QWEN3=(Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B)
QWEN35=(Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B)
QWEN38=Qwen3.8-27B
ADDON=Qwen3-32B
T0_EPOCH=$(date -u -d "2026-09-26T20:30:00Z" +%s)
INTERIM_AT_EPOCH=$((T0_EPOCH + 20 * 3600))
export PYTHONDONTWRITEBYTECODE=1
# ladder_run.sh passes this on to ladder_generate.py: the stage-2 size in force (ladder_config.STAGE2_EXTRA_USED)
export STAGE2_EXTRA="$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.STAGE2_EXTRA_USED)")"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

step() {  # never stops the script
  local name="$1"; shift
  log "start ${name}"
  if "$@" > "${LOG_DIR}/${name}.log" 2>&1; then log "done  ${name}"; else log "FAILED ${name}; see ${LOG_DIR}/${name}.log"; fi
}

# --- 1. wait for the chain ------------------------------------------------------------------------
chain_pid="$1"
log "waiting for ladder_chain.sh (pid ${chain_pid})"
interim_done=0
[[ -e results/exp02_vea_ladder/interim_T20/summary.json ]] && interim_done=1
while kill -0 "$chain_pid" 2>/dev/null; do
  if ((interim_done == 0 && $(date +%s) >= INTERIM_AT_EPOCH)); then
    log "T+20 reached with the chain still running: interim analysis of the labeled models"
    step tail_analysis_interim "$MAIN_PY" scripts/ladder_analysis.py --out-name interim_T20
    step tail_report_interim "$MAIN_PY" scripts/ladder_report.py --out-name interim_T20
    interim_done=1
  fi
  sleep 60
done
if grep -q "ladder_chain complete" "${LOG_DIR}/chain.log"; then
  log "chain completed"
else
  log "chain did not complete; retrying what is missing"
fi

# --- 2. retry anything unfinished -----------------------------------------------------------------
for model in "${QWEN3[@]}" "${QWEN35[@]}" "$QWEN38"; do
  if ! "${STATUS[@]}" generated "$model"; then
    if "${STATUS[@]}" weights "$model"; then
      keep=0; [[ "$model" == "$QWEN38" ]] && keep=1
      # ladder_run.sh reuses the log name; keep the failed attempt's log
      [[ -f "${LOG_DIR}/generate_${model}.log" ]] && mv "${LOG_DIR}/generate_${model}.log" "${LOG_DIR}/generate_${model}.attempt1.log"
      step "tail_generate_${model}" env KEEP_WEIGHTS="$keep" scripts/ladder_run.sh generate "$model"
    else
      log "SKIP ${model}: generation incomplete and weights missing"
    fi
  fi
done

generated=()
for model in "${QWEN3[@]}" "${QWEN35[@]}" "$QWEN38"; do
  "${STATUS[@]}" generated "$model" && generated+=("$model")
done
unlabeled=()
for model in "${generated[@]}"; do "${STATUS[@]}" labeled "$model" || unlabeled+=("$model"); done
if ((${#unlabeled[@]})); then
  step tail_judge_retry "$VLLM_PY" scripts/ladder_judge.py label --models "${unlabeled[@]}"
fi

labeled=()
for model in "${generated[@]}"; do "${STATUS[@]}" labeled "$model" && labeled+=("$model"); done
if ((${#labeled[@]})); then
  step tail_validate_thinking "$VLLM_PY" scripts/ladder_judge.py validate --engine thinking --models "${labeled[@]}"
  step tail_validate_fp8 "$VLLM_PY" scripts/ladder_judge.py validate --engine fp8 --models "${labeled[@]}"
fi

# --- 3. validated main-ladder results, on CPU while the GPU runs the add-on -----------------------
validated_pid=""
if ! "${STATUS[@]}" labeled "$ADDON"; then
  ( step tail_analysis_validated "$MAIN_PY" scripts/ladder_analysis.py --out-name all_models_validated
    step tail_report_validated "$MAIN_PY" scripts/ladder_report.py --out-name all_models_validated ) &
  validated_pid=$!
fi

# --- 4. Qwen3-32B add-on --------------------------------------------------------------------------
rev="$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.ADDON_SUBJECTS['${ADDON}']['revision'])")"
addon_generated() { "${STATUS[@]}" stage1 "$ADDON" && "${STATUS[@]}" logit "$ADDON"; }
if ! addon_generated && ! "${STATUS[@]}" weights "$ADDON"; then
  step tail_download_qwen3_32b /venv/main/bin/hf download "Qwen/${ADDON}" --revision "$rev"
fi
if ! addon_generated; then
  if "${STATUS[@]}" weights "$ADDON"; then
    # One engine load: logit monitor, then stage 1 (10 rollouts per question + benign twins x2); no stage 2.
    step tail_qwen3_32b_stage1 env STAGE1_ONLY=1 scripts/ladder_run.sh generate "$ADDON"
  else
    log "SKIP Qwen3-32B: weights missing after the download attempt"
  fi
fi
if "${STATUS[@]}" stage1 "$ADDON" && ! "${STATUS[@]}" labeled "$ADDON"; then
  step tail_judge_qwen3_32b "$VLLM_PY" scripts/ladder_judge.py label --models "$ADDON"
fi
[[ -n "$validated_pid" ]] && wait "$validated_pid"
for dir in "$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.hf_repo_dir(c.ADDON_SUBJECTS['${ADDON}']['repo']))")" \
           "$("$MAIN_PY" -c "import sys; sys.path.insert(0, 'scripts'); import ladder_config as c; print(c.hf_repo_dir(c.SUBJECTS['${QWEN38}']['repo']))")"; do
  [[ -d "$dir" ]] && { log "deleting weights ${dir}"; rm -rf "$dir"; }
done

# --- 5. final analysis, report, commit ------------------------------------------------------------
step tail_analysis_final "$MAIN_PY" scripts/ladder_analysis.py --out-name final --labeling-sheet 2
step tail_report_final "$MAIN_PY" scripts/ladder_report.py
git add -A results/exp02_vea_ladder figures
git commit -q -m "exp02_vea_ladder: final analysis, figures and auto-drafted report (UNVERIFIED)" \
  && log "committed $(git rev-parse --short HEAD)" || log "nothing to commit"
log "ladder_tail complete"
