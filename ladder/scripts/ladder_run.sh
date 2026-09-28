#!/usr/bin/env bash
# exp02 runner. Each stage logs to log/exp02/<stage>.log; the first failure stops everything.
#
#   scripts/ladder_run.sh smoke                                  # smoke test (cache only)
#   scripts/ladder_run.sh generate Qwen3-0.6B Qwen3-1.7B ...     # main run, needs the manifest
#   STAGE2_EXTRA=20 scripts/ladder_run.sh generate ...           # pre-agreed fallback
#   STAGE1_ONLY=1 scripts/ladder_run.sh generate Qwen3-32B       # add-on
#   LOGIT_ONLY=1 scripts/ladder_run.sh generate Qwen3-32B        # add-on, logit monitor first
#   KEEP_WEIGHTS=1 scripts/ladder_run.sh generate Qwen3.8-27B    # weights reused by the bf16 judge check
#
# Weights of a model are deleted after its main-run generation succeeds (disk), unless KEEP_WEIGHTS=1.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_PY=/venv/main/bin/python
VLLM_PY="${REPO_DIR}/scripts/vllm_python.sh"
LOG_DIR="${REPO_DIR}/log/exp02"
MANIFEST="${REPO_DIR}/results/exp02_vea_ladder/manifest.json"
export PYTHONDONTWRITEBYTECODE=1
cd "${REPO_DIR}/scripts"
mkdir -p "$LOG_DIR"

stage() {
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] start ${name}"
  "$@" > "${LOG_DIR}/${name}.log" 2>&1 || { echo "FAILED ${name}; see ${LOG_DIR}/${name}.log" >&2; exit 1; }
  echo "[$(date -u +%H:%M:%S)] done  ${name}"
}

weights_dir() {
  "$MAIN_PY" -c "import ladder_config as c; m = c.ALL_SUBJECTS['$1']; print(c.hf_repo_dir(m['repo']))"
}

mode="$1"; shift
case "$mode" in
  smoke)
    for model in $("$MAIN_PY" -c "import ladder_config as c; print(' '.join(c.SMOKE_SUBJECTS))"); do
      stage "smoke_${model}" "$VLLM_PY" ladder_generate.py --model "$model" --smoke
    done
    ;;
  generate)
    test -f "$MANIFEST" || { echo "manifest.json missing" >&2; exit 1; }
    extra_args=(--stage2-extra "${STAGE2_EXTRA:-40}")
    suffix=""
    [[ "${STAGE1_ONLY:-0}" == 1 ]] && extra_args+=(--stage1-only)
    # Logit-only runs keep the weights: stage 1 may still follow.
    [[ "${LOGIT_ONLY:-0}" == 1 ]] && { extra_args+=(--logit-only); suffix="_logit_only"; KEEP_WEIGHTS=1; }
    for model in "$@"; do
      stage "generate_${model}${suffix}" "$VLLM_PY" ladder_generate.py --model "$model" "${extra_args[@]}"
      if [[ "${KEEP_WEIGHTS:-0}" != 1 ]]; then
        dir="$(weights_dir "$model")"
        echo "[$(date -u +%H:%M:%S)] deleting weights ${dir}"
        rm -rf "$dir"
      fi
    done
    ;;
  *)
    echo "unknown mode ${mode}" >&2; exit 1
    ;;
esac
echo "[$(date -u +%H:%M:%S)] ladder_run ${mode} complete"
