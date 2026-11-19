# Generation helpers for the quantization-study jobs (sourced after common.sh by generate.sbatch and model.sbatch).

download_for() {  # download the pinned weights the given variants need (compute nodes have network access)
  echo "downloading for $* ($(date +%H:%M))"
  HF_HUB_OFFLINE=0 uv run --no-sync python -m eval_awareness.quant.download --variants "$@" \
    > "$QUANT_DATA/logs/download_${SLURM_JOB_ID}.log" 2>&1 && return 0
  echo "FAILED download (see download_${SLURM_JOB_ID}.log)"; tail -5 "$QUANT_DATA/logs/download_${SLURM_JOB_ID}.log"
  return 1
}

quantize_if_missing() {  # $1 variant; our recipes only; one retry (a transient $SCRATCH read error broke job 3559831)
  local v=$1 try
  case ${v#*@} in gptq-*|rtn-*) ;; *) return 0 ;; esac
  [[ -f "$QUANT_CHECKPOINTS/$v/recipe_provenance.json" ]] && return 0  # written last by quantize.py
  for try in 1 2; do
    echo "quantizing $v ($(date +%H:%M))"
    # one at a time per node: each run holds the bf16 model in CPU RAM (64 GB for 32B; nodes have 256 GB)
    flock "/tmp/quantize.${SLURM_JOB_ID:-0}.lock" env UV_PROJECT_ENVIRONMENT="$SCRATCH/venvs/quantize" uv run --no-sync \
      --project envs/quantize python src/eval_awareness/quant/quantize.py --variant "$v" > "$QUANT_DATA/logs/quantize_$v.log" 2>&1 && return 0
    echo "quantize attempt $try for $v failed (see quantize_$v.log)"; sleep 30
  done
  return 1
}

rollouts_complete() {  # $1 variant: every adversarial (489 x 10) and benign (489 x 2) rollout written
  local d=$QUANT_DATA/rollouts
  [[ -f "$d/$1.adversarial.jsonl" && -f "$d/$1.benign.jsonl" ]] || return 1
  (( $(wc -l < "$d/$1.adversarial.jsonl") >= 4890 && $(wc -l < "$d/$1.benign.jsonl") >= 978 ))
}

generate_all() {  # generate_all TP PP VARIANT...: the node's GPUs form slots of TP*PP; each slot takes the next
  # variant from a shared queue (list the longest first), makes our checkpoint if missing, then generates.
  local tp=$1 pp=$2; shift 2
  (( $# )) || return 0
  if (( tp * pp == 1 )); then  # 27B/32B do not fit one A100-40GB: generate them on the whole node first
    # (Qwen3.5-27B W4 is 24.5 GB with linear attention kept in bf16; vLLM OOMed in CUDA-graph profiling, job 3591734)
    local big=() rest=() v
    for v in "$@"; do case $v in Qwen3.5-27B@*|Qwen3-32B@*) big+=("$v") ;; *) rest+=("$v") ;; esac; done
    if (( ${#big[@]} )); then
      local bigfail=0
      generate_all 1 "$(nvidia-smi -L | wc -l)" "${big[@]}" || bigfail=1
      generate_all 1 1 "${rest[@]}" || bigfail=1
      return $bigfail
    fi
  fi
  local per=$((tp * pp)) slots queue p fail=0 pids=() slot
  slots=$(( $(nvidia-smi -L | wc -l) / per ))
  queue=$(mktemp -p "$SCRATCH"); printf '%s\n' "$@" > "$queue"
  _gen_worker() {  # $1 slot
    local devs v bad=0 log
    devs=$(seq -s, $(($1 * per)) $(($1 * per + per - 1)))
    export CUDA_VISIBLE_DEVICES=$devs
    while v=$(flock "$queue.lock" bash -c 'head -1 "$1"; sed -i 1d "$1"' _ "$queue") && [[ -n $v ]]; do
      echo "$v on GPUs $devs ($(date +%H:%M))"
      quantize_if_missing "$v" || { echo "FAILED quantize $v"; bad=1; continue; }
      log=$QUANT_DATA/logs/gen_${v}${GEN_ARGS:+.smoke}.log
      if uv run --no-sync python -m eval_awareness.quant.generate --variant "$v" --tp "$tp" --pp "$pp" ${GEN_ARGS:-} >> "$log" 2>&1; then
        echo "done $v ($(date +%H:%M)): $(grep -h 'tok/s' "$log" | tail -1)"
      else
        echo "FAILED generate $v (see gen_$v.log)"; bad=1
      fi
    done
    return $bad
  }
  for ((slot = 0; slot < slots; slot++)); do _gen_worker "$slot" & pids+=($!); sleep 5; done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  rm -f "$queue" "$queue.lock"
  return $fail
}
