#!/bin/bash
#SBATCH -J quant-gen
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o quant-gen-%j.out
#SBATCH --signal=B:TERM@180   # 3 min before the limit, so the EXIT trap stages data out
# Generate rollouts for VARIANTS (separated by spaces or +), TP*PP GPUs each. The node's GPUs form slots; each slot
# takes the next variant from a shared queue (so long and short variants balance), makes our checkpoint first if it
# is missing (quant/config.py recipes), then generates. Weights are downloaded once at the start.
#   tacc submit scripts/tacc/generate.sbatch --cd $SCRATCH --env VARIANTS=Qwen3.5-4B@bf16+Qwen3.5-4B@gptq-w4
#   PP=3 for bf16 27B/32B (and 14B) on a 3x A100-40GB node; GEN_ARGS=--smoke for the 20-question smoke test.
#   List the longest variants first: the queue is taken in order.
git -C "$WORK/eval_awareness_scaling" clean -fq -- results/quant  # untracked analysis outputs would block the pull
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
TP=${TP:-1}; PP=${PP:-1}; PER=$((TP * PP))
SLOTS=$(( $(nvidia-smi -L | wc -l) / PER ))
read -ra ALL <<< "${VARIANTS//[,+]/ }"

echo "downloading for ${ALL[*]} ($(date +%H:%M))"
HF_HUB_OFFLINE=0 uv run --no-sync python -m eval_awareness.quant.download --variants "${ALL[@]}" \
  > "$QUANT_DATA/logs/download_${SLURM_JOB_ID}.log" 2>&1 || { echo "FAILED download (see download_${SLURM_JOB_ID}.log)"; tail -5 "$QUANT_DATA/logs/download_${SLURM_JOB_ID}.log"; exit 1; }

QUEUE=$(mktemp -p "$SCRATCH"); printf '%s\n' "${ALL[@]}" > "$QUEUE"
next_variant() {  # pop the first line of the queue under a lock
  flock "$QUEUE.lock" bash -c 'head -1 "$1"; sed -i 1d "$1"' _ "$QUEUE"
}
quantize_if_missing() {  # $1 variant; our recipes only; one retry (a transient $SCRATCH read error broke job 3559831)
  local v=$1
  case ${v#*@} in gptq-*|rtn-*) ;; *) return 0 ;; esac
  [[ -f "$QUANT_CHECKPOINTS/$v/recipe_provenance.json" ]] && return 0  # written last by quantize.py
  for try in 1 2; do
    echo "quantizing $v ($(date +%H:%M))"
    UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/quantize uv run --no-sync --project envs/quantize \
      python src/eval_awareness/quant/quantize.py --variant "$v" > "$QUANT_DATA/logs/quantize_$v.log" 2>&1 && return 0
    echo "quantize attempt $try for $v failed (see quantize_$v.log)"; sleep 30
  done
  return 1
}
worker() {  # $1 slot
  local devs fail=0 v
  devs=$(seq -s, $(($1 * PER)) $(($1 * PER + PER - 1)))
  export CUDA_VISIBLE_DEVICES=$devs
  while v=$(next_variant) && [[ -n $v ]]; do
    echo "$v on GPUs $devs ($(date +%H:%M))"
    if ! quantize_if_missing "$v"; then echo "FAILED quantize $v"; fail=1; continue; fi
    if uv run --no-sync python -m eval_awareness.quant.generate --variant "$v" --tp "$TP" --pp "$PP" ${GEN_ARGS:-} \
         >> "$QUANT_DATA/logs/gen_${v}${GEN_ARGS:+.smoke}.log" 2>&1; then
      echo "done $v ($(date +%H:%M)): $(grep -h 'tok/s' "$QUANT_DATA/logs/gen_${v}${GEN_ARGS:+.smoke}.log" | tail -1)"
    else
      echo "FAILED generate $v (see gen_$v.log)"; fail=1
    fi
  done
  return $fail
}
pids=(); fail=0
for ((slot = 0; slot < SLOTS; slot++)); do worker "$slot" & pids+=($!); sleep 5; done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
rm -f "$QUEUE" "$QUEUE.lock"
[[ $fail -eq 0 ]] && echo "generation done"
exit $fail
