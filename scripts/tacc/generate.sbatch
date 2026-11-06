#!/bin/bash
#SBATCH -J quant-gen
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o quant-gen-%j.out
#SBATCH --signal=B:TERM@180   # 3 min before the limit, so the EXIT trap stages data out
# Generate rollouts for VARIANTS (separated by spaces or +), TP GPUs each, as many at once as the node's GPUs allow.
#   tacc submit scripts/tacc/generate.sbatch --cd $SCRATCH --env VARIANTS=Qwen3.5-4B@bf16+Qwen3.5-4B@gptq-w4
#   GEN_ARGS=--smoke for the 20-question smoke test; TP=2 for bf16 27B/32B on A100-40GB.
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
TP=${TP:-1}
SLOTS=$(( $(nvidia-smi -L | wc -l) / TP ))
read -ra ALL <<< "${VARIANTS//[,+]/ }"
fail=0
for ((start = 0; start < ${#ALL[@]}; start += SLOTS)); do
  pids=(); names=()
  for ((slot = 0; slot < SLOTS && start + slot < ${#ALL[@]}; slot++)); do
    v=${ALL[$((start + slot))]}
    devs=$(seq -s, $((slot * TP)) $((slot * TP + TP - 1)))
    echo "$v on GPUs $devs"
    CUDA_VISIBLE_DEVICES=$devs uv run --no-sync python -m eval_awareness.quant.generate --variant "$v" --tp "$TP" \
      ${GEN_ARGS:-} > "$QUANT_DATA/logs/gen_${v}${GEN_ARGS:+.smoke}.log" 2>&1 &
    pids+=($!); names+=("$v")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "FAILED ${names[$i]}"; fail=1; }; done
done
[[ $fail -eq 0 ]] && echo "generation done"
exit $fail
