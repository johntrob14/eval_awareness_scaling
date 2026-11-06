#!/bin/bash
#SBATCH -J quant-judge
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o quant-judge-%j.out
# Judge VARIANTS (separated by +; commas break sbatch --export) with Qwen3.8-27B bf16 (54 GB).
#   A100-40GB: one engine, TP=2 (default).
#   H100-80GB (-p gpu-h100, 2 GPUs): TP=1 runs one engine per GPU, variants split round-robin between them.
#   tacc submit scripts/tacc/judge.sbatch --cd $RESULTS/logs -- --export=ALL,VARIANTS=Qwen3.5-4B@bf16
#   tacc submit scripts/tacc/judge.sbatch --cd $RESULTS/logs -- -p gpu-h100 --export=ALL,TP=1,VARIANTS=a+b
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
TP=${TP:-2}
read -ra ALL <<< "${VARIANTS//[,+]/ }"
SLOTS=$(( $(nvidia-smi -L | wc -l) / TP ))
pids=(); fail=0
for ((slot = 0; slot < SLOTS; slot++)); do
  mine=(); for ((i = slot; i < ${#ALL[@]}; i += SLOTS)); do mine+=("${ALL[$i]}"); done
  (( ${#mine[@]} )) || continue
  devs=$(seq -s, $((slot * TP)) $((slot * TP + TP - 1)))
  echo "GPUs $devs: ${mine[*]}"
  VARIANTS="${mine[*]}" TP=$TP GPUS=$devs scripts/quant_judge.sh &
  pids+=($!)
  sleep 5  # distinct log timestamps
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail -eq 0 ]] && echo "judging done"
exit $fail
