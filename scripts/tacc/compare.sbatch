#!/bin/bash
#SBATCH -J quant-compare
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o quant-compare-%j.out
#SBATCH --signal=B:TERM@180   # 3 min before the limit, so the EXIT trap stages data out
# Weight-level comparison of quantized checkpoints (CPU only). BASE and VARIANTS (separated by spaces or +).
#   tacc submit scripts/tacc/compare.sbatch --cd $SCRATCH --env BASE=Qwen3.5-4B --env VARIANTS=gptq-w4+hf-w4+gptq-w4-calB+rtn-w4
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
read -ra ALL <<< "${VARIANTS//[,+]/ }"
mkdir -p "$QUANT_DATA/compare"
uv run --no-sync python -m eval_awareness.quant.compare_weights --base "$BASE" --variants "${ALL[@]}" \
  --out "$QUANT_DATA/compare/${BASE}_weights.json" 2>&1 | tee "$QUANT_DATA/logs/compare_${SLURM_JOB_ID}.log"
