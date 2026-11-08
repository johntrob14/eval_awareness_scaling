#!/bin/bash
#SBATCH -J quant-make
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o quant-make-%j.out
#SBATCH --signal=B:TERM@180   # 3 min before the limit, so the EXIT trap stages data out
# Download the pinned weights for VARIANTS (separated by spaces or +), then make our quantized
# checkpoints for those whose condition is one of our recipes (quant/config.py), one after another.
# DOWNLOAD_ONLY=1 skips quantization (e.g. bf16 and hf-w4 variants).
#   tacc submit scripts/tacc/quantize.sbatch --cd $SCRATCH --env VARIANTS=Qwen3.5-9B@gptq-w4+Qwen3.5-9B@hf-w4
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
read -ra ALL <<< "${VARIANTS//[,+]/ }"
echo "downloading for ${ALL[*]}"
HF_HUB_OFFLINE=0 uv run --no-sync python -m eval_awareness.quant.download --variants "${ALL[@]}" \
  > "$QUANT_DATA/logs/download_${SLURM_JOB_ID}.log" 2>&1 || { echo "FAILED download (see download_${SLURM_JOB_ID}.log)"; tail -5 "$QUANT_DATA/logs/download_${SLURM_JOB_ID}.log"; exit 1; }
[[ ${DOWNLOAD_ONLY:-0} == 1 ]] && { echo "download done"; exit 0; }
failed=()
for v in "${ALL[@]}"; do
  case ${v#*@} in gptq-*|rtn-*) ;; *) continue ;; esac
  echo "quantizing $v ($(date +%H:%M))"
  # one retry: a transient $SCRATCH read error once broke `import numpy` (job 3559831, 2026-11-07)
  for try in 1 2; do
    UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/quantize uv run --no-sync --project envs/quantize python src/eval_awareness/quant/quantize.py --variant "$v" \
      > "$QUANT_DATA/logs/quantize_$v.log" 2>&1 && break
    echo "attempt $try for $v failed (see quantize_$v.log)"; (( try == 2 )) && failed+=("$v"); sleep 30
  done
done
(( ${#failed[@]} )) && { echo "quantization failed for: ${failed[*]}"; exit 1; }
echo "quantization done"
