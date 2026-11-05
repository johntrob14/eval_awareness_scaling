#!/bin/bash
#SBATCH -J quant-make
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o quant-make-%j.out
# Make our quantized checkpoints for comma-separated VARIANTS (recipes in quant/config.py), one after another.
#   tacc submit scripts/tacc/quantize.sbatch --cd $RESULTS/logs -- --export=ALL,VARIANTS=Qwen3.5-4B@gptq-w4
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
for v in ${VARIANTS//,/ }; do
  echo "quantizing $v"
  uv run --no-sync --project envs/quantize python src/eval_awareness/quant/quantize.py --variant "$v" \
    > "$QUANT_DATA/logs/quantize_$v.log" 2>&1 || { echo "FAILED $v"; exit 1; }
done
echo "quantization done"
