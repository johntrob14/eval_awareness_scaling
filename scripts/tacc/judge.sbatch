#!/bin/bash
#SBATCH -J quant-judge
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o quant-judge-%j.out
# Judge VARIANTS separated by + (commas break sbatch --export) with Qwen3.8-27B bf16 (54 GB): TP=2 on two A100-40GB.
#   tacc submit scripts/tacc/judge.sbatch --cd $RESULTS/logs -- --export=ALL,VARIANTS=Qwen3.5-4B@bf16
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
VARIANTS=${VARIANTS//[,+]/ } TP=${TP:-2} GPUS=${GPUS:-0,1} scripts/quant_judge.sh
