#!/bin/bash
#SBATCH -J quant-gen
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o quant-gen-%j.out
#SBATCH --signal=B:TERM@180   # 3 min before the limit, so the EXIT trap stages data out
# Generate rollouts for VARIANTS (separated by spaces or +), TP*PP GPUs each (gen_lib.sh: a shared queue over the
# node's GPU slots; missing checkpoints of our recipes are made first). List the longest variants first.
#   tacc submit scripts/tacc/generate.sbatch --cd $SCRATCH --env VARIANTS=Qwen3.5-4B@bf16+Qwen3.5-4B@gptq-w4
#   PP=3 for bf16 27B/32B (and 14B) on a 3x A100-40GB node; GEN_ARGS=--smoke for the 20-question smoke test.
git -C "$WORK/eval_awareness_scaling" clean -fq -- results/quant  # untracked analysis outputs would block the pull
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
. scripts/tacc/gen_lib.sh
read -ra ALL <<< "${VARIANTS//[,+]/ }"
download_for "${ALL[@]}" || exit 1
generate_all "${TP:-1}" "${PP:-1}" "${ALL[@]}" && { echo "generation done"; exit 0; }
exit 1
