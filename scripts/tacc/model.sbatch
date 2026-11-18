#!/bin/bash
#SBATCH -J quant-model
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 36:00:00
#SBATCH -o quant-model-%j.out
#SBATCH --signal=B:TERM@180   # 3 min before the limit, so the EXIT trap stages data out
# One model, one node, one queue wait: generate its variants, then judge them on the same node.
#   GEN_PP3   variants generated first, one at a time on all three GPUs (bf16 14B)
#   GEN       variants generated next, one per GPU from a shared queue (missing checkpoints are made first)
#   EXTERNAL  variants another job generates (bf16/W8/hf-w4 27B and 32B): judged last, once their rollouts are
#             complete; waits at most WAIT_MAX_MIN (default 120) so the node does not idle long, then leaves them
#             for a judge-only job.
# Judging: Qwen3.8-27B, one engine on all three GPUs (PP=3) for GEN_PP3 + GEN, then another for EXTERNAL.
#   tacc submit scripts/tacc/model.sbatch --cd $SCRATCH --env GEN=a+b+c -- -t 30:00:00 -J quant-m2B
git -C "$WORK/eval_awareness_scaling" clean -fq -- results/quant  # untracked analysis outputs would block the pull
git -C "$WORK/eval_awareness_scaling" pull -q --ff-only origin quantization  # this job body is fixed at submit; the checkout is not
. "$WORK/eval_awareness_scaling/scripts/tacc/common.sh"
. scripts/tacc/gen_lib.sh
split() { local s=${1:-}; echo "${s//[,+]/ }"; }
read -ra BIG <<< "$(split "${GEN_PP3:-}")"
read -ra SMALL <<< "$(split "${GEN:-}")"
read -ra EXT <<< "$(split "${EXTERNAL:-}")"
OWN=("${BIG[@]}" "${SMALL[@]}")
fail=0

if (( ${#OWN[@]} )); then
  download_for "${OWN[@]}" || exit 1
  generate_all 1 3 "${BIG[@]}" || fail=1
  generate_all 1 1 "${SMALL[@]}" || fail=1
  echo "generation finished ($(date +%H:%M)), status $fail"
  ready=(); for v in "${OWN[@]}"; do rollouts_complete "$v" && ready+=("$v") || { echo "incomplete rollouts: $v"; fail=1; }; done
  if (( ${#ready[@]} )); then
    VARIANTS="${ready[*]}" TP=1 PP=3 GPUS=0,1,2 scripts/quant_judge.sh || fail=1
  fi
fi

if (( ${#EXT[@]} )); then
  waited=0
  while :; do
    pending=(); for v in "${EXT[@]}"; do rollouts_complete "$v" || pending+=("$v"); done
    (( ${#pending[@]} == 0 || waited >= ${WAIT_MAX_MIN:-120} )) && break
    echo "waiting for rollouts of ${pending[*]} ($(date +%H:%M))"; sleep 600; waited=$((waited + 10))
  done
  ready=(); for v in "${EXT[@]}"; do rollouts_complete "$v" && ready+=("$v"); done
  (( ${#pending[@]} )) && { echo "NOT JUDGED (rollouts incomplete): ${pending[*]}"; fail=1; }
  if (( ${#ready[@]} )); then
    VARIANTS="${ready[*]}" TP=1 PP=3 GPUS=0,1,2 scripts/quant_judge.sh || fail=1
  fi
fi
(( fail == 0 )) && echo "model done"
exit $fail
