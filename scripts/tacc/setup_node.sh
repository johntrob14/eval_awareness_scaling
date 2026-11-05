#!/bin/bash
# Set up the quantization study on a Lonestar6 compute node (never the login node). Idempotent.
#   tacc exec --write 'cd $WORK/eval_awareness_scaling && nohup scripts/tacc/setup_node.sh > $WORK/results/eval_awareness_scaling/logs/setup.log 2>&1 &'
# MODELS: extra "repo@revision" downloads beyond the pilot set (space-separated).
set -euo pipefail
cd "$WORK/eval_awareness_scaling"
grep -q QUANT_DATA .sauce/env.sh || cat >> .sauce/env.sh <<ENV
export QUANT_DATA=$WORK/results/eval_awareness_scaling/data
export QUANT_CHECKPOINTS=$WORK/results/eval_awareness_scaling/checkpoints
ENV
. .sauce/env.sh
mkdir -p "$QUANT_DATA/logs" "$QUANT_CHECKPOINTS"
echo "== node $(hostname), $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
for u in https://huggingface.co https://pypi.org/simple/ https://github.com; do
  printf '%s ' "$u"; curl -s -o /dev/null -m 15 -w '%{http_code}\n' "$u" || echo unreachable
done

echo "== code"
git fetch -q origin quantization && git checkout -q quantization && git merge -q --ff-only origin/quantization
git log --oneline -1

echo "== main venv (vLLM)"
uv sync 2>&1 | tail -3
uv run --no-sync python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'vllm', vllm.__version__)"

echo "== quantize venv (llm-compressor)"
uv sync --project envs/quantize 2>&1 | tail -3
uv run --no-sync --project envs/quantize python -c "import llmcompressor, transformers, torch; print('llmcompressor', llmcompressor.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, torch.cuda.is_available())"

echo "== downloads"
PILOT="Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
RedHatAI/Qwen3.5-4B-quantized.w4a16@7a613872f394578b0b52b683ff4ac47516b4bcaf
Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
for spec in $PILOT ${MODELS:-}; do
  echo "-- $spec"
  uv run --no-sync hf download "${spec%@*}" --revision "${spec#*@}" --quiet > /dev/null
done
uv run --no-sync --project envs/quantize python -c "
from datasets import load_dataset
ds = load_dataset('garage-bAInd/Open-Platypus', revision='37141edbdb7826378cce118c46a109b813e1f038', split='train')
print('Open-Platypus rows', len(ds))"

echo "== prompts"
uv run --no-sync python -m eval_awareness.quant.prepare
du -sh "$HF_HUB_CACHE"
echo "== setup done $(date -Is)"
