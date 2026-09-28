#!/bin/bash
# Refusal-judge comparison: the collaborator's judge on GPU 1, mine on GPU 3, both input views.
cd /home/jtr2875/eval_awareness_scaling
export HF_HOME=/home/jtr2875/.cache/huggingface HF_HUB_CACHE=/home/jtr2875/.cache/huggingface/hub VLLM_USE_FLASHINFER_SAMPLER=0
D=results/judge_compare/refusal
CUDA_VISIBLE_DEVICES=1 uv run python $D/run_theirs.py --reps 3 > logs/refusal_compare_theirs.log 2>&1 &
( for v in cut384 full; do
  CUDA_VISIBLE_DEVICES=3 uv run python -m eval_awareness.judge $D/set.$v.jsonl --task refusal --backend vllm \
    --judge-model Qwen/Qwen3.8-27B --tp 1 --text-only --max-model-len 40960 --max-num-seqs 256 \
    --gpu-memory-utilization 0.8 --judge-max-tokens 1024 --out-dir $D/mine || exit 1
done ) > logs/refusal_compare_mine.log 2>&1 &
wait
