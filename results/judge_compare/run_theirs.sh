#!/bin/bash
cd /home/jtr2875/eval_awareness_scaling
export HF_HOME=/home/jtr2875/.cache/huggingface HF_HUB_CACHE=/home/jtr2875/.cache/huggingface/hub VLLM_USE_FLASHINFER_SAMPLER=0
D=results/judge_compare

CUDA_VISIBLE_DEVICES=1 uv run python $D/run_theirs.py $D/heldout_set.jsonl $D/theirs.heldout.jsonl --reps 3
