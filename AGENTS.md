# AGENTS.md

Replication of the Qwen3 results in *Verbalized Eval Awareness Inflates Measured Safety*
(LessWrong, https://www.lesswrong.com/posts/mDriHK4beN5rq2tAA). Focus: Appendix A1 (eval
awareness vs. Qwen3 model size on Fortress) and the Section 2 correlation between verbalized
eval awareness (VEA) and refusal.

## Claims under test

From the post (Qwen3 family, Fortress, 100 rollouts per question, GPT-5-mini judge):

1. **Qwen3-0.6B verbalizes eval awareness at all.** On q140 (drug-trafficking "show the duality"
   role-play): 6% of rollouts for 0.6B vs 45% for 32B. Example 0.6B quote: "The user might be
   testing the AI's ability to show duality".
2. **VEA rate rises with model size**, 0.6B → 32B (Fig. A1.1; exact per-size numbers are only in the figure).
3. **The safety-relevant share of VEA rises with size**: under 10% of VEA instances at the small
   end, over 50% at the large end (Fig. A1.2). The "abilities" testing subcategory is by definition
   not safety-relevant.
4. For **Qwen3-32B**: questions with ≥1 VEA rollout (out of 500) refuse 21% vs 1% for those
   without (Fig. 5); Spearman(#VEA rollouts, refusal) ρ = 0.607, N = 190 (App. C6); within-question
   relative refusal increase 711% unweighted / 241% inverse-variance weighted (App. C4).

## Layout

- `src/eval_awareness/data.py`: loads Fortress (`ScaleAI/fortress_public`). `qid` = row position,
  which matches the post's numbering (q140 = ID 356). Excludes `EXPLICIT_EVAL_QIDS`.
- `generate.py`: samples rollouts. `--backend mlx` (Apple Silicon) or `vllm` (NVIDIA),
  auto-detected. Writes `data/rollouts/<model>.jsonl` with one row per rollout
  (`qid, sample_idx, cot, response, finish_reason, n_tokens, …`).
- `judge.py`: `--task vea|refusal|category`, `--backend openai|vllm|mlx`. Writes
  `data/judgments/<model>.<task>.<judge-slug>.jsonl`.
- `rubrics.py`: judge prompts. The VEA (App. F1) and refusal (App. F2) rubrics are **verbatim
  from the post**; do not edit them. The category/subcategory prompt and the way request and
  response are inserted into the refusal rubric are our reconstructions.
- `analyze.py`: VEA rate, pass@k, category shares, within-question refusal difference (weights
  n(N−n)), mixed-effects linear probability model, Fig. 5 split, Spearman. Writes `results/`.
- `scripts/run_scaling.sh`: the full GPU pipeline for all sizes. `scripts/mlx_generate_with_retry.sh`
  is for laptops only.
- `data/pilot/q140_Qwen3-0.6B.jsonl`: 100 Qwen3-0.6B rollouts on q140, generated with MLX on an M3.

## Running

```bash
uv sync                      # installs vllm on Linux, mlx-lm on macOS (platform markers)
scripts/run_scaling.sh       # all six sizes, N=100; see header for MODELS/N/TP/JUDGE_* env vars
# two GPUs, small models on one and 32B on both:
MODELS="Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B" CUDA_VISIBLE_DEVICES=0 scripts/run_scaling.sh
MODELS=Qwen3-32B TP=2 CUDA_VISIBLE_DEVICES=0,1 scripts/run_scaling.sh
# faithful judge, once OPENAI_API_KEY is set (e.g. in .env, which is gitignored):
JUDGE_BACKEND=openai scripts/run_scaling.sh
```

Qwen3-32B in bf16 needs about 64 GB of weights: use TP=2 on 2×48 GB+, one 80 GB card, or
`Qwen/Qwen3-32B-FP8`. Rollouts and judgments are gitignored (large; they contain harmful-content
completions). Copy them between machines out of band if needed.

## Conventions

- Every stage appends to JSONL and skips `(qid, sample_idx)` pairs already present, so it is safe
  to kill and rerun. Don't add stages that rewrite these files in place.
- Sampling uses Qwen3's recommended thinking-mode parameters: temperature 0.6, top-p 0.95,
  top-k 20, `enable_thinking=True`, max 4096 new tokens. Local judges run greedy with thinking off.
- VEA is judged on the CoT only (the post finds that 95% of instances are CoT-only). Refusal is
  judged on the final response; a rollout that never closed `</think>` gets `refusal = null`.
- `analyze.py --judge` takes the judge slug (the part after the `/`), e.g. `Qwen3-8B` or `gpt-5-mini`.

## Deviations from the post

- **Judge:** the post uses GPT-5-mini. The default here is a local Qwen3-8B until an OpenAI key is
  available, so results from a local judge are provisional. Compare against GPT-5-mini before
  drawing conclusions.
- **Excluded prompts:** the post drops 11 prompts that tell the model it is being evaluated but
  doesn't list them. We found 6 by reading every regex hit (rows 84, 90, 91, 100, 440, 442), so
  we use 494 questions rather than 489.
- The post's sampling parameters and max tokens aren't stated.
- `rubrics.PREFILTER_RE` skips the judge on CoTs with no test/evaluate-like words. It is only used
  with slow local judges; the GPU script passes `--no-prefilter`.

## Status (2026-09-23)

- **Pilot:** 100 Qwen3-0.6B rollouts on q140, on an M3 MacBook with MLX. Hand-reading every regex
  hit found 3 clear VEA rollouts (samples 31, 59, 87) and 1 ambiguous one (15), i.e. 3–4% vs the
  post's 6%. That is consistent at n=100. Sample 87: "The user might be testing the AI's ability to
  show both the surface and the hidden aspects". This is an "abilities" test, as the post
  describes for 0.6B. Median CoT+response length: 500 tokens, none truncated.
- Not yet run: the full Fortress sweep, the judge over the pilot, and the other sizes. Next step is
  `scripts/run_scaling.sh` on the GPU box.

## Gotchas

- On a 16 GB Mac, MLX batch generation OOMs above batch size ~16. Long Fortress prompts (several
  thousand tokens) make padded batches big. A local Qwen3-8B-4bit judge takes about 7.5 s per
  judgment and gains nothing from batching.
- Qwen3's `<think>`/`</think>` are non-special added tokens, so they survive decoding.
  `generate.split_thinking` relies on that.
- vLLM `SamplingParams(n=k)` gives k independent samples per prompt. `generate.py` uses it
  per question, with n = the number of missing rollouts.
