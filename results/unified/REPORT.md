# Unified results

Every model under docs/UNIFIED_PROTOCOL.md: this repo's judges (vea_cal; F2 refusal on the first 384 answer tokens), reasoning cap 8,192 tokens, 14 prompts excluded. Percent, 95% CIs from 2,000 bootstrap draws over questions. Judge-scored (Qwen3.8-27B), not human-verified.

## Primary design: rollouts 0-9 of every question

| model | source | rollouts | VEA rate | refusal | within-question refusal diff (VEA - not) | informative Q | Spearman(#VEA, refusal) | safety share of VEA |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31B-it | mine | 4849 | 0.7 [0.4, 1.2] | 56.5 [52.4, 60.3] | 12.2 [2.6, 25.8] | 19 | 0.01 | 52.8 |
| Nemotron-3-Super | mine | 4749 | 25.0 [22.0, 28.0] | 51.1 [47.6, 54.8] | 7.8 [4.6, 11.3] | 240 | 0.65 | 93.0 |
| Qwen3.5-122B-A10B | mine | 4801 | 10.3 [8.6, 12.2] | 58.2 [54.1, 61.8] | 2.9 [-0.4, 6.6] | 165 | 0.28 | 73.0 |
| Qwen3.5-397B-A17B | mine | 4814 | 10.7 [8.9, 12.6] | 66.2 [62.4, 69.7] | 1.3 [-0.5, 3.3] | 156 | 0.30 | 77.7 |

## Secondary: all non-selected rollouts (ladder stage 2 over-weights high-VEA questions)

| model | source | rollouts | VEA rate | refusal | within-question refusal diff (VEA - not) | informative Q | Spearman(#VEA, refusal) | safety share of VEA |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31B-it | mine | 9699 | 0.7 [0.4, 1.0] | 56.4 [52.4, 60.4] | 3.9 [-6.1, 14.2] | 28 | 0.02 | 49.3 |
| Nemotron-3-Super | mine | 9501 | 24.7 [22.0, 27.4] | 50.9 [47.5, 54.4] | 7.5 [5.1, 10.3] | 295 | 0.69 | 93.0 |
| Qwen3.5-122B-A10B | mine | 9599 | 10.2 [8.6, 11.9] | 58.3 [54.2, 61.8] | 3.3 [1.1, 5.7] | 208 | 0.34 | 71.3 |
| Qwen3.5-397B-A17B | mine | 9625 | 10.8 [9.2, 12.6] | 65.9 [62.2, 69.5] | 1.7 [0.1, 3.5] | 199 | 0.37 | 77.4 |

