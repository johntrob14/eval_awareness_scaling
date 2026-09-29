# Unified results (v2, FINAL)

Every model under docs/UNIFIED_PROTOCOL.md: this repo's judges (vea_cal; F2 refusal on the first 384 answer tokens), reasoning cap 8,192 tokens, 14 prompts excluded. Percent, 95% CIs from 2,000 bootstrap draws over questions. Judge-scored (Qwen3.8-27B), not human-verified.

## Primary design: rollouts 0-9 of every question

| model | source | scored / assigned | VEA rate | refusal | within-question refusal diff (VEA - not) | informative Q | Spearman(#VEA, refusal) | safety share of VEA |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31B-it | mine | 4849 / 4860 | 0.7 [0.4, 1.2] | 56.5 [52.4, 60.3] | 12.2 [2.6, 25.8] | 19 | 0.01 | 52.8 |
| Nemotron-3-Super | mine | 4732 / 4860 | 25.0 [22.1, 27.9] | 51.3 [47.7, 55.0] | 7.9 [4.7, 11.3] | 238 | 0.66 | 93.4 |
| Qwen3.5-122B-A10B | mine | 4798 / 4860 | 10.4 [8.7, 12.3] | 58.2 [54.1, 61.8] | 3.1 [-0.3, 6.7] | 164 | 0.28 | 72.6 |
| Qwen3.5-397B-A17B | mine | 4775 / 4860 | 10.5 [8.7, 12.4] | 66.0 [62.2, 69.6] | 1.3 [-0.5, 3.4] | 156 | 0.30 | 78.3 |

## Secondary: all non-selected rollouts (ladder stage 2 over-weights high-VEA questions)

| model | source | scored / assigned | VEA rate | refusal | within-question refusal diff (VEA - not) | informative Q | Spearman(#VEA, refusal) | safety share of VEA |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31B-it | mine | 9699 / 9720 | 0.7 [0.4, 1.0] | 56.4 [52.4, 60.4] | 3.9 [-6.1, 14.2] | 28 | 0.02 | 49.3 |
| Nemotron-3-Super | mine | 9465 / 9720 | 24.7 [22.0, 27.5] | 51.0 [47.6, 54.5] | 7.6 [5.1, 10.3] | 292 | 0.70 | 93.3 |
| Qwen3.5-122B-A10B | mine | 9591 / 9720 | 10.2 [8.6, 11.9] | 58.3 [54.2, 61.8] | 3.5 [1.3, 5.9] | 208 | 0.34 | 71.2 |
| Qwen3.5-397B-A17B | mine | 9562 / 9720 | 10.7 [9.0, 12.5] | 65.7 [62.0, 69.3] | 1.5 [-0.1, 3.2] | 199 | 0.37 | 77.9 |

## Coverage and label checks (primary design)

| model | status counts | valid VEA / refusal labels on scored | VEA with category | VEA quote found in reasoning (exact / fragments / near / fail) |
|---|---|---|---|---|
| gemma-4-31B-it | scored 4849, truncated 11 | 4849 / 4849 of 4849 | 36 / 36 | 36 / 0 / 0 / 0 |
| Nemotron-3-Super | scored 4732, truncated 127, no_think_end 1 | 4732 / 4732 of 4732 | 1183 / 1183 | 1181 / 0 / 1 / 1 |
| Qwen3.5-122B-A10B | scored 4798, truncated 61, no_think_end 1 | 4798 / 4798 of 4798 | 500 / 500 | 487 / 0 / 10 / 3 |
| Qwen3.5-397B-A17B | scored 4775, truncated 84, no_think_end 1 | 4775 / 4775 of 4775 | 502 / 502 | 489 / 1 / 10 / 2 |

