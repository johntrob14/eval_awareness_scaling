# Unified results (v2)

- Rollout tables: complete: every scored primary rollout has valid labels
- Interventions: 15 of 16 models fully judged; pending: Qwen3.5-397B-A17B
- Status: not final: re-run with --final once every part is complete

Every model under docs/UNIFIED_PROTOCOL.md: this repo's judges (vea_cal; F2 refusal on the first 384 answer tokens), reasoning cap 8,192 tokens, 14 prompts excluded. Percent, 95% CIs from 2,000 bootstrap draws over questions. Judge-scored (Qwen3.8-27B), not human-verified.

## Primary design: rollouts 0-9 of every question

| model | source | scored / assigned | VEA rate | refusal | within-question refusal diff (VEA - not) | informative Q | Spearman(#VEA, refusal) | safety share of VEA |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | ladder | 4824 / 4860 | 0.3 [0.1, 0.5] | 2.9 [2.2, 3.8] | -3.6 [-15.1, 13.0] | 12 | 0.17 | 28.6 |
| Qwen3-1.7B | ladder | 4771 / 4860 | 0.8 [0.5, 1.2] | 1.6 [0.8, 2.5] | 1.8 [-4.8, 11.4] | 25 | 0.23 | 43.6 |
| Qwen3-4B | ladder | 4785 / 4860 | 4.0 [2.9, 5.3] | 3.1 [1.9, 4.4] | 4.1 [-1.7, 10.1] | 61 | 0.44 | 59.6 |
| Qwen3-8B | ladder | 4743 / 4860 | 4.9 [3.6, 6.3] | 4.3 [3.0, 5.8] | 8.6 [2.6, 15.4] | 69 | 0.45 | 64.7 |
| Qwen3-14B | ladder | 4832 / 4860 | 5.0 [3.7, 6.4] | 7.5 [5.5, 9.5] | 13.1 [6.7, 20.0] | 76 | 0.59 | 76.5 |
| Qwen3.5-0.8B | ladder | 4683 / 4860 | 32.6 [29.9, 35.2] | 27.9 [25.2, 30.9] | 9.7 [6.4, 12.8] | 367 | 0.35 | 51.8 |
| Qwen3.5-2B | ladder | 4745 / 4860 | 28.1 [25.5, 30.8] | 36.9 [33.5, 40.4] | 5.7 [3.0, 8.8] | 304 | 0.44 | 60.2 |
| Qwen3.5-4B | ladder | 4708 / 4860 | 15.3 [13.2, 17.4] | 48.5 [44.7, 52.5] | 3.2 [0.5, 5.9] | 218 | 0.35 | 73.1 |
| Qwen3.5-9B | ladder | 4746 / 4860 | 13.2 [11.3, 15.2] | 51.4 [47.5, 55.4] | 3.0 [0.5, 5.6] | 196 | 0.26 | 73.2 |
| Qwen3.5-27B | ladder | 4821 / 4860 | 10.0 [8.4, 11.8] | 60.4 [56.7, 64.1] | 2.2 [-1.2, 5.6] | 168 | 0.25 | 71.3 |
| Qwen3.8-27B | ladder | 4783 / 4860 | 1.0 [0.6, 1.5] | 62.6 [58.8, 66.4] | -3.1 [-14.4, 9.4] | 30 | -0.05 | 56.2 |
| Qwen3-32B | ladder | 4839 / 4860 | 4.4 [3.2, 5.7] | 7.0 [5.2, 8.8] | 11.7 [3.0, 20.7] | 62 | 0.57 | 80.5 |
| gemma-4-31B-it | mine | 4849 / 4860 | 0.7 [0.4, 1.2] | 56.5 [52.4, 60.3] | 12.2 [2.6, 25.8] | 19 | 0.01 | 52.8 |
| Nemotron-3-Super | mine | 4732 / 4860 | 25.0 [22.1, 27.9] | 51.3 [47.7, 55.0] | 7.9 [4.7, 11.3] | 238 | 0.66 | 93.4 |
| Qwen3.5-122B-A10B | mine | 4798 / 4860 | 10.4 [8.7, 12.3] | 58.2 [54.1, 61.8] | 3.1 [-0.3, 6.7] | 164 | 0.28 | 72.6 |
| Qwen3.5-397B-A17B | mine | 4775 / 4860 | 10.5 [8.7, 12.4] | 66.0 [62.2, 69.6] | 1.3 [-0.5, 3.4] | 156 | 0.30 | 78.3 |

## Secondary: all non-selected rollouts (ladder stage 2 over-weights high-VEA questions)

| model | source | scored / assigned | VEA rate | refusal | within-question refusal diff (VEA - not) | informative Q | Spearman(#VEA, refusal) | safety share of VEA |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | ladder | 8631 / 8740 | 0.6 [0.2, 1.2] | 3.2 [2.2, 4.4] | -6.4 [-17.3, 2.9] | 13 | 0.24 | 24.1 |
| Qwen3-1.7B | ladder | 8494 / 8700 | 1.8 [0.9, 2.9] | 1.7 [0.9, 2.6] | 0.5 [-5.3, 5.9] | 32 | 0.38 | 48.4 |
| Qwen3-4B | ladder | 8410 / 8700 | 9.7 [7.0, 12.5] | 5.6 [3.3, 8.2] | 7.0 [2.6, 12.0] | 69 | 0.60 | 58.5 |
| Qwen3-8B | ladder | 8254 / 8700 | 10.1 [7.1, 13.2] | 6.8 [4.2, 9.6] | 5.3 [2.0, 9.6] | 81 | 0.53 | 65.1 |
| Qwen3-14B | ladder | 8620 / 8700 | 9.4 [6.8, 12.1] | 12.2 [8.5, 16.0] | 10.7 [6.0, 16.0] | 87 | 0.61 | 72.3 |
| Qwen3.5-0.8B | ladder | 6538 / 6780 | 42.4 [39.0, 45.5] | 33.1 [29.6, 36.8] | 8.6 [6.0, 11.3] | 377 | 0.36 | 51.4 |
| Qwen3.5-2B | ladder | 6542 / 6780 | 38.8 [35.2, 42.1] | 41.3 [37.1, 45.3] | 6.0 [3.8, 8.4] | 320 | 0.45 | 56.0 |
| Qwen3.5-4B | ladder | 6421 / 6780 | 23.4 [19.8, 26.7] | 51.5 [47.2, 56.0] | 2.8 [0.9, 4.8] | 225 | 0.35 | 68.3 |
| Qwen3.5-9B | ladder | 6559 / 6820 | 19.5 [16.4, 22.4] | 54.4 [49.9, 58.6] | 2.7 [0.6, 4.9] | 203 | 0.28 | 69.7 |
| Qwen3.5-27B | ladder | 6709 / 6820 | 13.8 [11.5, 16.4] | 62.3 [58.2, 66.2] | 4.5 [1.9, 7.3] | 174 | 0.25 | 65.5 |
| Qwen3.8-27B | ladder | 8546 / 8740 | 1.8 [1.0, 2.9] | 61.5 [56.4, 66.5] | -2.5 [-10.2, 5.1] | 35 | -0.02 | 43.0 |
| Qwen3-32B | ladder | 8652 / 8740 | 9.9 [7.2, 12.6] | 12.7 [9.4, 16.1] | 17.1 [11.9, 22.0] | 70 | 0.59 | 83.2 |
| gemma-4-31B-it | mine | 9699 / 9720 | 0.7 [0.4, 1.0] | 56.4 [52.4, 60.4] | 3.9 [-6.1, 14.2] | 28 | 0.02 | 49.3 |
| Nemotron-3-Super | mine | 9465 / 9720 | 24.7 [22.0, 27.5] | 51.0 [47.6, 54.5] | 7.6 [5.1, 10.3] | 292 | 0.70 | 93.3 |
| Qwen3.5-122B-A10B | mine | 9591 / 9720 | 10.2 [8.6, 11.9] | 58.3 [54.2, 61.8] | 3.5 [1.3, 5.9] | 208 | 0.34 | 71.2 |
| Qwen3.5-397B-A17B | mine | 9562 / 9720 | 10.7 [9.0, 12.5] | 65.7 [62.0, 69.3] | 1.5 [-0.1, 3.2] | 199 | 0.37 | 77.9 |

## Coverage and label checks (primary design)

| model | status counts | valid VEA / refusal labels on scored | VEA with category | VEA quote found in reasoning (exact / fragments / near / fail) |
|---|---|---|---|---|
| Qwen3-0.6B | scored 4824, truncated 36 | 4824 / 4824 of 4824 | 14 / 14 | 14 / 0 / 0 / 0 |
| Qwen3-1.7B | scored 4771, truncated 89 | 4771 / 4771 of 4771 | 39 / 39 | 39 / 0 / 0 / 0 |
| Qwen3-4B | scored 4785, truncated 75 | 4785 / 4785 of 4785 | 193 / 193 | 188 / 0 / 4 / 1 |
| Qwen3-8B | scored 4743, truncated 117 | 4743 / 4743 of 4743 | 232 / 232 | 230 / 0 / 2 / 0 |
| Qwen3-14B | scored 4832, truncated 28 | 4832 / 4832 of 4832 | 243 / 243 | 241 / 0 / 2 / 0 |
| Qwen3.5-0.8B | scored 4683, truncated 176, no_think_end 1 | 4683 / 4683 of 4683 | 1525 / 1525 | 1493 / 1 / 20 / 11 |
| Qwen3.5-2B | scored 4745, truncated 115 | 4745 / 4745 of 4745 | 1335 / 1335 | 1296 / 0 / 30 / 9 |
| Qwen3.5-4B | scored 4708, truncated 152 | 4708 / 4708 of 4708 | 720 / 720 | 704 / 0 / 11 / 5 |
| Qwen3.5-9B | scored 4746, truncated 114 | 4746 / 4746 of 4746 | 628 / 628 | 608 / 0 / 10 / 10 |
| Qwen3.5-27B | scored 4821, truncated 39 | 4821 / 4821 of 4821 | 481 / 481 | 464 / 0 / 10 / 7 |
| Qwen3.8-27B | scored 4783, truncated 73, no_think_end 4 | 4783 / 4783 of 4783 | 48 / 48 | 48 / 0 / 0 / 0 |
| Qwen3-32B | scored 4839, truncated 21 | 4839 / 4839 of 4839 | 215 / 215 | 213 / 0 / 1 / 1 |
| gemma-4-31B-it | scored 4849, truncated 11 | 4849 / 4849 of 4849 | 36 / 36 | 36 / 0 / 0 / 0 |
| Nemotron-3-Super | scored 4732, truncated 127, no_think_end 1 | 4732 / 4732 of 4732 | 1183 / 1183 | 1181 / 0 / 1 / 1 |
| Qwen3.5-122B-A10B | scored 4798, truncated 61, no_think_end 1 | 4798 / 4798 of 4798 | 500 / 500 | 487 / 0 / 10 / 3 |
| Qwen3.5-397B-A17B | scored 4775, truncated 84, no_think_end 1 | 4775 / 4775 of 4775 | 502 / 502 | 489 / 1 / 10 / 2 |

## Sentence insertion / removal (refusal, percentage points; ladder exp03 design)

Primary: refusal among observed continuations, paired within base, 95% CI over questions. Bounds: missing outcomes (not scored or not judged) set to the worst case for the contrast.

| model | contrast | estimate [95% CI] | bounds | bases / questions |
|---|---|---|---|---|
| Nemotron-3-Super | insert vea - none | +28.1 [+22.6, +34.1] | +26.9 to +28.7 | 150 / 150 |
| Nemotron-3-Super | insert placebo - none | +37.6 [+31.5, +43.6] | +36.2 to +37.8 | 150 / 150 |
| Nemotron-3-Super | insert neutral - none | +2.5 [+0.1, +4.9] | +1.4 to +3.5 | 150 / 150 |
| Nemotron-3-Super | insert vea - placebo | -9.5 [-12.9, -6.2] | -9.7 to -8.7 | 150 / 150 |
| Nemotron-3-Super | remove keep - remove | +2.1 [+0.3, +4.0] | +1.8 to +2.1 | 150 / 107 |
| Qwen3-0.6B | insert vea - none | +4.4 [+2.0, +7.1] | +3.8 to +4.8 | 147 / 147 |
| Qwen3-0.6B | insert placebo - none | +5.5 [+2.9, +8.3] | +5.0 to +6.0 | 147 / 147 |
| Qwen3-0.6B | insert neutral - none | -0.3 [-2.0, +1.7] | -0.9 to +0.3 | 147 / 147 |
| Qwen3-0.6B | insert vea - placebo | -1.0 [-3.2, +1.1] | -1.7 to -0.8 | 147 / 147 |
| Qwen3-0.6B | remove keep - remove | -2.3 [-12.9, +5.0] | -3.3 to -2.0 | 19 / 11 |
| Qwen3-1.7B | insert vea - none | +4.1 [+1.4, +7.4] | +1.5 to +5.2 | 91 / 91 |
| Qwen3-1.7B | insert placebo - none | +4.9 [+1.7, +8.6] | +2.2 to +5.9 | 91 / 91 |
| Qwen3-1.7B | insert neutral - none | +0.3 [-1.3, +1.8] | -2.2 to +1.4 | 91 / 91 |
| Qwen3-1.7B | insert vea - placebo | -0.8 [-2.8, +1.0] | -1.9 to +0.5 | 91 / 91 |
| Qwen3-1.7B | remove keep - remove | -1.5 [-4.0, +0.8] | -2.3 to +0.5 | 49 / 30 |
| Qwen3-14B | insert vea - none | +23.9 [+17.3, +31.1] | +22.9 to +24.0 | 78 / 78 |
| Qwen3-14B | insert placebo - none | +26.8 [+19.4, +34.5] | +25.8 to +27.1 | 78 / 78 |
| Qwen3-14B | insert neutral - none | +0.9 [-3.2, +5.1] | -0.2 to +2.1 | 78 / 78 |
| Qwen3-14B | insert vea - placebo | -2.9 [-6.6, +0.8] | -3.2 to -2.7 | 78 / 78 |
| Qwen3-14B | remove keep - remove | +0.0 [-2.0, +2.2] | -0.2 to +0.2 | 119 / 64 |
| Qwen3-32B | insert vea - none | +22.5 [+16.5, +28.6] | +21.4 to +23.5 | 66 / 66 |
| Qwen3-32B | insert placebo - none | +21.0 [+13.7, +28.2] | +19.9 to +21.4 | 66 / 66 |
| Qwen3-32B | insert neutral - none | +1.5 [-2.1, +4.9] | +0.4 to +1.7 | 66 / 66 |
| Qwen3-32B | insert vea - placebo | +1.5 [-3.2, +6.1] | +1.1 to +2.5 | 66 / 66 |
| Qwen3-32B | remove keep - remove | +3.1 [+0.4, +5.9] | +2.9 to +3.5 | 112 / 55 |
| Qwen3-4B | insert vea - none | +8.6 [+5.0, +12.7] | +8.2 to +9.7 | 81 / 81 |
| Qwen3-4B | insert placebo - none | +10.8 [+5.4, +16.4] | +10.5 to +11.7 | 81 / 81 |
| Qwen3-4B | insert neutral - none | +1.9 [-0.3, +4.3] | +1.5 to +3.2 | 81 / 81 |
| Qwen3-4B | insert vea - placebo | -2.2 [-5.9, +1.7] | -3.1 to -1.2 | 81 / 81 |
| Qwen3-4B | remove keep - remove | +1.1 [-1.4, +3.5] | -0.3 to +2.1 | 108 / 55 |
| Qwen3-8B | insert vea - none | +20.3 [+13.4, +27.7] | +17.9 to +21.0 | 79 / 79 |
| Qwen3-8B | insert placebo - none | +18.8 [+12.4, +25.3] | +16.5 to +19.6 | 79 / 79 |
| Qwen3-8B | insert neutral - none | -0.9 [-4.1, +2.3] | -3.2 to +1.4 | 79 / 79 |
| Qwen3-8B | insert vea - placebo | +1.5 [-3.1, +6.2] | +0.5 to +2.4 | 79 / 79 |
| Qwen3-8B | remove keep - remove | +0.3 [-1.5, +2.1] | -1.2 to +1.4 | 117 / 61 |
| Qwen3.5-0.8B | insert vea - none | +2.2 [+0.3, +4.2] | +0.8 to +3.4 | 149 / 149 |
| Qwen3.5-0.8B | insert placebo - none | +2.3 [+0.3, +4.1] | +0.8 to +3.3 | 149 / 149 |
| Qwen3.5-0.8B | insert neutral - none | -0.9 [-2.6, +0.8] | -2.3 to +0.4 | 149 / 149 |
| Qwen3.5-0.8B | insert vea - placebo | -0.0 [-1.8, +1.8] | -1.2 to +1.3 | 149 / 149 |
| Qwen3.5-0.8B | remove keep - remove | -0.8 [-3.4, +1.9] | -4.5 to +1.8 | 75 / 68 |
| Qwen3.5-122B-A10B | insert vea - none | +3.3 [+0.1, +6.7] | +3.3 to +3.4 | 84 / 84 |
| Qwen3.5-122B-A10B | insert placebo - none | +5.2 [+1.8, +8.6] | +5.2 to +5.4 | 84 / 84 |
| Qwen3.5-122B-A10B | insert neutral - none | -0.6 [-4.0, +2.7] | -0.6 to -0.4 | 84 / 84 |
| Qwen3.5-122B-A10B | insert vea - placebo | -1.9 [-4.9, +1.2] | -1.9 to -1.9 | 84 / 84 |
| Qwen3.5-122B-A10B | remove keep - remove | -1.1 [-2.5, +0.2] | -1.8 to -0.8 | 150 / 103 |
| Qwen3.5-27B | insert vea - none | +4.1 [+0.3, +8.2] | +3.8 to +5.8 | 75 / 75 |
| Qwen3.5-27B | insert placebo - none | +5.7 [+1.8, +9.9] | +5.0 to +7.2 | 75 / 75 |
| Qwen3.5-27B | insert neutral - none | -1.4 [-5.1, +2.0] | -1.8 to +1.3 | 75 / 75 |
| Qwen3.5-27B | insert vea - placebo | -1.7 [-5.7, +2.2] | -1.8 to -0.7 | 75 / 75 |
| Qwen3.5-27B | remove keep - remove | +0.1 [-1.6, +1.9] | -0.4 to +0.7 | 93 / 70 |
| Qwen3.5-2B | insert vea - none | +1.9 [-0.5, +4.3] | +1.1 to +2.5 | 79 / 79 |
| Qwen3.5-2B | insert placebo - none | +0.5 [-1.4, +2.2] | -0.3 to +0.5 | 79 / 79 |
| Qwen3.5-2B | insert neutral - none | -1.3 [-3.3, +0.6] | -2.1 to -0.8 | 79 / 79 |
| Qwen3.5-2B | insert vea - placebo | +1.4 [-0.6, +3.5] | +1.4 to +2.1 | 79 / 79 |
| Qwen3.5-2B | remove keep - remove | -0.9 [-3.7, +1.8] | -2.2 to +0.3 | 80 / 71 |
| Qwen3.5-4B | insert vea - none | +4.4 [+1.5, +7.4] | +4.4 to +4.4 | 66 / 66 |
| Qwen3.5-4B | insert placebo - none | +2.1 [-1.3, +5.5] | +2.1 to +2.1 | 66 / 66 |
| Qwen3.5-4B | insert neutral - none | +0.6 [-2.5, +3.2] | +0.6 to +0.6 | 66 / 66 |
| Qwen3.5-4B | insert vea - placebo | +2.3 [-1.3, +5.7] | +2.3 to +2.3 | 66 / 66 |
| Qwen3.5-4B | remove keep - remove | +1.4 [-0.3, +3.1] | -0.9 to +3.1 | 81 / 64 |
| Qwen3.5-9B | insert vea - none | +0.8 [-2.7, +4.4] | +0.8 to +0.8 | 74 / 74 |
| Qwen3.5-9B | insert placebo - none | +1.7 [-1.4, +4.7] | +1.7 to +1.7 | 74 / 74 |
| Qwen3.5-9B | insert neutral - none | -0.7 [-3.5, +2.2] | -0.7 to -0.5 | 74 / 74 |
| Qwen3.5-9B | insert vea - placebo | -0.8 [-3.9, +1.9] | -0.8 to -0.8 | 74 / 74 |
| Qwen3.5-9B | remove keep - remove | -1.8 [-3.4, -0.2] | -4.9 to +1.1 | 81 / 63 |
| Qwen3.8-27B | insert vea - none | +6.8 [+2.5, +11.3] | +6.4 to +7.8 | 106 / 106 |
| Qwen3.8-27B | insert placebo - none | +5.3 [+0.9, +9.8] | +5.1 to +6.1 | 106 / 106 |
| Qwen3.8-27B | insert neutral - none | -1.7 [-5.4, +1.7] | -2.8 to -0.7 | 105 / 105 |
| Qwen3.8-27B | insert vea - placebo | +1.5 [-1.7, +4.9] | +0.9 to +2.0 | 106 / 106 |
| Qwen3.8-27B | remove keep - remove | +2.3 [-1.9, +7.4] | +1.3 to +3.9 | 48 / 26 |
| gemma-4-31B-it | insert vea - none | +4.5 [+0.2, +9.0] | +4.5 to +4.5 | 106 / 106 |
| gemma-4-31B-it | insert placebo - none | +7.1 [+2.9, +11.9] | +7.1 to +7.1 | 106 / 106 |
| gemma-4-31B-it | insert neutral - none | +3.9 [+0.5, +7.4] | +3.9 to +3.9 | 106 / 106 |
| gemma-4-31B-it | insert vea - placebo | -2.6 [-5.9, +0.5] | -2.6 to -2.6 | 106 / 106 |
| gemma-4-31B-it | remove keep - remove | +0.8 [-2.0, +3.2] | +0.3 to +1.6 | 48 / 27 |

Per condition: scored / judged / assigned continuations, and re-verbalized VEA in the new reasoning (vea_cal, among judged).

| model | arm | condition | scored | judged | assigned | re-verbalized |
|---|---|---|---|---|---|---|
| Nemotron-3-Super | insert | neutral | 1189 | 1189 | 1200 | 9.8% |
| Nemotron-3-Super | insert | none | 1186 | 1186 | 1200 | 6.2% |
| Nemotron-3-Super | insert | placebo | 1196 | 1196 | 1200 | 28.1% |
| Nemotron-3-Super | insert | vea | 1192 | 1192 | 1200 | 21.1% |
| Nemotron-3-Super | remove | keep | 1200 | 1200 | 1200 | 9.7% |
| Nemotron-3-Super | remove | remove | 1197 | 1197 | 1200 | 75.9% |
| Qwen3-0.6B | insert | neutral | 1168 | 1168 | 1176 | 1.0% |
| Qwen3-0.6B | insert | none | 1170 | 1170 | 1176 | 0.4% |
| Qwen3-0.6B | insert | placebo | 1170 | 1170 | 1176 | 16.5% |
| Qwen3-0.6B | insert | vea | 1171 | 1171 | 1176 | 25.3% |
| Qwen3-0.6B | remove | keep | 152 | 152 | 152 | 11.8% |
| Qwen3-0.6B | remove | remove | 150 | 150 | 152 | 23.0% |
| Qwen3-1.7B | insert | neutral | 720 | 720 | 728 | 2.1% |
| Qwen3-1.7B | insert | none | 710 | 710 | 728 | 2.1% |
| Qwen3-1.7B | insert | placebo | 719 | 719 | 728 | 14.0% |
| Qwen3-1.7B | insert | vea | 719 | 719 | 728 | 22.5% |
| Qwen3-1.7B | remove | keep | 384 | 384 | 392 | 7.9% |
| Qwen3-1.7B | remove | remove | 389 | 389 | 392 | 41.1% |
| Qwen3-14B | insert | neutral | 616 | 616 | 624 | 11.1% |
| Qwen3-14B | insert | none | 618 | 618 | 624 | 7.7% |
| Qwen3-14B | insert | placebo | 622 | 622 | 624 | 40.7% |
| Qwen3-14B | insert | vea | 623 | 623 | 624 | 42.8% |
| Qwen3-14B | remove | keep | 950 | 950 | 952 | 8.8% |
| Qwen3-14B | remove | remove | 950 | 950 | 952 | 57.4% |
| Qwen3-32B | insert | neutral | 527 | 527 | 528 | - |
| Qwen3-32B | insert | none | 522 | 522 | 528 | - |
| Qwen3-32B | insert | placebo | 526 | 526 | 528 | - |
| Qwen3-32B | insert | vea | 523 | 523 | 528 | - |
| Qwen3-32B | remove | keep | 892 | 892 | 896 | - |
| Qwen3-32B | remove | remove | 895 | 895 | 896 | - |
| Qwen3-4B | insert | neutral | 640 | 640 | 648 | 10.8% |
| Qwen3-4B | insert | none | 645 | 645 | 648 | 5.6% |
| Qwen3-4B | insert | placebo | 643 | 643 | 648 | 25.6% |
| Qwen3-4B | insert | vea | 641 | 641 | 648 | 31.3% |
| Qwen3-4B | remove | keep | 856 | 856 | 864 | 16.3% |
| Qwen3-4B | remove | remove | 851 | 851 | 864 | 57.2% |
| Qwen3-8B | insert | neutral | 617 | 617 | 632 | 11.7% |
| Qwen3-8B | insert | none | 618 | 618 | 632 | 7.6% |
| Qwen3-8B | insert | placebo | 626 | 626 | 632 | 35.1% |
| Qwen3-8B | insert | vea | 626 | 626 | 632 | 42.2% |
| Qwen3-8B | remove | keep | 926 | 926 | 936 | 12.9% |
| Qwen3-8B | remove | remove | 922 | 922 | 936 | 54.1% |
| Qwen3.5-0.8B | insert | neutral | 1176 | 1176 | 1192 | 12.6% |
| Qwen3.5-0.8B | insert | none | 1176 | 1176 | 1192 | 8.9% |
| Qwen3.5-0.8B | insert | placebo | 1179 | 1179 | 1192 | 15.5% |
| Qwen3.5-0.8B | insert | vea | 1176 | 1176 | 1192 | 16.4% |
| Qwen3.5-0.8B | remove | keep | 583 | 583 | 600 | 51.3% |
| Qwen3.5-0.8B | remove | remove | 579 | 579 | 600 | 46.6% |
| Qwen3.5-122B-A10B | insert | neutral | 672 | 672 | 672 | 3.1% |
| Qwen3.5-122B-A10B | insert | none | 671 | 671 | 672 | 1.8% |
| Qwen3.5-122B-A10B | insert | placebo | 672 | 672 | 672 | 8.8% |
| Qwen3.5-122B-A10B | insert | vea | 672 | 672 | 672 | 7.6% |
| Qwen3.5-122B-A10B | remove | keep | 1195 | 1195 | 1200 | 22.9% |
| Qwen3.5-122B-A10B | remove | remove | 1193 | 1193 | 1200 | 38.2% |
| Qwen3.5-27B | insert | neutral | 590 | 590 | 600 | - |
| Qwen3.5-27B | insert | none | 591 | 591 | 600 | - |
| Qwen3.5-27B | insert | placebo | 596 | 596 | 600 | - |
| Qwen3.5-27B | insert | vea | 597 | 597 | 600 | - |
| Qwen3.5-27B | remove | keep | 741 | 741 | 744 | - |
| Qwen3.5-27B | remove | remove | 739 | 739 | 744 | - |
| Qwen3.5-2B | insert | neutral | 629 | 629 | 632 | - |
| Qwen3.5-2B | insert | none | 627 | 627 | 632 | - |
| Qwen3.5-2B | insert | placebo | 632 | 632 | 632 | - |
| Qwen3.5-2B | insert | vea | 628 | 628 | 632 | - |
| Qwen3.5-2B | remove | keep | 629 | 629 | 640 | - |
| Qwen3.5-2B | remove | remove | 635 | 635 | 640 | - |
| Qwen3.5-4B | insert | neutral | 528 | 528 | 528 | - |
| Qwen3.5-4B | insert | none | 528 | 528 | 528 | - |
| Qwen3.5-4B | insert | placebo | 528 | 528 | 528 | - |
| Qwen3.5-4B | insert | vea | 528 | 528 | 528 | - |
| Qwen3.5-4B | remove | keep | 636 | 636 | 648 | - |
| Qwen3.5-4B | remove | remove | 634 | 634 | 648 | - |
| Qwen3.5-9B | insert | neutral | 591 | 591 | 592 | - |
| Qwen3.5-9B | insert | none | 592 | 592 | 592 | - |
| Qwen3.5-9B | insert | placebo | 592 | 592 | 592 | - |
| Qwen3.5-9B | insert | vea | 592 | 592 | 592 | - |
| Qwen3.5-9B | remove | keep | 631 | 631 | 648 | - |
| Qwen3.5-9B | remove | remove | 626 | 626 | 648 | - |
| Qwen3.8-27B | insert | neutral | 836 | 836 | 848 | - |
| Qwen3.8-27B | insert | none | 842 | 842 | 848 | - |
| Qwen3.8-27B | insert | placebo | 845 | 845 | 848 | - |
| Qwen3.8-27B | insert | vea | 842 | 842 | 848 | - |
| Qwen3.8-27B | remove | keep | 378 | 378 | 384 | - |
| Qwen3.8-27B | remove | remove | 380 | 380 | 384 | - |
| gemma-4-31B-it | insert | neutral | 848 | 848 | 848 | 0.9% |
| gemma-4-31B-it | insert | none | 848 | 848 | 848 | 0.8% |
| gemma-4-31B-it | insert | placebo | 848 | 848 | 848 | 1.7% |
| gemma-4-31B-it | insert | vea | 848 | 848 | 848 | 1.4% |
| gemma-4-31B-it | remove | keep | 381 | 381 | 384 | 11.5% |
| gemma-4-31B-it | remove | remove | 382 | 382 | 384 | 28.1% |
