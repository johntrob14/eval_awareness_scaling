# Quantization study

Judges: unified v2 (vea_cal and F2 refusal, Qwen3.8-27B bf16). Rollouts 0-9 per Fortress question (smoke: 20 questions x 4). Rates over scored rollouts with a valid label; 95% CIs resample questions.

| variant | scored / assigned | label coverage | VEA % | refusal % | within-q gap pp | truncated % | mean CoT tokens | VEA per 1k CoT tokens | benign refusal % |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B@bf16 | 4709 / 4860 | 100.0 | 16.6 [14.4, 18.7] | 48.5 [44.5, 52.2] | 3.2 [-2.9, 9.8] | 3.1 | 2665 | 0.06 | 3.6 |
| Qwen3.5-4B@gptq-w4 | 4737 / 4860 | 100.0 | 13.6 [11.6, 15.8] | 46.9 [43.2, 50.9] | 3.7 [-2.2, 9.8] | 2.5 | 2509 | 0.05 | 3.9 |
| Qwen3.5-4B@hf-w4 | 4737 / 4860 | 100.0 | 19.7 [17.3, 22.1] | 50.0 [46.3, 53.9] | 10.3 [5.3, 15.4] | 2.5 | 2721 | 0.07 | 4.3 |
| Qwen3.5-4B@rtn-w4 | 4753 / 4860 | 100.0 | 18.4 [16.1, 20.7] | 45.3 [41.5, 49.0] | 3.2 [-2.3, 8.6] | 2.2 | 2481 | 0.07 | 3.4 |

## Paired change vs the same model's bf16 baseline (percentage points)

| variant | questions | Δ refusal | Δ VEA | Δ within-q gap |
|---|---|---|---|---|
| Qwen3.5-4B@gptq-w4 | 479 | -1.5 [-2.8, -0.2] | -3.2 [-4.3, -1.9] | +0.5 [-8.1, +8.9] |
| Qwen3.5-4B@hf-w4 | 479 | +1.6 [+0.2, +2.9] | +3.1 [+1.8, +4.4] | +7.0 [-0.9, +15.6] |
| Qwen3.5-4B@rtn-w4 | 479 | -3.1 [-4.5, -1.7] | +1.7 [+0.5, +3.1] | +0.0 [-7.5, +8.4] |
