# Quantization study (smoke)

Judges: unified v2 (vea_cal and F2 refusal, Qwen3.8-27B bf16). Rollouts 0-9 per Fortress question (smoke: 20 questions x 4). Rates over scored rollouts with a valid label; 95% CIs resample questions.

| variant | scored / assigned | label coverage | VEA % | refusal % | within-q gap pp | truncated % | mean CoT tokens | VEA per 1k CoT tokens | benign refusal % |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B@bf16.smoke | 76 / 76 | 100.0 | 15.8 [2.6, 31.6] | 52.6 [32.9, 72.4] | - | 0.0 | 2105 | 0.07 | - |
| Qwen3.5-4B@gptq-w4-calB.smoke | 76 / 76 | 100.0 | 18.4 [6.6, 31.6] | 50.0 [30.3, 68.4] | -33.3 [-66.7, 0.0] | 0.0 | 2313 | 0.08 | - |
| Qwen3.5-4B@gptq-w4.smoke | 76 / 76 | 100.0 | 7.9 [0.0, 19.7] | 51.3 [30.3, 71.1] | - | 0.0 | 2206 | 0.04 | - |
| Qwen3.5-4B@gptq-w8.smoke | 76 / 76 | 100.0 | 14.5 [5.3, 26.3] | 55.3 [35.5, 75.0] | 0.0 [-100.0, 100.0] | 0.0 | 2299 | 0.06 | - |
| Qwen3.5-4B@hf-w4.smoke | 76 / 76 | 100.0 | 14.5 [5.3, 26.3] | 47.4 [27.6, 67.1] | 0.0 [-100.0, 100.0] | 0.0 | 2332 | 0.06 | - |
| Qwen3.5-4B@rtn-w4.smoke | 76 / 76 | 100.0 | 19.7 [7.9, 32.9] | 46.1 [26.3, 65.8] | 5.6 [-50.0, 66.7] | 0.0 | 2143 | 0.09 | - |

## Paired change vs the same model's bf16 baseline (percentage points)

| variant | questions | Δ refusal | Δ VEA | Δ within-q gap |
|---|---|---|---|---|
| Qwen3.5-4B@gptq-w4-calB.smoke | 19 | -2.6 [-11.8, +6.6] | +2.6 [-3.9, +9.2] | - |
| Qwen3.5-4B@gptq-w4.smoke | 19 | -1.3 [-9.2, +7.9] | -7.9 [-22.4, +2.6] | - |
| Qwen3.5-4B@gptq-w8.smoke | 19 | +2.6 [+0.0, +6.6] | -1.3 [-11.8, +7.9] | - |
| Qwen3.5-4B@hf-w4.smoke | 19 | -5.3 [-13.2, +1.3] | -1.3 [-13.2, +10.5] | - |
| Qwen3.5-4B@rtn-w4.smoke | 19 | -6.6 [-15.8, +1.3] | +3.9 [-5.3, +11.8] | - |
