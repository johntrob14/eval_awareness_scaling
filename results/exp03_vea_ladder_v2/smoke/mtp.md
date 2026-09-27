# exp03 MTP smoke test (Qwen3.8-27B)

600 rollouts (300 Fortress questions x 2, exp02 seeds).

- no MTP: 8.7 min, 1295 generated tokens/s
- MTP (2 tokens): 8.8 min, 1301 tokens/s -> 1.00x tokens/s, 0.99x wall clock

| run | mean CoT | median CoT | regex hit | truncated | mean answer |
|---|---|---|---|---|---|
| 0 | 808 | 446 | 0.758 | 0.015 | 318 |
| 2 | 826 | 436 | 0.755 | 0.015 | 320 |
| exp02 | 820 | 434 | 0.757 | 0.012 | 316 |

- 2 minus no-MTP: CoT +19 tokens [-41, +79], regex rate -0.003 [-0.037, +0.030], KS p=0.92, identical CoT text 0.0%
- exp02 minus no-MTP: CoT +12 tokens [-32, +58], regex rate -0.002 [-0.028, +0.027], KS p=0.92, identical CoT text 10.8%

Decision: use MTP = **False**
