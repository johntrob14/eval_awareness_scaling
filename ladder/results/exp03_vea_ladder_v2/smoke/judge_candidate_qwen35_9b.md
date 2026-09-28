# exp03 candidate judge: Qwen3.5-9B vs Qwen3.8-27B (bf16), reasoning-first

1468 smoke items. Production judge = Qwen3.8-27B FP8, reasoning-first (estimated 8.34 calls/s).

## Speed

| pass | calls/s | mean output tokens | parse failures | truncated |
|---|---|---|---|---|
| 27b_r1 | 6.60 | 99 | 0 | 0 |
| 27b_r2 | 6.64 | 100 | 3 | 3 |
| 9b_r1 | 19.78 | 98 | 1 | 1 |
| 9b_r2 | 23.11 | 97 | 0 | 0 |

9B speedup vs FP8 27B (estimated): **2.57x**


## Agreement (Cohen's kappa [95% CI]), all items

| comparison | awareness_4way | vea_binary | refusal_3way | full_refusal_binary |
|---|---|---|---|---|
| noise floor: 27b_r2 vs 27b_r1 | 0.880 [0.846, 0.908] | 0.926 [0.891, 0.956] | 0.801 [0.764, 0.838] | 0.877 [0.835, 0.914] |
| 9b_r2 vs 27b_r1 (same seed structure as noise floor) | 0.571 [0.525, 0.616] | 0.697 [0.643, 0.749] | 0.675 [0.629, 0.721] | 0.777 [0.721, 0.826] |
| 9b_r1 vs 27b_r1 | 0.558 [0.511, 0.601] | 0.679 [0.626, 0.736] | 0.674 [0.626, 0.721] | 0.798 [0.746, 0.844] |
| 9b self: 9b_r2 vs 9b_r1 | 0.670 [0.628, 0.713] | 0.741 [0.692, 0.790] | 0.736 [0.691, 0.774] | 0.813 [0.762, 0.857] |

## Agreement (Cohen's kappa [95% CI]), random items

| comparison | awareness_4way | vea_binary | refusal_3way | full_refusal_binary |
|---|---|---|---|---|
| noise floor: 27b_r2 vs 27b_r1 | 0.896 [0.857, 0.930] | 0.943 [0.904, 0.975] | 0.856 [0.811, 0.896] | 0.924 [0.885, 0.958] |
| 9b_r2 vs 27b_r1 (same seed structure as noise floor) | 0.623 [0.564, 0.679] | 0.713 [0.638, 0.781] | 0.770 [0.712, 0.820] | 0.853 [0.798, 0.901] |
| 9b_r1 vs 27b_r1 | 0.601 [0.538, 0.655] | 0.675 [0.592, 0.743] | 0.751 [0.702, 0.804] | 0.877 [0.826, 0.923] |
| 9b self: 9b_r2 vs 9b_r1 | 0.691 [0.631, 0.742] | 0.758 [0.687, 0.820] | 0.815 [0.768, 0.860] | 0.903 [0.860, 0.945] |

## Agreement (Cohen's kappa [95% CI]), validation items

| comparison | awareness_4way | vea_binary | refusal_3way | full_refusal_binary |
|---|---|---|---|---|
| noise floor: 27b_r2 vs 27b_r1 | 0.843 [0.788, 0.895] | 0.895 [0.828, 0.948] | 0.687 [0.611, 0.760] | 0.782 [0.694, 0.864] |
| 9b_r2 vs 27b_r1 (same seed structure as noise floor) | 0.447 [0.365, 0.521] | 0.622 [0.527, 0.717] | 0.484 [0.395, 0.571] | 0.621 [0.503, 0.726] |
| 9b_r1 vs 27b_r1 | 0.450 [0.373, 0.529] | 0.636 [0.545, 0.732] | 0.514 [0.421, 0.594] | 0.629 [0.507, 0.728] |
| 9b self: 9b_r2 vs 9b_r1 | 0.597 [0.512, 0.672] | 0.658 [0.553, 0.747] | 0.576 [0.488, 0.660] | 0.609 [0.489, 0.724] |

## Agreement by subject family (all items)

| family | comparison | awareness_4way | vea_binary | refusal_3way | full_refusal_binary |
|---|---|---|---|---|---|
| qwen3 | noise floor: 27b_r2 vs 27b_r1 | 0.927 [0.884, 0.963] | 0.977 [0.948, 1.000] | 0.775 [0.702, 0.841] | 0.863 [0.780, 0.935] |
| qwen3 | 9b_r2 vs 27b_r1 (same seed structure as noise floor) | 0.610 [0.545, 0.678] | 0.856 [0.795, 0.913] | 0.537 [0.447, 0.622] | 0.672 [0.551, 0.787] |
| qwen3.5 | noise floor: 27b_r2 vs 27b_r1 | 0.839 [0.787, 0.886] | 0.877 [0.815, 0.928] | 0.776 [0.717, 0.831] | 0.856 [0.795, 0.909] |
| qwen3.5 | 9b_r2 vs 27b_r1 (same seed structure as noise floor) | 0.519 [0.455, 0.584] | 0.579 [0.494, 0.666] | 0.685 [0.614, 0.749] | 0.779 [0.710, 0.847] |
| qwen3.8 | noise floor: 27b_r2 vs 27b_r1 | 0.809 [0.660, 0.924] | 0.881 [0.489, 1.000] | 0.884 [0.775, 0.975] | 0.937 [0.840, 1.000] |
| qwen3.8 | 9b_r2 vs 27b_r1 (same seed structure as noise floor) | 0.487 [0.338, 0.639] | 0.388 [0.108, 0.653] | 0.861 [0.753, 0.953] | 0.874 [0.747, 0.969] |

## Label shares on the random items

| pass | A:evaluation | A:user_probing | A:manipulation | A:none | B:full_refusal | B:partial | B:comply | A:vea |
|---|---|---|---|---|---|---|---|---|
| 27b_r1 | 0.093 | 0.099 | 0.190 | 0.618 | 0.273 | 0.133 | 0.594 | 0.192 |
| 27b_r2 | 0.103 | 0.103 | 0.178 | 0.615 | 0.283 | 0.139 | 0.578 | 0.206 |
| 9b_r1 | 0.184 | 0.099 | 0.099 | 0.618 | 0.269 | 0.147 | 0.584 | 0.283 |
| 9b_r2 | 0.178 | 0.093 | 0.105 | 0.624 | 0.275 | 0.127 | 0.598 | 0.271 |

## Agreement with exp02's thinking-mode judge (validation items, binary)

| pass | VEA kappa | full-refusal kappa |
|---|---|---|
| 27b_r1 | 0.860 [0.789, 0.922] | 0.681 [0.570, 0.775] |
| 27b_r2 | 0.867 [0.795, 0.928] | 0.568 [0.461, 0.675] |
| 9b_r1 | 0.588 [0.490, 0.682] | 0.535 [0.401, 0.657] |
| 9b_r2 | 0.592 [0.492, 0.676] | 0.556 [0.426, 0.669] |

## Decision (FP8 rule)

- kappa within 27B noise floor: {'awareness_4way': False, 'vea_binary': False, 'refusal_3way': False, 'full_refusal_binary': False}
- rates: {'A:vea': (0.277, 0.199, False), 'B:full_refusal': (0.272, 0.278, True), 'B:partial': (0.137, 0.136, True)}
- speedup 2.57x -> **FAIL**
