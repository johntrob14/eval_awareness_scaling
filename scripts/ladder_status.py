"""exp02: what is already on disk, for the unattended tail script. Exit code 0 = yes, 1 = no.

    python scripts/ladder_status.py generated Qwen3.5-9B        # stage 1 + stage 2 (stage 1 for the add-on)
    python scripts/ladder_status.py stage1 Qwen3-32B
    python scripts/ladder_status.py logit Qwen3-32B
    python scripts/ladder_status.py labeled Qwen3.5-9B
    python scripts/ladder_status.py weights Qwen3.8-27B
    python scripts/ladder_status.py hours-left-after-32b-stage1  # prints hours to the T+20 cutoff after the estimate
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import ladder_config as cfg
import ladder_generate
import ladder_judge

T0 = datetime(2026, 9, 26, 20, 30, tzinfo=timezone.utc)  # plan approval
STAGE1_CUTOFF_HOURS = 20.0  # the 32B add-on (stage 1 + judge) must end by T+20: 1 h for analysis and report
QWEN3_32B_VS_14B_TIME = 2.5  # ~2.3x the weights per decode step, similar KV room at 0.92 vs 0.85
QWEN3_32B_JUDGE_HOURS = 0.6


def paths(model: str) -> dict:
    return ladder_generate.cache_paths(model, SimpleNamespace(smoke=False, stage2_extra=cfg.STAGE2_EXTRA_ROLLOUTS))


def generated(model: str) -> bool:
    return paths(model)["stage1" if model in cfg.ADDON_SUBJECTS else "stage2"].exists()


def labeled(model: str) -> bool:
    try:
        return ladder_judge.label_path(model, False).exists()
    except FileNotFoundError:
        return False


def weights(model: str) -> bool:
    """Every safetensors shard listed in the index (or the single file) is present."""
    m = cfg.ALL_SUBJECTS[model]
    d = cfg.model_dir(m["repo"], m["revision"])
    index = d / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        return all((d / s).exists() for s in shards)
    return (d / "model.safetensors").exists()


def hours_left_after_32b_stage1() -> float:
    """Hours to spare at the T+20 cutoff if the 32B stage 1 and its judging start now (negative: skip)."""
    timing = json.loads(paths("Qwen3-14B")["timing"].read_text())
    estimate_h = (QWEN3_32B_VS_14B_TIME * timing["stage1_s"] + 600) / 3600 + QWEN3_32B_JUDGE_HOURS  # +10 min load
    elapsed_h = (time.time() - T0.timestamp()) / 3600
    return STAGE1_CUTOFF_HOURS - elapsed_h - estimate_h


def main() -> None:
    check = sys.argv[1]
    if check == "hours-left-after-32b-stage1":
        print(f"{hours_left_after_32b_stage1():.2f}")
        return
    model = sys.argv[2]
    ok = {"generated": generated, "stage1": lambda m: paths(m)["stage1"].exists(),
          "logit": lambda m: paths(m)["logit"].exists(), "labeled": labeled, "weights": weights}[check](model)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
