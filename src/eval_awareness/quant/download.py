"""Download the pinned weights a set of variants needs (on a compute node; login nodes are for job access only).

    uv run python -m eval_awareness.quant.download --variants Qwen3.5-9B@gptq-w4 Qwen3.5-9B@hf-w4

For every variant: the base model's bf16 snapshot (needed for bf16 runs and as the source of our quantized
checkpoints), and for hf-w4 the off-the-shelf checkpoint, each at its pinned revision, into HF_HUB_CACHE.
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from eval_awareness.quant import config as qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()
    cfg = qc.ladder_config()
    wanted = {}
    for v in args.variants:
        base, condition = qc.split_variant(v)
        m = cfg.ALL_SUBJECTS[base]
        wanted[(m["repo"], m["revision"])] = None
        if condition == "hf-w4":
            wanted[qc.OFF_THE_SHELF[base]] = None
    for repo, revision in wanted:
        path = snapshot_download(repo, revision=revision)
        print(f"{repo}@{revision[:8]} -> {path}", flush=True)


if __name__ == "__main__":
    main()
