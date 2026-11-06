"""Judge quantization-study rollouts with the unified v2 judges, loading the judge model once for every pass.

    uv run python -m eval_awareness.quant.judge_all --variants Qwen3.5-4B@bf16 Qwen3.5-4B@gptq-w4 --tp 2
    uv run python -m eval_awareness.quant.judge_all --variants Qwen3.5-4B@bf16.smoke --splits adversarial

The passes and arguments are those of scripts/run_unified.sh, each a call to eval_awareness.judge.main on the
same vLLM engine: vea_cal (low thinking 4,096 tokens, rescue at 16,384, then thinking off at 1,024 for outputs
still unparseable), refusal (F2, thinking off), category (on each VEA quote). Benign splits get refusal only.
Every call passes --redo-invalid, so a rerun skips valid judgments. The judge is Qwen3.8-27B bf16 at the revision
the v2 judgments used. Exits non-zero if a pass raises; the remaining passes are not run.
"""

from __future__ import annotations

import argparse
import sys

from eval_awareness import judge
from eval_awareness.quant import config as qc

JUDGE = "Qwen/Qwen3.8-27B"
JUDGE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VEA_PASSES = [["--judge-thinking", "low", "--judge-max-tokens", "4096"],
              ["--judge-thinking", "low", "--judge-max-tokens", "16384"],
              ["--judge-thinking", "off", "--judge-max-tokens", "1024"]]


def passes(variant: str, split: str, tasks: list[str]) -> list[list[str]]:
    rollouts = qc.rollouts_path(variant, split)
    if not rollouts.exists():
        return []
    out = []
    for task in tasks:
        if split == "benign" and task != "refusal":
            continue
        base = [str(rollouts), "--task", task, "--split", split]
        if task == "vea_cal":
            out += [base + p for p in VEA_PASSES]
        elif task == "refusal":
            out.append(base + ["--judge-max-tokens", "1024"])
        elif task == "category":
            vea = qc.DATA / "judgments" / f"{variant}.{split}.vea_cal.Qwen3.8-27B.jsonl"
            out.append(base + ["--judge-max-tokens", "1024", "--vea-judgments", str(vea)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True, help="e.g. Qwen3.5-4B@gptq-w4 (add .smoke for smoke runs)")
    ap.add_argument("--splits", nargs="+", default=["adversarial", "benign"])
    ap.add_argument("--tasks", nargs="+", default=["vea_cal", "refusal", "category"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--pp", type=int, default=1, help="pipeline stages (3 = all three A100-40GB of a node, TP=1)")
    ap.add_argument("--gpu-memory-utilization", default="0.85")
    ap.add_argument("--max-num-seqs", default="256")
    args = ap.parse_args()
    engine_args = ["--backend", "vllm", "--judge-model", JUDGE, "--judge-revision", JUDGE_REVISION,
                   "--tp", str(args.tp), "--pp", str(args.pp), "--text-only", "--no-prefilter", "--max-model-len", "40960",
                   "--max-num-seqs", args.max_num_seqs, "--gpu-memory-utilization", args.gpu_memory_utilization,
                   "--out-dir", str(qc.DATA / "judgments"), "--redo-invalid"]
    engine = None
    for variant in args.variants:
        for split in args.splits:
            for argv in passes(variant, split, args.tasks):
                print(f"== {variant} {split} {' '.join(argv[1:])}", flush=True)
                engine = judge.main(argv + engine_args, engine=engine) or engine
    print("all judging done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
