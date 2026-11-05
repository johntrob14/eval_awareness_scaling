"""Generate Fortress rollouts for one variant with the ladder's exact prompts, sampling, caps and seeds.

    uv run python -m eval_awareness.quant.generate --variant Qwen3.5-4B@gptq-w4
    uv run python -m eval_awareness.quant.generate --variant Qwen3.5-27B@bf16 --tp 2
    uv run python -m eval_awareness.quant.generate --variant Qwen3.5-4B@bf16 --smoke   # 20 questions x 4

Rollouts 0-9 per adversarial question and 0-1 per benign twin (the unified primary design), written to
QUANT_DATA/rollouts/<variant>.<split>.jsonl in the unified format (qid, sample_idx, status, cot, response, ...).
Requests run in chunks and each chunk is appended, so a killed job resumes and skips (qid, sample_idx) pairs
already written. Every invocation appends its settings, weights, versions and throughput to <file>.runs.jsonl.
Needs QUANT_DATA/prompts.jsonl (python -m eval_awareness.quant.prepare, on a node with network access).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from eval_awareness.io import append_jsonl, read_jsonl
from eval_awareness.quant import config as qc


def ladder():
    cfg = qc.ladder_config()
    import ladder_generate
    return cfg, ladder_generate


def load_prompts(family: str) -> dict[str, list[dict]]:
    by_set: dict[str, list[dict]] = {}
    for line in (qc.DATA / "prompts.jsonl").open():
        row = json.loads(line)
        if row["family"] == family:
            by_set.setdefault(row["prompt_set"], []).append(row)
    return by_set


def git_revision() -> str:
    rev = subprocess.run(["git", "-C", str(qc.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(qc.ROOT), "status", "--porcelain", "src", "ladder/scripts"],
                           capture_output=True, text=True).stdout.strip()
    return rev + ("+dirty" if dirty else "")


def fortress_qids() -> dict[int, int]:
    """Fortress ID -> row position (the unified qid)."""
    from eval_awareness.data import load_fortress
    df = load_fortress(exclude_explicit_eval=False)
    return {int(i): int(q) for q, i in zip(df.qid, df.ID)}


def requests_for(cfg, gen, prompts, split: str, smoke: bool) -> list[dict]:
    prompt_set, n = qc.SPLITS[split]
    items = prompts[prompt_set]
    if smoke:  # the ladder's smoke questions, so smoke runs of every variant see the same 20
        chosen = set(random.Random(gen.SMOKE_SELECTION_SEED).sample([i["item_id"] for i in prompts["fortress_adv"]],
                                                                    gen.SMOKE_QUESTIONS))
        items, n = [i for i in items if i["item_id"] in chosen], gen.SMOKE_ROLLOUTS
    return gen.rollout_requests(items, range(n), "stage1")


def unified_row(variant: str, split: str, qid_of: dict, r: dict) -> dict:
    status = r["status"]
    if status == "scored" and not (r["answer_text"] or "").strip():
        status = "empty_answer"
    return {"model": variant, "source": "quant", "split": split, "qid": qid_of[int(r["item_id"])],
            "sample_idx": r["rollout_index"], "stage": "stage1", "seed": r["seed"], "status": status,
            "attempt": 0, "n_attempts": 1, "n_prompt_tokens": r["n_prompt_tokens"], "n_cot_tokens": r["n_cot_tokens"],
            "n_answer_tokens": r["n_answer_tokens"], "answer_cut": r["answer_cut"],
            "finish_reasons": r["finish_reasons"], "cot": r["cot_text"],
            "response": r["answer_text"] if status == "scored" else None}


def make_llm(variant: str, family: str, args):
    from vllm import LLM

    cfg, _ = ladder()
    w = qc.weights(variant)
    if not Path(w["path"]).exists():
        raise SystemExit(f"{w['path']} missing: download or quantize {variant} first")
    extra = {} if family == "qwen3" else {"limit_mm_per_prompt": {"image": 0, "video": 0}}
    return LLM(model=str(w["path"]), tokenizer=str(w["tokenizer"]), dtype="bfloat16", seed=cfg.VLLM_ENGINE_SEED,
               max_model_len=cfg.VLLM_MAX_MODEL_LEN, gpu_memory_utilization=args.gpu_memory_utilization,
               max_num_seqs=cfg.VLLM_MAX_NUM_SEQS, tensor_parallel_size=args.tp,
               generation_config="vllm",  # sampling is always explicit; never inherit model defaults
               enable_prefix_caching=True, **extra)


def log_prompt(path: Path, tokenizer, request: dict) -> None:
    """The exact first prompt, as token ids and decoded text, so the rendered input can be checked."""
    ids = request["item"]["prompt_token_ids"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompt_set": request["item"]["prompt_set"], "item_id": request["item"]["item_id"],
                                "n_tokens": len(ids), "first_tokens": tokenizer.convert_ids_to_tokens(ids[:8]),
                                "last_tokens": tokenizer.convert_ids_to_tokens(ids[-8:]),
                                "text": tokenizer.decode(ids)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="<base>@<condition>, e.g. Qwen3.5-4B@gptq-w4")
    ap.add_argument("--splits", nargs="+", default=list(qc.SPLITS), choices=list(qc.SPLITS))
    ap.add_argument("--smoke", action="store_true", help="adversarial only, the ladder's 20 smoke questions x 4")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--chunk", type=int, default=1200, help="requests per append")
    args = ap.parse_args()

    cfg, gen = ladder()
    base, condition = qc.split_variant(args.variant)
    family_name = cfg.ALL_SUBJECTS[base]["family"]
    prompts = load_prompts(family_name)
    qid_of = fortress_qids()
    splits = ["adversarial"] if args.smoke else args.splits
    suffix = ".smoke" if args.smoke else ""

    todo = {}
    for split in splits:
        out = qc.rollouts_path(args.variant + suffix, split)
        done = {(r["qid"], r["sample_idx"]) for r in read_jsonl(out)} if out.exists() else set()
        reqs = [r for r in requests_for(cfg, gen, prompts, split, args.smoke)
                if (qid_of[int(r["item"]["item_id"])], r["rollout_index"]) not in done]
        print(f"{args.variant} {split}: {len(done)} done, {len(reqs)} to generate -> {out}", flush=True)
        if reqs:
            todo[split] = (out, reqs)
    if not todo:
        return

    t0 = time.time()
    llm = make_llm(args.variant, family_name, args)
    tokenizer = llm.get_tokenizer()
    load_s = time.time() - t0
    import torch
    import vllm
    for split, (out, reqs) in todo.items():
        log_prompt(qc.DATA / "prompts" / f"{args.variant}.{split}.json", tokenizer, reqs[0])
        t1, n_tok, status = time.time(), 0, {}
        for i in range(0, len(reqs), args.chunk):
            rows = gen.generate(llm, base, reqs[i:i + args.chunk], tokenizer)
            urows = [unified_row(args.variant, split, qid_of, r) for r in rows]
            append_jsonl(out, urows)
            n_tok += sum(r["n_cot_tokens"] + r["n_answer_tokens"] for r in rows)
            for r in urows:
                status[r["status"]] = status.get(r["status"], 0) + 1
            print(f"  {split}: {i + len(rows)}/{len(reqs)} rollouts, {n_tok / (time.time() - t1):.0f} tok/s", flush=True)
        run = {"variant": args.variant, "base": base, "condition": condition, "split": split, "smoke": args.smoke,
               "weights": {k: str(v) for k, v in qc.weights(args.variant).items()},
               "family": family_name, "sampling": cfg.FAMILIES[family_name]["sampling"],
               "generation_mode": cfg.FAMILIES[family_name]["generation_mode"],
               "caps": {"think": cfg.THINK_CAP_TOKENS, "answer": cfg.ANSWER_CAP_TOKENS,
                        "one_pass_max": cfg.QWEN35_ONE_PASS_MAX_TOKENS},
               "engine": {"seed": cfg.VLLM_ENGINE_SEED, "max_model_len": cfg.VLLM_MAX_MODEL_LEN, "tp": args.tp,
                          "gpu_memory_utilization": args.gpu_memory_utilization, "dtype": "bfloat16"},
               "n_generated": len(reqs), "status_counts": status, "generated_tokens": n_tok,
               "seconds": time.time() - t1, "engine_load_s": load_s, "tokens_per_s": n_tok / (time.time() - t1),
               "code_revision": git_revision(), "vllm": vllm.__version__, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0), "n_gpus": torch.cuda.device_count(),
               "host": platform.node(), "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t1)),
               "argv": sys.argv}
        append_jsonl(out.with_suffix(".runs.jsonl"), [run])
        print(f"{args.variant} {split}: done {json.dumps(status)}", flush=True)


if __name__ == "__main__":
    main()
