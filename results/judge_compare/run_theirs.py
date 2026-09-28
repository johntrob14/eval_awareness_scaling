"""Run the collaborator's exp03 Call A (awareness) judge, unmodified, on our labeled/held-out CoT sets.

Imports their code from /home/jtr2875/eval-awareness/scripts: regex gate + excerpt windows (ladder_judge.cot_windows),
the 4-way rubric (exp03_config.CALL_A_RUBRIC, reasoning_first as chosen in judge_engine_decision.json), JSON-schema
structured output, their sampling (T=0.7, top-p 0.8, top-k 20, presence 1.5, max 384 tokens) and FP8 load-time
quantization of Qwen3.8-27B. Only change: gpu_memory_utilization 0.80 (another user holds ~15 GB per GPU).
Regex-negative CoTs get awareness "none" without a call, as in their pipeline. VEA = evaluation or user_probing.

    HF_HOME=/home/jtr2875/.cache/huggingface uv run python run_theirs.py <set.jsonl> <out.jsonl> --reps 3
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/jtr2875/eval-awareness/scripts")
os.environ["VLLM_USE_DEEP_GEMM"] = "0"
import ladder_config as base
import exp03_config as cfg
from ladder_judge import cot_windows, recall_hit
import exp03_judge as ej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    FMT = "reasoning_first"
    rows = [json.loads(l) for l in open(a.inp)]
    tok = ej.judge_tokenizer("fp8")
    from vllm import LLM
    llm = LLM(model=str(ej.judge_dir("fp8")), tokenizer=str(ej.judge_dir("fp8")), seed=base.VLLM_ENGINE_SEED,
              max_model_len=base.JUDGE_MAX_MODEL_LEN, gpu_memory_utilization=0.80, max_num_seqs=base.VLLM_MAX_NUM_SEQS,
              generation_config="vllm", enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0},
              quantization="fp8")
    out = []
    for rep in range(a.reps):
        calls = []
        for i, r in enumerate(rows):
            key = f"{os.path.basename(a.inp)}|{i}|rep{rep}"
            stub = {"i": i, "rep": rep, "regex_hit": recall_hit(r["cot"])}
            if not stub["regex_hit"]:
                out.append({**stub, "awareness": "none", "vea": False, "quote": None, "window": ""})
                continue
            w = cot_windows(r["cot"], tok)
            calls.append({"call": "A", "variant": "main", "sort_key": i, "seed": ej.call_seed(key, "A", "main"),
                          "message": ej.call_a_message(w, FMT), "i": i, "rep": rep, "regex_hit": True, "window": w})
        res, timing = ej.run_calls(llm, tok, calls, FMT)
        print("rep", rep, timing, flush=True)
        for c in res:
            aw = c.get("awareness")
            out.append({k: c[k] for k in ("i", "rep", "regex_hit", "window", "parse_ok")} |
                       {"awareness": aw, "vea": None if aw is None else aw in cfg.VEA_TYPES,
                        "framing": (c["labels"] or {}).get("framing"), "quote": (c["labels"] or {}).get("quote"),
                        "labels": c["labels"]})
    with open(a.out, "w") as f:
        for r in sorted(out, key=lambda r: (r["rep"], r["i"])):
            f.write(json.dumps(r) + "\n")
    from exp03_exit import exit_without_teardown
    exit_without_teardown()  # kills the engine subprocesses, then os._exit


if __name__ == "__main__":
    main()
