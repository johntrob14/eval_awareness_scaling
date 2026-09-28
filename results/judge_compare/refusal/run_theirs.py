"""The collaborator's exp03 Call B (graded refusal) judge, unmodified, on set.{full,cut384}.jsonl.

Their rubric (full_refusal / partial / comply), reasoning-first JSON with schema-constrained output, sampling
T=0.7 / top-p 0.8 / top-k 20 / presence 1.5, max 384 tokens, Qwen3.8-27B quantized to FP8 at load time.
Request = the Fortress adversarial prompt (row qid). Their cut note says "first 384 tokens"; for the 6,000-character
view it is reworded to "first 6000 characters". Only change: gpu_memory_utilization 0.80.
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/jtr2875/eval-awareness/scripts")
os.environ["VLLM_USE_DEEP_GEMM"] = "0"


def main():
    import ladder_config as base
    import exp03_judge as ej
    from eval_awareness.data import load_fortress
    from vllm import LLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    FMT = "reasoning_first"
    D = os.path.dirname(os.path.abspath(__file__))
    prompts = dict(zip(*load_fortress(exclude_explicit_eval=False)[["qid", "adversarial_prompt"]].T.values))
    tok = ej.judge_tokenizer("fp8")
    llm = LLM(model=str(ej.judge_dir("fp8")), tokenizer=str(ej.judge_dir("fp8")), seed=base.VLLM_ENGINE_SEED,
              max_model_len=base.JUDGE_MAX_MODEL_LEN, gpu_memory_utilization=0.80, max_num_seqs=base.VLLM_MAX_NUM_SEQS,
              generation_config="vllm", enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0},
              quantization="fp8")
    for view in ("cut384", "full"):
        rows = [json.loads(l) for l in open(f"{D}/set.{view}.jsonl")]
        out = []
        for rep in range(a.reps):
            calls = []
            for i, r in enumerate(rows):
                msg = ej.call_b_message(prompts[r["qid"]], r["response"], r["answer_cut"], FMT)
                if view == "full":
                    msg = msg.replace(f"first {base.ANSWER_CAP_TOKENS} tokens", "first 6000 characters")
                calls.append({"call": "B", "variant": "main", "sort_key": i, "i": i, "rep": rep,
                              "seed": ej.call_seed(f"refusal|{view}|{i}|rep{rep}", "B", "main"), "message": msg})
            res, timing = ej.run_calls(llm, tok, calls, FMT)
            print(view, "rep", rep, timing, flush=True)
            out += [{"i": c["i"], "rep": c["rep"], "parse_ok": c["parse_ok"], "refusal": c.get("refusal"),
                     "labels": c["labels"]} for c in res]
        with open(f"{D}/theirs.{view}.jsonl", "w") as f:
            for r in sorted(out, key=lambda r: (r["rep"], r["i"])):
                f.write(json.dumps(r) + "\n")
    from exp03_exit import exit_without_teardown
    exit_without_teardown()


if __name__ == "__main__":
    main()
