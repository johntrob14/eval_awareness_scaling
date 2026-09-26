"""Sample Fortress rollouts from a Qwen3 or Gemma 4 model (MLX on Apple Silicon, vLLM on NVIDIA).

Finished rollouts are appended to a JSONL file as they complete, so an
interrupted run resumes where it left off.

    uv run python -m eval_awareness.generate --model Qwen/Qwen3-0.6B --n-rollouts 10
    uv run python -m eval_awareness.generate --model Qwen/Qwen3-32B --n-rollouts 100 --tp 2
    uv run python -m eval_awareness.generate --model google/gemma-4-31B-it --tp 4 --seed 1 \
        --generation-config-sampling --split benign --out results/gemma4_pilot/rollouts/x.jsonl

Gemma 4 (vLLM only) thinks only when <|think|> opens the system turn. Its chat template does
that for enable_thinking=True; a system message containing "<|think|>" is NOT enough, because
without enable_thinking the template also prefills an empty "<|channel>thought\\n<channel|>".
The thought delimiters are special tokens, so decoding keeps special tokens.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from eval_awareness.backends import add_vllm_args, default_backend, load_vllm
from eval_awareness.data import load_fortress
from eval_awareness.io import append_jsonl, read_jsonl

THINK_RE = re.compile(r"^\s*(?:<think>)?(.*?)</think>(.*)$", re.S)
GEMMA_OPEN, GEMMA_CLOSE = "<|channel>thought", "<channel|>"
# token that ends the chain of thought, used to count CoT vs response tokens
THINK_END_TOKEN = {"qwen": "</think>", "gemma": GEMMA_CLOSE}
SPLIT_CODE = {"adversarial": 0, "benign": 1}


def model_slug(model: str) -> str:
    return model.split("/")[-1]


def thinking_style(model: str) -> str:
    return "gemma" if "gemma" in model.lower() else "qwen"


def rollout_seed(base: int, split: str, qid: int, idx: int) -> int:
    """Deterministic per-rollout seed; unique for qid < 1000, idx < 1000."""
    return base * 10_000_000 + SPLIT_CODE[split] * 1_000_000 + qid * 1000 + idx


def split_thinking(text: str) -> tuple[str, str | None]:
    """Return (chain_of_thought, response). response is None if thinking never closed."""
    m = THINK_RE.match(text)
    if not m:
        return text.replace("<think>", "", 1).strip(), None
    return m.group(1).strip(), m.group(2).strip()


def split_thinking_gemma(text: str) -> tuple[str, str | None]:
    """CoT = text between <|channel>thought and <channel|>; response = everything after.

    No thought block at the start gives an empty CoT (a failure). A block that never closes
    (truncated) gives response None.
    """
    body = text.lstrip()
    if not body.startswith(GEMMA_OPEN):
        return "", text.replace("<turn|>", "").strip()
    body = body[len(GEMMA_OPEN):]
    if GEMMA_CLOSE not in body:
        return body.strip(), None
    cot, response = body.split(GEMMA_CLOSE, 1)
    return cot.strip(), response.replace("<turn|>", "").strip()


def chat(tok, text: str, tokenize: bool):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=True, tokenize=tokenize)


def gemma_prompt_ids(tok, text: str) -> list[int]:
    # Token ids, not a string: the rendered string starts with <bos>, and vLLM would add another.
    s = chat(tok, text, tokenize=False)
    return tok(s, add_special_tokens=False).input_ids


def sample_mlx(args, jobs, texts, emit):
    """jobs: [(qid, sample_idx)]; emit([(qid, idx, text, n_tokens, finish_reason, seed, cot_tokens)])."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(args.model)
    prompts = {qid: chat(tok, t, tokenize=True) for qid, t in texts.items()}
    gen = BatchGenerator(
        model,
        max_tokens=args.max_tokens,
        stop_tokens=[[t] for t in tok.eos_token_ids],
        sampler=make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k),
        completion_batch_size=args.batch_size,
        prefill_batch_size=min(8, args.batch_size),
    )
    # Similar-length prompts share a batch, which keeps padded KV-cache memory down.
    jobs = sorted(jobs, key=lambda j: len(prompts[j[0]]))
    uids = gen.insert([prompts[q] for q, _ in jobs], [args.max_tokens] * len(jobs))
    job_of = dict(zip(uids, jobs))
    tokens = {uid: [] for uid in uids}
    while responses := gen.next_generated():
        finished = []
        for r in responses:
            if r.finish_reason != "stop":
                tokens[r.uid].append(r.token)
            if r.finish_reason is None:
                continue
            toks = tokens.pop(r.uid)
            finished.append((*job_of[r.uid], tok.decode(toks), len(toks), r.finish_reason, None, None))
        if finished:
            mx.clear_cache()  # keeps Metal memory from fragmenting on 16GB machines
            emit(finished)
    gen.close()


def sample_vllm(args, jobs, texts, emit):
    from vllm import SamplingParams

    extra = {"seed": args.seed} if args.seed is not None else {}
    if args.text_only:
        extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = load_vllm(args.model, args.tensor_parallel_size, args.gpu_memory_utilization, args.max_model_len,
                         pipeline_parallel_size=args.pipeline_parallel_size, **extra)
    args.gen_started = time.time()  # throughput excludes engine start-up
    tok = llm.get_tokenizer()
    gemma = args.thinking == "gemma"
    end_id = tok.convert_tokens_to_ids(THINK_END_TOKEN[args.thinking])

    def prompt(q):
        return {"prompt_token_ids": gemma_prompt_ids(tok, texts[q])} if gemma else chat(tok, texts[q], tokenize=False)

    def cot_len(ids):
        return ids.index(end_id) if end_id in ids else None

    if args.log_prompt:
        q0 = next(iter(texts))
        ids = gemma_prompt_ids(tok, texts[q0]) if gemma else tok(chat(tok, texts[q0], tokenize=False)).input_ids
        Path(args.log_prompt).write_text(json.dumps({
            "qid": int(q0), "rendered": tok.decode(ids, skip_special_tokens=False),
            "first_tokens": tok.convert_ids_to_tokens(ids[:6]),
            "think_token_present": "<|think|>" in tok.convert_ids_to_tokens(ids[:8]),
        }, indent=2))

    sp = dict(temperature=args.temp, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens,
              skip_special_tokens=not gemma)
    by_q = defaultdict(list)
    for qid, idx in jobs:
        by_q[qid].append(idx)
    qids = list(by_q)
    # Each chunk is one vLLM call; results are written after every chunk.
    for i in range(0, len(qids), args.questions_per_chunk):
        chunk = qids[i: i + args.questions_per_chunk]
        if args.seed is None:
            params = [SamplingParams(n=len(by_q[q]), **sp) for q in chunk]
            outs = llm.generate([prompt(q) for q in chunk], params)
            emit([(q, idx, o.text, len(o.token_ids), o.finish_reason, None, cot_len(list(o.token_ids)))
                  for q, req in zip(chunk, outs) for idx, o in zip(by_q[q], req.outputs)])
        else:  # one request per rollout so each has its own logged seed
            reqs = [(q, idx, rollout_seed(args.seed, args.split, q, idx)) for q in chunk for idx in by_q[q]]
            outs = llm.generate([prompt(q) for q, _, _ in reqs],
                                [SamplingParams(n=1, seed=s, **sp) for _, _, s in reqs])
            emit([(q, idx, o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason, s,
                   cot_len(list(o.outputs[0].token_ids))) for (q, idx, s), o in zip(reqs, outs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--backend", choices=["mlx", "vllm"], default=default_backend())
    ap.add_argument("--n-rollouts", type=int, default=10, help="rollouts per question")
    ap.add_argument("--qids", type=int, nargs="*", help="restrict to these question ids")
    ap.add_argument("--max-tokens", type=int, default=4096)
    # Qwen3's recommended thinking-mode sampling parameters
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", type=Path, help="default: data/rollouts/<model>.jsonl")
    ap.add_argument("--batch-size", type=int, default=16, help="MLX: concurrent sequences")
    ap.add_argument("--questions-per-chunk", type=int, default=50, help="vLLM: questions per generate call")
    ap.add_argument("--split", choices=["adversarial", "benign"], default="adversarial",
                    help="which Fortress prompt to sample from")
    ap.add_argument("--exclude-qids", type=int, nargs="*",
                    help="questions to drop (default: data.EXPLICIT_EVAL_QIDS)")
    ap.add_argument("--thinking", choices=["qwen", "gemma"], help="CoT format (default: from model name)")
    ap.add_argument("--generation-config-sampling", action="store_true",
                    help="take temperature/top-p/top-k from the model's generation_config.json")
    ap.add_argument("--seed", type=int, help="vLLM: base seed; each rollout gets rollout_seed(seed, split, qid, idx)")
    ap.add_argument("--text-only", action="store_true", help="vLLM: disable image/video inputs (multimodal models)")
    ap.add_argument("--retry-truncated", action="store_true",
                    help="regenerate rollouts whose latest row hit a smaller max_tokens; the new row is "
                         "appended and readers keep the last row per (qid, sample_idx)")
    ap.add_argument("--log-prompt", type=Path, help="write one fully rendered prompt to this JSON file")
    add_vllm_args(ap)
    args = ap.parse_args()
    args.thinking = args.thinking or thinking_style(args.model)
    if args.generation_config_sampling:
        from transformers import GenerationConfig

        gc = GenerationConfig.from_pretrained(args.model)
        args.temp, args.top_p, args.top_k = gc.temperature, gc.top_p, gc.top_k

    out = args.out or Path("data/rollouts") / f"{model_slug(args.model)}.jsonl"
    df = load_fortress(exclude_explicit_eval=args.exclude_qids is None)
    if args.exclude_qids is not None:
        df = df[~df.qid.isin(args.exclude_qids)]
    if args.qids:
        df = df[df.qid.isin(args.qids)]

    done, latest = defaultdict(set), {}
    for r in read_jsonl(out):
        done[r["qid"]].add(r["sample_idx"])
        latest[r["qid"], r["sample_idx"]] = r
    if args.retry_truncated:
        for (qid, idx), r in latest.items():
            if r["finish_reason"] == "length" and r.get("max_tokens", 0) < args.max_tokens:
                done[qid].discard(idx)
    jobs = [(qid, i) for qid in df.qid for i in range(args.n_rollouts) if i not in done[qid]]
    print(f"{out}: {sum(map(len, done.values()))} rollouts kept, {len(jobs)} to generate "
          f"({args.backend}, {args.split}, T={args.temp} top_p={args.top_p} top_k={args.top_k})", flush=True)
    if not jobs:
        return

    t0, n_done, n_tok = time.time(), 0, 0
    split = split_thinking_gemma if args.thinking == "gemma" else split_thinking

    def emit(finished):
        nonlocal n_done, n_tok
        rows = []
        for qid, idx, text, n, reason, seed, cot_tokens in finished:
            cot, response = split(text)
            row = {
                "model": args.model, "qid": int(qid), "sample_idx": idx,
                "cot": cot, "response": response, "finish_reason": reason,
                "n_tokens": n, "temp": args.temp, "top_p": args.top_p, "top_k": args.top_k,
            }
            if args.thinking == "gemma" or args.seed is not None:
                row |= {"prompt_id": int(qid), "split": args.split, "rollout_idx": idx, "seed": seed,
                        "cot_tokens": cot_tokens if cot_tokens is not None else n,
                        "response_tokens": n - cot_tokens - 1 if cot_tokens is not None else 0,
                        "max_tokens": args.max_tokens, "thought_empty": not cot, "raw": text}
            rows.append(row)
            n_tok += n
        append_jsonl(out, rows)
        prev, n_done = n_done, n_done + len(rows)
        if n_done // 50 > prev // 50 or n_done == len(jobs):
            el = time.time() - t0
            print(f"{n_done}/{len(jobs)} done, {n_tok / el:.0f} tok/s, "
                  f"eta {el / n_done * (len(jobs) - n_done) / 60:.0f} min", flush=True)

    col = "adversarial_prompt" if args.split == "adversarial" else "benign_prompt"
    texts = dict(zip(df.qid, df[col]))
    t_start = time.time()
    (sample_mlx if args.backend == "mlx" else sample_vllm)(args, jobs, texts, emit)
    # One line per invocation: throughput for budgeting and the exact config for the record.
    el = time.time() - getattr(args, "gen_started", t_start)
    append_jsonl(out.with_suffix(".runs.jsonl"), [{
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)), "wall_s": time.time() - t_start,
        "gen_s": el, "n_rollouts": n_done, "output_tokens": n_tok, "output_tok_per_s": n_tok / el,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }])


if __name__ == "__main__":
    main()
