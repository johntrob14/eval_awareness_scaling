"""Sample Fortress rollouts from a Qwen3 model (MLX on Apple Silicon, vLLM on NVIDIA).

Finished rollouts are appended to a JSONL file as they complete, so an
interrupted run resumes where it left off.

    uv run python -m eval_awareness.generate --model Qwen/Qwen3-0.6B --n-rollouts 10
    uv run python -m eval_awareness.generate --model Qwen/Qwen3-32B --n-rollouts 100 --tp 2
"""

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

from eval_awareness.backends import add_vllm_args, default_backend, load_vllm
from eval_awareness.data import load_fortress
from eval_awareness.io import append_jsonl, read_jsonl

THINK_RE = re.compile(r"^\s*(?:<think>)?(.*?)</think>(.*)$", re.S)


def model_slug(model: str) -> str:
    return model.split("/")[-1]


def split_thinking(text: str) -> tuple[str, str | None]:
    """Return (chain_of_thought, response). response is None if thinking never closed."""
    m = THINK_RE.match(text)
    if not m:
        return text.replace("<think>", "", 1).strip(), None
    return m.group(1).strip(), m.group(2).strip()


def chat(tok, text: str, tokenize: bool):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=True, tokenize=tokenize)


def sample_mlx(args, jobs, texts, emit):
    """jobs: [(qid, sample_idx)]; emit([(qid, idx, text, n_tokens, finish_reason)])."""
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
            finished.append((*job_of[r.uid], tok.decode(toks), len(toks), r.finish_reason))
        if finished:
            mx.clear_cache()  # keeps Metal memory from fragmenting on 16GB machines
            emit(finished)
    gen.close()


def sample_vllm(args, jobs, texts, emit):
    from vllm import SamplingParams

    llm = load_vllm(args.model, args.tensor_parallel_size, args.gpu_memory_utilization, args.max_model_len)
    tok = llm.get_tokenizer()
    by_q = defaultdict(list)
    for qid, idx in jobs:
        by_q[qid].append(idx)
    qids = list(by_q)
    # Each chunk is one vLLM call; results are written after every chunk.
    for i in range(0, len(qids), args.questions_per_chunk):
        chunk = qids[i: i + args.questions_per_chunk]
        params = [SamplingParams(n=len(by_q[q]), temperature=args.temp, top_p=args.top_p,
                                 top_k=args.top_k, max_tokens=args.max_tokens) for q in chunk]
        outs = llm.generate([chat(tok, texts[q], tokenize=False) for q in chunk], params)
        emit([(q, idx, o.text, len(o.token_ids), o.finish_reason)
              for q, req in zip(chunk, outs) for idx, o in zip(by_q[q], req.outputs)])


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
    add_vllm_args(ap)
    args = ap.parse_args()

    out = args.out or Path("data/rollouts") / f"{model_slug(args.model)}.jsonl"
    df = load_fortress()
    if args.qids:
        df = df[df.qid.isin(args.qids)]

    done = defaultdict(set)
    for r in read_jsonl(out):
        done[r["qid"]].add(r["sample_idx"])
    jobs = [(qid, i) for qid in df.qid for i in range(args.n_rollouts) if i not in done[qid]]
    print(f"{out}: {sum(map(len, done.values()))} rollouts exist, {len(jobs)} to generate "
          f"({args.backend})", flush=True)
    if not jobs:
        return

    t0, n_done, n_tok = time.time(), 0, 0

    def emit(finished):
        nonlocal n_done, n_tok
        rows = []
        for qid, idx, text, n, reason in finished:
            cot, response = split_thinking(text)
            rows.append({
                "model": args.model, "qid": int(qid), "sample_idx": idx,
                "cot": cot, "response": response, "finish_reason": reason,
                "n_tokens": n, "temp": args.temp, "top_p": args.top_p, "top_k": args.top_k,
            })
            n_tok += n
        append_jsonl(out, rows)
        prev, n_done = n_done, n_done + len(rows)
        if n_done // 50 > prev // 50 or n_done == len(jobs):
            el = time.time() - t0
            print(f"{n_done}/{len(jobs)} done, {n_tok / el:.0f} tok/s, "
                  f"eta {el / n_done * (len(jobs) - n_done) / 60:.0f} min", flush=True)

    texts = dict(zip(df.qid, df.adversarial_prompt))
    (sample_mlx if args.backend == "mlx" else sample_vllm)(args, jobs, texts, emit)


if __name__ == "__main__":
    main()
