"""Refusal-judge comparison set: 60 random finished adversarial rollouts per model (seed 20260929).

Two views of each answer, fed to both judges:
  full : the response (my judge cuts it at 6,000 characters; the other judge gets the same 6,000 characters)
  cut  : the first 384 tokens of the response under the subject model's own tokenizer (the collaborator's
         ANSWER_CAP_TOKENS), flagged `answer_cut` when longer
Rows keep the Fortress row `qid` (for the request text); `sample_idx` = 1000 * model index + rollout index so
keys are unique across models.
"""
import json, random
from transformers import AutoTokenizer

SEED, N, CAP, CHARS = 20260929, 60, 384, 6000
SOURCES = [
    ("gemma-4-31B-it", "google/gemma-4-31B-it", "results/gemma4_pilot/rollouts/gemma-4-31B-it.adversarial.jsonl"),
    ("Nemotron-3-Super", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
     "results/nemotron3_super/rollouts/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.adversarial.jsonl"),
    ("Qwen3.5-122B", "Qwen/Qwen3.5-122B-A10B-FP8", "results/qwen35_122b/rollouts/Qwen3.5-122B-A10B-FP8.adversarial.jsonl"),
    ("Qwen3.5-397B", "Qwen/Qwen3.5-397B-A17B-FP8", "results/qwen35_397b/rollouts/Qwen3.5-397B-A17B-FP8.adversarial.jsonl"),
]
rng = random.Random(SEED)
full, cut = [], []
for m_i, (name, repo, path) in enumerate(SOURCES):
    tok = AutoTokenizer.from_pretrained(repo)
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["qid"], r["sample_idx"])] = r
    rows = [r for _, r in sorted(rows.items()) if r.get("response")]
    for r in rng.sample(rows, N):
        ids = tok.encode(r["response"], add_special_tokens=False)
        base = {"model": name, "qid": r["qid"], "sample_idx": 1000 * m_i + r["sample_idx"],
                "orig_idx": r["sample_idx"], "n_answer_tokens": len(ids)}
        full.append(base | {"response": r["response"][:CHARS], "answer_cut": len(r["response"]) > CHARS})
        cut.append(base | {"response": tok.decode(ids[:CAP]) if len(ids) > CAP else r["response"],
                           "answer_cut": len(ids) > CAP})
    print(name, "answers >384 tokens:", sum(c["answer_cut"] for c in cut if c["model"] == name), "of", N)
for fname, rows in (("set.full.jsonl", full), ("set.cut384.jsonl", cut)):
    with open(f"results/judge_compare/refusal/{fname}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
