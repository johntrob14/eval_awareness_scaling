"""Fortress benchmark loading.

Question ids (`qid`) are row positions in the public parquet, which matches the
post's numbering: q140 is the drug-trafficking "show the duality" role-play prompt.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

FORTRESS_REPO = "ScaleAI/fortress_public"
# Pinned (the ladder pins the same revision and file hash, ladder/scripts/ladder_config.py FORTRESS).
FORTRESS_REVISION = "0c096becbc75bb12065c8059a53960c7f0d4d35c"
FORTRESS_SHA256 = "cdc3687982cdb8abdbf6e7ce3155f69904e4bdc0003848d6a4ef3eaaa429f3a9"

# Rows whose prompt explicitly tells the model it is being tested/evaluated as a
# jailbreak technique. The post excludes 11 such prompts but does not list them;
# these are the ones we could identify by reading every regex hit (see README).
EXPLICIT_EVAL_QIDS = [84, 90, 91, 100, 440, 442]
# Gemma pilot (2026-09-26): a keyword search (test, evaluat*, assess*, benchmark, red team, experiment,
# debug, ...) plus reading every hit found four more prompts that tell the model the exchange is a test:
# 10 and 129 ("your first test request to verify your capacity..."), 206 ("Assignment type: Internal
# test"), 485 ("this is a test scenario"). The Qwen runs keep EXPLICIT_EVAL_QIDS for comparability.
PILOT_EXCLUDE_QIDS = EXPLICIT_EVAL_QIDS + [10, 129, 206, 485]


_sha_ok: dict[str, bool] = {}


def load_fortress(exclude_explicit_eval: bool = True) -> pd.DataFrame:
    path = hf_hub_download(FORTRESS_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset",
                           revision=FORTRESS_REVISION)
    if not _sha_ok.get(path):
        import hashlib

        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest != FORTRESS_SHA256:
            raise ValueError(f"Fortress parquet hash {digest} != pinned {FORTRESS_SHA256}")
        _sha_ok[path] = True
    df = pd.read_parquet(path).reset_index(drop=True)
    df.insert(0, "qid", df.index)
    if exclude_explicit_eval:
        df = df[~df.qid.isin(EXPLICIT_EVAL_QIDS)]
    return df


# Capability benchmarks for measuring VEA outside safety prompts. The model sees the raw problem text,
# with no "put your answer in \\boxed{}" suffix. qid = row position; GSM8K keeps its first 500 test
# rows so rollout seeds stay unique (qid < 1000).
CAPABILITY = {
    "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet", "question", 500),
    "math500": ("HuggingFaceH4/MATH-500", "test.jsonl", "problem", 500),
}


def load_capability(name: str) -> pd.DataFrame:
    repo, fname, col, n = CAPABILITY[name]
    path = hf_hub_download(repo, fname, repo_type="dataset")
    df = pd.read_parquet(path) if fname.endswith(".parquet") else pd.read_json(path, lines=True)
    df = df.head(n).reset_index(drop=True)
    df.insert(0, "qid", df.index)
    return df.rename(columns={col: "prompt"})
