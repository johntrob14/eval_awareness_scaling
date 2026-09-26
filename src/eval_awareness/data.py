"""Fortress benchmark loading.

Question ids (`qid`) are row positions in the public parquet, which matches the
post's numbering: q140 is the drug-trafficking "show the duality" role-play prompt.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

FORTRESS_REPO = "ScaleAI/fortress_public"

# Rows whose prompt explicitly tells the model it is being tested/evaluated as a
# jailbreak technique. The post excludes 11 such prompts but does not list them;
# these are the ones we could identify by reading every regex hit (see README).
EXPLICIT_EVAL_QIDS = [84, 90, 91, 100, 440, 442]
# Gemma pilot (2026-09-26): a keyword search (test, evaluat*, assess*, benchmark, red team, experiment,
# debug, ...) plus reading every hit found four more prompts that tell the model the exchange is a test:
# 10 and 129 ("your first test request to verify your capacity..."), 206 ("Assignment type: Internal
# test"), 485 ("this is a test scenario"). The Qwen runs keep EXPLICIT_EVAL_QIDS for comparability.
PILOT_EXCLUDE_QIDS = EXPLICIT_EVAL_QIDS + [10, 129, 206, 485]


def load_fortress(exclude_explicit_eval: bool = True) -> pd.DataFrame:
    path = hf_hub_download(FORTRESS_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path).reset_index(drop=True)
    df.insert(0, "qid", df.index)
    if exclude_explicit_eval:
        df = df[~df.qid.isin(EXPLICIT_EVAL_QIDS)]
    return df
