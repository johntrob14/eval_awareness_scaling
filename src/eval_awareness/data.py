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


def load_fortress(exclude_explicit_eval: bool = True) -> pd.DataFrame:
    path = hf_hub_download(FORTRESS_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path).reset_index(drop=True)
    df.insert(0, "qid", df.index)
    if exclude_explicit_eval:
        df = df[~df.qid.isin(EXPLICIT_EVAL_QIDS)]
    return df
