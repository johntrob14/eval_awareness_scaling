"""Build the Fortress prompt sets for the quantization study with the ladder's own code.

    uv run python -m eval_awareness.quant.prepare        # on a node with network access (login node)

Same rendering as ladder/scripts/ladder_prepare_prompts.py (pinned, hash-checked Fortress; each family's chat
template with enable_thinking=True; special-token and per-member rendering checks), restricted to the three
Fortress sets. The ladder's WildChat / ToxicChat sets feed only its logit monitor, which this study does not run.
Writes QUANT_DATA/prompts.jsonl and QUANT_DATA/prompt_prep.json.
"""

from __future__ import annotations

import hashlib
import json

from eval_awareness.quant import config as qc

FORTRESS_SETS = ["fortress_adv", "fortress_benign", "fortress_eval_stating"]


def main() -> None:
    cfg = qc.ladder_config()
    import ladder_prepare_prompts as prep
    from transformers import AutoTokenizer

    fortress, fortress_info = prep.load_fortress()
    from eval_awareness.data import load_fortress
    load_fortress()  # also cache the copy generate.py and judge.py read, for offline compute nodes
    excluded = set(cfg.EXCLUDED_FORTRESS_IDS)
    included = fortress[~fortress.ID.isin(excluded)].sort_values("ID")
    held_out = fortress[fortress.ID.isin(excluded)].sort_values("ID")

    def item(row, column: str) -> dict:
        return {"item_id": str(row.ID), "text": getattr(row, column),
                "risk_domain": row.risk_domain, "risk_subdomain": row.risk_subdomain}

    items = {"fortress_adv": [item(r, "adversarial_prompt") for r in included.itertuples()],
             "fortress_benign": [item(r, "benign_prompt") for r in included.itertuples()],
             "fortress_eval_stating": [item(r, "adversarial_prompt") for r in held_out.itertuples()]}

    rows, family_info = [], {}
    for family_name, family in cfg.FAMILIES.items():
        members = [n for n, m in cfg.ALL_SUBJECTS.items() if m["family"] == family_name]
        prints = {n: prep.tokenizer_fingerprint(cfg.ALL_SUBJECTS[n]["repo"], cfg.ALL_SUBJECTS[n]["revision"], family)
                  for n in members}
        reference = prints[family["tokenizer_from"]]
        mismatched = [n for n in members if not prep.same_rendering(prints[n], reference)]
        if mismatched:
            raise ValueError(f"{family_name}: tokenizer/template differs for {mismatched}")
        source = cfg.ALL_SUBJECTS[family["tokenizer_from"]]
        tokenizer = AutoTokenizer.from_pretrained(source["repo"], revision=source["revision"])
        token_ids = prep.check_family_tokens(tokenizer, family)
        for prompt_set in FORTRESS_SETS:
            for it in items[prompt_set]:
                _, ids = prep.render(tokenizer, family, it["text"])
                rows.append({"family": family_name, "prompt_set": prompt_set, **it, "prompt_token_ids": ids,
                             "n_prompt_tokens": len(ids), "n_chars": len(it["text"])})
        family_info[family_name] = {"members": members, "tokenizer_fingerprint": reference,
                                    "special_token_ids": token_ids}

    qc.DATA.mkdir(parents=True, exist_ok=True)
    path = qc.DATA / "prompts.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    info = {"fortress": {**cfg.FORTRESS, **fortress_info, "excluded_ids": sorted(excluded)},
            "counts": {s: len(items[s]) for s in FORTRESS_SETS}, "families": family_info,
            "prompts_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (qc.DATA / "prompt_prep.json").write_text(json.dumps(info, indent=1))
    print(f"{len(rows)} prompt rows -> {path}; counts {info['counts']}")


if __name__ == "__main__":
    main()
