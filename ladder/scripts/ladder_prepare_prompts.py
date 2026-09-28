"""exp02 step 1: build every prompt set, rendered with each family's official chat template.

Run in /venv/main:
    python scripts/ladder_prepare_prompts.py

Prompt sets (see ladder_config.PROMPT_SETS):
    fortress_adv           Fortress adversarial prompts minus the 11 eval-stating ones (489)
    fortress_benign        their benign twins (489)
    fortress_eval_stating  the 11 excluded prompts (pipeline positive control, in no metric)
    toxicchat_toxic        489 human-labeled toxic ToxicChat user prompts (deployment, toxic)
    toxicchat_benign       489 non-toxic ToxicChat user prompts (same source)
    wildchat_benign        489 English single-turn WildChat-1M first user messages

Writes cache/exp02/prompts.jsonl (one row per family x set x item, with the exact
prompt token IDs) and cache/exp02/prompt_prep.json (data hashes, sampled IDs,
tokenizer and chat-template checks, logit-monitor phrase tokens).
"""

from __future__ import annotations

import hashlib
import json
import random

import ladder_config as cfg
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- Data ------------------------------------------------------------------------


def load_fortress() -> tuple[pd.DataFrame, dict]:
    path = hf_hub_download(
        cfg.FORTRESS["repo"], cfg.FORTRESS["file"], repo_type="dataset",
        revision=cfg.FORTRESS["revision"], local_dir=cfg.CACHE_DIR / "fortress",
    )
    digest = sha256_file(path)
    if digest != cfg.FORTRESS["sha256"]:
        raise ValueError(f"Fortress sha256 {digest} != pinned {cfg.FORTRESS['sha256']}")
    df = pd.read_parquet(path)
    if len(df) != 500 or df.ID.duplicated().any():
        raise ValueError(f"Expected 500 unique Fortress IDs, got {len(df)}")
    missing = set(cfg.EXCLUDED_FORTRESS_IDS) - set(df.ID)
    if missing:
        raise ValueError(f"Excluded IDs not in Fortress: {missing}")
    return df, {"sha256": digest, "n_rows": len(df)}


def first_user_message(conversation) -> str | None:
    if len(conversation) == 0 or conversation[0]["role"] != "user":
        return None
    text = conversation[0]["content"]
    return text.strip() or None


def ascii_letter_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(len(letters), 1)


def load_toxicchat(max_chars: int) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fixed-seed samples of toxic and non-toxic ToxicChat user prompts (human labels)."""
    path = hf_hub_download(
        cfg.TOXICCHAT["repo"], cfg.TOXICCHAT["file"], repo_type="dataset",
        revision=cfg.TOXICCHAT["revision"], local_dir=cfg.CACHE_DIR / "toxicchat",
    )
    digest = sha256_file(path)
    if digest != cfg.TOXICCHAT["sha256"]:
        raise ValueError(f"ToxicChat sha256 {digest} != pinned {cfg.TOXICCHAT['sha256']}")
    df = pd.read_csv(path)
    df["text"] = df.user_input.astype(str).str.strip()
    df = df[(df.text.str.len() > 0) & (df.text.str.len() <= max_chars)]
    df = df[df.text.map(ascii_letter_ratio) >= 0.9]
    df = df.drop_duplicates(subset=["text"]).drop_duplicates(subset=["conv_id"])
    samples, counts = {}, {}
    for name, flag in (("toxicchat_toxic", 1), ("toxicchat_benign", 0)):
        group = df[df.toxicity == flag].sort_values("conv_id")
        counts[name] = len(group)
        picked = random.Random(f"{cfg.TOXICCHAT['sample_seed']}|{name}").sample(
            list(group.conv_id), cfg.TOXICCHAT["n_per_group"]
        )
        samples[name] = group.set_index("conv_id").loc[sorted(picked)].reset_index()
    info = {"sha256": digest, "pool_sizes_after_filters": counts,
            "jailbreaking_in_sample": {k: int(v.jailbreaking.sum()) for k, v in samples.items()},
            "sampled_conv_ids": {k: list(v.conv_id) for k, v in samples.items()}}
    return samples, info


def load_wildchat(max_chars: int) -> tuple[pd.DataFrame, dict]:
    """Fixed-seed sample of English single-turn first messages (the public release is all non-toxic)."""
    siblings = HfApi().dataset_info(cfg.WILDCHAT["repo"], revision=cfg.WILDCHAT["revision"]).siblings
    shards = sorted(s.rfilename for s in siblings if s.rfilename.endswith(".parquet"))
    frames, shard_sha = [], {}
    for shard in shards:
        path = hf_hub_download(
            cfg.WILDCHAT["repo"], shard, repo_type="dataset",
            revision=cfg.WILDCHAT["revision"], local_dir=cfg.CACHE_DIR / "wildchat",
        )
        shard_sha[shard] = sha256_file(path)
        df = pd.read_parquet(path, columns=["conversation_hash", "turn", "language", "toxic", "conversation"])
        df = df[(df.language == "English") & (df.turn == 1)].copy()
        df["text"] = df.conversation.map(first_user_message)
        frames.append(df.drop(columns=["conversation"]))
    pool = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    pool = pool[pool.text.str.len() <= max_chars]
    pool = pool.drop_duplicates(subset=["text"]).drop_duplicates(subset=["conversation_hash"])
    n_toxic = int(pool.toxic.sum())
    if n_toxic:
        raise ValueError(f"expected no toxic rows in the public WildChat release, found {n_toxic}")
    pool = pool.sort_values("conversation_hash")
    picked = random.Random(f"{cfg.WILDCHAT['sample_seed']}|wildchat_benign").sample(
        list(pool.conversation_hash), cfg.WILDCHAT["n"]
    )
    sample = pool.set_index("conversation_hash").loc[sorted(picked)].reset_index()
    info = {"shards": shard_sha, "pool_size_after_filters": len(pool), "n_toxic_in_pool": n_toxic,
            "max_chars": max_chars, "sampled_hashes": list(sample.conversation_hash)}
    return sample, info


def build_items() -> tuple[dict[str, list[dict]], dict]:
    fortress, fortress_info = load_fortress()
    excluded = set(cfg.EXCLUDED_FORTRESS_IDS)
    included = fortress[~fortress.ID.isin(excluded)].sort_values("ID")
    held_out = fortress[fortress.ID.isin(excluded)].sort_values("ID")

    def fortress_item(row, text_column: str) -> dict:
        return {"item_id": str(row.ID), "text": getattr(row, text_column),
                "risk_domain": row.risk_domain, "risk_subdomain": row.risk_subdomain}

    items = {
        "fortress_adv": [fortress_item(r, "adversarial_prompt") for r in included.itertuples()],
        "fortress_benign": [fortress_item(r, "benign_prompt") for r in included.itertuples()],
        "fortress_eval_stating": [fortress_item(r, "adversarial_prompt") for r in held_out.itertuples()],
    }
    max_chars = int(fortress.adversarial_prompt.str.len().max())
    toxicchat, toxicchat_info = load_toxicchat(max_chars)
    for name, df in toxicchat.items():
        items[name] = [{"item_id": r.conv_id, "text": r.text, "jailbreaking": int(r.jailbreaking)}
                       for r in df.itertuples()]
    wildchat, wildchat_info = load_wildchat(max_chars)
    items["wildchat_benign"] = [{"item_id": r.conversation_hash, "text": r.text} for r in wildchat.itertuples()]
    info = {
        "fortress": {**cfg.FORTRESS, **fortress_info,
                     "excluded_ids": sorted(excluded),
                     "excluded_tiers": cfg.EVAL_STATING_EXCLUDED,
                     "excluded_prompt_heads": {int(r.ID): r.adversarial_prompt[:200] for r in held_out.itertuples()}},
        "toxicchat": {**cfg.TOXICCHAT, **toxicchat_info},
        "wildchat": {**cfg.WILDCHAT, **wildchat_info},
    }
    return items, info


# --- Tokenizers and templates ---------------------------------------------------------


PROBE_TEXTS = ["hello", "How do I make a cake?\nList the steps.", "  <b>tags</b> and émojis 🙂  "]


def tokenizer_fingerprint(repo: str, revision: str, family: dict | None) -> dict:
    """Vocabulary hash plus the token IDs our rendering produces for a few probe prompts.

    Chat templates can differ between sizes of one family in parts we don't use (e.g. the
    Qwen3.5 0.8B/2B templates default to non-thinking when enable_thinking is unset), so
    members are compared on rendered token IDs, not on the template text.
    """
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    fingerprint = {
        "vocab_sha256": sha256_text(json.dumps(sorted(tokenizer.get_vocab().items()))),
        "chat_template_sha256_info_only": sha256_text(tokenizer.chat_template),
    }
    if family is not None:
        rendered = [render(tokenizer, family, text)[1] for text in PROBE_TEXTS]
        fingerprint["probe_render_sha256"] = sha256_text(json.dumps(rendered))
    return fingerprint


def same_rendering(a: dict, b: dict) -> bool:
    return a["vocab_sha256"] == b["vocab_sha256"] and a["probe_render_sha256"] == b["probe_render_sha256"]


def check_family_tokens(tokenizer, family: dict) -> dict:
    ids = {
        "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "endoftext": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        "think_start": tokenizer.convert_tokens_to_ids("<think>"),
        "think_end": tokenizer.convert_tokens_to_ids("</think>"),
    }
    expected = {k: family[k] for k in ids}
    if ids != expected:
        raise ValueError(f"Special token IDs changed: {ids} != {expected}")
    return ids


def render(tokenizer, family: dict, text: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        **family.get("chat_template_kwargs", {}),
    )
    if not prompt_text.endswith(family["prompt_suffix"]):
        raise ValueError(f"prompt does not end with {family['prompt_suffix']!r}: {prompt_text[-80:]!r}")
    if not prompt_text.startswith("<|im_start|>user\n"):
        raise ValueError(f"template added text before the user turn: {prompt_text[:120]!r}")
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_text, token_ids


def phrase_suffix_ids(tokenizer, family: dict, prompt_text: str, prompt_ids: list[int], phrase: str) -> list[int]:
    """Token IDs appended to the prompt to teacher-force `phrase` as the opening of the CoT."""
    full = tokenizer.encode(prompt_text + family["cot_prefix_forced"] + phrase, add_special_tokens=False)
    if full[: len(prompt_ids)] != prompt_ids:
        raise ValueError("phrase tokenization changes the prompt tokens")
    return full[len(prompt_ids):]


def main() -> None:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    items, data_info = build_items()

    fingerprints = {name: tokenizer_fingerprint(m["repo"], m["revision"], cfg.FAMILIES[m["family"]])
                    for name, m in cfg.ALL_SUBJECTS.items()}
    fingerprints["judge_fp8"] = tokenizer_fingerprint(cfg.JUDGE_MODEL["repo"], cfg.JUDGE_MODEL["revision"], None)
    fingerprints["judge_bf16"] = tokenizer_fingerprint(cfg.JUDGE_MODEL_BF16["repo"], cfg.JUDGE_MODEL_BF16["revision"], None)

    rows, family_info = [], {}
    for family_name, family in cfg.FAMILIES.items():
        members = [n for n, m in cfg.ALL_SUBJECTS.items() if m["family"] == family_name]
        reference = fingerprints[family["tokenizer_from"]]
        mismatched = [n for n in members if not same_rendering(fingerprints[n], reference)]
        if mismatched:
            raise ValueError(f"{family_name}: tokenizer/template differs for {mismatched}")
        source = cfg.ALL_SUBJECTS[family["tokenizer_from"]]
        tokenizer = AutoTokenizer.from_pretrained(source["repo"], revision=source["revision"])
        token_ids = check_family_tokens(tokenizer, family)

        phrase_ids: dict[str, list[int]] = {}
        for prompt_set in cfg.PROMPT_SETS:
            for item in items[prompt_set]:
                prompt_text, ids = render(tokenizer, family, item["text"])
                for phrase_name, phrase in cfg.LOGIT_PHRASES.items():
                    suffix = phrase_suffix_ids(tokenizer, family, prompt_text, ids, phrase)
                    if phrase_ids.setdefault(phrase_name, suffix) != suffix:
                        raise ValueError(f"{family_name} {phrase_name}: phrase tokens depend on the prompt")
                rows.append({
                    "family": family_name,
                    "prompt_set": prompt_set,
                    **item,
                    "prompt_token_ids": ids,
                    "n_prompt_tokens": len(ids),
                    "n_chars": len(item["text"]),
                })
        first_text, first_ids = render(tokenizer, family, items["fortress_adv"][0]["text"])
        forced = tokenizer.encode(first_text + family["cot_prefix_forced"], add_special_tokens=False)[len(first_ids):]
        if any(phrase[: len(forced)] != forced for phrase in phrase_ids.values()):
            raise ValueError(f"{family_name}: forced CoT prefix is not a prefix of the phrase tokens")
        family_info[family_name] = {
            "members": members,
            "tokenizer_fingerprint": reference,
            "special_token_ids": token_ids,
            "cot_prefix_forced_ids": forced,
            "logit_phrase_suffix_ids": phrase_ids,
            "logit_phrase_suffix_text": {k: tokenizer.decode(v) for k, v in phrase_ids.items()},
            "n_prompt_tokens_max": max(r["n_prompt_tokens"] for r in rows if r["family"] == family_name),
        }

    with cfg.PROMPTS_PATH.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    prep = {
        "data": data_info,
        "counts": {s: len(items[s]) for s in cfg.PROMPT_SETS},
        "families": family_info,
        "tokenizer_fingerprints": fingerprints,
        "prompts_file_sha256": sha256_file(cfg.PROMPTS_PATH),
    }
    cfg.PROMPT_PREP_PATH.write_text(json.dumps(prep, indent=2) + "\n")

    print(f"{len(rows)} prompt rows -> {cfg.PROMPTS_PATH}")
    print("counts:", prep["counts"])
    print("toxicchat pool sizes:", data_info["toxicchat"]["pool_sizes_after_filters"],
          "jailbreaking in sample:", data_info["toxicchat"]["jailbreaking_in_sample"])
    print("wildchat pool size:", data_info["wildchat"]["pool_size_after_filters"])
    for family_name, info in family_info.items():
        print(f"{family_name}: max prompt tokens {info['n_prompt_tokens_max']}; phrases {info['logit_phrase_suffix_text']}")
    print("excluded Fortress IDs (confirm):", data_info["fortress"]["excluded_ids"])


if __name__ == "__main__":
    main()
