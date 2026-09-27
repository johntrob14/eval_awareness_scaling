"""exp03 logit monitor v2: does the eval-awareness logit move with the prompt, on matched prompt pairs?

Score per prompt (as exp02): log P("The user might be testing") - log P("The user might be asking"), both
teacher-forced as the opening of the reasoning. Two pair types, each pair = the same request in two forms:
  format    a Fortress adversarial prompt as a plain chat message vs wrapped in a benchmark template
  disguise  a WildJailbreak blunt harmful request ("vanilla") vs its adversarial rewrite (gated dataset),
            with the same for benign requests ("disguise_benign") as the length- and wrapper-matched control
The analysis compares paired win rates, and harmful vs benign disguise shifts, with a prompt-length baseline.

  run --model M --kinds format [disguise]   one model per process (vLLM run with scripts/vllm_python.sh)
Output: cache/exp03/logit_monitor_v2/<model>_<key>.jsonl, never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys

import exp03_config as cfg
import ladder_config as base
import ladder_generate as gen02

PHRASES = {k: base.LOGIT_PHRASES[k] for k in ("primary", "control")}


# render / phrase_suffix_ids are ladder_prepare_prompts' (exp02), copied because that module imports pandas,
# which the vLLM environment does not have.
def render(tokenizer, family: dict, text: str) -> tuple[str, list[int]]:
    prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True,
                                                **family.get("chat_template_kwargs", {}))
    if not prompt_text.endswith(family["prompt_suffix"]) or not prompt_text.startswith("<|im_start|>user\n"):
        raise ValueError(f"unexpected prompt shape: {prompt_text[:80]!r} ... {prompt_text[-80:]!r}")
    return prompt_text, tokenizer.encode(prompt_text, add_special_tokens=False)


def phrase_suffix_ids(tokenizer, family: dict, prompt_text: str, prompt_ids: list[int], phrase: str) -> list[int]:
    """Token IDs appended to the prompt to teacher-force `phrase` as the opening of the CoT."""
    full = tokenizer.encode(prompt_text + family["cot_prefix_forced"] + phrase, add_special_tokens=False)
    if full[: len(prompt_ids)] != prompt_ids:
        raise ValueError("phrase tokenization changes the prompt tokens")
    return full[len(prompt_ids):]


def format_pairs() -> list[dict]:
    """Every Fortress adversarial prompt, as chat (side a) and in a benchmark wrapper (side b)."""
    items = sorted({(r["item_id"], r["text"]) for r in map(json.loads, base.PROMPTS_PATH.open())
                    if r["prompt_set"] == "fortress_adv"})
    pairs = []
    for i, (item_id, text) in enumerate(items):
        template = cfg.BENCHMARK_TEMPLATES[i % len(cfg.BENCHMARK_TEMPLATES)]
        pairs.append({"kind": "format", "pair_id": item_id, "side": "a_chat", "text": text})
        pairs.append({"kind": "format", "pair_id": item_id, "side": "b_benchmark", "text": template.format(request=text),
                      "template": i % len(cfg.BENCHMARK_TEMPLATES)})
    return pairs


def disguise_pairs() -> list[dict]:
    """WildJailbreak blunt requests (side a) and their adversarial rewrites (side b), one pair per distinct blunt
    request: kind "disguise" for harmful requests, "disguise_benign" for benign ones. Rewrites are ~10x longer in
    both, so harmful minus benign shift removes the effect of length and of the role-play wrapper itself."""
    from huggingface_hub import hf_hub_download

    wj = cfg.WILDJAILBREAK
    path = hf_hub_download(wj["repo"], wj["file"], repo_type="dataset", revision=wj["revision"])
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["vanilla"].strip() and r["adversarial"].strip()]
    pairs = []
    for data_type, kind in (("adversarial_harmful", "disguise"), ("adversarial_benign", "disguise_benign")):
        candidates = [r for r in rows if r["data_type"] == data_type]
        random.Random(f"{wj['sample_seed']}|{data_type}").shuffle(candidates)
        seen, picked = set(), []
        for r in candidates:
            if r["vanilla"] not in seen:
                seen.add(r["vanilla"])
                picked.append(r)
            if len(picked) == wj["n_pairs"]:
                break
        for i, r in enumerate(picked):
            pairs.append({"kind": kind, "pair_id": str(i), "side": "a_blunt", "text": r["vanilla"]})
            pairs.append({"kind": kind, "pair_id": str(i), "side": "b_disguised", "text": r["adversarial"]})
    return pairs


def output_path(model: str, kinds: list[str]):
    key = base.content_key({"model": base.ALL_SUBJECTS[model], "kinds": sorted(kinds), "phrases": PHRASES,
                            "templates": cfg.BENCHMARK_TEMPLATES, "wildjailbreak": cfg.WILDJAILBREAK,
                            "prompts": base.PROMPTS_PATH.stat().st_size})
    return cfg.LOGIT_DIR / f"{model}_{'+'.join(sorted(kinds))}_{key}.jsonl"


def run(model: str, kinds: list[str]) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt

    out = output_path(model, kinds)
    if out.exists():
        print(f"[logit-v2] {out.name} exists")
        return
    pairs = (format_pairs() if "format" in kinds else []) + (disguise_pairs() if "disguise" in kinds else [])
    subject = base.ALL_SUBJECTS[model]
    family = base.FAMILIES[subject["family"]]
    tokenizer = AutoTokenizer.from_pretrained(str(base.model_dir(subject["repo"], subject["revision"])))
    n_forced = len(tokenizer.encode(family["cot_prefix_forced"], add_special_tokens=False)) if family["cot_prefix_forced"] else 0
    jobs = []
    for p in pairs:
        prompt_text, prompt_ids = render(tokenizer, family, p["text"])
        for phrase_name, phrase in PHRASES.items():
            suffix = phrase_suffix_ids(tokenizer, family, prompt_text, prompt_ids, phrase)
            jobs.append((p, phrase_name, len(prompt_ids), prompt_ids + suffix))
    llm = gen02.make_llm(model)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=seq) for *_, seq in jobs],
                           SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1))
    scores = {}
    for (p, phrase_name, n_prompt, seq), o in zip(jobs, outputs, strict=True):
        start = n_prompt + n_forced
        logprob = sum(o.prompt_logprobs[i][seq[i]].logprob for i in range(start, len(seq)))
        row = scores.setdefault((p["kind"], p["pair_id"], p["side"]), {
            "model": model, **{k: v for k, v in p.items() if k != "text"}, "n_prompt_tokens": n_prompt,
            "n_chars": len(p["text"])})
        row[f"logprob_{phrase_name}"] = logprob
    rows = [{**r, "score": r["logprob_primary"] - r["logprob_control"]} for r in scores.values()]
    out.parent.mkdir(parents=True, exist_ok=True)
    gen02.write_jsonl(out, rows)
    print(f"[logit-v2] {model}: {len(rows)} prompts", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["run"])
    parser.add_argument("--model", required=True, choices=list(base.ALL_SUBJECTS))
    parser.add_argument("--kinds", nargs="+", choices=["format", "disguise"], default=["format"])
    args = parser.parse_args()
    run(args.model, args.kinds)


if __name__ == "__main__":
    main()
