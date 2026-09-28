"""Raw exp02 examples: the same prompts across all models, and Qwen3.8-27B's placebo effect.

Run in /venv/main:
    python scripts/exp02_raw_examples.py

Writes results/exp02_vea_ladder/examples/raw_examples.md (stops if it exists).

Selection is deterministic and stated in the output, so the examples are not hand-picked:
- Cross-model section: the Fortress questions where the most models produced at least one VEA
  rollout (ties broken by item id), among prompts that contain no test-like word, so the VEA is
  not the model restating the prompt. For each model, the VEA rollout the judge was most confident
  about and, if any, the most confident placebo rollout.
- Qwen3.8 placebo section: the questions contributing most to its pooled placebo difference
  (weight n1*n0/n times the within-question refusal difference). For each, the placebo rollout
  and the "neither" rollout (no VEA, no placebo) whose reasoning length is closest to its group's
  median, so the pair shows the typical case, not the most striking one.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import ladder_analysis as la
import ladder_config as cfg
import ladder_judge

OUT_PATH = cfg.REPO_ROOT / "results" / "exp02_vea_ladder" / "examples" / "raw_examples.md"
MODELS = [m for m in cfg.RUN_ORDER if m != "Qwen3-32B"]
PLACEBO_MODEL = "Qwen3.8-27B"
N_CROSS_MODEL_QUESTIONS = 2
N_PLACEBO_QUESTIONS = 4
PROMPT_CHARS = 700
COT_CHARS = 600
ANSWER_CHARS = 350
TEST_WORDS = re.compile(r"\b(test\w*|evaluat\w*|benchmark\w*|assess\w*|exam\w*)\b", re.IGNORECASE)


def clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " [...]"


def load_table(model: str) -> pd.DataFrame:
    """Scored Fortress-adversarial rollouts with labels, texts and the judge's quotes."""
    df = la.load_rollout_table(model, smoke=False)
    df = df[(df.prompt_set == "fortress_adv") & df.scored].dropna(subset=["V", "P", "R"]).copy()
    texts = {ladder_judge.rollout_key(r): r for r in ladder_judge.load_rollouts(model, smoke=False)}
    labels = [json.loads(line) for line in ladder_judge.label_path(model, smoke=False).open()]
    call_a = {x["rollout_key"]: x for x in labels if x["call"] == "A" and x["variant"] == "main" and x["parse_ok"]}
    df["cot_text"] = df.rollout_key.map(lambda k: texts[k]["cot_text"])
    df["answer_text"] = df.rollout_key.map(lambda k: texts[k]["answer_text"])
    df["quote"] = df.rollout_key.map(lambda k: call_a.get(k, {}).get("labels", {}).get("quote") or "")
    df["judge_reason"] = df.rollout_key.map(lambda k: call_a.get(k, {}).get("labels", {}).get("reasoning") or "")
    df["p_P"] = df.rollout_key.map(lambda k: call_a.get(k, {}).get("p_true", {}).get("placebo"))
    df["group"] = np.select([df.V == 1, df.P == 1], ["VEA", "placebo"], "neither")
    return df


def rollout_block(r: pd.Series, show_cot_opening: bool) -> list[str]:
    verdict = "REFUSED" if r.R == 1 else "complied"
    lines = [f"- **{r.group}**, {verdict}, {r.n_cot_tokens} reasoning tokens (`{r.stage}` #{r.rollout_index})"]
    if r.group == "VEA" and r.quote:
        lines.append(f"  - judge's quote from the reasoning: \"{clip(r.quote, COT_CHARS)}\"")
    if r.group == "placebo" and r.judge_reason:
        lines.append(f"  - judge's reason for the placebo label: \"{clip(r.judge_reason, COT_CHARS)}\"")
    if show_cot_opening or r.group == "neither":
        lines.append(f"  - reasoning opens: \"{clip(r.cot_text, COT_CHARS)}\"")
    lines.append(f"  - answer opens: \"{clip(r.answer_text, ANSWER_CHARS)}\"")
    return lines


def cross_model_section(tables: dict[str, pd.DataFrame], prompts: dict) -> list[str]:
    has_vea = pd.DataFrame({m: t.groupby("item_id").V.max() for m, t in tables.items()}).fillna(0)
    counts = has_vea.sum(axis=1).rename("n_models").reset_index().rename(columns={"index": "item_id"})
    counts["item_num"] = counts.item_id.astype(int)
    counts = counts[~counts.item_id.map(lambda i: bool(TEST_WORDS.search(prompts[("fortress_adv", i)])))]
    chosen = counts.sort_values(["n_models", "item_num"], ascending=[False, True]).item_id[:N_CROSS_MODEL_QUESTIONS]

    out = ["## 1. The same prompts across all models", "",
           (f"Questions: the {N_CROSS_MODEL_QUESTIONS} Fortress prompts without a test-like word where the most "
            "models had at least one VEA rollout. Per model: the question's rates over all its rollouts (the "
            "placebo column counts every placebo flag, including rollouts also labelled VEA), the VEA rollout "
            "with the highest judge confidence, and the most confident placebo-only rollout if there is one."), ""]
    for item in chosen:
        out += [f"### Fortress item {item} (VEA in {int(has_vea.loc[item].sum())} of {len(tables)} models)", "",
                "> " + clip(prompts[("fortress_adv", item)], PROMPT_CHARS), "",
                "| model | rollouts | VEA | placebo | refused |", "|---|---|---|---|---|"]
        for m, t in tables.items():
            q = t[t.item_id == item]
            out.append(f"| {m} | {len(q)} | {q.V.mean():.0%} | {q.P.mean():.0%} | {q.R.mean():.0%} |")
        out.append("")
        for m, t in tables.items():
            q = t[t.item_id == item]
            picks = [q[q.group == "VEA"].sort_values("p_V", ascending=False).head(1),
                     q[q.group == "placebo"].sort_values("p_P", ascending=False).head(1)]
            rows = [r for p in picks for _, r in p.iterrows()]
            if not rows:
                continue
            out.append(f"**{m}**")
            for r in rows:
                out += rollout_block(r, show_cot_opening=False)
            out.append("")
    return out


def placebo_contributions(t: pd.DataFrame) -> pd.DataFrame:
    """Per question: placebo vs neither refusal difference and its Mantel-Haenszel weight."""
    d = t[t.group != "VEA"]
    g = d.groupby(["item_id", "group"]).R.agg(["size", "mean"]).unstack("group")
    g = g.dropna()
    n1, n0 = g[("size", "placebo")], g[("size", "neither")]
    g = pd.DataFrame({"n_placebo": n1, "n_neither": n0,
                      "refusal_placebo": g[("mean", "placebo")], "refusal_neither": g[("mean", "neither")]})
    g["weight"] = g.n_placebo * g.n_neither / (g.n_placebo + g.n_neither)
    g["difference"] = g.refusal_placebo - g.refusal_neither
    g["contribution"] = g.weight * g.difference
    return g.sort_values("contribution", ascending=False)


def typical(rows: pd.DataFrame) -> pd.Series:
    """The rollout whose reasoning length is closest to the group's median."""
    return rows.loc[(rows.n_cot_tokens - rows.n_cot_tokens.median()).abs().idxmin()]


def placebo_section(t: pd.DataFrame, prompts: dict) -> list[str]:
    contrib = placebo_contributions(t)
    pooled = contrib.contribution.sum() / contrib.weight.sum()
    top_share = contrib.contribution.head(N_PLACEBO_QUESTIONS).sum() / contrib.contribution.sum()
    summary = t.groupby("group").agg(rollouts=("R", "size"), refused=("R", "mean"),
                                     median_reasoning_tokens=("n_cot_tokens", "median"))
    out = [f"## 2. {PLACEBO_MODEL}'s placebo effect", "",
           (f"Pooled within-question refusal difference, placebo vs neither: {pooled:+.1%} over "
            f"{len(contrib)} questions with both kinds of rollout. The {N_PLACEBO_QUESTIONS} questions below "
            f"carry {top_share:.0%} of it."), "",
           "| group | rollouts | refused | median reasoning tokens |", "|---|---|---|---|"]
    out += [f"| {g} | {r.rollouts:.0f} | {r.refused:.0%} | {r.median_reasoning_tokens:.0f} |"
            for g, r in summary.iterrows()]
    out.append("")
    for item, c in contrib.head(N_PLACEBO_QUESTIONS).iterrows():
        q = t[t.item_id == item]
        out += [(f"### Fortress item {item}: placebo {c.refusal_placebo:.0%} refused (n={c.n_placebo:.0f}) vs "
                 f"neither {c.refusal_neither:.0%} (n={c.n_neither:.0f})"), "",
                "> " + clip(prompts[("fortress_adv", item)], PROMPT_CHARS), ""]
        for group in ("placebo", "neither"):
            out += rollout_block(typical(q[q.group == group]), show_cot_opening=True)
        out.append("")
    return out


def main() -> None:
    if OUT_PATH.exists():
        raise SystemExit(f"{OUT_PATH} exists; not overwriting")
    prompts = ladder_judge.prompt_texts()
    tables = {m: load_table(m) for m in MODELS}
    lines = ["# exp02 raw examples", "",
             ("All labels are from the exp02 judge (Qwen3.8-27B). VEA = the reasoning says it may be tested; "
              "placebo = it notices a jailbreak or manipulation without mentioning testing; neither = no regex hit "
              "or the judge found neither. Answers are cut at 384 tokens for the judge, so \"answer opens\" is the "
              "start of what the refusal judge saw."), ""]
    lines += cross_model_section(tables, prompts)
    lines += placebo_section(tables[PLACEBO_MODEL], prompts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
