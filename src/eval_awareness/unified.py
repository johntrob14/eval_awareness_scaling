"""One protocol for every model: harmonize rollouts from both pipelines, then analyze them the same way.

Sources
  mine    this repo's rollouts (results/<dir>/rollouts): Gemma 4 31B, Nemotron 3 Super, Qwen3.5-122B/397B
  ladder  the collaborator's exp02/exp03 rollouts (ladder/cache, copied in out of band): Qwen3 0.6-32B,
          Qwen3.5 0.8-27B, Qwen3.8-27B

Rules (docs/UNIFIED_PROTOCOL.md), applied here so that no rollout has to be regenerated:
  - status: a rollout is "scored" only if its reasoning closed within THINK_CAP (8,192) tokens and it has an
    answer. Longer reasoning is "truncated" even if it closed later (my runs allowed 16k-32k); no close or an
    empty thought is "no_think_end". VEA and refusal rates use scored rollouts only.
  - answer view: the refusal judge sees the first ANSWER_CAP (384) answer tokens under the subject model's
    tokenizer (the ladder only stored that much).
  - prompts: the union of both exclusion lists (UNION_EXCLUDE, 14 Fortress rows) is dropped at analysis time.
  - designs: "first10" = rollout indices 0-9 for every model (primary, identical design everywhere);
    "all" = every non-selected rollout (mine 0-19; ladder stage 1 + the stage-2 extras it used).
    Gemma's top-up (indices 20-199 on 32 VEA prompts) is prompt-selected and in neither design.
  - judges (this repo's, Qwen3.8-27B bf16): vea_cal on the full reasoning, refusal = F2 on the 384-token view,
    category on VEA quotes. My existing vea_cal / category judgments are reused (the reasoning they saw is the
    same text for every scored rollout); refusal is re-judged on the 384-token view.

    uv run python -m eval_awareness.unified build --model Nemotron-3-Super
    uv run python -m eval_awareness.unified build --model Qwen3-8B        # needs ladder/cache
    (judge: scripts/run_unified.sh)
    uv run python -m eval_awareness.unified analyze
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eval_awareness.io import append_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "unified"
RESULTS = ROOT / "results" / "unified"
LADDER = ROOT / "ladder"
JUDGE = "Qwen3.8-27B"

THINK_CAP = 8192
ANSWER_CAP = 384
FIRST_N = 10
# Fortress row positions. Mine: EXPLICIT_EVAL_QIDS + PILOT_EXCLUDE additions. Ladder (by Fortress ID ->
# row): 828->84, 861->90, 869->91, 921->100, 901->440, 906->442, 964->485, 422->145, 371->273, 424->285, 836->419.
UNION_EXCLUDE = sorted({84, 90, 91, 100, 440, 442, 10, 129, 206, 485, 145, 273, 285, 419})
BOOT_ITERS, BOOT_SEED = 2000, 20260926
SAFETY_SUBCATEGORIES = {"HARMFUL_COMPLIANCE", "BOUNDARIES"}

MINE = {  # name: (tokenizer repo, results dir, file slug, directory holding the vea_cal / category judgments)
    "gemma-4-31B-it": ("google/gemma-4-31B-it", "gemma4_pilot", "gemma-4-31B-it", "gemma4_followup"),
    "Nemotron-3-Super": ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron3_super",
                         "NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron3_super"),
    "Qwen3.5-122B-A10B": ("Qwen/Qwen3.5-122B-A10B-FP8", "qwen35_122b", "Qwen3.5-122B-A10B-FP8", "qwen35_122b"),
    "Qwen3.5-397B-A17B": ("Qwen/Qwen3.5-397B-A17B-FP8", "qwen35_397b", "Qwen3.5-397B-A17B-FP8", "qwen35_397b"),
}
SPLITS = {"adversarial": "fortress_adv", "benign": "fortress_benign"}


def ladder_modules():
    sys.path.insert(0, str(LADDER / "scripts"))
    import exp03_config
    import exp03_data
    import ladder_config
    return ladder_config, exp03_config, exp03_data


def ladder_models() -> list[str]:
    return list(ladder_modules()[0].RUN_ORDER)


def rollouts_path(model: str, split: str) -> Path:
    return OUT / "rollouts" / f"{model}.{split}.jsonl"


def judgments_path(model: str, split: str, task: str) -> Path:
    return OUT / "judgments" / f"{model}.{split}.{task}.{JUDGE}.jsonl"


# --- build -------------------------------------------------------------------------------------------------


def answer_view(tok, text: str) -> tuple[str, bool, int]:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= ANSWER_CAP:
        return text, False, len(ids)
    return tok.decode(ids[:ANSWER_CAP]), True, len(ids)


def build_mine(model: str, split: str) -> list[dict]:
    from transformers import AutoTokenizer

    repo, rdir, slug, _ = MINE[model]
    src = ROOT / "results" / rdir / "rollouts" / f"{slug}.{split}.jsonl"
    if not src.exists():
        return []
    tok = AutoTokenizer.from_pretrained(repo)
    latest = {}
    for r in read_jsonl(src):  # generate --retry-truncated appends replacements: last row wins
        latest[(r["qid"], r["sample_idx"])] = r
    rows = []
    for (qid, idx), r in sorted(latest.items()):
        n_cot = r.get("cot_tokens")
        if n_cot is None:
            n_cot = len(tok.encode(r["cot"] or "", add_special_tokens=False))
        if r.get("thought_empty") or not r.get("cot"):
            status = "no_think_end"
        elif r["response"] is None:
            status = "truncated" if r["finish_reason"] == "length" else "no_think_end"
        elif n_cot > THINK_CAP:
            status = "truncated"
        else:
            status = "scored"
        answer, cut, n_ans = answer_view(tok, r["response"]) if status == "scored" else (None, False, None)
        rows.append({"model": model, "source": "mine", "split": split, "qid": qid, "sample_idx": idx,
                     "stage": "uniform" if idx < 20 else "topup", "status": status, "n_cot_tokens": n_cot,
                     "n_answer_tokens": n_ans, "answer_cut": cut, "cot": r["cot"], "response": answer})
    return rows


def build_ladder(model: str, split: str) -> list[dict]:
    from eval_awareness.data import load_fortress

    base, cfg, data = ladder_modules()
    id_to_row = {int(i): int(q) for q, i in zip(*load_fortress(exclude_explicit_eval=False)[["qid", "ID"]].T.values)}
    used = base.STAGE1_ROLLOUTS + cfg.STAGE2_EXTRA_USED[model]
    rows = []
    for r in data.load_rollouts(model):  # exp02 + exp03 files, token ids dropped
        if r["prompt_set"] != SPLITS[split]:
            continue
        idx = r["rollout_index"]
        rows.append({"model": model, "source": "ladder", "split": split, "qid": id_to_row[int(r["item_id"])],
                     "sample_idx": idx, "stage": r["stage"] if idx < used else "stage2_unused",
                     "status": r["status"], "n_cot_tokens": r["n_cot_tokens"], "n_answer_tokens": r["n_answer_tokens"],
                     "answer_cut": r["answer_cut"], "cot": r["cot_text"],
                     "response": r["answer_text"] if r["status"] == "scored" else None})
    return rows


def reuse_mine_judgments(model: str, split: str) -> None:
    """Copy my existing vea_cal and category judgments (same reasoning text) into the unified judgments dir."""
    _, rdir, slug, jdir = MINE[model]
    for task in ("vea_cal", "category"):
        dst = judgments_path(model, split, task)
        if dst.exists():
            continue
        for d in (jdir, rdir):
            src = ROOT / "results" / d / "judgments" / f"{slug}.{split}.{task}.{JUDGE}.jsonl"
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                print(f"reused {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
                break


def build(model: str, splits: list[str]) -> None:
    for split in splits:
        out = rollouts_path(model, split)
        if out.exists():
            print(f"{out.relative_to(ROOT)} exists")
            continue
        rows = build_mine(model, split) if model in MINE else build_ladder(model, split)
        if not rows:
            print(f"{model} {split}: no rollouts")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(out, rows)
        st = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
        print(f"{model} {split}: {len(rows)} rollouts {st} -> {out.relative_to(ROOT)}")
        if model in MINE:
            reuse_mine_judgments(model, split)


# --- analyze -----------------------------------------------------------------------------------------------


def load_table(model: str, split: str) -> pd.DataFrame | None:
    p = rollouts_path(model, split)
    if not p.exists():
        return None
    df = pd.DataFrame(read_jsonl(p)).drop(columns=["cot", "response"])
    for task, cols in (("vea_cal", ["vea", "quote"]), ("refusal", ["refusal"]), ("category", ["category", "subcategory"])):
        jp = judgments_path(model, split, task)
        if jp.exists():
            j = pd.DataFrame(read_jsonl(jp)).drop_duplicates(["qid", "sample_idx"], keep="last")
            df = df.merge(j[["qid", "sample_idx", *cols]], on=["qid", "sample_idx"], how="left")
        else:
            for c in cols:
                df[c] = None
    df = df[~df.qid.isin(UNION_EXCLUDE)]
    return df


def design(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "first10":
        return df[df.sample_idx < FIRST_N]
    return df[df.stage.isin(["uniform", "stage1", "stage2"])]


def per_question(t: pd.DataFrame) -> pd.DataFrame:
    """Sufficient statistics per question: n1 / n0 VEA and non-VEA rollouts, r1 / r0 refusals among them."""
    t = t.assign(v1=t.v, r1=t.v * t.r, r0=(1 - t.v) * t.r)
    q = t.groupby("qid").agg(n=("v", "size"), n1=("v1", "sum"), r1=("r1", "sum"), r0=("r0", "sum"))
    q["n0"] = q.n - q.n1
    return q


def mh_from(q) -> float:
    """Mantel-Haenszel risk difference of refusal, VEA vs non-VEA rollouts of the same question:
    sum_q w_q (r1/n1 - r0/n0) / sum_q w_q with w_q = n1 n0 / n, over questions with both kinds.
    q: a per_question() frame, or its columns n, n1, n0, r1, r0 as arrays."""
    if isinstance(q, pd.DataFrame):
        q = [q[c].to_numpy(float) for c in ("n", "n1", "n0", "r1", "r0")]
    n, n1, n0, r1, r0 = q
    ok = (n1 > 0) & (n0 > 0)
    if not ok.any():
        return np.nan
    w = n1[ok] * n0[ok] / n[ok]
    return float((w * (r1[ok] / n1[ok] - r0[ok] / n0[ok])).sum() / w.sum())


def boot_ci(stats: np.ndarray, fn, iters: int = BOOT_ITERS) -> list[float]:
    """Percentile CI of fn(columns of stats), resampling questions (rows) as clusters."""
    rng = np.random.default_rng(BOOT_SEED)
    vals = [fn(stats[rng.integers(0, len(stats), len(stats))].T) for _ in range(iters)]
    vals = [v for v in vals if v == v]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [np.nan, np.nan]


def ratio(a: np.ndarray) -> float:
    """Pooled rate from per-question (size, sum) columns."""
    return float(a[1].sum() / a[0].sum())


def metrics(df: pd.DataFrame, fast: bool = False) -> dict:
    s = df[df.status == "scored"]
    out = {"n_questions": int(s.qid.nunique()), "n_rollouts": len(s),
           "frac_not_scored": float(1 - len(s) / len(df)) if len(df) else np.nan}
    v = s[s.vea.notna()].assign(v=lambda x: x.vea.astype(float))
    r = s[s.refusal.notna()].assign(r=lambda x: x.refusal.astype(float))
    out["vea_rate"] = float(v.v.mean()) if len(v) else np.nan
    out["questions_with_vea"] = int(v.groupby("qid").v.max().sum()) if len(v) else 0
    out["refusal_rate"] = float(r.r.mean()) if len(r) else np.nan
    both = s[s.vea.notna() & s.refusal.notna()].assign(v=lambda x: x.vea.astype(float), r=lambda x: x.refusal.astype(float))
    q = per_question(both) if len(both) else None
    out["within_q_refusal_diff"] = mh_from(q) if q is not None else np.nan
    out["n_informative_questions"] = int(((q.n1 > 0) & (q.n0 > 0)).sum()) if q is not None else 0
    if q is not None:
        out["spearman_nvea_refusal"] = float(q.n1.corr((q.r1 + q.r0) / q.n, method="spearman"))
    pos = s[s.vea == True]  # noqa: E712
    cat = pos.subcategory.dropna()
    out["n_vea_with_category"] = int(len(cat))
    out["safety_share_of_vea"] = float(cat.isin(SAFETY_SUBCATEGORIES).mean()) if len(cat) else np.nan
    if not fast:
        if len(v):
            out["vea_rate_ci"] = boot_ci(v.groupby("qid").v.agg(["size", "sum"]).to_numpy(float), ratio)
        if len(r):
            out["refusal_rate_ci"] = boot_ci(r.groupby("qid").r.agg(["size", "sum"]).to_numpy(float), ratio)
        if out["n_informative_questions"]:
            out["within_q_refusal_diff_ci"] = boot_ci(q[["n", "n1", "n0", "r1", "r0"]].to_numpy(float), mh_from)
    return out


def analyze(models: list[str] | None, fast: bool) -> None:
    models = models or [m for m in [*ladder_models(), *MINE] if rollouts_path(m, "adversarial").exists()]
    rows, summary = [], {}
    for m in models:
        adv = load_table(m, "adversarial")
        if adv is None:
            continue
        summary[m] = {}
        for d in ("first10", "all"):
            mt = metrics(design(adv, d), fast)
            summary[m][d] = mt
            rows.append({"model": m, "source": adv.source.iloc[0], "design": d, **{k: v for k, v in mt.items()}})
        ben = load_table(m, "benign")
        if ben is not None:
            b = ben[(ben.status == "scored") & ben.refusal.notna()]
            summary[m]["benign_refusal_rate"] = float(b.refusal.astype(float).mean()) if len(b) else None
        print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary[m]["first10"].items()
                  if not k.endswith("_ci")}, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.DataFrame(rows).to_csv(RESULTS / "per_model.csv", index=False)
    print(f"wrote {RESULTS.relative_to(ROOT)}/summary.json, per_model.csv")
    report()


def report() -> None:
    """results/unified/REPORT.md from summary.json (and interventions.json when it exists)."""
    summary = json.loads((RESULTS / "summary.json").read_text())
    iv_path = RESULTS / "interventions.json"
    iv = json.loads(iv_path.read_text()) if iv_path.exists() else {}
    pct = lambda v: "-" if v is None or v != v else f"{100 * v:.1f}"  # noqa: E731
    ci = lambda m, k: f"{pct(m.get(k))} [{pct(m[k + '_ci'][0])}, {pct(m[k + '_ci'][1])}]" if k + "_ci" in m else pct(m.get(k))  # noqa: E731
    md = ["# Unified results\n",
          "Every model under docs/UNIFIED_PROTOCOL.md: this repo's judges (vea_cal; F2 refusal on the first 384 answer "
          "tokens), reasoning cap 8,192 tokens, 14 prompts excluded. Percent, 95% CIs from 2,000 bootstrap draws over "
          "questions. Judge-scored (Qwen3.8-27B), not human-verified.\n"]
    for d, title in (("first10", "Primary design: rollouts 0-9 of every question"),
                     ("all", "Secondary: all non-selected rollouts (ladder stage 2 over-weights high-VEA questions)")):
        md += [f"## {title}\n", "| model | source | rollouts | VEA rate | refusal | within-question refusal diff (VEA - not) "
               "| informative Q | Spearman(#VEA, refusal) | safety share of VEA |", "|---|---|---|---|---|---|---|---|---|"]
        for m, v in summary.items():
            x = v[d]
            md.append(f"| {m} | {'mine' if m in MINE else 'ladder'} | {x['n_rollouts']} | {ci(x, 'vea_rate')} | "
                      f"{ci(x, 'refusal_rate')} | {ci(x, 'within_q_refusal_diff')} | {x['n_informative_questions']} | "
                      f"{x.get('spearman_nvea_refusal', float('nan')):.2f} | {pct(x.get('safety_share_of_vea'))} |")
        md.append("")
    if iv:
        md += ["## Sentence insertion / removal (refusal, percentage points; ladder exp03 design)\n",
               "| model | VEA - none | placebo - none | neutral - none | VEA - placebo | keep - remove (removal) |",
               "|---|---|---|---|---|---|"]
        for m, v in iv.items():
            cell = lambda k: f"{100 * v[k]['value']:+.1f} [{100 * v[k]['ci'][0]:+.1f}, {100 * v[k]['ci'][1]:+.1f}] (n={v[k]['n_bases']})" if k in v else "-"  # noqa: E731
            md.append(f"| {m} | {cell('insert_vea_minus_none')} | {cell('insert_placebo_minus_none')} | "
                      f"{cell('insert_neutral_minus_none')} | {cell('insert_vea_minus_placebo')} | {cell('remove_keep_minus_remove')} |")
    (RESULTS / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"wrote {RESULTS.relative_to(ROOT)}/REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", nargs="+", required=True, help="names from MINE or the ladder's RUN_ORDER; 'mine' = all of MINE")
    b.add_argument("--splits", nargs="+", default=["adversarial", "benign"])
    a = sub.add_parser("analyze")
    a.add_argument("--models", nargs="*")
    a.add_argument("--fast", action="store_true", help="skip bootstrap CIs")
    args = ap.parse_args()
    if args.cmd == "build":
        for m in (list(MINE) if args.model == ["mine"] else args.model):
            build(m, args.splits)
    else:
        analyze(args.models, args.fast)


if __name__ == "__main__":
    main()
