"""exp03 smoke-test report: judge engine (FP8 vs bf16), judge answer format, MTP, intervention splices.

Run with /venv/main/bin/python after the smoke runs. Writes results/exp03_vea_ladder_v2/smoke/<section>.{json,md}
and, for the judge and MTP sections, the decision files the full run reads. Nothing is overwritten.

  judge   agreement (Cohen's kappa, bootstrap CI over items) between passes on the same 1,468 items:
          bf16 rep1 vs bf16 rep2 (different seeds) is the noise floor; FP8 (rep2 seeds) vs bf16 rep1 has the
          same seed structure; reasoning-first vs label-first. Reference: exp02's thinking-mode judge on the
          478 validation items. Also: rates on the 990 random items, speed.
  mtp     MTP on vs off on Qwen3.8-27B (same 600 requests, exp02 seeds), plus exp02's own rollouts.
  interventions   status counts and example continuations of the intervention smoke run.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import exp03_config as cfg
import ladder_config as base
import numpy as np
from scipy import stats

OUT_DIR = cfg.RESULTS_DIR / "smoke"
RNG_SEED = 20260928
BOOT = 1000


def write_new(name: str, summary: dict, markdown: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in ((OUT_DIR / f"{name}.json", json.dumps(summary, indent=1)), (OUT_DIR / f"{name}.md", markdown)):
        if path.exists():
            raise FileExistsError(f"{path} exists; smoke reports are never overwritten")
        path.write_text(text)


def kappa(a: list, b: list) -> float:
    """Cohen's kappa for two label lists (any number of categories)."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    cats = sorted(set(a) | set(b), key=str)
    observed = np.mean(a == b)
    expected = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def kappa_ci(a: list, b: list) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    draws = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(BOOT))]
    return {"kappa": kappa(a, b), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "agreement": float(np.mean(a == b)), "n": len(a)}


# --- judge -----------------------------------------------------------------------------------------------------

JUDGE_PASSES = {"bf16_r1": ("bf16", "label_first", "rep1"), "bf16_r2": ("bf16", "label_first", "rep2"),
                "fp8_r2": ("fp8", "label_first", "rep2"), "bf16_reasoning_r1": ("bf16", "reasoning_first", "rep1")}
# What is compared on each call type: the labels themselves and the binary outcome Bloom's measures use.
MEASURES = {"A": {"awareness_4way": lambda r: r["awareness"],
                  "vea_binary": lambda r: r["awareness"] in cfg.VEA_TYPES},
            "B": {"refusal_3way": lambda r: r["refusal"],
                  "full_refusal_binary": lambda r: r["refusal"] == cfg.REFUSAL_BINARY}}


def load_pass(engine: str, answer_format: str, rep: str) -> dict[int, dict] | None:
    path = cfg.SMOKE_DIR / f"judge_{engine}_{answer_format}_{rep}.jsonl"
    if not path.exists():
        return None
    return {r["item"]: r for r in map(json.loads, path.open())}


def load_timing(engine: str, answer_format: str, rep: str) -> dict | None:
    path = cfg.SMOKE_DIR / f"judge_{engine}_{answer_format}_{rep}_timing.json"
    return json.loads(path.read_text()) if path.exists() else None


def exp02_thinking_reference() -> dict[tuple[str, str], dict]:
    (path,) = glob.glob(str(base.CACHE_DIR / "judge" / "validation_thinking_*.jsonl"))
    return {(r["rollout_key"], r["call"]): r["labels"] for r in map(json.loads, open(path)) if r["parse_ok"]}


def compare(x: dict, y: dict, items: list[dict], subset: str) -> dict:
    """Kappa per measure between two passes, on items both parsed."""
    out = {}
    for call, measures in MEASURES.items():
        idx = [i for i, it in enumerate(items) if it["call"] == call and (subset == "all" or it["source"] == subset)
               and x[i]["parse_ok"] and y[i]["parse_ok"]]
        for name, f in measures.items():
            out[name] = kappa_ci([f(x[i]) for i in idx], [f(y[i]) for i in idx])
    return out


def rates(p: dict, items: list[dict]) -> dict:
    """Label shares on the random items (Call A: among regex-positive windows; Call B: among scored answers)."""
    out = {}
    for call, field, options in (("A", "awareness", cfg.AWARENESS_TYPES), ("B", "refusal", cfg.REFUSAL_TYPES)):
        rows = [p[i] for i, it in enumerate(items) if it["call"] == call and it["source"] == "random" and p[i]["parse_ok"]]
        out.update({f"{call}:{o}": float(np.mean([r[field] == o for r in rows])) for o in options})
    out["A:vea"] = out["A:evaluation"] + out["A:user_probing"]
    return out


def judge_report() -> None:
    items = json.loads((cfg.SMOKE_DIR / "judge_items.json").read_text())
    passes = {name: load_pass(*spec) for name, spec in JUDGE_PASSES.items()}
    missing = [n for n, p in passes.items() if p is None]
    if missing:
        sys.exit(f"missing smoke passes: {missing}")
    timing = {name: load_timing(*spec) for name, spec in JUDGE_PASSES.items()}
    summary = {"n_items": len(items), "n_random": sum(i["source"] == "random" for i in items),
               "n_validation": sum(i["source"] == "validation" for i in items),
               "timing": timing, "parse_failures": {n: sum(not r["parse_ok"] for r in p.values()) for n, p in passes.items()},
               "truncated": {n: sum(r["finish_reason"] == "length" for r in p.values()) for n, p in passes.items()}}

    # the judge's own "awareness" field vs the label derived from its three flags
    summary["awareness_field_matches_flags"] = {
        n: float(np.mean([r["labels"]["awareness"] == r["awareness"] for r in p.values()
                          if r["call"] == "A" and r["parse_ok"]])) for n, p in passes.items()}

    pairs = {"noise_floor: bf16_r2 vs bf16_r1": ("bf16_r2", "bf16_r1"),
             "fp8_r2 vs bf16_r1 (same seed structure as noise floor)": ("fp8_r2", "bf16_r1"),
             "fp8_r2 vs bf16_r2 (identical seeds)": ("fp8_r2", "bf16_r2"),
             "reasoning_first vs label_first (bf16, rep1 seeds)": ("bf16_reasoning_r1", "bf16_r1")}
    summary["agreement"] = {subset: {label: compare(passes[a], passes[b], items, subset) for label, (a, b) in pairs.items()}
                            for subset in ("all", "random", "validation")}
    summary["rates_random_items"] = {n: rates(p, items) for n, p in passes.items()}

    # agreement with exp02's thinking-mode judge (binary labels) on the validation items
    reference = exp02_thinking_reference()
    ref = {}
    for n, p in passes.items():
        ref[n] = {}
        for call, field, measure in (("A", "eval_aware", "vea_binary"), ("B", "refusal", "full_refusal_binary")):
            idx = [i for i, it in enumerate(items) if it["call"] == call and it["source"] == "validation"
                   and p[i]["parse_ok"] and (it["rollout_key"], call) in reference]
            ref[n][measure] = kappa_ci([MEASURES[call][measure](p[i]) for i in idx],
                                       [bool(reference[(items[i]["rollout_key"], call)][field]) for i in idx])
    summary["vs_exp02_thinking_judge_validation_items"] = ref

    # decision rules fixed before the smoke test (see exp03_config.SMOKE_RATE_TOLERANCE)
    floor = summary["agreement"]["all"]["noise_floor: bf16_r2 vs bf16_r1"]
    fp8 = summary["agreement"]["all"]["fp8_r2 vs bf16_r1 (same seed structure as noise floor)"]
    r = summary["rates_random_items"]
    rate_checks = {}
    for key in ("A:vea", "B:full_refusal"):
        bf16_mean = (r["bf16_r1"][key] + r["bf16_r2"][key]) / 2
        tolerance = max(cfg.SMOKE_RATE_TOLERANCE, abs(r["bf16_r2"][key] - r["bf16_r1"][key]))
        rate_checks[key] = {"fp8": r["fp8_r2"][key], "bf16_mean": bf16_mean, "tolerance": tolerance,
                            "ok": abs(r["fp8_r2"][key] - bf16_mean) <= tolerance}
    speedup = timing["fp8_r2"]["calls_per_s"] / timing["bf16_r2"]["calls_per_s"]
    kappa_ok = {m: fp8[m]["kappa"] >= floor[m]["lo"] for m in floor}
    fp8_pass = speedup > 1.0 and all(kappa_ok.values()) and all(c["ok"] for c in rate_checks.values())
    # Same rule for the answer format: label-first only if faster and it changes labels no more than resampling does.
    formats = summary["agreement"]["all"]["reasoning_first vs label_first (bf16, rep1 seeds)"]
    format_kappa_ok = {m: formats[m]["kappa"] >= floor[m]["lo"] for m in floor}
    format_speedup = timing["bf16_r1"]["calls_per_s"] / timing["bf16_reasoning_r1"]["calls_per_s"]
    label_first_pass = format_speedup > 1.0 and all(format_kappa_ok.values())
    summary["decision"] = {"fp8_speedup_vs_bf16": speedup, "fp8_kappa_within_noise_floor": kappa_ok,
                           "fp8_rate_checks": rate_checks, "fp8_pass": fp8_pass,
                           "engine": "fp8" if fp8_pass else "bf16",
                           "label_first_speedup_vs_reasoning_first": format_speedup,
                           "label_first_kappa_within_noise_floor": format_kappa_ok, "label_first_pass": label_first_pass,
                           "answer_format": "label_first" if label_first_pass else "reasoning_first"}

    md = ["# exp03 judge smoke test\n", (f"{len(items)} items: {summary['n_random']} random exp02 calls "
          f"(90 per model), {summary['n_validation']} exp02 validation items (uncertainty-weighted).\n"),
          "## Speed\n", "| pass | calls/s | mean output tokens | parse failures |", "|---|---|---|---|"]
    for n in JUDGE_PASSES:
        md.append(f"| {n} | {timing[n]['calls_per_s']:.2f} | {timing[n]['mean_output_tokens']:.0f} | "
                  f"{summary['parse_failures'][n]} |")
    md += ["\n## Agreement (Cohen's kappa [95% CI]), all items\n",
           "| comparison | " + " | ".join(floor) + " |", "|---|" + "---|" * len(floor)]
    for label, res in summary["agreement"]["all"].items():
        md.append(f"| {label} | " + " | ".join(f"{v['kappa']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for v in res.values()) + " |")
    md += ["\n## Label shares on the random items\n", "| pass | " + " | ".join(r["bf16_r1"]) + " |",
           "|---|" + "---|" * len(r["bf16_r1"])]
    for n, v in r.items():
        md.append(f"| {n} | " + " | ".join(f"{x:.3f}" for x in v.values()) + " |")
    md += ["\n## Agreement with exp02's thinking-mode judge (validation items, binary)\n",
           "| pass | VEA kappa | full-refusal kappa |", "|---|---|---|"]
    for n, v in ref.items():
        md.append(f"| {n} | {v['vea_binary']['kappa']:.3f} [{v['vea_binary']['lo']:.3f}, {v['vea_binary']['hi']:.3f}] | "
                  f"{v['full_refusal_binary']['kappa']:.3f} [{v['full_refusal_binary']['lo']:.3f}, "
                  f"{v['full_refusal_binary']['hi']:.3f}] |")
    md += ["\n## Decision\n", (f"FP8 speedup {speedup:.2f}x; kappa within noise floor: {kappa_ok}; "
           f"rate checks: { {k: c['ok'] for k, c in rate_checks.items()} } -> engine **{summary['decision']['engine']}**")]
    write_new("judge", summary, "\n".join(md) + "\n")
    decision = {"engine": summary["decision"]["engine"], "answer_format": summary["decision"]["answer_format"],
                "source": str(OUT_DIR / "judge.json")}
    if cfg.JUDGE_ENGINE_DECISION_PATH.exists():
        raise FileExistsError(cfg.JUDGE_ENGINE_DECISION_PATH)
    cfg.JUDGE_ENGINE_DECISION_PATH.write_text(json.dumps(decision, indent=1))
    print("\n".join(md))


# --- candidate judge: Qwen3.5-9B -------------------------------------------------------------------------------

CANDIDATE_PASSES = {"27b_r1": ("bf16", "reasoning_first", "rep1"), "27b_r2": ("bf16", "reasoning_first", "rep2"),
                    "9b_r1": ("qwen35_9b", "reasoning_first", "rep1"), "9b_r2": ("qwen35_9b", "reasoning_first", "rep2")}


def subject_family(model: str) -> str:
    return base.SUBJECTS[model]["family"]


def candidate_report() -> None:
    """Qwen3.5-9B vs Qwen3.8-27B (bf16), both reasoning-first, with the FP8 pass rule.

    Pass: 9B is faster than the production judge (FP8 27B; its reasoning-first speed is estimated from the measured
    bf16 reasoning-first speed times the measured FP8/bf16 label-first speedup), 9B(rep2) vs 27B(rep1) kappa is at
    least the lower CI bound of 27B(rep2) vs 27B(rep1) on every measure, and 9B's VEA and full-refusal rates on the
    random items are within tolerance of 27B's.
    """
    items = json.loads((cfg.SMOKE_DIR / "judge_items.json").read_text())
    passes = {name: load_pass(*spec) for name, spec in CANDIDATE_PASSES.items()}
    missing = [n for n, p in passes.items() if p is None]
    if missing:
        sys.exit(f"missing smoke passes: {missing}")
    timing = {name: load_timing(*spec) for name, spec in CANDIDATE_PASSES.items()}
    fp8_label = load_timing("fp8", "label_first", "rep2")["calls_per_s"]
    bf16_label = load_timing("bf16", "label_first", "rep2")["calls_per_s"]
    bf16_reasoning = (timing["27b_r1"]["calls_per_s"] + timing["27b_r2"]["calls_per_s"]) / 2
    fp8_reasoning_est = bf16_reasoning * fp8_label / bf16_label
    speed_9b = (timing["9b_r1"]["calls_per_s"] + timing["9b_r2"]["calls_per_s"]) / 2

    pairs = {"noise floor: 27b_r2 vs 27b_r1": ("27b_r2", "27b_r1"),
             "9b_r2 vs 27b_r1 (same seed structure as noise floor)": ("9b_r2", "27b_r1"),
             "9b_r1 vs 27b_r1": ("9b_r1", "27b_r1"),
             "9b self: 9b_r2 vs 9b_r1": ("9b_r2", "9b_r1")}
    agreement = {subset: {label: compare(passes[a], passes[b], items, subset) for label, (a, b) in pairs.items()}
                 for subset in ("all", "random", "validation")}

    # Same-family check: Qwen3.5-9B judging Qwen3.5 rollouts vs the other families.
    by_family = {}
    for family in sorted({subject_family(i["model"]) for i in items}):
        idx = [k for k, item in enumerate(items) if subject_family(item["model"]) == family]
        sub = [items[i] for i in idx]
        by_family[family] = {label: compare({k: passes[a][i] for k, i in enumerate(idx)},
                                            {k: passes[b][i] for k, i in enumerate(idx)}, sub, "all")
                             for label, (a, b) in list(pairs.items())[:2]}

    r = {n: rates(p, items) for n, p in passes.items()}
    rate_checks = {}
    for key in ("A:vea", "B:full_refusal", "B:partial"):
        ref_mean = (r["27b_r1"][key] + r["27b_r2"][key]) / 2
        tolerance = max(cfg.SMOKE_RATE_TOLERANCE, abs(r["27b_r2"][key] - r["27b_r1"][key]))
        cand_mean = (r["9b_r1"][key] + r["9b_r2"][key]) / 2
        rate_checks[key] = {"9b_mean": cand_mean, "27b_mean": ref_mean, "tolerance": tolerance,
                            "ok": abs(cand_mean - ref_mean) <= tolerance}

    reference = exp02_thinking_reference()
    vs_exp02 = {}
    for n, p in passes.items():
        vs_exp02[n] = {}
        for call, field, measure in (("A", "eval_aware", "vea_binary"), ("B", "refusal", "full_refusal_binary")):
            idx = [i for i, it in enumerate(items) if it["call"] == call and it["source"] == "validation"
                   and p[i]["parse_ok"] and (it["rollout_key"], call) in reference]
            vs_exp02[n][measure] = kappa_ci([MEASURES[call][measure](p[i]) for i in idx],
                                            [bool(reference[(items[i]["rollout_key"], call)][field]) for i in idx])

    floor = agreement["all"]["noise floor: 27b_r2 vs 27b_r1"]
    cand = agreement["all"]["9b_r2 vs 27b_r1 (same seed structure as noise floor)"]
    kappa_ok = {m: cand[m]["kappa"] >= floor[m]["lo"] for m in floor}
    speedup = speed_9b / fp8_reasoning_est
    decision = {"speedup_vs_fp8_27b_reasoning_first_est": speedup, "kappa_within_noise_floor": kappa_ok,
                "rate_checks": rate_checks,
                "pass": speedup > 1.0 and all(kappa_ok.values()) and all(c["ok"] for c in rate_checks.values())}
    summary = {"timing": timing, "fp8_27b_reasoning_first_calls_per_s_est": fp8_reasoning_est,
               "parse_failures": {n: sum(not x["parse_ok"] for x in p.values()) for n, p in passes.items()},
               "truncated": {n: sum(x["finish_reason"] == "length" for x in p.values()) for n, p in passes.items()},
               "agreement": agreement, "agreement_by_subject_family": by_family, "rates_random_items": r,
               "vs_exp02_thinking_judge_validation_items": vs_exp02, "decision": decision}

    def fmt(v: dict) -> str:
        return f"{v['kappa']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"

    md = ["# exp03 candidate judge: Qwen3.5-9B vs Qwen3.8-27B (bf16), reasoning-first\n",
          f"{len(items)} smoke items. Production judge = Qwen3.8-27B FP8, reasoning-first "
          f"(estimated {fp8_reasoning_est:.2f} calls/s).\n",
          "## Speed\n", "| pass | calls/s | mean output tokens | parse failures | truncated |", "|---|---|---|---|---|"]
    for n in CANDIDATE_PASSES:
        md.append(f"| {n} | {timing[n]['calls_per_s']:.2f} | {timing[n]['mean_output_tokens']:.0f} | "
                  f"{summary['parse_failures'][n]} | {summary['truncated'][n]} |")
    md.append(f"\n9B speedup vs FP8 27B (estimated): **{speedup:.2f}x**\n")
    for subset in ("all", "random", "validation"):
        md += [f"\n## Agreement (Cohen's kappa [95% CI]), {subset} items\n",
               "| comparison | " + " | ".join(floor) + " |", "|---|" + "---|" * len(floor)]
        for label, res in agreement[subset].items():
            md.append(f"| {label} | " + " | ".join(fmt(v) for v in res.values()) + " |")
    md += ["\n## Agreement by subject family (all items)\n", "| family | comparison | " + " | ".join(floor) + " |",
           "|---|---|" + "---|" * len(floor)]
    for family, res in by_family.items():
        for label, v in res.items():
            md.append(f"| {family} | {label} | " + " | ".join(fmt(x) for x in v.values()) + " |")
    md += ["\n## Label shares on the random items\n", "| pass | " + " | ".join(r["27b_r1"]) + " |",
           "|---|" + "---|" * len(r["27b_r1"])]
    for n, v in r.items():
        md.append(f"| {n} | " + " | ".join(f"{x:.3f}" for x in v.values()) + " |")
    md += ["\n## Agreement with exp02's thinking-mode judge (validation items, binary)\n",
           "| pass | VEA kappa | full-refusal kappa |", "|---|---|---|"]
    for n, v in vs_exp02.items():
        md.append(f"| {n} | {fmt(v['vea_binary'])} | {fmt(v['full_refusal_binary'])} |")
    md += ["\n## Decision (FP8 rule)\n", f"- kappa within 27B noise floor: {kappa_ok}",
           f"- rates: { {k: (round(c['9b_mean'], 3), round(c['27b_mean'], 3), c['ok']) for k, c in rate_checks.items()} }",
           f"- speedup {speedup:.2f}x -> **{'PASS' if decision['pass'] else 'FAIL'}**"]
    write_new("judge_candidate_qwen35_9b", summary, "\n".join(md) + "\n")
    print("\n".join(md))


# --- mtp -------------------------------------------------------------------------------------------------------


def mtp_report() -> None:
    import exp03_data as data

    runs = {}
    for k in (0, cfg.MTP_SPECULATIVE_TOKENS):
        path = cfg.SMOKE_DIR / f"mtp_{cfg.MTP_SMOKE_MODEL}_tokens{k}.jsonl"
        runs[k] = {(r["item_id"], r["rollout_index"]): r for r in map(json.loads, path.open())}
        runs[k]["_timing"] = json.loads(path.with_name(path.stem + "_timing.json").read_text())
    keys = [k for k in runs[0] if k != "_timing"]
    exp02 = {(r["item_id"], r["rollout_index"]): r for r in data.load_rollouts(cfg.MTP_SMOKE_MODEL, exp02_only=True)
             if r["stage"] == "stage1" and r["prompt_set"] == "fortress_adv"}
    runs["exp02"] = {k: exp02[k] for k in keys}
    recall = base.compiled(base.RECALL_EVAL_PATTERNS + base.RECALL_PLACEBO_PATTERNS)

    def describe(rows: list[dict]) -> dict:
        cot = np.array([r["n_cot_tokens"] for r in rows])
        return {"mean_cot_tokens": float(cot.mean()), "median_cot_tokens": float(np.median(cot)),
                "regex_hit_rate": float(np.mean([any(p.search(r["cot_text"]) for p in recall) for r in rows])),
                "truncated_rate": float(np.mean([r["status"] != "scored" for r in rows])),
                "mean_answer_tokens": float(np.mean([r["n_answer_tokens"] for r in rows]))}

    rng = np.random.default_rng(RNG_SEED)
    summary = {"n_rollouts": len(keys), "timing": {str(k): runs[k]["_timing"] for k in (0, cfg.MTP_SPECULATIVE_TOKENS)},
               "describe": {str(k): describe([runs[k][x] for x in keys]) for k in runs}}
    diffs = {}
    for other in (cfg.MTP_SPECULATIVE_TOKENS, "exp02"):
        a = np.array([runs[other][x]["n_cot_tokens"] for x in keys], float)
        b = np.array([runs[0][x]["n_cot_tokens"] for x in keys], float)
        ha = np.array([any(p.search(runs[other][x]["cot_text"]) for p in recall) for x in keys], float)
        hb = np.array([any(p.search(runs[0][x]["cot_text"]) for p in recall) for x in keys], float)
        boot = [(a[i].mean() - b[i].mean(), ha[i].mean() - hb[i].mean())
                for i in (rng.integers(0, len(keys), len(keys)) for _ in range(BOOT))]
        d_len, d_hit = np.array(boot).T
        diffs[f"{other} minus no-MTP"] = {
            "mean_cot_diff": float(a.mean() - b.mean()), "mean_cot_ci": [float(np.percentile(d_len, 2.5)), float(np.percentile(d_len, 97.5))],
            "regex_rate_diff": float(ha.mean() - hb.mean()), "regex_rate_ci": [float(np.percentile(d_hit, 2.5)), float(np.percentile(d_hit, 97.5))],
            "ks_pvalue_cot_length": float(stats.ks_2samp(a, b).pvalue),
            "identical_cot_text": float(np.mean([runs[other][x]["cot_text"] == runs[0][x]["cot_text"] for x in keys]))}
    summary["differences"] = diffs
    t0, t1 = summary["timing"]["0"], summary["timing"][str(cfg.MTP_SPECULATIVE_TOKENS)]
    speedup = t1["tokens_per_s"] / t0["tokens_per_s"]
    d = diffs[f"{cfg.MTP_SPECULATIVE_TOKENS} minus no-MTP"]
    same = (d["mean_cot_ci"][0] <= 0 <= d["mean_cot_ci"][1] and d["regex_rate_ci"][0] <= 0 <= d["regex_rate_ci"][1]
            and d["ks_pvalue_cot_length"] > 0.01)
    use = speedup >= cfg.MTP_MIN_SPEEDUP and same
    summary["decision"] = {"speedup_tokens_per_s": speedup, "wall_clock_ratio": t0["seconds"] / t1["seconds"],
                           "outputs_statistically_same": same, "use_mtp": use}
    md = ["# exp03 MTP smoke test (Qwen3.8-27B)\n", f"{len(keys)} rollouts (300 Fortress questions x 2, exp02 seeds).\n",
          f"- no MTP: {t0['seconds'] / 60:.1f} min, {t0['tokens_per_s']:.0f} generated tokens/s",
          (f"- MTP ({cfg.MTP_SPECULATIVE_TOKENS} tokens): {t1['seconds'] / 60:.1f} min, {t1['tokens_per_s']:.0f} tokens/s "
           f"-> {speedup:.2f}x tokens/s, {summary['decision']['wall_clock_ratio']:.2f}x wall clock\n"),
          "| run | mean CoT | median CoT | regex hit | truncated | mean answer |", "|---|---|---|---|---|---|"]
    for k, v in summary["describe"].items():
        md.append(f"| {k} | {v['mean_cot_tokens']:.0f} | {v['median_cot_tokens']:.0f} | {v['regex_hit_rate']:.3f} | "
                  f"{v['truncated_rate']:.3f} | {v['mean_answer_tokens']:.0f} |")
    md.append("")
    for label, v in diffs.items():
        md.append(f"- {label}: CoT {v['mean_cot_diff']:+.0f} tokens [{v['mean_cot_ci'][0]:+.0f}, {v['mean_cot_ci'][1]:+.0f}], "
                  f"regex rate {v['regex_rate_diff']:+.3f} [{v['regex_rate_ci'][0]:+.3f}, {v['regex_rate_ci'][1]:+.3f}], "
                  f"KS p={v['ks_pvalue_cot_length']:.2f}, identical CoT text {v['identical_cot_text']:.1%}")
    md.append(f"\nDecision: use MTP = **{use}**")
    write_new("mtp", summary, "\n".join(md) + "\n")
    if cfg.MTP_DECISION_PATH.exists():
        raise FileExistsError(cfg.MTP_DECISION_PATH)
    cfg.MTP_DECISION_PATH.write_text(json.dumps({"mtp_tokens": cfg.MTP_SPECULATIVE_TOKENS if use else 0,
                                                 "source": str(OUT_DIR / "mtp.json")}, indent=1))
    print("\n".join(md))


# --- interventions ---------------------------------------------------------------------------------------------


def interventions_report(model: str, n_bases: int) -> None:
    path = cfg.SMOKE_DIR / f"interventions_{model}_first{n_bases}.jsonl"
    rows = [json.loads(line) for line in path.open()]
    judged_path = path.with_name(path.stem + "_callB.jsonl")
    judged = {}
    if judged_path.exists():
        judged = {r["continuation_key"]: r.get("refusal") for r in map(json.loads, judged_path.open())}
    counts = {}
    for r in rows:
        key = f"{r['experiment']}:{r['condition']}"
        c = counts.setdefault(key, {"n": 0, "scored": 0, "mean_new_cot_tokens": 0.0, "full_refusal": 0, "judged": 0})
        c["n"] += 1
        c["scored"] += r["status"] == "scored"
        c["mean_new_cot_tokens"] += r["n_new_cot_tokens"] / 8 / n_bases
        verdict = judged.get(f"{r['base_key']}|{r['experiment']}|{r['condition']}|{r['continuation']}")
        if verdict:
            c["judged"] += 1
            c["full_refusal"] += verdict == cfg.REFUSAL_BINARY
    md = [f"# exp03 intervention smoke ({model}, first {n_bases} bases per experiment, 8 continuations each)\n",
          "| experiment:condition | continuations | scored | mean new CoT tokens | full refusals / judged |",
          "|---|---|---|---|---|"]
    for k, c in counts.items():
        md.append(f"| {k} | {c['n']} | {c['scored']} | {c['mean_new_cot_tokens']:.0f} | {c['full_refusal']}/{c['judged']} |")
    md.append("\n## Example continuations (first 300 characters of the new reasoning)\n")
    for r in rows:
        if r["continuation"] == 0 and r["condition"] in ("vea", "none", "remove", "keep"):
            md.append(f"- {r['experiment']}:{r['condition']} ({r['base_key']}): {r['new_cot_text'][:300]!r}")
    write_new(f"interventions_{model}", {"counts": counts}, "\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("section", choices=["judge", "judge-candidate", "mtp", "interventions"])
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--n-bases", type=int, default=3)
    args = parser.parse_args()
    {"judge": judge_report, "judge-candidate": candidate_report,
     "mtp": mtp_report}.get(args.section, lambda: interventions_report(args.model, args.n_bases))()


if __name__ == "__main__":
    main()
