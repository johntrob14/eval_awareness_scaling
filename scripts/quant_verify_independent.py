# Usage: python3 scripts/quant_verify_independent.py data/quant_tacc   (set BASES below to change the model set)
# Independent recomputation from raw JSONL (no repo code). Rollouts 0-9, status scored, valid vea_cal and refusal labels
# (last line wins per (qid, sample_idx)). Rates pooled over questions common to all variants of the 8 seed models.
import json, collections, random, statistics as st, sys
from math import comb
D = sys.argv[1]
J = "Qwen3.8-27B"
BASES = ["Qwen3.5-0.8B","Qwen3.5-2B","Qwen3.5-4B","Qwen3.5-9B","Qwen3.5-27B","Qwen3-8B","Qwen3-14B","Qwen3-32B"]
SEEDS = ["gptq-w4","gptq-w4-calB","gptq-w4-calC"]
def last(p):
    d = {}
    for l in open(p):
        r = json.loads(l); d[(r["qid"], r["sample_idx"])] = r
    return d
def load(v):
    roll = [json.loads(l) for l in open(f"{D}/rollouts/{v}.adversarial.jsonl")]
    vea = last(f"{D}/judgments/{v}.adversarial.vea_cal.{J}.jsonl"); ref = last(f"{D}/judgments/{v}.adversarial.refusal.{J}.jsonl")
    q = collections.defaultdict(lambda: [0, 0, 0]); cot = []; trunc = 0; n10 = 0; scored = 0; labeled = 0
    for r in roll:
        if r["sample_idx"] >= 10: continue
        n10 += 1; trunc += r["status"] == "truncated"
        if r["status"] != "scored": continue
        scored += 1; cot.append(r["n_cot_tokens"])
        k = (r["qid"], r["sample_idx"]); a, b = vea.get(k), ref.get(k)
        if not a or a.get("parse_ok") is False or a.get("vea") is None or not b or b.get("parse_ok") is False or b.get("refusal") is None: continue
        labeled += 1; q[r["qid"]][0] += 1; q[r["qid"]][1] += bool(a["vea"]); q[r["qid"]][2] += bool(b["refusal"])
    return dict(q), {"trunc": trunc / n10, "cot": st.mean(cot), "coverage": labeled / scored}
V = {}
for b in BASES:
    for c in ["bf16", *SEEDS, "gptq-w8", "hf-w4"]:
        try: V[(b, c)] = load(f"{b}@{c}")
        except FileNotFoundError: pass
common = sorted(set.intersection(*(set(q) for q, _ in V.values())))
print("variants", len(V), "common questions", len(common))
def rate(q, qs, i):
    return sum(q[x][i] for x in qs) / sum(q[x][0] for x in qs)
def delta(b, c, qs, i):
    return rate(V[(b, c)][0], qs, i) - rate(V[(b, "bf16")][0], qs, i)
out = {}
for i, name in ((2, "refusal"), (1, "vea")):
    print(f"\n{name}: per-model W4 seeds (pp), mean; W8")
    allck = []
    for b in BASES:
        s = [100 * delta(b, c, common, i) for c in SEEDS]; allck += s
        w8 = 100 * delta(b, "gptq-w8", common, i)
        print(f"  {b:13s} {s[0]:+5.1f} {s[1]:+5.1f} {s[2]:+5.1f}  mean {st.mean(s):+5.2f}  W8 {w8:+5.2f}")
    pooled = st.mean(st.mean(100 * delta(b, c, common, i) for c in SEEDS) for b in BASES)
    k = sum(x < 0 for x in allck); n = len(allck)
    p = 2 * sum(comb(n, j) for j in range(max(k, n - k), n + 1)) / 2 ** n
    rng = random.Random(1); hb = []; qb = []
    for _ in range(2000):
        qs = [rng.choice(common) for _ in common]
        hb.append(st.mean(st.mean(100 * delta(b, rng.choice(SEEDS), qs, i) for _ in range(3)) for b in BASES))
        qb.append(st.mean(st.mean(100 * delta(b, c, qs, i) for c in SEEDS) for b in BASES))
    hb.sort(); qb.sort()
    print(f"  pooled W4 mean {pooled:+.2f}; question-only CI [{qb[50]:+.2f}, {qb[1949]:+.2f}]; "
          f"hierarchical (questions + seeds) CI [{hb[50]:+.2f}, {hb[1949]:+.2f}]; checkpoints negative {k}/{n}, sign test p={min(p,1):.3f}")
    w8 = st.mean(100 * delta(b, "gptq-w8", common, i) for b in BASES)
    print(f"  pooled W8 mean {w8:+.2f}")
print("\nconfounds (W4 seed mean minus bf16): truncation pp, mean scored CoT tokens, label coverage pp")
for b in BASES:
    m0 = V[(b, "bf16")][1]; ms = [V[(b, c)][1] for c in SEEDS]
    print(f"  {b:13s} trunc {100*(st.mean(m['trunc'] for m in ms)-m0['trunc']):+5.1f}  cot {st.mean(m['cot'] for m in ms)-m0['cot']:+6.0f} (bf16 {m0['cot']:.0f})  coverage {100*(st.mean(m['coverage'] for m in ms)-m0['coverage']):+5.2f}")
