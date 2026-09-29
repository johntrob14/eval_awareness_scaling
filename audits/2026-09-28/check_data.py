"""Read-only evidence audit. Run with .venv/bin/python audits/2026-09-28/check_data.py."""
import json
from collections import Counter
from pathlib import Path
from eval_awareness import unified as U
from eval_awareness.io import read_jsonl
from eval_awareness.pilot import locate_quote

out = {}
saved = json.loads((U.RESULTS / 'summary.json').read_text())
for model, (_, rdir, slug, _) in U.MINE.items():
    df = U.load_table(model, 'adversarial')
    rows = read_jsonl(U.rollouts_path(model,'adversarial'))
    bykey = {(r['qid'],r['sample_idx']): r for r in rows}
    primary = U.design(df,'first10')
    scored = primary[primary.status=='scored']
    got = U.metrics(primary,fast=True)
    delta = {k: [saved[model]['first10'].get(k),v] for k,v in got.items() if saved[model]['first10'].get(k)!=v}
    record = {'rows':len(rows),'unique_keys':len(bykey),'primary_rows':len(primary),
              'primary_scored':len(scored),'primary_vea_labels':int(scored.vea.notna().sum()),
              'primary_refusal_labels':int(scored.refusal.notna().sum()),'recomputed_differences':delta,
              'empty_scored_responses':sum(r['status']=='scored' and not r['response'] for r in rows)}
    for task in ['vea_cal','refusal','category']:
        js=read_jsonl(U.judgments_path(model,'adversarial',task))
        matched=[j for j in js if (j['qid'],j['sample_idx']) in bykey and bykey[j['qid'],j['sample_idx']]['status']=='scored']
        record[task]={'rows':len(js),'duplicate_keys':len(js)-len({(j['qid'],j['sample_idx']) for j in js}),
                      'parse_false_scored':sum(j.get('parse_ok') is False for j in matched)}
        if task=='vea_cal':
            pos=[j for j in matched if j.get('vea')]
            checks=Counter(locate_quote(j['quote'],bykey[j['qid'],j['sample_idx']]['cot'])[0] for j in pos)
            record[task]['quote_checks']=dict(checks)
            record[task]['failed_parse_primary_keys']=[[j['qid'],j['sample_idx']] for j in matched if j.get('parse_ok') is False and j['sample_idx']<10 and j['qid'] not in U.UNION_EXCLUDE]
    src=read_jsonl(U.ROOT/'results'/rdir/'rollouts'/f'{slug}.adversarial.jsonl')
    hist={}; changed=set()
    for r in src:
        k=(r['qid'],r['sample_idx'])
        if k in hist and (r['cot'],r['response'])!=(hist[k]['cot'],hist[k]['response']): changed.add(k)
        hist[k]=r
    record['replaced_source_keys']=len(changed)
    record['replaced_now_scored']=sum(k in bykey and bykey[k]['status']=='scored' for k in changed)
    out[model]=record
print(json.dumps(out,indent=2))
