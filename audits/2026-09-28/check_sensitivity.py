"""Retry/cap audit and exclusion sensitivity; never substitutes new labels for old text."""
import json
from pathlib import Path
from collections import defaultdict
from eval_awareness import unified as U
from eval_awareness.io import read_jsonl
out={}
for m,(_,d,slug,_) in U.MINE.items():
 hist=defaultdict(list)
 for r in read_jsonl(Path('results')/d/'rollouts'/f'{slug}.adversarial.jsonl'):hist[r['qid'],r['sample_idx']].append(r)
 flips=[]; changed=[]
 for k,v in hist.items():
  if len(v)<2:continue
  def scored(r):return bool(r['cot']) and r['response'] is not None and r['cot_tokens']<=8192
  if not scored(v[0]) and scored(v[-1]):flips.append(k)
  if v[0]['cot']!=v[-1]['cot'] or v[0]['response']!=v[-1]['response']:changed.append(k)
 p=U.design(U.load_table(m,'adversarial'),'first10')
 bad=set(changed)
 keep=p[[ (r.qid,r.sample_idx) not in bad for r in p.itertuples() ]]
 keys=['n_rollouts','vea_rate','refusal_rate','within_q_refusal_diff']
 out[m]={'failed_original_scored_retry':len(flips),'primary_failed_original_scored_retry':sum(q not in U.UNION_EXCLUDE and i<10 for q,i in flips),
         'current':{k:v for k,v in U.metrics(p,fast=True).items() if k in keys},
         'exclude_all_changed_retries_sensitivity_only':{k:v for k,v in U.metrics(keep,fast=True).items() if k in keys}}
print(json.dumps(out,indent=2))
