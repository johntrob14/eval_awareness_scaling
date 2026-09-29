"""Inspect saved splice spans with cached tokenizers only; no model loading."""
import json,os
from collections import Counter
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_CACHE']='/home/jtr2875/.cache/huggingface/hub'
from transformers import AutoTokenizer
from eval_awareness import unified as U
from eval_awareness.io import read_jsonl
out={}
for model,(repo,*_) in U.MINE.items():
 tok=AutoTokenizer.from_pretrained(repo,local_files_only=True)
 b=json.loads((U.OUT/'interventions'/f'{model}.bases.json').read_text())
 texts={(r['qid'],r['sample_idx']):r['cot'] for r in read_jsonl(U.rollouts_path(model,'adversarial'))}
 missing=[]; lengths=[]; starts=[]
 for r in b['remove']:
  ids=tok.encode(texts[r['qid'],r['sample_idx']],add_special_tokens=False)
  span=tok.decode(ids[r['cut_remove']:r['cut_keep']])
  if r['quote'] not in span:missing.append([r['qid'],r['sample_idx'],r['cut_remove'],r['cut_keep']])
  lengths.append(r['cut_keep']-r['cut_remove']);starts.append(r['cut_remove']==0)
 out[model]={'removal_bases':len(lengths),'span_does_not_contain_full_quote':missing,
             'remove_from_start':sum(starts),'removed_tokens_min_median_max': [min(lengths),sorted(lengths)[len(lengths)//2],max(lengths)]}
 print(model, out[model],flush=True)
Path('audits/2026-09-28/splice_checks.json').write_text(json.dumps(out,indent=2))
