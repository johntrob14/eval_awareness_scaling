"""CPU-only reproductions of audit findings; no production files are modified."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from eval_awareness import unified as U, intervene_unified as I, judge
from eval_awareness.io import read_jsonl

results={}
results['unparseable_vea']=judge.parse('vea_cal','Output interrupted before verdict')
results['unclosed_judge_thinking']=judge.parse('vea_cal','<think>I should consider whether QUOTE: a test is supported')
# MH checked against a hand-computed unequal-size example: weights 1 and 2.4.
q=pd.DataFrame({'n':[4.,10.],'n1':[2.,4.],'n0':[2.,6.],'r1':[2.,1.],'r0':[0.,3.]})
expected=(1*1+2.4*(-.25))/3.4
assert np.isclose(U.mh_from(q),expected)
results['mh_unequal_sizes_matches_independent_calculation']=True
# A model still generating, with only one available condition.
partial=pd.DataFrame([dict(experiment='insert',condition='vea',base_key='b',prompt_qid=1,R=1.)])
try:I.contrasts(partial)
except Exception as e:results['partial_intervention_analysis']=f'{type(e).__name__}: {e}'
# A cached prefix is treated as a complete build, so appended source rows are ignored.
with tempfile.TemporaryDirectory() as t:
    out=Path(t)
    with patch.object(U,'OUT',out),patch.object(U,'ROOT',out),patch.object(U,'build_mine') as builder:
        p=U.rollouts_path('gemma-4-31B-it','adversarial');p.parent.mkdir();p.write_text('{}\n')
        U.build('gemma-4-31B-it',['adversarial'])
        results['existing_build_never_reads_source']=not builder.called
    p=out/'partial.jsonl';p.write_text('{"qid":1}\n{"qid":')
    try:read_jsonl(p)
    except Exception as e:results['interrupted_append_unreadable']=type(e).__name__
# Status must require a nonempty answer, but builder only tests response is None.
with tempfile.TemporaryDirectory() as t:
    root=Path(t);p=root/'results/gemma4_pilot/rollouts/gemma-4-31B-it.adversarial.jsonl';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(qid=1,sample_idx=0,cot='reasoning',cot_tokens=3,response='',finish_reason='stop'))+'\n')
    fake=SimpleNamespace(encode=lambda s,**kw:list(s),decode=lambda ids:''.join(ids))
    with patch.object(U,'ROOT',root),patch('transformers.AutoTokenizer.from_pretrained',return_value=fake):
        results['empty_answer_status']=U.build_mine('gemma-4-31B-it','adversarial')[0]['status']
print(json.dumps(results,indent=2))
