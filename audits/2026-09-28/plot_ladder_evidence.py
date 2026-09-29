"""Plot saved legacy ladder estimates; does not pool them with unified results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[2]
s=json.loads((root/'ladder/results/exp03_vea_ladder_v2/final/summary.json').read_text())
models=list(s['per_model'])
fig,axes=plt.subplots(1,3,figsize=(13,7),sharey=True)
contrasts=[('insert_vea_minus_none_R','Insert VEA − insert nothing','#176b9b'),
           ('insert_vea_minus_placebo_R','Insert VEA − manipulation control','#a34710'),
           ('remove_keep_minus_remove_R','Keep − remove natural sentence','#486047')]
y=np.arange(len(models))
for ax,(key,title,color) in zip(axes,contrasts):
 for i,m in enumerate(models):
  v=s['interventions'][m][key];x=100*v['value'];lo,hi=np.array(v['ci'])*100
  ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=3,markersize=5)
 ax.axvline(0,color='0.45',lw=1,linestyle='--')
 ax.set_title(title,fontsize=11,pad=14);ax.set_xlabel('Refusal difference (percentage points)')
 ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(y,models);axes[0].invert_yaxis()
axes[0].set_xlim(-6,36);axes[1].set_xlim(-9,9);axes[2].set_xlim(-9,9)
fig.suptitle('Existing ladder interventions: large sentence effects need not be evaluation-specific',fontsize=14,y=.985)
fig.text(.5,.035,'Historical exp03 judge; not the unified rejudge. 95% prompt-bootstrap intervals, unadjusted for multiple comparisons.\nSelected bases differ by model. Missing outcomes are excluded in the saved estimates.',ha='center',fontsize=9,color='0.3')
fig.tight_layout(rect=[0,.085,1,.96])
fig.savefig(root/'audits/2026-09-28/ladder_causal_evidence.png',dpi=180)
