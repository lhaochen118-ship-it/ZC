import json,sys,itertools
sys.path.insert(0,'/tmp/claude-0/-home-user-ZC/67cec8b5-c044-570d-a1c2-4ef17349b57a/scratchpad')
from an1 import *
from an3 import canon
from collections import Counter
runs=['pva3k16','g1v3apf2','pvs1k16','pvs2k16','pvsoup16','pv3bk16','g1v3bpf','g1v4s1pf','g1v4s2pf','q3l70pf','q3a70pf']
LL={r:load(r) for r in runs}
def pooled(i,rs,theta=0.5,denom='all',equal=False):
    votes=Counter(); rep={}; n=0; nv=0
    for rn in rs:
        r=LL[rn].get(i)
        if not r: continue
        SS=r['samples'].get('samples') or []
        w0=1.0/max(1,len(SS)) if equal else 1.0
        for s in SS:
            n+=w0
            if not s.get('ok'): continue
            nv+=w0
            c=canon(s['answer']); votes[c]+=(0.5 if s.get('flags') else 1.0)*w0; rep.setdefault(c,s['answer'])
    if not votes: return {'type':'abstain'}
    top=votes.most_common(1)[0][0]; d=n if denom=='all' else nv
    return rep[top] if votes[top]/max(d,1e-9)>=theta else {'type':'abstain'}
def ev(rs,**kw):
    c=Counter(verdict(pooled(i,rs,**kw),R[i]) for i in R)
    return c['correct'],c['wrong']+c['ref_abstain'],c['abstain'],(c['correct']-c['wrong']-c['ref_abstain'])/1000
for rs in [['pva3k16'],['pva3k16','g1v3apf2'],['pva3k16','pvs1k16'],['pva3k16','pvs2k16'],['pva3k16','pv3bk16'],['pva3k16','pvs1k16','pvs2k16'],['pva3k16','pvs2k16','pv3bk16'],['pva3k16','pvs1k16','pvs2k16','pv3bk16'],['pva3k16','pvs1k16','pvs2k16','pv3bk16','pvsoup16'],runs]:
    print(len(rs),rs[-3:],'all:',ev(rs),'| equal:',ev(rs,equal=True),'| valid-denom:',ev(rs,denom='valid'),'| th.4:',ev(rs,theta=0.4,equal=True), '| th.35 valid',ev(rs,theta=0.35,denom='valid',equal=True))
