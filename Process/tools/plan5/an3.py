import json,sys
sys.path.insert(0,'/tmp/claude-0/-home-user-ZC/67cec8b5-c044-570d-a1c2-4ef17349b57a/scratchpad')
from an1 import *
from collections import Counter
def canon(a):
    return json.dumps(a,ensure_ascii=False,sort_keys=True,default=str).lower()
def plain(r,theta=0.5):
    SS=r['samples'].get('samples') or []
    votes=Counter(); rep={}
    for s in SS:
        if not s.get('ok'): continue
        c=canon(s['answer']); votes[c]+=0.5 if s.get('flags') else 1.0; rep.setdefault(c,s['answer'])
    n=max(1,len(SS))
    if not votes: return {'type':'abstain'},0,votes,rep
    top=votes.most_common(1)[0][0]; conf=votes[top]/n
    return (rep[top] if conf>=theta else {'type':'abstain'}),conf,votes,rep
if __name__=='__main__':
    run=sys.argv[1]; L=load(run)
    c=Counter(); cat=Counter(); rows=[]
    for i,r in L.items():
        fin,conf,votes,rep=plain(r)
        v=verdict(fin,R[i]); c[v]+=1
        if v=='correct': continue
        SS=r['samples'].get('samples') or []
        nr=sum(1 for s in SS if s.get('ok') and verdict(s['answer'],R[i])=='correct')
        nok=sum(1 for s in SS if s.get('ok'))
        top_right = votes and verdict(rep[votes.most_common(1)[0][0]],R[i])=='correct'
        k='ref?' if v=='ref_abstain' else ('top_right_but_abstain' if top_right else ('has_right' if nr else 'none_right'))
        cat[k]+=1
        rows.append((k,v,i,r['business_id'],len(SS),nok,nr,round(conf,2),r['question'][:90]))
    print(c, 'PA~',(c['correct']-c['wrong'])/1000); print(cat)
    for x in sorted(rows): print(x)
