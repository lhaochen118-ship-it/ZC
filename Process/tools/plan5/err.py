import sys,json
sys.path.insert(0,'/home/user/ZC/src'); import judge, score
from collections import Counter
Rf=score.load('/home/user/ZC/data/raw/public_gt.jsonl')
f=sys.argv[1]; tau=float(sys.argv[2]) if len(sys.argv)>2 else 0.8
rows=[json.loads(l) for l in open(f)]; mode=rows[0]['mode']; cat=Counter()
for r in rows:
    cands=r['cands']; n=r['n_samples']; ch=r['choices'] if mode=='list' else {int(k):x for k,x in r['choices'].items()}
    if judge.unanimous(cands,n): fin,_=judge.vote_decision(cands,n,0.5,'valid'); how='unan'
    else: fin,_,how=judge.decide(cands,n,ch,mode,tau=tau,denom='valid')
    v='abstain' if fin.get('type')=='abstain' else score.judge(fin,Rf[r['id']])
    if v=='correct': continue
    hasr=any(score.judge(c['answer'],Rf[r['id']])=='correct' for c in cands)
    cat[(v,'unan' if how=='unan' else 'split','has_right' if hasr else 'no_right')]+=1
for k,x in sorted(cat.items()): print(k,x)
