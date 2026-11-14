import sys,json,itertools
sys.path.insert(0,'/home/user/ZC/src'); import judge, score
from collections import Counter, defaultdict
Rf=score.load('/home/user/ZC/data/raw/public_gt.jsonl')
d=sys.argv[1]; variants=sys.argv[2].split(',')
V={v:{json.loads(l)['id']:json.loads(l) for l in open(f'{d}/{v}/verdicts.jsonl')} for v in variants}
ids=list(V[variants[0]])
def dist(r):
    cands=r['cands']; k=len(cands); out=defaultdict(float)
    if r['mode']=='list':
        ch=[c for c in r['choices'] if c is not None]
        for c in ch: out[str(c)]+=1/len(ch)
    else:
        for i,x in r['choices'].items():
            x=[c for c in x if c is not None]
            if x: out[str(int(i)+1)]+=sum(1 for c in x if c==1)/len(x)/1.0
        s=sum(out.values())
        if s: out={k2:v/s for k2,v in out.items()}
        else: out={'0':1.0}
    return out
def run(vs, margin):
    c=Counter()
    for i in ids:
        r=V[vs[0]][i]; cands=r['cands']; n=r['n_samples']
        if judge.unanimous(cands,n) or not any(V[v][i]['choices'] for v in vs): fin,_=judge.vote_decision(cands,n,0.5,'valid')
        else:
            tot=defaultdict(float)
            for v in vs:
                for k,x in dist(V[v][i]).items(): tot[k]+=x/len(vs)
            top=max(tot,key=tot.get) if tot else None
            if top is None or tot[top]<margin: fin,_=judge.vote_decision(cands,n,0.5,'valid')
            elif top in ('0',): fin={'type':'abstain'}
            elif top=='U': fin={'type':'unanswerable','answer':None}
            else: fin=cands[int(top)-1]['answer']
        c['abstain' if fin.get('type')=='abstain' else score.judge(fin,Rf[i])]+=1
    return (c['correct']-c['wrong'])/len(ids),dict(c)
res=[]
for m in range(1,len(variants)+1):
    for vs in itertools.combinations(variants,m):
        for margin in (0.0,0.5,0.6,0.7):
            res.append((run(list(vs),margin),vs,margin))
res.sort(key=lambda x:-x[0][0])
for r in res[:8]: print(r)
