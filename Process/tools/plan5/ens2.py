import sys,json
sys.path.insert(0,'/home/user/ZC/src'); import judge, score
from collections import Counter, defaultdict
Rf=score.load('/home/user/ZC/data/raw/public_gt.jsonl')
def dist(r):
    out=defaultdict(float)
    if r['mode']=='list':
        ch=[c for c in r['choices'] if c is not None]
        for c in ch: out[str(c)]+=1/len(ch)
    else:
        for i,x in r['choices'].items():
            x=[c for c in x if c is not None]
            if x: out[str(int(i)+1)]+=sum(1 for c in x if c==1)/len(x)
        s=sum(out.values()); out={k:v/s for k,v in out.items()} if s else {'0':1.0}
    return out
def run(V,vs,margin):
    c=Counter()
    for i in V[vs[0]]:
        r=V[vs[0]][i]; cands=r['cands']; n=r['n_samples']
        if judge.unanimous(cands,n) or not any(V[v][i]['choices'] for v in vs): fin,_=judge.vote_decision(cands,n,0.5,'valid')
        else:
            tot=defaultdict(float)
            for v in vs:
                for k,x in dist(V[v][i]).items(): tot[k]+=x/len(vs)
            top=max(tot,key=tot.get) if tot else None
            if top is None or tot[top]<margin: fin,_=judge.vote_decision(cands,n,0.5,'valid')
            elif top=='0': fin={'type':'abstain'}
            elif top=='U': fin={'type':'unanswerable','answer':None}
            else: fin=cands[int(top)-1]['answer']
        c['abstain' if fin.get('type')=='abstain' else score.judge(fin,Rf[i])]+=1
    return (c['correct']-c['wrong'])/1000
dirs=sys.argv[1].split(',')
combos=[('list',),('point',),('list','point'),('list','point','point_think'),('point','point_think'),('list','list_think','point','point_think')]
for vs in combos:
  for margin in (0.0,0.5,0.7):
    out=[]
    for d in dirs:
        V={v:{json.loads(l)['id']:json.loads(l) for l in open(f'{d}/{v}/verdicts.jsonl')} for v in vs}
        out.append(run(V,vs,margin))
    print('%.4f'%(sum(out)/len(out)),vs,margin,['%.3f'%x for x in out])
