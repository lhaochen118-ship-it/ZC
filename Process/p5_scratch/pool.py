import sys,json,gzip
sys.path.insert(0,'/home/user/ZC/src'); sys.path.insert(0,'/home/user/ZC/Process/tools/p5')
import score, agent
from new7_score import vote
gt=score.load('/home/user/ZC/data/raw/public_gt.jsonl')
P={'h0a':'Process/runs/v2/h0a/h0a/flow_p5_final_h/solver_logs.jsonl.gz','h0a2':'Process/runs/v2/h0a2/h0a2/flow_p5_final_h/solver_logs.jsonl.gz',
   'e8':'Process/runs/p5_results/e8/flow_e8/solver_logs.jsonl.gz','e7':'Process/runs/p5_results/e7/flow_e7/solver_logs.jsonl.gz',
   'p02t':'Process/runs/p5_results/p02t/flow_p02t/solver_logs.jsonl.gz'}
L={}
for k,p in P.items():
    try: L[k]={r['id']:(r.get('samples') or {}).get('samples') or [] for r in map(json.loads,gzip.open(p,'rt'))}
    except Exception as e: print(k,e)
def pa(names,th=0.5,w=None):
    sub={}
    for q in gt:
        smp=[]
        for n in names:
            s=L[n].get(q,[]); smp+= s
        sub[q]=vote(smp,theta=th)
    return score.score(gt,sub)[0]['PA']
for combo in [['h0a'],['h0a2'],['e8'],['h0a','h0a2'],['h0a','e8'],['h0a2','e8'],['h0a','h0a2','e8'],['h0a','p02t'],['h0a','e8','p02t']]:
    if all(c in L for c in combo): print(combo,[pa(combo,t) for t in (0.4,0.5,0.6)])
def oracle(names):
    n=0
    for q,g in gt.items():
        ok=False
        for nm in names:
            for s in L[nm].get(q,[]):
                if s.get('ok') and isinstance(s.get('answer'),dict):
                    r,v=score.score({q:g},{q:s['answer']})
                    if v[q]=='correct': ok=True;break
            if ok: break
        n+=ok
    return n
for c in [['h0a'],['h0a','h0a2'],['h0a','e8'],['h0a','h0a2','e8']]: print('oracle',c,oracle(c))
