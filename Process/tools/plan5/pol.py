import sys, json
sys.path.insert(0,'/home/user/ZC/src'); sys.path.insert(0,'/home/user/ZC/tests')
import judge, score
from collections import Counter
R=score.load('/home/user/ZC/data/raw/public_gt.jsonl')
def v(fin,ref):
    if not fin or fin.get('type')=='abstain': return 'abstain'
    return score.judge(fin,ref)
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f)]
    mode=rows[0]['mode']
    for only_split in (False,True):
      for tau in (0.5,0.67,0.8,1.0):
        for den in ('all','valid'):
          c=Counter()
          for r in rows:
            ch=r['choices'] if mode=='list' else {int(k):x for k,x in r['choices'].items()}
            cands=r['cands']; n=r['n_samples']
            unan = len(cands)==1 and cands[0]['n']==n
            if only_split and unan: fin,_=judge.vote_decision(cands,n,0.5,den)
            else: fin,_,_=judge.decide(cands,n,ch,mode,tau=tau,denom=den)
            c[v(fin,R[r['id']])]+=1
          print(f.split('/')[-2],'only_split' if only_split else 'all     ',tau,den,(c['correct']-c['wrong'])/1000,dict(c))
