import json,sys,glob,gzip
sys.path.insert(0,'/home/user/ZC/src'); sys.path.insert(0,'/tmp/claude-0/-home-user-ZC/67cec8b5-c044-570d-a1c2-4ef17349b57a/scratchpad')
import score
from an3 import canon
from collections import Counter
A={}
for f in glob.glob('/home/user/ZC/data/raw/train/**/answers.jsonl',recursive=True):
    for l in open(f):
        r=json.loads(l); A[r['id']]=r
print('train answers',len(A))
tot=Counter()
for run in ['c1a','c1b','rq5_tr_dev_para_a','rq5_tr_dev_para_b','v2v4s1a','v2v4s1b','v2v4s2a','v2v4s2b','rq5_tr_dev_all','ptr2']:
    for p in glob.glob(f'/home/user/ZC/Process/runs/{run}/**/logs.jsonl.gz',recursive=True):
        c=Counter()
        for l in gzip.open(p,'rt',encoding='utf-8'):
            r=json.loads(l); ref=A.get(r['id'].split('~')[0])
            if not ref: c['noref']+=1; continue
            SS=(r['samples'].get('samples') or []) if isinstance(r.get('samples'),dict) else []
            g={}
            for s in SS:
                if s.get('ok'): g.setdefault(canon(s['answer']),s['answer'])
            v=[score.judge(a,ref)=='correct' for a in g.values()]
            c['q']+=1; c['samples']+=len(SS)
            fin=r.get('final') or {}
            fv='abstain' if fin.get('type') in (None,'abstain') else score.judge(fin,ref); c['fin_'+fv]+=1
            if len(g)>=2 and any(v) and not all(v): c['mixed(right+wrong groups)']+=1
            elif len(g)>=1 and not any(v): c['all_wrong']+=1
            elif len(g)==0: c['no_valid']+=1
            else: c['all_right']+=1
            c['wrong_groups']+=sum(1 for x in v if not x); c['right_groups']+=sum(v)
        print(run,p.split('/')[-2],dict(c)); tot.update(c)
print('TOTAL',dict(tot))
