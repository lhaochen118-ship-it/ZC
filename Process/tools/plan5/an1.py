import json,gzip,sys,os
sys.path.insert(0,'/home/user/ZC/src')
import score
R={}
for l in open('/home/user/ZC/data/raw/public_gt.jsonl'):
    r=json.loads(l); R[r['id']]=r
def load(run):
    import glob
    p=glob.glob(f'/home/user/ZC/Process/runs/{run}/**/logs.jsonl.gz',recursive=True)[0]
    L={}
    for l in gzip.open(p,'rt',encoding='utf-8'):
        r=json.loads(l); L[r['id']]=r
    return L
def verdict(fin,ref):
    if not fin or fin.get('type')=='abstain': return 'abstain'
    if ref.get('type')=='abstain': return 'ref_abstain'
    try: return score.judge(fin,{'type':ref['type'],'answer':ref.get('answer'),'alias':ref.get('alias')})
    except Exception as e: return 'err:'+type(e).__name__
if __name__=='__main__':
    L=load(sys.argv[1])
    from collections import Counter
    c=Counter(); bad=[]
    for i,r in L.items():
        if i not in R: continue
        v=verdict(r.get('final'),R[i]); c[v]+=1
        if v!='correct': bad.append((i,v))
    print(c)
    json.dump(bad,open(f'{os.path.dirname(os.path.abspath(__file__))}/bad_{sys.argv[1]}.json','w'))
