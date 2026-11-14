import json,sys,glob
sys.path.insert(0,'/tmp/claude-0/-home-user-ZC/67cec8b5-c044-570d-a1c2-4ef17349b57a/scratchpad')
from an1 import *
from an3 import plain
from collections import Counter,defaultdict
runs=['pva3k16','g1v3apf2','pvs1k16','pvs2k16','pvsoup16','pvsoup8','pv3bk16','g1v3bpf','g1v4s1pf','g1v4s2pf','g1v3apf','q3l70pf','q3a70pf','f2a3']
LL={}
for r in runs:
    try: LL[r]=load(r)
    except Exception as e: print('skip',r,e)
base=LL['pva3k16']
bad=[i for i,r in base.items() if verdict(plain(r)[0],R[i])!='correct']
anyright=defaultdict(dict)
for i in bad:
    for rn,L in LL.items():
        r=L.get(i)
        if not r: continue
        SS=r['samples'].get('samples') or []
        nr=sum(1 for s in SS if s.get('ok') and verdict(s['answer'],R[i])=='correct')
        fin=verdict(plain(r)[0],R[i])
        anyright[i][rn]=(nr,len(SS),fin[0])
for i in bad:
    print(i,base[i]['business_id'],base[i]['question'][:60])
    print('   ',' '.join(f"{rn}:{a}/{b}{c}" for rn,(a,b,c) in anyright[i].items()))
never=[i for i in bad if not any(a for a,b,c in anyright[i].values())]
print('never right in any run:',len(never),never)
# union oracle over all runs for all questions
allq=list(base.keys()); c=0
for i in allq:
    ok=False
    for rn,L in LL.items():
        r=L.get(i)
        if r and any(s.get('ok') and verdict(s['answer'],R[i])=='correct' for s in (r['samples'].get('samples') or [])): ok=True;break
    c+=ok
print('union oracle has-right over all runs:',c,'/',len(allq))
for rn,L in LL.items():
    cc=Counter(verdict(plain(r)[0],R[i]) for i,r in L.items() if i in R)
    print(rn,len(L),dict(cc))
