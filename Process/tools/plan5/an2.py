import json,sys
sys.path.insert(0,'/tmp/claude-0/-home-user-ZC/67cec8b5-c044-570d-a1c2-4ef17349b57a/scratchpad')
from an1 import *
run=sys.argv[1]
L=load(run); P=load('p2r17ps1')
from collections import Counter
bybiz=Counter(); n=0
for i,r in L.items():
    v=verdict(r.get('final'),R[i])
    if v=='correct': continue
    n+=1; bybiz[r['business_id']]+=1
    lg=r['samples']; SS=lg.get('samples') or []
    anyright=sum(1 for s in SS if s.get('answer') and verdict(s['answer'],R[i])=='correct')
    print('='*100); print(n,i,r['business_id'],r['as_of_date'],v,'| nsamp',len(SS),'right',anyright)
    print('Q:',r['question'])
    print('REF:',json.dumps(R[i],ensure_ascii=False)[:300])
    print('FIN:',json.dumps(r.get('final'),ensure_ascii=False)[:300],'conf',r.get('conf'))
    print('VOTES:',str(r.get('votes'))[:400])
    seen=set()
    for s in SS:
        t=(s.get('turns') or [{}])[-1]; code=(t.get('code') or '').strip()
        if code in seen: continue
        seen.add(code)
        ok=s.get('answer') and verdict(s['answer'],R[i])=='correct'
        print('  [%s%s] %s'%(s.get('variant'),' OK' if ok else '',code[:500].replace('\n','\n      ')))
        if len(seen)>=4: break
    p=P.get(i)
    if p:
        ps=(p['samples'].get('samples') or [])
        for s in ps:
            if s.get('answer') and verdict(s['answer'],R[i])=='correct':
                t=(s.get('turns') or [{}])[-1]
                print('  P-HINTS:',json.dumps(p['samples'].get('hints'),ensure_ascii=False)[:500])
                print('  P-CODE:',(t.get('code') or '')[:500].replace('\n','\n      ')); break
print(bybiz)
