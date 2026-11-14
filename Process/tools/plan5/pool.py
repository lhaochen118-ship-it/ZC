import sys,json,gzip
def load(p):
    op=gzip.open if p.endswith('gz') else open
    return {json.loads(l)['id']:json.loads(l) for l in op(p,'rt')}
out=sys.argv[1]; srcs=[load(p) for p in sys.argv[2:]]
with open(out,'w') as f:
    for i,r in srcs[0].items():
        ss=[]
        for s in srcs:
            if i in s: ss+=(s[i]['samples'].get('samples') or [])
        r=dict(r); r['samples']=dict(r['samples'], samples=ss)
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
