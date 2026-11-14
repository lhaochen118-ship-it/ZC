import json,sys,glob
sys.path.insert(0,'/tmp/claude-0/-home-user-ZC/67cec8b5-c044-570d-a1c2-4ef17349b57a/scratchpad')
from an1 import *
L=load('pva3k16')
cards={}
for f in glob.glob('/home/user/ZC/src/refs/lrl/*.jsonl'):
    for l in open(f):
        try: c=json.loads(l); cards[c.get('id')]=(f.split('/')[-1],c)
        except Exception as e: pass
print(len(cards))
for i in ['q_735ea0888aab','q_a7dd46bc4b38','q_e0ac213834f6','q_05cde771f4c3','q_792e1070baaf','q_76bdadbeb802','q_9aa0981148ef']:
    r=L[i]; print('='*80); print(r['question']); print('rules',r['samples'].get('rules'))
    for rid in r['samples'].get('rules') or []:
        if rid in cards: print('   ',cards[rid][0],json.dumps(cards[rid][1],ensure_ascii=False)[:600])
    t=r['samples']['samples'][0]['turns'][0]['text']; print('  OUT:',t[-500:])
