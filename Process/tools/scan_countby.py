"""Signal scan for a candidate precheck: the question counts something (hints.counted) but the program calls a by= toolkit
function with a money metric (explicit or the default). Per sample (last turn with code): right / wrong vs the reference.
Dev ids only: pub_full, tr_dev_all, d3sel, tr_dev_para."""
import os, sys, json, ast, glob
ROOT = "/home/user/ZC"; sys.path.insert(0, os.path.join(ROOT, "src"))
import score, api_spec, inspect, dm, industry
MONEY = {"revenue", "amount", "aov", "profit", "commission"}
DEF = {}
for n, opts in api_spec.BY.items():
    f = getattr(dm, n, None) or getattr(industry, n, None)
    p = inspect.signature(f).parameters.get("by") if f else None
    DEF[n] = p.default if p is not None and p.default is not inspect._empty else opts[0]
ids = set()
for s in ("pub_full", "tr_dev_all", "d3sel", "tr_dev_para"):
    ids |= {l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()}
refs = {}
for p in (os.path.join(ROOT, "data/raw/public_gt.jsonl"), os.path.join(ROOT, "data/raw/train/train/answers.jsonl"), os.path.join(ROOT, "tests/splits/d3sel.answers.jsonl")):
    refs.update(score.load(p))
def money_calls(code):
    try: t = ast.parse(code)
    except Exception: return []
    out = []
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in api_spec.BY:
            by = DEF[n.func.id]
            for kw in n.keywords:
                if kw.arg == "by": by = kw.value.value if isinstance(kw.value, ast.Constant) else None
            if by in MONEY: out.append((n.func.id, by))
    return out
tot = {}
ex = []
for run in sys.argv[1].split(","):
    f = os.path.join(run, "logs.jsonl")
    if not os.path.isfile(f): continue
    for l in open(f):
        r = json.loads(l)
        if r["id"] not in ids: continue
        S = r.get("samples") or {}; h = S.get("hints") or {}
        if not h.get("counted"): continue
        ref = refs.get(r["id"].split("~")[0])
        if ref is None: continue
        for s in S.get("samples") or []:
            turns = [t for t in s.get("turns") or [] if t.get("code")]
            if not turns: continue
            mc = money_calls(turns[-1]["code"])
            if not mc or s.get("answer") is None: continue
            try: ok = score.judge(s["answer"], ref) == "correct"
            except Exception: ok = False
            key = (mc[0][0], mc[0][1]); tot.setdefault(key, [0, 0])[0 if ok else 1] += 1
            if len(ex) < 400: ex.append((os.path.basename(run), r["id"], ok, h.get("counted"), r["question"][:90], turns[-1]["code"][:160]))
for k, (a, b) in sorted(tot.items(), key=lambda x: -sum(x[1])): print(k, "right", a, "wrong", b)
if "--ex" in sys.argv:
    for e in ex: print(e)
