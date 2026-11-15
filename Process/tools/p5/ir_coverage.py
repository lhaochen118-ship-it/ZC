#!/usr/bin/env python3
"""P5 plan I01 gate (EXP-070, CPU): share of the verified training programs (src/refs/bank/examples.jsonl) expressible in the typed IR
{metric, entities, time_filter, group_by, op, aggregation, topk}: ANSWER = {"type": T, "answer": <one toolkit metric call>} whose
arguments are period constructors (month(...), quarter(TODAY)...), constants / TODAY and constant keyword filters (item=..., by=...),
optionally inside round(). Anything else (arithmetic between metrics, comprehensions, several statements, subscripts) needs the
Python fallback. Gate of the plan: build the compiler only when coverage >= 80 %."""
import ast, json, sys, collections
sys.path.insert(0, "/home/user/ZC/src")
import api_spec, dm, inspect
G = api_spec.GROUP_OF
METRICS = {n for n, f in inspect.getmembers(dm.Shop, inspect.isfunction) if not n.startswith("_") and G.get(n) not in ("period", "number")}


def const_like(n):
    if isinstance(n, ast.Constant): return True
    if isinstance(n, ast.Name) and n.id in ("TODAY", "True", "False", "None"): return True
    if isinstance(n, ast.UnaryOp) and isinstance(n.operand, ast.Constant): return True
    if isinstance(n, (ast.List, ast.Tuple)): return all(const_like(x) for x in n.elts)
    return False


def period_like(n):
    if const_like(n): return True
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and G.get(n.func.id) in ("period", "number") and \
        all(period_like(x) for x in n.args) and all(period_like(k.value) for k in n.keywords)


def ir_ok(code):
    try: t = ast.parse(code)
    except SyntaxError: return False, "syntax"
    if len(t.body) != 1 or not isinstance(t.body[0], ast.Assign): return False, "statements"
    v = t.body[0].value
    if not isinstance(v, ast.Dict): return False, "not a dict"
    ans = next((val for k, val in zip(v.keys, v.values) if isinstance(k, ast.Constant) and k.value == "answer"), None)
    if ans is None: return False, "no answer"
    if isinstance(ans, ast.Call) and isinstance(ans.func, ast.Name) and ans.func.id == "round" and ans.args: ans = ans.args[0]
    if const_like(ans): return True, "constant"
    if not (isinstance(ans, ast.Call) and isinstance(ans.func, ast.Name)): return False, type(ans).__name__
    f = ans.func.id
    if f not in METRICS: return False, f"not a toolkit metric: {f}"
    if all(period_like(x) for x in ans.args) and all(period_like(k.value) for k in ans.keywords): return True, "one call"
    return False, "nested / computed arguments"


rows = [json.loads(l) for l in open("/home/user/ZC/src/refs/bank/examples.jsonl", encoding="utf-8") if l.strip()]
why = collections.Counter(); ok = 0
for r in rows:
    o, w = ir_ok(r.get("code") or ""); ok += o; why[("IR " if o else "fallback ") + w] += 1
print(f"{len(rows)} verified training programs; IR-expressible {ok} = {ok / len(rows):.1%}")
for k, v in why.most_common(10): print(f"  {k}: {v}")
