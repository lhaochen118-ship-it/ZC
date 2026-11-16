#!/usr/bin/env python3
import os, re, sys, ast, json, glob, argparse, unicodedata
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training"))


def slug(term):
    s = unicodedata.normalize("NFD", term.replace("đ", "d").replace("Đ", "D"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return (s or "ham")[:40]


def convert(c):
    import dm
    try: tree = ast.parse(c["program"])
    except SyntaxError: return None
    asg = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ANSWER" for t in n.targets)]
    if len(asg) != 1 or not isinstance(asg[0].value, ast.Dict) or len(tree.body) != 1: return None
    d = asg[0].value; kv = {k.value: v for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant)}
    if "answer" not in kv or not isinstance(kv.get("type"), ast.Constant): return None
    expr = kv["answer"]; src = c["program"]
    periods = [n for n in ast.walk(expr) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in dm.PERIOD_API]
    segs = {ast.get_source_segment(src, n) for n in periods}
    if len(segs) != 1: return None
    per = segs.pop()

    seg = ast.get_source_segment(src, expr)

    class R(ast.NodeTransformer):
        def visit_Call(self, n):
            if isinstance(n.func, ast.Name) and n.func.id in dm.PERIOD_API and ast.get_source_segment(seg, n) == per:
                return ast.copy_location(ast.Name(id="p", ctx=ast.Load()), n)
            return self.generic_visit(n)
    body = R().visit(ast.parse(seg, mode="eval")).body
    if any(isinstance(n, ast.Name) and n.id == "TODAY" for n in ast.walk(body)): return None
    name = slug(c["term"])
    return f"def {name}(p):\n    return {ast.unparse(body)}", name, per, kv["type"].value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cards"); ap.add_argument("out"); ap.add_argument("--split", default=os.path.join(ROOT, "data", "raw", "train", "train"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import contrast_job, judge
    cards = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.cards, "**", "cards_newterm*.jsonl"), recursive=True)) for l in open(f, encoding="utf-8") if l.strip()]
    out, st = [], Counter()
    for c in cards:
        r = convert(c)
        if r is None: st["not_convertible"] += 1; continue
        code, name, per, typ = r
        ok, ans = contrast_job.run_program(a.split, c["business_id"], c["as_of_date"], code + f'\nANSWER = {{"type": "{typ}", "answer": {name}({per})}}')
        if not ok or judge.canon(ans) != judge.canon(c.get("answer")): st["mismatch"] += 1; continue
        st["kept"] += 1
        out.append({"kind": "unit", "bid": c["business_id"], "term": c["term"], "definition": c["definition"][:400], "section": "thuật ngữ mới",
                    "name": name, "signature": f"{name}(p)", "doc": c["definition"][:300], "triggers": [c["term"]], "code": code,
                    "plan": f"định nghĩa = {ast.parse(code).body[0].body[0].value.func.id if isinstance(ast.parse(code).body[0].body[0].value, ast.Call) and isinstance(ast.parse(code).body[0].body[0].value.func, ast.Name) else 'biểu thức'} trên kỳ p",
                    "src": "newterm_prog", "qid": c.get("id"), "verified": {"agree": "checked program"}})
    with open(os.path.join(a.out, "units_teacher.jsonl"), "w", encoding="utf-8") as w:
        for u in out: w.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(json.dumps({"cards": len(cards), **st}))


if __name__ == "__main__":
    main()
