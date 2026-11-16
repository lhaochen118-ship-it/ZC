#!/usr/bin/env python3
import os, sys, ast, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import api_spec, dm

PERIOD = set(dm.PERIOD_API) | {n for n, r in api_spec.RETURNS.items() if r in ("period", "periods")}
NUMBER = set(dm.NUM_API)
SKIP = PERIOD | {"len", "int", "float", "round", "sum", "list", "sorted", "min", "max", "abs", "str", "dict", "set", "key",
                          "isinstance", "any", "all", "zip", "range", "enumerate", "print"}
PARAM_KINDS = {"item", "group", "customer", "customer_group", "staff", "supplier", "channel", "payment", "status", "vehicle", "brand",
               "size", "line", "unit", "shift", "weekday", "by", "text"}


def _name(f):
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _const(v):
    try: return ast.literal_eval(v)
    except Exception: return None


def spec_of(code):
    try: tree = ast.parse(code or "")
    except SyntaxError: return {}
    periods, measure, filters, numbers, atype = [], [], {}, [], None
    inside_period = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node.func) in PERIOD and id(node) not in inside_period:
            e = ast.unparse(node)
            if e not in periods: periods.append(e)
            for sub in ast.walk(node):
                if sub is not node: inside_period.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            n = _name(node.func)
            if id(node) in inside_period or n is None: continue
            if n not in SKIP and n not in measure: measure.append(n)
            for kw in node.keywords:
                if kw.arg and api_spec.KIND.get(kw.arg) in PARAM_KINDS | {"bool", "bool_or_none", "int", "number"}:
                    v = _const(kw.value)
                    if v is not None and v not in filters.setdefault(kw.arg, []): filters[kw.arg].append(v)
            if n not in PERIOD:
                for arg in node.args[1:]:
                    v = _const(arg)
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and v not in numbers: numbers.append(v)
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "type" and isinstance(v, ast.Constant): atype = v.value
    spec = {}
    if periods: spec["period"] = periods if len(periods) > 1 else periods[0]
    if measure: spec["measure"] = measure
    if filters: spec["params"] = {k: v[0] if len(v) == 1 else v for k, v in filters.items()}
    if numbers: spec["numbers"] = numbers
    if atype: spec["answer_type"] = atype
    return spec


def spec_block(spec):
    return "<spec>" + json.dumps(spec, ensure_ascii=False, separators=(", ", ": "), default=str) + "</spec>\n"


def add_spec(r):
    lab = (r.get("messages") or [{}])[-1]
    if lab.get("role") != "assistant" or "<spec>" in lab.get("content", "") or not r.get("program"): return r, False
    sp = spec_of(r["program"])
    if not sp: return r, False
    return dict(r, messages=r["messages"][:-1] + [dict(lab, content=spec_block(sp) + lab["content"])]), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs"); ap.add_argument("out_dir"); ap.add_argument("--suffix", default="s")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    for f in a.inputs.split(","):
        t0 = time.time(); rows = []; n = 0
        for r in common.jl(f):
            r, ok = add_spec(r); rows.append(r); n += ok
        out = os.path.join(a.out_dir, os.path.basename(f).replace(".jsonl", a.suffix + ".jsonl"))
        common.write_jl(out, rows)
        print(f"{f}: {n} / {len(rows)} labels with a <spec> -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
