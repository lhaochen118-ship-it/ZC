#!/usr/bin/env python3
import os, sys, ast, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def fn_help(code, spec_by):
    import re, dm, inspect
    lines = []
    for f in dict.fromkeys(re.findall(r"\b([a-z_][a-z0-9_]*)\(", code)):
        e = spec_by.get(f)
        if e is None: continue
        fn = getattr(dm.Shop, f, None) or dm.PERIOD_API.get(f) or dm.NUM_API.get(f)
        doc = " ".join(((fn.__doc__ if fn is not None else "") or e["doc"] or "").split())[:260]
        lines.append(f"- {e['signature']}: {doc}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("programs"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    import api_spec, prompting, teacher_prompt, precheck
    by = api_spec.by_name(api_spec.build_api_spec())
    cards, out = {}, []
    for r in common.jl(a.programs):
        b = r["business_id"]
        if b not in cards:
            _, C, _, _ = common.canon(a.split, a.cache, a.canon_dir, b); cards[b] = prompting.compact_card(C, "", max_chars=1200)
        per = precheck.first_period(ast.parse(r["code"]), r["as_of_date"])
        code = r["code"] + (f"\n(kỳ trong chương trình: {per[0]} .. {per[1]})" if per else "")
        out.append({"id": r["id"], "messages": teacher_prompt.messages(cards[b], r["as_of_date"], code, r["answer"], fn_help(r["code"], by), a.n)})
    common.write_jl(a.out, out)
    print(f"teacher prompts: {len(out)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
