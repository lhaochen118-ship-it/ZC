#!/usr/bin/env python3
"""SFT records follow the inference prompt (docs/08 §5.3): system = # ROLE, # API, # OUTPUT FORMAT; user = # BUSINESS
(+ # EXAMPLES, # HINTS) + # QUESTION in this order; the last message is the assistant label (<plan> + <code>) and the only
target; the label parses with prompting.parse_output and is accepted by its shop's grammar; hints carry no metric name.
Usage: test_sft_format.py <train_split> <scratch> <file.jsonl>[,...] [--n 200]"""
import os, sys, json, random, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))


def main():
    split, S, files = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
    import common, prompting, grammar, api_spec
    names = {e["name"] for e in api_spec.build_api_spec() if e["group"] in ("metric", "industry")}
    cache, canon = os.environ.get("CACHE") or os.path.join(S, "cache"), os.path.join(S, "canon")
    rng = random.Random(20261108); bad = []; chk = {}
    for f in files:
        rows = common.jl(f)
        for r in rng.sample(rows, min(n, len(rows))):
            m = r["messages"]
            if m[0]["role"] != "system" or m[-1]["role"] != "assistant" or m[1]["role"] != "user": bad.append((r["id"], "roles")); continue
            s = m[0]["content"]
            if not (s.index("# ROLE") < s.index("# API") < s.index("# OUTPUT FORMAT")): bad.append((r["id"], "system order"))
            u = m[1]["content"]; pos = [u.find(x) for x in ("# BUSINESS", "# EXAMPLES", "# HINTS", "# QUESTION")]
            seq = [p for p in pos if p >= 0]
            if pos[0] != 0 or pos[3] < 0 or seq != sorted(seq): bad.append((r["id"], "user order", pos))
            h = u[u.find("# HINTS"):u.find("# QUESTION")] if pos[2] >= 0 else ""
            leak = [x for x in names if re.search(r"\b" + x + r"\(", h)]
            if leak: bad.append((r["id"], "hint names a metric", leak[:3]))
            out = prompting.parse_output(m[-1]["content"])
            if not out["code"] or out["escape"] or not out["plan"]: bad.append((r["id"], "label format")); continue
            b = r["business_id"]
            if b not in chk:
                c = common.ctx(split, cache, canon, b, "compact"); chk[b] = grammar.Checker(c.grammar())
            if not chk[b](m[-1]["content"]): bad.append((r["id"], "label not in the shop grammar", out["code"][:100]))
    for x in bad[:15]: print("PROBLEM", x)
    print(f"checked {sum(min(n, len(common.jl(f))) for f in files)} records of {len(files)} files")
    print("test_sft_format " + ("PASS" if not bad else "FAIL")); sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
