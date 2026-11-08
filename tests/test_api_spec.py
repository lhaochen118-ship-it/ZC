#!/usr/bin/env python3
"""src/api_spec.py: every name a program can call (dm.api of a real shop: periods, number helpers, industry helpers, the
shop's metrics) is in the spec with its parameters; every tool of src/tools.py is in it; 'industry' hides every industry
name; help() gives a signature + example for the common metrics; no entry has an unknown parameter kind for metric filters.
Usage: test_api_spec.py <train_split> <scratch>"""
import os, sys, inspect
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
import api_spec, dm, industry, tools


def main():
    split, S = sys.argv[1], sys.argv[2]
    import common
    c = common.ctx(split, os.environ.get("CACHE") or os.path.join(S, "cache"), os.path.join(S, "canon"), "biz_101", "compact")
    shop = c.shop("2026-01-01")
    ns = dm.api(shop, "2026-01-01")
    spec = api_spec.build_api_spec(); names = {e["name"] for e in spec}
    callables = {k for k, v in ns.items() if callable(v) and not isinstance(v, type) and k not in ("pd", "np", "math", "re", "Period", "Unanswerable")}
    callables -= {"D"}
    missing = sorted(callables - names)
    assert not missing, f"callable names missing from the spec: {missing}"
    tl = set(tools.bind_tools(shop, "2026-01-01"))
    assert tl <= names, f"tools missing: {tl - names}"
    hid = {e["name"] for e in api_spec.build_api_spec(hide=("industry",))}
    assert not (hid & (set(industry.INDUSTRY_NAMES) | set(industry.HELPERS))), "industry names not hidden"
    for n in ("revenue", "top_items", "n_orders", "item_qty", "share", "growth"):
        h = api_spec.help_text(spec, n)
        assert n + "(" in h and "example:" in h, (n, h)
    for e in spec:
        if e["group"] in ("metric", "industry"):
            for p in e["params"]: assert p["kind"] != "any" or p["name"] in ("bins",), (e["name"], p)
    print(f"spec {len(spec)} entries, {len(callables)} namespace callables covered, {len(tl)} tools")
    print("test_api_spec PASS")


if __name__ == "__main__":
    main()
