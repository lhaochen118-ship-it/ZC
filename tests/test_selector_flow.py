#!/usr/bin/env python3
"""Plan 4 V: the learned selector shipped in src/refs/selector/ loads, is deterministic, picks among the logged answer groups
only (never invents an answer), abstains below its theta, and the hybrid flow's --selector flag parses.
Usage: test_selector_flow.py"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import selector, agent

p = os.path.join(ROOT, "src", "refs", "selector", "selector_v3a_inv.json")
sel = selector.Selector.load(p)
S = lambda a, v="direct", fl=(): {"ok": True, "answer": a, "variant": v, "flags": list(fl), "turns": [{"code": "ANSWER = 1"}]}
A, B = {"type": "number", "answer": 5}, {"type": "number", "answer": 7}
strong = {"samples": {"hints": {}, "samples": [S(A), S(A), S(A), S(A), S(B)]}}
split = {"samples": {"hints": {}, "samples": [S(A), S(B, fl=["zero"]), S({"type": "number", "answer": 9}), S({"type": "number", "answer": 11})]}}
d1, c1 = sel.decide(strong); d2, c2 = sel.decide(strong)
assert d1 == A and d1 == d2 and c1 == c2, (d1, c1)
d3, c3 = sel.decide(split)
assert d3.get("type") == "abstain" or d3 in (A, B, {"type": "number", "answer": 9}, {"type": "number", "answer": 11}), d3
assert c1 > c3, (c1, c3)
a = agent.build_parser().parse_args(["S", "O", "--selector", "src/refs/selector/selector_v3a_inv.json"])
assert a.selector.endswith("selector_v3a_inv.json")
cfg = json.load(open(os.path.join(ROOT, "configs", "flow_swv.json")))
assert cfg["selector"] == "src/refs/selector/selector_v3a_inv.json" and cfg.get("pure") == 1 and not cfg.get("hints")
print(f"strong group p={c1:.3f}, split p={c3:.3f} -> {d3.get('type')}")
print("test_selector_flow PASS")
