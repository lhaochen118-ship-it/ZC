"""v2 I4: several functions per rulebook row (lrl_gen to_cards on {'functions': [...]}, select_units multi keeps every agreed signature)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import lrl_gen
meta = {"bid": "b1", "term": "Dòng máy", "definition": "dòng máy = phần tên trước GB", "section": "x", "rulebook": "r"}
fs = {"plan": "p", "functions": [{"name": "ban_theo_dong_may", "signature": "ban_theo_dong_may(p)", "doc": "d", "triggers": ["dòng máy"], "code": ["def ban_theo_dong_may(p):", "    return 1"]},
                                 {"name": "so_dong_may", "signature": "so_dong_may(p)", "doc": "d", "triggers": [], "code": ["def so_dong_may(p):", "    return 2"]}]}
cards = lrl_gen.to_cards("unit", [(meta, "")], [json.dumps(fs)], set())
assert [c["name"] for c in cards] == ["ban_theo_dong_may", "so_dong_may"], cards
one = lrl_gen.to_cards("unit", [(meta, "")], [json.dumps({"plan": "p", "name": "f", "signature": "f(p)", "doc": "", "triggers": [], "code": ["def f(p):", "    return 1"]})], set())
assert [c["name"] for c in one] == ["f"]
# 3 samples: sig A agreed by 3, sig B by 2, sig C by 1
cands = []
for i, sig in enumerate(["A", "A", "B", "A", "B", "C"]):
    c = dict(cards[0] if sig != "B" else cards[1]); c.update(_ok=True, _sig=sig, _nerr=0); cands.append(c)
out, fails = lrl_gen.select_units(cands, 2, multi=1)
assert len(out) == 2 and {o["verified"]["agree"].split("/")[0] for o in out} == {"3", "2"}, out
out1, _ = lrl_gen.select_units(cands, 2)
assert len(out1) == 1
# same name for two different agreed signatures -> renamed
cands2 = [dict(cards[0], _ok=True, _sig=s, _nerr=0) for s in ["A", "A", "B", "B"]]
out2, _ = lrl_gen.select_units(cands2, 2, multi=1)
assert len({o["name"] for o in out2}) == 2 and all(f"def {o['name']}(" in o["code"] for o in out2), out2
print("units_multi PASS")
